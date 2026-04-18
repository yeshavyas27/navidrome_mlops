"""
Navidrome - Session Dataset Builder (GRU4Rec + SessionKNN format)
Reads sessions chunks from Swift, builds chronological train/eval splits.
Run: source ~/.chi_auth.sh && python3 pipeline/build_dataset.py
"""
import os, json, subprocess, io, ast
import pandas as pd
import numpy as np
from datetime import datetime, timezone

CONTAINER    = "navidrome-bucket-proj05"
RUN_ID       = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
VERSION      = f"v{datetime.now(timezone.utc).strftime('%Y%m%d')}-001"
HOLDOUT_FRAC = 0.15

AUTH_ARGS = [
    "--os-auth-url", os.environ["OS_AUTH_URL"],
    "--os-auth-type", "v3applicationcredential",
    "--os-application-credential-id", os.environ["OS_APPLICATION_CREDENTIAL_ID"],
    "--os-application-credential-secret", os.environ["OS_APPLICATION_CREDENTIAL_SECRET"],
]

def swift_run(args):
    return subprocess.run(["swift"] + AUTH_ARGS + args, capture_output=True, text=True)

def swift_upload(local, name):
    swift_run(["upload", "--object-name", name, CONTAINER, local])
    print(f"  uploaded -> {name} ({os.path.getsize(local)/1e6:.1f} MB)")

def swift_upload_bytes(data, name):
    tmp = f"/tmp/ds_{RUN_ID}.bin"
    with open(tmp, "wb") as f:
        f.write(data)
    swift_upload(tmp, name)
    os.remove(tmp)

def swift_download(name, local):
    swift_run(["download", "--output", local, CONTAINER, name])

def list_objects(prefix):
    r = swift_run(["list", CONTAINER, "--prefix", prefix])
    return [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]

def load_sessions(max_chunks=5):
    print("\n[STEP 1] Loading 30Music sessions...")
    chunks = list_objects("processed/30music/chunks/sessions/")
    print(f"  found {len(chunks)} session chunks")
    if not chunks:
        print("  sessions not ready yet")
        return pd.DataFrame()
    chunks = chunks[:max_chunks]
    dfs = []
    for i, chunk in enumerate(chunks):
        local = f"/tmp/sess_{i}.parquet"
        swift_download(chunk, local)
        try:
            df = pd.read_parquet(local, engine="pyarrow")
            dfs.append(df)
        except Exception as e:
            print(f"  skip: {e}")
        finally:
            if os.path.exists(local):
                os.remove(local)
        if (i+1) % 10 == 0:
            print(f"  loaded {i+1}/{len(chunks)} chunks...")
    if not dfs:
        return pd.DataFrame()
    sessions = pd.concat(dfs, ignore_index=True)
    print(f"  sessions loaded: {len(sessions):,}")
    print(f"  columns: {list(sessions.columns)}")
    return sessions

def parse_sessions(sessions_df):
    print("\n[STEP 2] Parsing session sequences...")
    import ast, json

    def extract_session(row):
        try:
            relations = row["relations"]
            if isinstance(relations, str):
                try:
                    relations = json.loads(relations.replace("'", '"'))
                except:
                    relations = ast.literal_eval(relations)
            if not isinstance(relations, dict):
                return None
            subjects = relations.get("subjects", [])
            objects  = relations.get("objects", [])
            if not subjects or not objects:
                return None
            user_id = None
            for s in subjects:
                if s.get("type") == "user":
                    user_id = s.get("id")
                    break
            if user_id is None:
                return None
            track_ids   = []
            play_ratios = []
            for obj in objects:
                if obj.get("type") == "track":
                    tid = obj.get("id")
                    if tid is not None:
                        track_ids.append(tid)
                        ratio = obj.get("playratio", 1.0)
                        try:
                            ratio = float(ratio) if ratio is not None else 1.0
                        except:
                            ratio = 1.0
                        play_ratios.append(min(ratio, 2.0))
            if len(track_ids) < 2:
                return None
            return {
                "session_id":  str(row["id"]),
                "user_id":     int(user_id),
                "timestamp":   pd.to_datetime(row["timestamp"], unit="s", errors="coerce"),
                "track_ids":   track_ids,
                "play_ratios": play_ratios,
                "session_len": len(track_ids),
                "source":      "30music_sessions"
            }
        except Exception:
            return None

    print(f"  processing {len(sessions_df):,} sessions...", flush=True)
    results = sessions_df.apply(extract_session, axis=1)
    valid = [r for r in results if r is not None]
    skipped = len(sessions_df) - len(valid)
    df = pd.DataFrame(valid)
    print(f"  parsed: {len(df):,} sessions, {skipped:,} skipped")
    if len(df) > 0:
        print(f"  avg session length: {df['session_len'].mean():.1f} tracks")
        print(f"  unique users: {df['user_id'].nunique():,}")
    return df

def build_split(sessions_df):
    print("\n[STEP 3] Building vocab and train/eval split...")
    np.random.seed(42)
    all_tracks = []
    for tids in sessions_df["track_ids"]:
        all_tracks.extend(tids)
    unique_tracks = sorted(set(all_tracks))
    track2idx = {tid: i+1 for i, tid in enumerate(unique_tracks)}
    idx2track = {i+1: tid for tid, i in track2idx.items()}
    print(f"  vocab size: {len(track2idx):,} tracks")
    sessions_df = sessions_df.copy()
    sessions_df["item_idxs"] = sessions_df["track_ids"].apply(
        lambda tids: [track2idx[t] for t in tids if t in track2idx])
    sessions_df["timestamp"] = pd.to_datetime(sessions_df["timestamp"], utc=True, errors="coerce")
    sessions_df = sessions_df.sort_values("timestamp").reset_index(drop=True)
    all_users = sessions_df["user_id"].unique()
    holdout = set(np.random.choice(all_users, size=int(len(all_users)*HOLDOUT_FRAC), replace=False))
    train_pool = sessions_df[~sessions_df["user_id"].isin(holdout)]
    eval_pool  = sessions_df[sessions_df["user_id"].isin(holdout)]
    cutoff_idx = int(len(train_pool) * 0.8)
    train_cutoff = train_pool.iloc[cutoff_idx]["timestamp"]
    train_final  = train_pool[train_pool["timestamp"] <= train_cutoff]
    val_extra    = train_pool[train_pool["timestamp"] > train_cutoff]
    eval_combined = pd.concat([eval_pool, val_extra], ignore_index=True)
    print(f"  train: {len(train_final):,} | eval: {len(eval_combined):,}")
    print(f"  holdout users: {len(holdout):,}")
    return train_final, eval_combined, track2idx, idx2track

def upload_dataset(train_df, eval_df, track2idx, idx2track):
    print(f"\n[STEP 4] Uploading dataset {VERSION}...")
    prefix = f"datasets/{VERSION}"
    tmp = "/tmp/train_sessions.parquet"
    train_df.to_parquet(tmp, index=False, engine="pyarrow")
    swift_upload(tmp, f"{prefix}/train_sessions.parquet")
    os.remove(tmp)
    tmp = "/tmp/eval_sessions.parquet"
    eval_df.to_parquet(tmp, index=False, engine="pyarrow")
    swift_upload(tmp, f"{prefix}/eval_sessions.parquet")
    os.remove(tmp)
    vocab = {
        "track2idx": {str(k): v for k, v in track2idx.items()},
        "idx2track": {str(k): v for k, v in idx2track.items()}
    }
    swift_upload_bytes(json.dumps(vocab).encode(), f"{prefix}/vocab.json")
    manifest = {
        "version_id":            VERSION,
        "run_id":                RUN_ID,
        "created_at":            datetime.now(timezone.utc).isoformat(),
        "format":                "session_sequences",
        "models":                ["GRU4Rec", "SessionKNN"],
        "leakage_check":         "chronological_strict_user_holdout",
        "holdout_user_fraction": HOLDOUT_FRAC,
        "vocab_size":            len(track2idx),
        "train_sessions":        len(train_df),
        "eval_sessions":         len(eval_df),
        "train_users":           int(train_df["user_id"].nunique()),
        "eval_users":            int(eval_df["user_id"].nunique()),
        "avg_session_len":       float(train_df["session_len"].mean()),
        "sources":               ["30music_sessions"],
        "schema": {
            "session_id":  "str",
            "user_id":     "int",
            "timestamp":   "datetime UTC",
            "track_ids":   "list[int] raw 30Music track IDs",
            "play_ratios": "list[float] per-track play ratio capped at 2.0",
            "item_idxs":   "list[int] vocab-mapped 1-based indices"
        },
        "candidate_selection": {
            "min_session_len": 2,
            "rationale": "Sessions with fewer than 2 tracks excluded"
        }
    }
    swift_upload_bytes(json.dumps(manifest, indent=2).encode(), f"{prefix}/manifest.json")
    return manifest

if __name__ == "__main__":
    print(f"=== Navidrome Session Dataset Builder | run {RUN_ID} ===")
    sessions_raw = load_sessions(max_chunks=5)
    if sessions_raw.empty:
        print("No sessions yet. Check: tail -f ~/sessions_parse.log")
        exit(1)
    sessions_df = parse_sessions(sessions_raw)
    if sessions_df.empty:
        print("No valid sessions parsed.")
        exit(1)
    train_df, eval_df, track2idx, idx2track = build_split(sessions_df)
    manifest = upload_dataset(train_df, eval_df, track2idx, idx2track)
    print("\n=== SESSION DATASET COMPLETE ===")
    print(json.dumps(manifest, indent=2))

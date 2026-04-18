"""
Navidrome - Batch Dataset Builder
Reads 30Music session chunks from Swift, builds versioned train/eval dataset
in interaction_df format compatible with GRU4Rec + SessionKNN training scripts.

Output format matches gru4rec.py expectations:
  interactions.parquet: session_id, user_id, position, track_id, playtime, playratio, skipped, split
  sessions.parquet:     session_id, user_id, timestamp, num_tracks, split
  vocab.json:           track2idx, idx2track

Run: source ~/.chi_auth.sh && python3 pipeline/build_dataset.py
"""
import os, json, subprocess, ast
import pandas as pd
import numpy as np
from datetime import datetime, timezone

CONTAINER            = "navidrome-bucket-proj05"
RUN_ID               = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
VERSION              = f"v{datetime.now(timezone.utc).strftime('%Y%m%d')}-001"
HOLDOUT_FRAC         = 0.15
MAX_CHUNKS           = 5
SKIP_RATIO_THRESHOLD = 0.25

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

def load_sessions(max_chunks=MAX_CHUNKS):
    print(f"\n[STEP 1] Loading session chunks (max {max_chunks})...")
    chunks = list_objects("processed/30music/chunks/sessions/")
    print(f"  found {len(chunks)} chunks, using {min(max_chunks, len(chunks))}")
    chunks = chunks[:max_chunks]

    dfs = []
    for i, chunk in enumerate(chunks):
        local = f"/tmp/sess_{i}.parquet"
        swift_download(chunk, local)
        try:
            df = pd.read_parquet(local, engine="pyarrow")
            dfs.append(df)
        except Exception as e:
            print(f"  skip {chunk}: {e}")
        finally:
            if os.path.exists(local):
                os.remove(local)
        if (i+1) % 5 == 0:
            print(f"  loaded {i+1}/{len(chunks)} chunks...")

    sessions = pd.concat(dfs, ignore_index=True)
    print(f"  total rows: {len(sessions):,}")
    return sessions

def build_interactions(sessions_df):
    print("\n[STEP 2] Building interaction rows...")
    np.random.seed(42)

    # chronological split
    sessions_df["timestamp"] = pd.to_numeric(sessions_df["timestamp"], errors="coerce")
    sessions_df = sessions_df.sort_values("timestamp").reset_index(drop=True)

    # 15% user holdout
    all_users = sessions_df["id"].unique()
    holdout   = set(np.random.choice(all_users, size=int(len(all_users)*HOLDOUT_FRAC), replace=False))

    train_pool = sessions_df[~sessions_df["id"].isin(holdout)]
    eval_pool  = sessions_df[sessions_df["id"].isin(holdout)]

    cutoff_idx   = int(len(train_pool) * 0.8)
    train_cutoff = train_pool.iloc[cutoff_idx]["timestamp"]
    train_final  = train_pool[train_pool["timestamp"] <= train_cutoff].copy()
    val_extra    = train_pool[train_pool["timestamp"] > train_cutoff].copy()
    eval_combined = pd.concat([eval_pool, val_extra], ignore_index=True)

    train_final["split"]  = "train"
    eval_combined["split"] = "eval"
    all_sessions = pd.concat([train_final, eval_combined], ignore_index=True)

    print(f"  train sessions: {len(train_final):,} | eval sessions: {len(eval_combined):,}")

    # parse relations to get track_ids and play_ratios
    session_rows     = []
    interaction_rows = []
    skipped_sessions = 0

    for _, row in all_sessions.iterrows():
        try:
            relations = row.get("relations", None)
            if isinstance(relations, str):
                relations = ast.literal_eval(relations)
            if not isinstance(relations, dict):
                skipped_sessions += 1
                continue

            subjects = relations.get("subjects", [])
            objects  = relations.get("objects",  [])
            if not subjects or not objects:
                skipped_sessions += 1
                continue

            user_id = None
            for s in subjects:
                if s.get("type") == "user":
                    user_id = s.get("id")
                    break
            if user_id is None:
                skipped_sessions += 1
                continue

            track_ids   = []
            play_ratios = []
            for obj in objects:
                if obj.get("type") == "track":
                    tid = obj.get("id")
                    pr  = obj.get("playratio", 1.0)
                    try:
                        pr = float(pr) if pr is not None else None
                    except:
                        pr = None
                    if tid:
                        track_ids.append(int(tid))
                        play_ratios.append(pr)

            if len(track_ids) < 2:
                skipped_sessions += 1
                continue

            session_id = str(row["id"])
            timestamp  = int(row["timestamp"]) if row["timestamp"] else 0
            split      = row["split"]

            session_rows.append({
                "session_id": session_id,
                "user_id":    int(user_id),
                "timestamp":  timestamp,
                "num_tracks": len(track_ids),
                "split":      split,
            })

            for pos, (tid, pr) in enumerate(zip(track_ids, play_ratios)):
                skipped = pr is not None and pr <= SKIP_RATIO_THRESHOLD
                interaction_rows.append({
                    "session_id": session_id,
                    "user_id":    int(user_id),
                    "position":   pos,
                    "track_id":   int(tid),
                    "playtime":   0,
                    "playratio":  pr,
                    "skipped":    skipped,
                    "split":      split,
                })

        except Exception:
            skipped_sessions += 1
            continue

    session_df     = pd.DataFrame(session_rows)
    interaction_df = pd.DataFrame(interaction_rows)

    session_df["timestamp"]    = pd.to_numeric(session_df["timestamp"],    errors="coerce").astype("Int64")
    session_df["user_id"]      = pd.to_numeric(session_df["user_id"],      errors="coerce").astype("Int64")
    interaction_df["track_id"] = pd.to_numeric(interaction_df["track_id"], errors="coerce").astype("Int64")
    interaction_df["user_id"]  = pd.to_numeric(interaction_df["user_id"],  errors="coerce").astype("Int64")

    print(f"  interaction_df: {len(interaction_df):,} rows, {skipped_sessions:,} sessions skipped")
    print(f"  unique tracks:  {interaction_df['track_id'].nunique():,}")
    print(f"  unique users:   {interaction_df['user_id'].nunique():,}")

    return session_df, interaction_df

def build_vocab(interaction_df):
    print("\n[STEP 3] Building vocab...")
    unique_tracks = sorted(interaction_df["track_id"].dropna().unique())
    track2idx = {int(tid): i+1 for i, tid in enumerate(unique_tracks)}
    idx2track = {i+1: int(tid) for tid, i in track2idx.items()}
    print(f"  vocab size: {len(track2idx):,} tracks")
    return track2idx, idx2track

def upload_dataset(session_df, interaction_df, track2idx, idx2track):
    print(f"\n[STEP 4] Uploading dataset {VERSION}...")
    prefix = f"datasets/{VERSION}"

    # delete old version
    old = list_objects(f"{prefix}/")
    for obj in old:
        swift_run(["delete", CONTAINER, obj])

    # interactions
    tmp = "/tmp/interactions.parquet"
    interaction_df.to_parquet(tmp, index=False, engine="pyarrow")
    swift_upload(tmp, f"{prefix}/interactions.parquet")
    os.remove(tmp)

    # sessions
    tmp = "/tmp/sessions.parquet"
    session_df.to_parquet(tmp, index=False, engine="pyarrow")
    swift_upload(tmp, f"{prefix}/sessions.parquet")
    os.remove(tmp)

    # vocab
    vocab = {
        "track2idx": {str(k): v for k, v in track2idx.items()},
        "idx2track": {str(k): v for k, v in idx2track.items()}
    }
    swift_upload_bytes(json.dumps(vocab).encode(), f"{prefix}/vocab.json")

    # manifest
    train_df = interaction_df[interaction_df["split"] == "train"]
    eval_df  = interaction_df[interaction_df["split"] == "eval"]

    manifest = {
        "version_id":        VERSION,
        "run_id":            RUN_ID,
        "created_at":        datetime.now(timezone.utc).isoformat(),
        "format":            "interaction_df",
        "models":            ["GRU4Rec", "SessionKNN"],
        "leakage_check":     "chronological_strict_user_holdout",
        "holdout_fraction":  HOLDOUT_FRAC,
        "vocab_size":        len(track2idx),
        "train_interactions": len(train_df),
        "eval_interactions":  len(eval_df),
        "train_sessions":    int(session_df[session_df["split"]=="train"]["session_id"].nunique()),
        "eval_sessions":     int(session_df[session_df["split"]=="eval"]["session_id"].nunique()),
        "unique_tracks":     int(interaction_df["track_id"].nunique()),
        "unique_users":      int(interaction_df["user_id"].nunique()),
        "skip_ratio_threshold": SKIP_RATIO_THRESHOLD,
        "columns": {
            "interactions": list(interaction_df.columns),
            "sessions":     list(session_df.columns)
        },
        "swift_public_url": f"https://chi.uc.chameleoncloud.org:7480/swift/v1/AUTH_7c0a7a1952e44c94aa75cae1ff5dc9b4/navidrome-bucket-proj05/datasets/{VERSION}/"
    }
    swift_upload_bytes(json.dumps(manifest, indent=2).encode(), f"{prefix}/manifest.json")

    print(f"\n  Public URL: {manifest['swift_public_url']}")
    return manifest

if __name__ == "__main__":
    print(f"=== Navidrome Build Dataset | run {RUN_ID} ===")
    print(f"Version: {VERSION}")

    sessions_raw              = load_sessions()
    session_df, interaction_df = build_interactions(sessions_raw)
    track2idx, idx2track       = build_vocab(interaction_df)
    manifest                   = upload_dataset(session_df, interaction_df, track2idx, idx2track)

    print("\n=== BUILD DATASET COMPLETE ===")
    print(json.dumps(manifest, indent=2))

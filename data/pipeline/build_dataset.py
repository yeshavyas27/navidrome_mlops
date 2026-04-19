"""
Navidrome - Build Dataset Pipeline
===================================
Reads 30Music session chunks from Swift, runs full preprocessing pipeline
identical to Yesha's gru4rec.py preprocessing, and uploads versioned
train/test pkl files ready for direct use in SessionDataset.

Pipeline:
  1. Load session chunks from Swift
  2. Parse idomaar relations into interaction_df
  3. filter_data()      - exact copy of Yesha's logic
  4. build_vocabs()     - exact copy of Yesha's logic
  5. build_sequences()  - exact copy of Yesha's logic
  6. chronological_split() - session-level: 80% train / 20% test by timestamp
  7. Upload pkl files to Swift as versioned dataset

Output in Swift:
  datasets/v{date}-001/
    train_sequences.pkl   - list of {session_id, user_idx, item_idxs, playratios}
    test_sequences.pkl    - list of {session_id, user_idx, item_idxs, playratios}
    item2idx.json         - track_id -> model index mapping
    user2idx.json         - user_id  -> model index mapping
    interactions.parquet  - raw interaction_df for drift monitoring
    sessions.parquet      - raw session_df for drift monitoring
    manifest.json         - full provenance record

Run: source ~/.chi_auth.sh && python3 pipeline/build_dataset.py
"""
import os, json, subprocess, ast, pickle, logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
CONTAINER  = "navidrome-bucket-proj05"
RUN_ID     = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
VERSION    = f"v{datetime.now(timezone.utc).strftime('%Y%m%d')}-001"
CHUNK_SIZE = 100000

# Yesha's filtering thresholds — keep in sync with gru4rec.py cfg
MIN_SESSION_LENGTH   = 3
MAX_SESSION_LENGTH   = 100
MIN_ITEM_SUPPORT     = 5
MIN_USER_SESSIONS    = 3
SKIP_RATIO_THRESHOLD = 0.25
TEST_FRACTION        = 0.2    # 80% train / 20% test chronological split
HOLDOUT_FRAC         = 0.15   # 15% user holdout for eval

AUTH_ARGS = [
    "--os-auth-url",   os.environ.get("OS_AUTH_URL", ""),
    "--os-auth-type",  "v3applicationcredential",
    "--os-application-credential-id",     os.environ.get("OS_APPLICATION_CREDENTIAL_ID", ""),
    "--os-application-credential-secret", os.environ.get("OS_APPLICATION_CREDENTIAL_SECRET", ""),
]

# ============================================================
# SWIFT HELPERS
# ============================================================
def swift_run(args):
    return subprocess.run(["swift"] + AUTH_ARGS + args, capture_output=True, text=True)

def swift_upload(local, name):
    swift_run(["upload", "--object-name", name, CONTAINER, local])
    log.info(f"  uploaded -> {name} ({os.path.getsize(local)/1e6:.1f} MB)")

def swift_upload_bytes(data, name):
    tmp = f"/tmp/ds_tmp_{RUN_ID}.bin"
    with open(tmp, "wb") as f: f.write(data)
    swift_upload(tmp, name)
    os.remove(tmp)

def swift_download(name, local):
    swift_run(["download", "--output", local, CONTAINER, name])

def list_objects(prefix):
    r = swift_run(["list", CONTAINER, "--prefix", prefix])
    return [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]

# ============================================================
# STEP 1 — Load raw session chunks from Swift
# ============================================================
def load_sessions():
    log.info("\n[STEP 1] Loading session chunks from Swift...")
    chunks = list_objects("processed/30music/chunks/sessions/")
    log.info(f"  found {len(chunks)} chunks")

    dfs = []
    for i, chunk in enumerate(chunks):
        local = f"/tmp/sess_{i}.parquet"
        swift_download(chunk, local)
        df = pd.read_parquet(local, engine="pyarrow")
        dfs.append(df)
        os.remove(local)
        if (i+1) % 5 == 0:
            log.info(f"  loaded {i+1}/{len(chunks)} chunks...")

    sessions = pd.concat(dfs, ignore_index=True)
    log.info(f"  total rows: {len(sessions):,}")
    return sessions

# ============================================================
# STEP 2 — Parse idomaar relations into interaction_df
# ============================================================
def parse_to_interaction_df(sessions_raw):
    log.info("\n[STEP 2] Parsing into interaction_df...")
    session_rows, interaction_rows = [], []
    skipped = 0

    for i in range(0, len(sessions_raw), CHUNK_SIZE):
        chunk = sessions_raw.iloc[i:i+CHUNK_SIZE]
        for _, row in chunk.iterrows():
            try:
                relations = row.get("relations", None)
                if isinstance(relations, str):
                    relations = ast.literal_eval(relations)
                if not isinstance(relations, dict):
                    skipped += 1
                    continue

                subjects = relations.get("subjects", [])
                objects  = relations.get("objects",  [])
                if not subjects or not objects:
                    skipped += 1
                    continue

                user_id = None
                for s in subjects:
                    if s.get("type") == "user":
                        user_id = s.get("id")
                        break
                if user_id is None:
                    skipped += 1
                    continue

                track_ids, play_ratios, playtimes = [], [], []
                for obj in objects:
                    if obj.get("type") == "track":
                        tid = obj.get("id")
                        pr  = obj.get("playratio", 1.0)
                        pt  = obj.get("playtime",  0)
                        try: pr = float(pr) if pr is not None else None
                        except: pr = None
                        if tid:
                            track_ids.append(int(tid))
                            play_ratios.append(pr)
                            playtimes.append(float(pt) if pt else 0.0)

                if len(track_ids) < 2:
                    skipped += 1
                    continue

                session_id = str(row["id"])
                timestamp  = int(row["timestamp"]) if row["timestamp"] else 0
                skipped_flags = [
                    pr is not None and pr <= SKIP_RATIO_THRESHOLD
                    for pr in play_ratios
                ]

                session_rows.append({
                    "session_id": session_id,
                    "user_id":    int(user_id),
                    "timestamp":  timestamp,
                    "num_tracks": len(track_ids),
                })

                for pos, (tid, pr, pt) in enumerate(zip(track_ids, play_ratios, playtimes)):
                    interaction_rows.append({
                        "session_id": session_id,
                        "user_id":    int(user_id),
                        "position":   pos,
                        "track_id":   int(tid),
                        "playtime":   pt,
                        "playratio":  pr,
                        "skipped":    pr is not None and pr <= SKIP_RATIO_THRESHOLD,
                    })

            except Exception:
                skipped += 1
                continue

        if (i // CHUNK_SIZE + 1) % 5 == 0:
            log.info(f"  parsed {i+len(chunk):,}/{len(sessions_raw):,} rows...")

    session_df     = pd.DataFrame(session_rows)
    interaction_df = pd.DataFrame(interaction_rows)

    session_df["timestamp"]    = pd.to_numeric(session_df["timestamp"],    errors="coerce").astype("Int64")
    session_df["user_id"]      = pd.to_numeric(session_df["user_id"],      errors="coerce").astype("Int64")
    interaction_df["track_id"] = pd.to_numeric(interaction_df["track_id"], errors="coerce").astype("Int64")
    interaction_df["user_id"]  = pd.to_numeric(interaction_df["user_id"],  errors="coerce").astype("Int64")

    log.info(f"  interaction_df: {len(interaction_df):,} rows ({skipped:,} sessions skipped)")
    log.info(f"  session_df:     {len(session_df):,} rows")
    return session_df, interaction_df

# ============================================================
# STEP 3 — filter_data() — exact copy of Yesha's logic
# ============================================================
def filter_data(session_df, interaction_df):
    log.info("\n[STEP 3] Filtering (exact copy of Yesha's filter_data)...")
    log.info(f"  before: {len(session_df):,} sessions, {len(interaction_df):,} interactions")

    # only non-skipped tracks
    engaged_df = interaction_df[~interaction_df["skipped"]].copy()

    # session length filter
    lengths    = engaged_df.groupby("session_id").size().reset_index(name="engaged_length")
    session_df = session_df.merge(lengths, on="session_id", how="left")
    session_df["engaged_length"] = session_df["engaged_length"].fillna(0).astype(int)

    valid_sessions = session_df[
        (session_df["engaged_length"] >= MIN_SESSION_LENGTH) &
        (session_df["engaged_length"] <= MAX_SESSION_LENGTH)
    ]["session_id"]
    engaged_df = engaged_df[engaged_df["session_id"].isin(valid_sessions)]

    # iterative co-occurrence filtering
    for iteration in range(5):
        prev = len(engaged_df)
        item_counts = engaged_df["track_id"].value_counts()
        engaged_df  = engaged_df[engaged_df["track_id"].isin(
            item_counts[item_counts >= MIN_ITEM_SUPPORT].index)]
        sess_lens   = engaged_df.groupby("session_id").size()
        engaged_df  = engaged_df[engaged_df["session_id"].isin(
            sess_lens[sess_lens >= MIN_SESSION_LENGTH].index)]
        user_sess   = engaged_df.groupby("user_id")["session_id"].nunique()
        engaged_df  = engaged_df[engaged_df["user_id"].isin(
            user_sess[user_sess >= MIN_USER_SESSIONS].index)]
        if len(engaged_df) == prev:
            log.info(f"  converged at iteration {iteration+1}")
            break

    session_df = session_df[session_df["session_id"].isin(
        engaged_df["session_id"].unique())]

    log.info(f"  after: {session_df['session_id'].nunique():,} sessions, "
             f"{engaged_df['track_id'].nunique():,} unique tracks, "
             f"{engaged_df['user_id'].nunique():,} users, "
             f"{len(engaged_df):,} interactions")
    return session_df, engaged_df

# ============================================================
# STEP 4 — build_vocabs() — exact copy of Yesha's logic
# ============================================================
def build_vocabs(interaction_df):
    log.info("\n[STEP 4] Building vocabs (exact copy of Yesha's build_vocabs)...")
    items    = sorted(interaction_df["track_id"].unique())
    users    = sorted(interaction_df["user_id"].unique())
    item2idx = {int(item): idx + 1 for idx, item in enumerate(items)}
    user2idx = {int(user): idx + 1 for idx, user in enumerate(users)}
    log.info(f"  item vocab: {len(item2idx):,} tracks")
    log.info(f"  user vocab: {len(user2idx):,} users")
    return item2idx, user2idx

# ============================================================
# STEP 5 — build_sequences() — exact copy of Yesha's logic
# ============================================================
def build_sequences(interaction_df, item2idx, user2idx):
    log.info("\n[STEP 5] Building sequences (exact copy of Yesha's build_sequences)...")
    sequences = []

    for session_id, group in interaction_df.sort_values("position").groupby("session_id"):
        user_id   = group["user_id"].iloc[0]
        item_ids  = group["track_id"].tolist()
        ratios    = group["playratio"].tolist()
        item_idxs = [item2idx[int(i)] for i in item_ids if int(i) in item2idx]

        clean_ratios = []
        for r in ratios:
            try:
                v = float(r)
                clean_ratios.append(float(np.clip(v, 0.0, 1.0)) if not np.isnan(v) else 1.0)
            except (TypeError, ValueError):
                clean_ratios.append(1.0)

        if len(item_idxs) >= 2:
            sequences.append({
                "session_id": session_id,
                "user_idx":   user2idx.get(int(user_id), 0),
                "item_idxs":  item_idxs,
                "playratios": clean_ratios,
            })

    log.info(f"  total sequences: {len(sequences):,}")
    return sequences

# ============================================================
# STEP 6 — Chronological split — exact copy of Yesha's temporal_split()
# ============================================================
def chronological_split(session_df, sequences):
    log.info("\n[STEP 6] Chronological split (exact copy of Yesha's temporal_split)...")
    log.info(f"  test_fraction: {TEST_FRACTION} | holdout_frac: {HOLDOUT_FRAC}")

    # sort sessions by timestamp
    session_df = session_df.sort_values("timestamp").reset_index(drop=True)

    # 15% user holdout — these users never appear in training
    np.random.seed(42)
    all_users  = session_df["user_id"].unique()
    holdout    = set(np.random.choice(
        all_users, size=int(len(all_users) * HOLDOUT_FRAC), replace=False
    ))

    train_pool = session_df[~session_df["user_id"].isin(holdout)]
    eval_pool  = session_df[session_df["user_id"].isin(holdout)]

    # chronological 80/20 split on non-holdout users
    split_idx    = int(len(train_pool) * (1 - TEST_FRACTION))
    train_ids    = set(train_pool.iloc[:split_idx]["session_id"])
    test_ids_chron = set(train_pool.iloc[split_idx:]["session_id"])
    test_ids_holdout = set(eval_pool["session_id"])

    # combine: test = holdout users + last 20% of train users
    test_ids = test_ids_chron | test_ids_holdout

    train_seqs = [s for s in sequences if s["session_id"] in train_ids]
    test_seqs  = [s for s in sequences if s["session_id"] in test_ids]

    log.info(f"  train sessions: {len(train_seqs):,}")
    log.info(f"  test sessions:  {len(test_seqs):,}")
    log.info(f"  holdout users:  {len(holdout):,}")
    return train_seqs, test_seqs

# ============================================================
# STEP 7 — Upload to Swift
# ============================================================
def upload_dataset(train_seqs, test_seqs, item2idx, user2idx,
                   session_df, interaction_df):
    log.info(f"\n[STEP 7] Uploading dataset {VERSION}...")
    prefix = f"datasets/{VERSION}"

    # delete old version if exists
    for obj in list_objects(f"{prefix}/"):
        swift_run(["delete", CONTAINER, obj])

    # train_sequences.pkl
    tmp = "/tmp/train_sequences.pkl"
    with open(tmp, "wb") as f: pickle.dump(train_seqs, f)
    swift_upload(tmp, f"{prefix}/train_sequences.pkl")
    os.remove(tmp)

    # test_sequences.pkl
    tmp = "/tmp/test_sequences.pkl"
    with open(tmp, "wb") as f: pickle.dump(test_seqs, f)
    swift_upload(tmp, f"{prefix}/test_sequences.pkl")
    os.remove(tmp)

    # item2idx.json
    swift_upload_bytes(
        json.dumps({str(k): v for k, v in item2idx.items()}).encode(),
        f"{prefix}/item2idx.json"
    )

    # user2idx.json
    swift_upload_bytes(
        json.dumps({str(k): v for k, v in user2idx.items()}).encode(),
        f"{prefix}/user2idx.json"
    )

    # interactions.parquet — for drift monitoring
    tmp = "/tmp/interactions.parquet"
    interaction_df.to_parquet(tmp, index=False, engine="pyarrow")
    swift_upload(tmp, f"{prefix}/interactions.parquet")
    os.remove(tmp)

    # sessions.parquet — for drift monitoring
    tmp = "/tmp/sessions.parquet"
    session_df.to_parquet(tmp, index=False, engine="pyarrow")
    swift_upload(tmp, f"{prefix}/sessions.parquet")
    os.remove(tmp)

    # manifest.json
    manifest = {
        "version_id":      VERSION,
        "run_id":          RUN_ID,
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "format":          "gru4rec_ready",
        "pipeline": {
            "step1": "load session chunks from Swift",
            "step2": "parse idomaar relations → interaction_df",
            "step3": "filter_data() — Yesha's exact logic",
            "step4": "build_vocabs() — Yesha's exact logic",
            "step5": "build_sequences() — Yesha's exact logic",
            "step6": "chronological_split() — Yesha's temporal_split() logic",
        },
        "filter_params": {
            "min_session_length":    MIN_SESSION_LENGTH,
            "max_session_length":    MAX_SESSION_LENGTH,
            "min_item_support":      MIN_ITEM_SUPPORT,
            "min_user_sessions":     MIN_USER_SESSIONS,
            "skip_ratio_threshold":  SKIP_RATIO_THRESHOLD,
            "test_fraction":         TEST_FRACTION,
            "holdout_fraction":      HOLDOUT_FRAC,
        },
        "stats": {
            "train_sequences":  len(train_seqs),
            "test_sequences":   len(test_seqs),
            "num_items":        len(item2idx),
            "num_users":        len(user2idx),
            "raw_interactions": len(interaction_df),
            "raw_sessions":     len(session_df),
        },
        "files": {
            "train_sequences": f"{prefix}/train_sequences.pkl",
            "test_sequences":  f"{prefix}/test_sequences.pkl",
            "item2idx":        f"{prefix}/item2idx.json",
            "user2idx":        f"{prefix}/user2idx.json",
            "interactions":    f"{prefix}/interactions.parquet",
            "sessions":        f"{prefix}/sessions.parquet",
        },
        "usage": {
            "prepare_data": "load train_sequences.pkl and test_sequences.pkl directly",
            "no_preprocessing_needed": True,
            "compatible_with": "SessionDataset, evaluate() in gru4rec.py",
        },
        "swift_public_url": (
            f"https://chi.uc.chameleoncloud.org:7480/swift/v1/"
            f"AUTH_7c0a7a1952e44c94aa75cae1ff5dc9b4/"
            f"navidrome-bucket-proj05/{prefix}/"
        ),
    }
    swift_upload_bytes(
        json.dumps(manifest, indent=2).encode(),
        f"{prefix}/manifest.json"
    )

    log.info(f"\n  Public URL: {manifest['swift_public_url']}")
    return manifest

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    log.info(f"=== Navidrome Build Dataset | run {RUN_ID} ===")
    log.info(f"Version: {VERSION}")
    log.info(f"Config: min_session={MIN_SESSION_LENGTH}, min_item_support={MIN_ITEM_SUPPORT}, "
             f"test_fraction={TEST_FRACTION}, holdout={HOLDOUT_FRAC}")

    # Step 1 — load
    sessions_raw = load_sessions()

    # Step 2 — parse
    session_df, interaction_df = parse_to_interaction_df(sessions_raw)

    # Step 3 — filter (Yesha's logic)
    session_df, interaction_df = filter_data(session_df, interaction_df)

    # Step 4 — vocab (Yesha's logic)
    item2idx, user2idx = build_vocabs(interaction_df)

    # Step 5 — sequences (Yesha's logic)
    sequences = build_sequences(interaction_df, item2idx, user2idx)

    # Step 6 — chronological split (Yesha's temporal_split logic)
    train_seqs, test_seqs = chronological_split(session_df, sequences)

    # Step 7 — upload
    manifest = upload_dataset(
        train_seqs, test_seqs, item2idx, user2idx,
        session_df, interaction_df
    )

    log.info("\n=== BUILD DATASET COMPLETE ===")
    log.info(json.dumps(manifest["stats"], indent=2))
    log.info(f"Swift URL: {manifest['swift_public_url']}")

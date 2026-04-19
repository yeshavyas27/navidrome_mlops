"""
Navidrome - Live Dataset Builder
=================================
Runs every 24 hours via K8S CronJob.
Reads live sessions from PostgreSQL, combines with 30Music historical data,
runs full preprocessing pipeline, uploads versioned dataset to Swift.

Pipeline:
  1. Load live sessions from PostgreSQL (last 24h or all accumulated)
  2. Optionally combine with historical 30Music data
  3. filter_data()         - Yesha's exact logic
  4. build_vocabs()        - Yesha's exact logic
  5. build_sequences()     - Yesha's exact logic
  6. chronological_split() - Yesha's temporal_split() logic
  7. Upload pkl files to Swift as versioned dataset
  8. Log metrics to MLflow

Run: python3 pipeline/build_dataset_live.py
K8S: runs as CronJob every 24 hours
"""
import os, json, subprocess, pickle, logging
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
CONTAINER  = "navidrome-bucket-proj05"
RUN_ID     = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
VERSION    = f"v{datetime.now(timezone.utc).strftime('%Y%m%d')}-live-001"

# PostgreSQL
PG_HOST = os.getenv("PG_HOST", "postgres.navidrome-platform.svc.cluster.local")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB   = os.getenv("PG_DB",   "navidrome")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "navidrome2026")

# MLflow
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.navidrome-platform.svc.cluster.local:8000")

# Yesha's filtering thresholds
MIN_SESSION_LENGTH   = 2
MAX_SESSION_LENGTH   = 100
MIN_ITEM_SUPPORT     = 1    # lower than 30Music since less data
MIN_USER_SESSIONS    = 1    # lower threshold for live data
SKIP_RATIO_THRESHOLD = 0.25
TEST_FRACTION        = 0.2
HOLDOUT_FRAC         = 0.15

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
    tmp = f"/tmp/live_tmp_{RUN_ID}.bin"
    with open(tmp, "wb") as f: f.write(data)
    swift_upload(tmp, name)
    os.remove(tmp)

def list_objects(prefix):
    r = swift_run(["list", CONTAINER, "--prefix", prefix])
    return [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]

# ============================================================
# STEP 1 — Load live sessions from PostgreSQL
# ============================================================
def load_postgres_sessions(since_hours=2):
    """
    Load live sessions from PostgreSQL.
    since_hours: None = all data, 24 = last 24 hours
    """
    log.info(f"\n[STEP 1] Loading sessions from PostgreSQL...")
    log.info(f"  host: {PG_HOST} | db: {PG_DB}")

    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        dbname=PG_DB, user=PG_USER, password=PG_PASS
    )

    if since_hours:
        since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        query = f"""
            SELECT session_id, user_id, track_ids, play_ratios,
                   num_tracks, timestamp, source
            FROM sessions
            WHERE timestamp >= '{since.isoformat()}'
              AND source = 'navidrome_live'
            ORDER BY timestamp ASC
        """
        log.info(f"  loading sessions since {since.strftime('%Y-%m-%d %H:%M')} UTC")
    else:
        query = """
            SELECT session_id, user_id, track_ids, play_ratios,
                   num_tracks, timestamp, source
            FROM sessions
            WHERE source = 'navidrome_live'
            ORDER BY timestamp ASC
        """
        log.info(f"  loading ALL navidrome_live sessions")

    df = pd.read_sql(query, conn)
    conn.close()

    log.info(f"  loaded {len(df):,} sessions from PostgreSQL")
    log.info(f"  unique users: {df['user_id'].nunique():,}")
    log.info(f"  date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df

# ============================================================
# STEP 2 — Convert PostgreSQL rows to interaction_df format
# ============================================================
def postgres_to_interaction_df(pg_df):
    """
    Convert PostgreSQL sessions to interaction_df format
    matching Yesha's filter_data() input expectations.
    """
    log.info("\n[STEP 2] Converting to interaction_df format...")
    session_rows     = []
    interaction_rows = []
    skipped          = 0

    for _, row in pg_df.iterrows():
        try:
            track_ids   = list(row["track_ids"])
            play_ratios = list(row["play_ratios"])
            session_id  = str(row["session_id"])
            user_id     = str(row["user_id"])
            timestamp   = row["timestamp"]

            if len(track_ids) < 2:
                skipped += 1
                continue

            # convert timestamp to unix int
            if hasattr(timestamp, "timestamp"):
                ts_int = int(timestamp.timestamp())
            else:
                ts_int = int(pd.Timestamp(timestamp).timestamp())

            session_rows.append({
                "session_id": session_id,
                "user_id":    user_id,
                "timestamp":  ts_int,
                "num_tracks": len(track_ids),
            })

            for pos, (tid, pr) in enumerate(zip(track_ids, play_ratios)):
                try:
                    pr_float = float(pr) if pr is not None else 1.0
                except: pr_float = 1.0
                skipped_flag = pr_float <= SKIP_RATIO_THRESHOLD

                interaction_rows.append({
                    "session_id": session_id,
                    "user_id":    user_id,
                    "position":   pos,
                    "track_id":   int(tid),
                    "playtime":   0,
                    "playratio":  pr_float,
                    "skipped":    skipped_flag,
                })

        except Exception as e:
            skipped += 1
            continue

    session_df     = pd.DataFrame(session_rows)
    interaction_df = pd.DataFrame(interaction_rows)

    session_df["timestamp"]    = pd.to_numeric(session_df["timestamp"],    errors="coerce").astype("Int64")
    session_df["user_id"]      = session_df["user_id"].astype(str)
    interaction_df["track_id"] = pd.to_numeric(interaction_df["track_id"], errors="coerce").astype("Int64")
    interaction_df["user_id"]  = interaction_df["user_id"].astype(str)

    log.info(f"  interaction_df: {len(interaction_df):,} rows ({skipped:,} skipped)")
    log.info(f"  session_df:     {len(session_df):,} rows")
    return session_df, interaction_df

# ============================================================
# STEP 3 — filter_data() — exact copy of Yesha's logic
# ============================================================
def filter_data(session_df, interaction_df):
    log.info("\n[STEP 3] Filtering (Yesha's filter_data logic)...")
    log.info(f"  before: {len(session_df):,} sessions, {len(interaction_df):,} interactions")

    engaged_df = interaction_df[~interaction_df["skipped"]].copy()

    lengths    = engaged_df.groupby("session_id").size().reset_index(name="engaged_length")
    session_df = session_df.merge(lengths, on="session_id", how="left")
    session_df["engaged_length"] = session_df["engaged_length"].fillna(0).astype(int)

    valid_sessions = session_df[
        (session_df["engaged_length"] >= MIN_SESSION_LENGTH) &
        (session_df["engaged_length"] <= MAX_SESSION_LENGTH)
    ]["session_id"]
    engaged_df = engaged_df[engaged_df["session_id"].isin(valid_sessions)]

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
    log.info("\n[STEP 4] Building vocabs...")
    items    = sorted(interaction_df["track_id"].unique())
    users    = sorted(interaction_df["user_id"].unique())
    item2idx = {int(item): idx + 1 for idx, item in enumerate(items)}
    user2idx = {str(user): idx + 1 for idx, user in enumerate(users)}
    log.info(f"  item vocab: {len(item2idx):,} tracks")
    log.info(f"  user vocab: {len(user2idx):,} users")
    return item2idx, user2idx

# ============================================================
# STEP 5 — build_sequences() — exact copy of Yesha's logic
# ============================================================
def build_sequences(interaction_df, item2idx, user2idx):
    log.info("\n[STEP 5] Building sequences...")
    sequences = []

    for session_id, group in interaction_df.sort_values("position").groupby("session_id"):
        user_id   = str(group["user_id"].iloc[0])
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
                "user_idx":   user2idx.get(user_id, 0),
                "item_idxs":  item_idxs,
                "playratios": clean_ratios,
            })

    log.info(f"  total sequences: {len(sequences):,}")
    return sequences

# ============================================================
# STEP 6 — Chronological split — exact copy of Yesha's temporal_split()
# ============================================================
def chronological_split(session_df, sequences):
    log.info("\n[STEP 6] Chronological split...")
    session_df = session_df.sort_values("timestamp").reset_index(drop=True)

    np.random.seed(42)
    all_users = session_df["user_id"].unique()
    holdout   = set(np.random.choice(
        all_users, size=max(1, int(len(all_users) * HOLDOUT_FRAC)), replace=False
    ))

    train_pool = session_df[~session_df["user_id"].isin(holdout)]
    eval_pool  = session_df[session_df["user_id"].isin(holdout)]

    split_idx      = int(len(train_pool) * (1 - TEST_FRACTION))
    train_ids      = set(train_pool.iloc[:split_idx]["session_id"])
    test_ids_chron = set(train_pool.iloc[split_idx:]["session_id"])
    test_ids_hold  = set(eval_pool["session_id"])
    test_ids       = test_ids_chron | test_ids_hold

    train_seqs = [s for s in sequences if s["session_id"] in train_ids]
    test_seqs  = [s for s in sequences if s["session_id"] in test_ids]

    log.info(f"  train: {len(train_seqs):,} | test: {len(test_seqs):,}")
    log.info(f"  holdout users: {len(holdout):,}")
    return train_seqs, test_seqs

# ============================================================
# STEP 7 — Upload to Swift
# ============================================================
def upload_dataset(train_seqs, test_seqs, item2idx, user2idx,
                   session_df, interaction_df, pg_row_count):
    log.info(f"\n[STEP 7] Uploading dataset {VERSION}...")
    prefix = f"datasets/{VERSION}"

    for obj in list_objects(f"{prefix}/"):
        swift_run(["delete", CONTAINER, obj])

    tmp = "/tmp/train_sequences.pkl"
    with open(tmp, "wb") as f: pickle.dump(train_seqs, f)
    swift_upload(tmp, f"{prefix}/train_sequences.pkl")
    os.remove(tmp)

    tmp = "/tmp/test_sequences.pkl"
    with open(tmp, "wb") as f: pickle.dump(test_seqs, f)
    swift_upload(tmp, f"{prefix}/test_sequences.pkl")
    os.remove(tmp)

    swift_upload_bytes(
        json.dumps({str(k): v for k, v in item2idx.items()}).encode(),
        f"{prefix}/item2idx.json"
    )
    swift_upload_bytes(
        json.dumps({str(k): v for k, v in user2idx.items()}).encode(),
        f"{prefix}/user2idx.json"
    )

    tmp = "/tmp/interactions.parquet"
    interaction_df.to_parquet(tmp, index=False, engine="pyarrow")
    swift_upload(tmp, f"{prefix}/interactions.parquet")
    os.remove(tmp)

    tmp = "/tmp/sessions.parquet"
    session_df.to_parquet(tmp, index=False, engine="pyarrow")
    swift_upload(tmp, f"{prefix}/sessions.parquet")
    os.remove(tmp)

    manifest = {
        "version_id":      VERSION,
        "run_id":          RUN_ID,
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "format":          "gru4rec_ready",
        "data_source":     "postgresql_live",
        "pipeline": {
            "step1": "load navidrome_live sessions from PostgreSQL",
            "step2": "convert to interaction_df format",
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
            "pg_sessions_loaded":  pg_row_count,
            "train_sequences":     len(train_seqs),
            "test_sequences":      len(test_seqs),
            "num_items":           len(item2idx),
            "num_users":           len(user2idx),
            "raw_interactions":    len(interaction_df),
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
# STEP 8 — Log to MLflow
# ============================================================
def log_to_mlflow(manifest):
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("navidrome-live-dataset")
        with mlflow.start_run(run_name=f"live_dataset_{VERSION}"):
            mlflow.log_params({
                "version":          VERSION,
                "data_source":      "postgresql",
                "min_item_support": MIN_ITEM_SUPPORT,
                "test_fraction":    TEST_FRACTION,
            })
            mlflow.log_metrics({
                "train_sequences":  manifest["stats"]["train_sequences"],
                "test_sequences":   manifest["stats"]["test_sequences"],
                "num_items":        manifest["stats"]["num_items"],
                "num_users":        manifest["stats"]["num_users"],
                "pg_sessions":      manifest["stats"]["pg_sessions_loaded"],
            })
            mlflow.set_tags({
                "dataset_version": VERSION,
                "data_source":     "postgresql_live",
            })
        log.info("  MLflow logged successfully")
    except Exception as e:
        log.warning(f"  MLflow logging failed (non-fatal): {e}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    log.info(f"=== Navidrome Live Dataset Builder | {RUN_ID} ===")
    log.info(f"Version: {VERSION}")

    # Step 1 — load from PostgreSQL
    pg_df = load_postgres_sessions(since_hours=2)  # all data

    if len(pg_df) < 100:
        log.warning(f"Only {len(pg_df)} sessions — skipping build (need at least 100)")
        exit(0)

    # Step 2 — convert to interaction_df
    session_df, interaction_df = postgres_to_interaction_df(pg_df)

    # Step 3 — filter
    session_df, interaction_df = filter_data(session_df, interaction_df)

    if len(session_df) < 50:
        log.warning("Not enough sessions after filtering — skipping")
        exit(0)

    # Step 4 — vocab
    item2idx, user2idx = build_vocabs(interaction_df)

    # Step 5 — sequences
    sequences = build_sequences(interaction_df, item2idx, user2idx)

    # Step 6 — split
    train_seqs, test_seqs = chronological_split(session_df, sequences)

    # Step 7 — upload
    manifest = upload_dataset(
        train_seqs, test_seqs, item2idx, user2idx,
        session_df, interaction_df, len(pg_df)
    )

    # Step 8 — MLflow
    log_to_mlflow(manifest)

    log.info("\n=== LIVE DATASET BUILD COMPLETE ===")
    log.info(json.dumps(manifest["stats"], indent=2))
    log.info(f"Swift URL: {manifest['swift_public_url']}")

"""
Navidrome - Live Dataset Builder
=================================
Runs every 24 hours via K8S CronJob.
Reads last 24 hours of sessions from PostgreSQL,
runs full preprocessing pipeline, saves to MinIO + Swift, logs to MLflow.

Run: python3 pipeline/build_dataset_live.py
K8S: CronJob at 0 2 * * * (2am UTC daily)
"""
import os, json, io, pickle, logging
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timezone, timedelta

from rollup import rollup_stale_users

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
RUN_ID  = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
VERSION = f"v{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-live"

PG_HOST = os.getenv("PG_HOST", "postgres.navidrome-platform.svc.cluster.local")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB   = os.getenv("PG_DB",   "navidrome")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "navidrome2026")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio.navidrome-platform.svc.cluster.local:9000")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "navidrome2026")
MINIO_BUCKET   = "navidrome-datasets"

SWIFT_CONTAINER = "navidrome-bucket-proj05"
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.navidrome-platform.svc.cluster.local:8000")

# Pipeline gates — both are env-configurable so a small cluster can tune
# them down without a code change.
MIN_SESSIONS_PER_RUN = int(os.getenv("MIN_SESSIONS_PER_RUN", "5"))
MIN_SESSIONS_AFTER_FILTER = int(os.getenv("MIN_SESSIONS_AFTER_FILTER", "5"))

# Filtering thresholds
MIN_SESSION_LENGTH   = 2          # need >=1 prefix + 1 target after split
MAX_SESSION_LENGTH   = 100
MIN_ITEM_SUPPORT     = 1
MIN_USER_SESSIONS    = 1
SKIP_RATIO_THRESHOLD = 0.2        # tracks with play_ratio <= 0.2 are dropped
TEST_FRACTION        = 0.2        # legacy — unused with leave-one-out split
HOLDOUT_FRAC         = 0.0

# ============================================================
# MINIO UPLOAD (primary — always works inside K8S)
# ============================================================
def get_s3():
    import boto3
    return boto3.client("s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS,
        aws_secret_access_key=MINIO_SECRET,
        region_name="us-east-1",
    )

def ensure_minio_bucket():
    s3 = get_s3()
    try:
        s3.head_bucket(Bucket=MINIO_BUCKET)
    except:
        s3.create_bucket(Bucket=MINIO_BUCKET)
        log.info(f"  created MinIO bucket: {MINIO_BUCKET}")

def upload_to_minio(data_bytes, key):
    s3 = get_s3()
    s3.upload_fileobj(io.BytesIO(data_bytes), MINIO_BUCKET, key)
    size = len(data_bytes) / 1e6
    log.info(f"  MinIO -> {key} ({size:.1f} MB)")

# ============================================================
# SWIFT UPLOAD (secondary — best effort)
# ============================================================
def upload_to_swift(data_bytes, key):
    try:
        import swiftclient
        conn = swiftclient.Connection(
            authurl=os.environ.get("OS_AUTH_URL", ""),
            auth_version="3",
            os_options={
                "auth_type": "v3applicationcredential",
                "application_credential_id":     os.environ.get("OS_APPLICATION_CREDENTIAL_ID", ""),
                "application_credential_secret": os.environ.get("OS_APPLICATION_CREDENTIAL_SECRET", ""),
            }
        )
        conn.put_object(SWIFT_CONTAINER, key, data_bytes)
        log.info(f"  Swift -> {key}")
    except Exception as e:
        log.warning(f"  Swift upload skipped (non-fatal): {e}")

def upload(data_bytes, key):
    """Upload to MinIO (required) + Swift (best effort)."""
    upload_to_minio(data_bytes, key)
    upload_to_swift(data_bytes, key)

# ============================================================
# STEP 1 — Load last 24 hours from PostgreSQL
# ============================================================
def load_postgres_sessions():
    log.info("\n[STEP 1] Loading last 24h sessions from PostgreSQL...")
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    log.info(f"  since: {since.strftime('%Y-%m-%d %H:%M')} UTC")

    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        dbname=PG_DB, user=PG_USER, password=PG_PASS
    )
    # First flush any sub-threshold users (>24h-old unassigned activity)
    # so casual listeners aren't invisible to training.
    stale = rollup_stale_users(conn, max_age_hours=24, min_size=1)
    if stale:
        log.info(f"  stale-rollup: created {stale} catch-up session(s)")

    # Both source labels feed training:
    #   navidrome_live → 50-track rollup (and stale catch-up) — primary
    #   inference     → 30-min snapshots from recommend clicks — added
    # The activities behind navidrome_live sessions are stamped (consumed),
    # while inference snapshots overlay activities that may also live in
    # a navidrome_live session — that's intentional duplication: more
    # recent contextual training samples on top of the canonical buckets.
    df = pd.read_sql(f"""
        SELECT session_id, user_id, track_ids, play_ratios,
               num_tracks, timestamp, source
        FROM sessions
        WHERE source IN ('navidrome_live', 'inference')
          AND timestamp >= '{since.isoformat()}'
        ORDER BY timestamp ASC
    """, conn)
    conn.close()

    log.info(f"  loaded: {len(df):,} sessions | {df['user_id'].nunique():,} users")
    if len(df) > 0:
        log.info(f"  range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df

# ============================================================
# STEP 2 — Convert to interaction_df
# ============================================================
def to_interaction_df(pg_df):
    log.info("\n[STEP 2] Converting to interaction_df...")
    session_rows, interaction_rows, skipped = [], [], 0

    for _, row in pg_df.iterrows():
        try:
            track_ids   = list(row["track_ids"])
            play_ratios = list(row["play_ratios"])
            if len(track_ids) < 2:
                skipped += 1
                continue

            ts = row["timestamp"]
            ts_int = int(ts.timestamp()) if hasattr(ts, "timestamp") else int(pd.Timestamp(ts).timestamp())

            session_rows.append({
                "session_id": str(row["session_id"]),
                "user_id":    str(row["user_id"]),
                "timestamp":  ts_int,
                "num_tracks": len(track_ids),
            })
            for pos, (tid, pr) in enumerate(zip(track_ids, play_ratios)):
                try: pr_f = float(pr) if pr is not None else 1.0
                except: pr_f = 1.0
                interaction_rows.append({
                    "session_id": str(row["session_id"]),
                    "user_id":    str(row["user_id"]),
                    "position":   pos,
                    "track_id":   int(tid),
                    "playtime":   0,
                    "playratio":  pr_f,
                    "skipped":    pr_f <= SKIP_RATIO_THRESHOLD,
                })
        except:
            skipped += 1

    session_df     = pd.DataFrame(session_rows)
    interaction_df = pd.DataFrame(interaction_rows)
    session_df["timestamp"]    = pd.to_numeric(session_df["timestamp"], errors="coerce").astype("Int64")
    interaction_df["track_id"] = pd.to_numeric(interaction_df["track_id"], errors="coerce").astype("Int64")

    log.info(f"  interactions: {len(interaction_df):,} | skipped: {skipped:,}")
    return session_df, interaction_df

# ============================================================
# STEP 3 — filter_data() — Yesha's exact logic
# ============================================================
def filter_data(session_df, interaction_df):
    log.info("\n[STEP 3] Filtering...")
    log.info(f"  before: {len(session_df):,} sessions")

    engaged_df = interaction_df[~interaction_df["skipped"]].copy()
    lengths    = engaged_df.groupby("session_id").size().reset_index(name="engaged_length")
    session_df = session_df.merge(lengths, on="session_id", how="left")
    session_df["engaged_length"] = session_df["engaged_length"].fillna(0).astype(int)

    valid = session_df[
        (session_df["engaged_length"] >= MIN_SESSION_LENGTH) &
        (session_df["engaged_length"] <= MAX_SESSION_LENGTH)
    ]["session_id"]
    engaged_df = engaged_df[engaged_df["session_id"].isin(valid)]

    for i in range(5):
        prev = len(engaged_df)
        item_counts = engaged_df["track_id"].value_counts()
        engaged_df  = engaged_df[engaged_df["track_id"].isin(item_counts[item_counts >= MIN_ITEM_SUPPORT].index)]
        sess_lens   = engaged_df.groupby("session_id").size()
        engaged_df  = engaged_df[engaged_df["session_id"].isin(sess_lens[sess_lens >= MIN_SESSION_LENGTH].index)]
        user_sess   = engaged_df.groupby("user_id")["session_id"].nunique()
        engaged_df  = engaged_df[engaged_df["user_id"].isin(user_sess[user_sess >= MIN_USER_SESSIONS].index)]
        if len(engaged_df) == prev:
            break

    session_df = session_df[session_df["session_id"].isin(engaged_df["session_id"].unique())]
    log.info(f"  after: {session_df['session_id'].nunique():,} sessions | "
             f"{engaged_df['track_id'].nunique():,} tracks | "
             f"{engaged_df['user_id'].nunique():,} users")
    return session_df, engaged_df

# ============================================================
# STEP 4 — build_vocabs() — Yesha's exact logic
# ============================================================
def build_vocabs(interaction_df):
    log.info("\n[STEP 4] Building vocabs...")
    items    = sorted(interaction_df["track_id"].unique())
    users    = sorted(interaction_df["user_id"].unique())
    item2idx = {int(item): idx + 1 for idx, item in enumerate(items)}
    user2idx = {str(user): idx + 1 for idx, user in enumerate(users)}
    log.info(f"  items: {len(item2idx):,} | users: {len(user2idx):,}")
    return item2idx, user2idx

# ============================================================
# STEP 5 — build_sequences() — Yesha's exact logic
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
            except: clean_ratios.append(1.0)

        if len(item_idxs) >= 2:
            sequences.append({
                "session_id": session_id,
                "user_idx":   user2idx.get(user_id, 0),
                "item_idxs":  item_idxs,
                "playratios": clean_ratios,
            })
    log.info(f"  sequences: {len(sequences):,}")
    return sequences

# ============================================================
# STEP 6 — In-session leave-one-out split
# ============================================================
def leave_one_out_split(sequences):
    """For each session [a, b, c, d, e]:
        train  -> [a, b, c, d]            (prefix only, model never sees e)
        test   -> [a, b, c, d, e]         (full sequence + target_idx = e)

    The whole sequence is in test so the model can score "given prefix,
    what comes next?"; target_idx is the held-out ground truth.

    Each session contributes one train sequence and one test sample.
    Sessions shorter than 2 tracks are skipped (need at least 1 prefix
    + 1 target).
    """
    log.info("\n[STEP 6] In-session leave-one-out split...")
    train_seqs, test_seqs = [], []
    for s in sequences:
        if len(s["item_idxs"]) < 2:
            continue
        train_seqs.append({
            "session_id": s["session_id"],
            "user_idx":   s["user_idx"],
            "item_idxs":  s["item_idxs"][:-1],
            "playratios": s["playratios"][:-1],
        })
        test_seqs.append({
            "session_id": s["session_id"],
            "user_idx":   s["user_idx"],
            "item_idxs":  s["item_idxs"],
            "playratios": s["playratios"],
            "target_idx": s["item_idxs"][-1],
        })

    train_users = set(s["user_idx"] for s in train_seqs)
    test_users  = set(s["user_idx"] for s in test_seqs)
    unseen      = test_users - train_users
    log.info(f"  train: {len(train_seqs):,} | test: {len(test_seqs):,}")
    log.info(f"  unseen users in test: {len(unseen)} (expected 0 — same sessions in both)")
    return train_seqs, test_seqs

# ============================================================
# STEP 7 — Upload to MinIO + Swift
# ============================================================
def upload_dataset(train_seqs, test_seqs, item2idx, user2idx, session_df, interaction_df, pg_count):
    log.info(f"\n[STEP 7] Uploading dataset {VERSION}...")
    ensure_minio_bucket()
    prefix = f"datasets/{VERSION}"

    upload(pickle.dumps(train_seqs),                                          f"{prefix}/train_sequences.pkl")
    upload(pickle.dumps(test_seqs),                                           f"{prefix}/test_sequences.pkl")
    upload(json.dumps({str(k): v for k, v in item2idx.items()}).encode(),     f"{prefix}/item2idx.json")
    upload(json.dumps({str(k): v for k, v in user2idx.items()}).encode(),     f"{prefix}/user2idx.json")

    buf = io.BytesIO()
    interaction_df.to_parquet(buf, index=False, engine="pyarrow")
    upload(buf.getvalue(), f"{prefix}/interactions.parquet")

    buf = io.BytesIO()
    session_df.to_parquet(buf, index=False, engine="pyarrow")
    upload(buf.getvalue(), f"{prefix}/sessions.parquet")

    manifest = {
        "version_id":   VERSION,
        "run_id":       RUN_ID,
        "created_at":   datetime.now(timezone.utc).isoformat(),
        "data_source":  "postgresql_live_24h",
        "split_strategy": "in_session_leave_one_out",
        "filter_params": {
            "min_session_length":   MIN_SESSION_LENGTH,
            "min_item_support":     MIN_ITEM_SUPPORT,
            "skip_ratio_threshold": SKIP_RATIO_THRESHOLD,
        },
        "stats": {
            "pg_sessions_loaded": pg_count,
            "train_sequences":    len(train_seqs),
            "test_sequences":     len(test_seqs),
            "num_items":          len(item2idx),
            "num_users":          len(user2idx),
        },
        "minio_url":    f"{MINIO_ENDPOINT}/{MINIO_BUCKET}/{prefix}/",
        "swift_url":    f"https://chi.uc.chameleoncloud.org:7480/swift/v1/AUTH_7c0a7a1952e44c94aa75cae1ff5dc9b4/{SWIFT_CONTAINER}/{prefix}/",
    }
    upload(json.dumps(manifest, indent=2).encode(), f"{prefix}/manifest.json")
    log.info(f"  MinIO URL: {manifest['minio_url']}")
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
                "data_source":      "postgresql_24h",
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
                "minio_url":       manifest["minio_url"],
            })
        log.info("  MLflow logged successfully")
    except Exception as e:
        log.warning(f"  MLflow logging failed (non-fatal): {e}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    log.info(f"=== Live Dataset Builder | {RUN_ID} ===")
    log.info(f"Version: {VERSION}")

    pg_df = load_postgres_sessions()

    if len(pg_df) < MIN_SESSIONS_PER_RUN:
        log.warning(f"Only {len(pg_df)} sessions in last 24h (need >= {MIN_SESSIONS_PER_RUN}) — skipping")
        exit(0)

    session_df, interaction_df = to_interaction_df(pg_df)
    session_df, interaction_df = filter_data(session_df, interaction_df)

    if len(session_df) < MIN_SESSIONS_AFTER_FILTER:
        log.warning(f"Only {len(session_df)} sessions after filtering (need >= {MIN_SESSIONS_AFTER_FILTER}) — skipping")
        exit(0)

    item2idx, user2idx = build_vocabs(interaction_df)
    sequences          = build_sequences(interaction_df, item2idx, user2idx)
    train_seqs, test_seqs = leave_one_out_split(sequences)
    manifest           = upload_dataset(train_seqs, test_seqs, item2idx, user2idx,
                                        session_df, interaction_df, len(pg_df))
    log_to_mlflow(manifest)

    log.info("\n=== COMPLETE ===")
    log.info(json.dumps(manifest["stats"], indent=2))

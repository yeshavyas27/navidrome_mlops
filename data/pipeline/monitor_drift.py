"""
Data Drift Monitor for Navidrome ML Pipeline
Reads live sessions from PostgreSQL, compares against training baseline.
Pushes metrics to MLflow for monitoring.

Run: python3 pipeline/monitor_drift.py
K8S: runs as CronJob every hour
"""
import os, json, logging
import pandas as pd
import numpy as np
import psycopg2
import requests
from datetime import datetime, timezone, timedelta
import mlflow
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
PG_HOST  = os.getenv("PG_HOST",  "postgres.navidrome-platform.svc.cluster.local")
PG_PORT  = int(os.getenv("PG_PORT", "5432"))
PG_DB    = os.getenv("PG_DB",    "navidrome")
PG_USER  = os.getenv("PG_USER",  "postgres")
PG_PASS  = os.getenv("PG_PASS",  "navidrome2026")

MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.navidrome-platform.svc.cluster.local:8000")
DATASET_VERSION = os.getenv("DATASET_VERSION", "v20260419-001")
SWIFT_BASE      = "https://chi.uc.chameleoncloud.org:7480/swift/v1/AUTH_7c0a7a1952e44c94aa75cae1ff5dc9b4/navidrome-bucket-proj05"
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "pushgateway.navidrome-monitoring.svc.cluster.local:9091")

# ============================================================
# STEP 1 — Load training baseline from Swift manifest
# ============================================================
def load_training_stats():
    log.info(f"Loading training manifest for {DATASET_VERSION}...")
    try:
        url  = f"{SWIFT_BASE}/datasets/{DATASET_VERSION}/manifest.json"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        manifest = resp.json()
        stats = manifest.get("stats", {})
        log.info(f"  training stats: {json.dumps(stats)}")
        return stats
    except Exception as e:
        log.warning(f"Could not load manifest: {e}")
        return {}

# ============================================================
# STEP 2 — Load last 24h sessions from PostgreSQL
# ============================================================
def load_production_sessions():
    log.info("Loading last 24h production sessions from PostgreSQL...")
    since = datetime.now(timezone.utc) - timedelta(hours=24)

    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        dbname=PG_DB, user=PG_USER, password=PG_PASS
    )
    df = pd.read_sql(f"""
        SELECT session_id, user_id, track_ids, play_ratios, num_tracks, timestamp
        FROM sessions
        WHERE source = 'navidrome_live'
          AND timestamp >= '{since.isoformat()}'
        ORDER BY timestamp ASC
    """, conn)
    conn.close()

    log.info(f"  loaded: {len(df):,} sessions | {df['user_id'].nunique():,} unique users")
    return df

# ============================================================
# STEP 3 — Load all-time stats from PostgreSQL for comparison
# ============================================================
def load_alltime_stats():
    log.info("Loading all-time PostgreSQL stats...")
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        dbname=PG_DB, user=PG_USER, password=PG_PASS
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*) as total_sessions,
            COUNT(DISTINCT user_id) as unique_users,
            AVG(num_tracks) as avg_tracks,
            MIN(timestamp) as earliest,
            MAX(timestamp) as latest
        FROM sessions
        WHERE source = 'navidrome_live'
    """)
    row = cur.fetchone()
    conn.close()
    return {
        "total_sessions": row[0],
        "unique_users":   row[1],
        "avg_tracks":     float(row[2] or 0),
        "earliest":       str(row[3]),
        "latest":         str(row[4]),
    }

# ============================================================
# STEP 4 — Compute drift metrics
# ============================================================
def compute_drift_metrics(training_stats, prod_df, alltime_stats):
    log.info("Computing drift metrics...")
    metrics = {}
    now = datetime.now(timezone.utc).isoformat()

    # production volume (last 24h)
    metrics["prod_sessions_24h"]      = len(prod_df)
    metrics["prod_unique_users_24h"]  = int(prod_df["user_id"].nunique()) if len(prod_df) > 0 else 0
    metrics["prod_avg_session_len"]   = float(prod_df["num_tracks"].mean()) if len(prod_df) > 0 else 0

    # all time stats
    metrics["total_pg_sessions"]      = int(alltime_stats["total_sessions"])
    metrics["total_pg_users"]         = int(alltime_stats["unique_users"])

    # session length drift vs training
    train_avg_len = training_stats.get("raw_interactions", 0) / max(training_stats.get("raw_sessions", 1), 1)
    prod_avg_len  = metrics["prod_avg_session_len"]
    metrics["session_length_drift"]   = abs(prod_avg_len - train_avg_len) / max(train_avg_len, 1) if train_avg_len > 0 else 0

    # new user rate
    train_users = training_stats.get("num_users", 0)
    prod_users  = metrics["prod_unique_users_24h"]
    metrics["new_user_rate"]          = max(0, prod_users - train_users) / max(train_users, 1) if train_users > 0 else 0

    # play ratio stats
    if len(prod_df) > 0 and "play_ratios" in prod_df.columns:
        all_ratios = []
        for ratios in prod_df["play_ratios"].dropna():
            if isinstance(ratios, (list, tuple)):
                all_ratios.extend([float(r) for r in ratios if r is not None])
        if all_ratios:
            metrics["prod_avg_playratio"] = float(np.mean(all_ratios))
            metrics["prod_skip_rate"]     = float(np.mean([r <= 0.25 for r in all_ratios]))
            metrics["prod_full_play_rate"] = float(np.mean([r >= 0.75 for r in all_ratios]))

    # track vocab coverage
    train_items = training_stats.get("num_items", 0)
    if len(prod_df) > 0 and train_items > 0:
        all_track_ids = set()
        for tids in prod_df["track_ids"].dropna():
            if isinstance(tids, (list, tuple)):
                all_track_ids.update([str(t) for t in tids])
        metrics["prod_unique_tracks_24h"] = len(all_track_ids)
        metrics["vocab_coverage"]         = min(1.0, len(all_track_ids) / train_items)

    # drift alert
    metrics["drift_alert"] = int(
        metrics.get("session_length_drift", 0) > 0.3 or
        metrics.get("new_user_rate", 0) > 0.5
    )

    metrics["monitor_timestamp"] = now
    log.info(f"  session_length_drift: {metrics.get('session_length_drift', 0):.3f}")
    log.info(f"  new_user_rate:        {metrics.get('new_user_rate', 0):.3f}")
    log.info(f"  prod_skip_rate:       {metrics.get('prod_skip_rate', 0):.3f}")
    log.info(f"  drift_alert:          {metrics['drift_alert']}")
    return metrics

# ============================================================
# STEP 5 — Push to MLflow
# ============================================================
def push_to_mlflow(metrics):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("navidrome-data-drift")

    run_name = f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "monitor_type":    "data_drift",
            "dataset_version": DATASET_VERSION,
            "data_source":     "postgresql",
        })

        numeric = {k: v for k, v in metrics.items()
                   if isinstance(v, (int, float)) and k != "drift_alert"}
        mlflow.log_metrics(numeric)
        mlflow.log_param("drift_alert",      metrics.get("drift_alert", 0))
        mlflow.log_param("dataset_version",  DATASET_VERSION)
        mlflow.log_param("monitor_timestamp", metrics.get("monitor_timestamp", ""))

        if metrics.get("drift_alert"):
            mlflow.set_tag("alert", "DRIFT_DETECTED — retrain needed")
            log.warning("DRIFT ALERT — trigger retraining!")

    log.info(f"MLflow logged -> experiment: navidrome-data-drift | run: {run_name}")

# ============================================================
# STEP 6 — Push to Prometheus Pushgateway
# ============================================================
def push_to_prometheus(metrics):
    """Expose drift metrics as Prometheus gauges via the Pushgateway.

    Each numeric metric becomes navidrome_<key>; the Pushgateway holds
    them indefinitely, Prometheus scrapes the gateway, and Grafana
    queries Prometheus for the dashboard.
    """
    registry = CollectorRegistry()
    for k, v in metrics.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        gauge = Gauge(
            f"navidrome_{k}",
            f"{k} from drift monitor",
            registry=registry,
        )
        gauge.set(float(v))
    try:
        push_to_gateway(
            PUSHGATEWAY_URL,
            job="drift_monitor",
            registry=registry,
            grouping_key={"dataset_version": DATASET_VERSION},
        )
        log.info(f"Prometheus pushed -> {PUSHGATEWAY_URL} (job=drift_monitor)")
    except Exception as e:
        log.error(f"Pushgateway push failed (non-fatal): {e}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    log.info("=== Navidrome Drift Monitor ===")
    log.info(f"Dataset baseline: {DATASET_VERSION}")

    training_stats = load_training_stats()
    alltime_stats  = load_alltime_stats()
    prod_df        = load_production_sessions()

    if len(prod_df) == 0:
        log.warning("No production sessions in last 24h — logging zero metrics")
        metrics = {
            "prod_sessions_24h":     0,
            "prod_unique_users_24h": 0,
            "total_pg_sessions":     int(alltime_stats["total_sessions"]),
            "total_pg_users":        int(alltime_stats["unique_users"]),
            "drift_alert":           0,
            "monitor_timestamp":     datetime.now(timezone.utc).isoformat(),
        }
    else:
        metrics = compute_drift_metrics(training_stats, prod_df, alltime_stats)

    push_to_mlflow(metrics)
    push_to_prometheus(metrics)

    log.info("=== Drift Monitor Complete ===")
    if metrics.get("drift_alert"):
        log.warning("ACTION REQUIRED: Drift detected — trigger retraining")

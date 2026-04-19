"""
Data Drift Monitor for Navidrome ML Pipeline
Runs periodically to detect drift between training data and live production data.
Pushes metrics to MLflow for monitoring.

Run: source ~/.chi_auth.sh && python3 pipeline/monitor_drift.py
K8S: runs as CronJob every hour
"""
import os, json, subprocess, logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

CONTAINER       = "navidrome-bucket-proj05"
DATASET_VERSION = os.getenv("DATASET_VERSION", "v20260418-001")
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://129.114.27.204:8000")

AUTH_ARGS = [
    "--os-auth-url",   os.environ.get("OS_AUTH_URL", ""),
    "--os-auth-type",  "v3applicationcredential",
    "--os-application-credential-id",     os.environ.get("OS_APPLICATION_CREDENTIAL_ID", ""),
    "--os-application-credential-secret", os.environ.get("OS_APPLICATION_CREDENTIAL_SECRET", ""),
]

def swift_run(args):
    return subprocess.run(["swift"] + AUTH_ARGS + args, capture_output=True, text=True)

def swift_download(name, local):
    swift_run(["download", "--output", local, CONTAINER, name])

def list_objects(prefix):
    r = swift_run(["list", CONTAINER, "--prefix", prefix])
    return [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]

def load_training_stats():
    """Load baseline stats from training dataset manifest."""
    log.info(f"Loading training manifest for {DATASET_VERSION}...")
    local = "/tmp/manifest.json"
    swift_download(f"datasets/{DATASET_VERSION}/manifest.json", local)
    with open(local) as f:
        return json.load(f)

def load_production_sessions():
    """Load recent production sessions from Swift."""
    log.info("Loading production sessions...")
    chunks = list_objects("production/sessions/")
    if not chunks:
        log.warning("No production sessions found")
        return None

    dfs = []
    for chunk in chunks[-10:]:  # last 10 batches
        local = "/tmp/prod_chunk.parquet"
        swift_download(chunk, local)
        try:
            df = pd.read_parquet(local, engine="pyarrow")
            dfs.append(df)
        except Exception as e:
            log.warning(f"Skip {chunk}: {e}")

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)

def compute_drift_metrics(training_stats, prod_df):
    """Compare production data distribution vs training baseline."""
    metrics = {}
    now = datetime.now(timezone.utc).isoformat()

    # production volume
    metrics["prod_total_sessions"]    = len(prod_df)
    metrics["prod_unique_users"]      = prod_df["user_id"].nunique()
    metrics["prod_avg_session_length"] = prod_df["num_tracks"].mean()

    # session length drift
    train_avg_len = training_stats.get("train_sequences", 0) / max(training_stats.get("train_sessions", 1), 1)
    prod_avg_len  = prod_df["num_tracks"].mean()
    metrics["session_length_drift"]   = abs(prod_avg_len - train_avg_len) / max(train_avg_len, 1)

    # new user rate (users not in training)
    train_users = training_stats.get("unique_users", 0)
    prod_users  = prod_df["user_id"].nunique()
    metrics["new_user_rate"]          = max(0, prod_users - train_users) / max(train_users, 1)

    # play ratio distribution
    if "play_ratios" in prod_df.columns:
        all_ratios = []
        for ratios in prod_df["play_ratios"].dropna():
            if isinstance(ratios, list):
                all_ratios.extend(ratios)
        if all_ratios:
            metrics["prod_avg_playratio"] = float(np.mean(all_ratios))
            metrics["prod_skip_rate"]     = float(np.mean([r <= 0.25 for r in all_ratios]))

    # drift alert thresholds
    metrics["drift_alert"] = int(
        metrics["session_length_drift"] > 0.3 or
        metrics["new_user_rate"] > 0.5
    )

    metrics["monitor_timestamp"] = now
    log.info(f"Drift metrics: {json.dumps(metrics, indent=2, default=str)}")
    return metrics

def push_to_mlflow(metrics):
    """Log drift metrics to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("navidrome-data-drift")

    with mlflow.start_run(run_name=f"drift_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tags({
            "monitor_type":    "data_drift",
            "dataset_version": DATASET_VERSION,
        })
        numeric = {k: v for k, v in metrics.items()
                   if isinstance(v, (int, float)) and k != "drift_alert"}
        mlflow.log_metrics(numeric)
        mlflow.log_param("drift_alert", metrics.get("drift_alert", 0))
        mlflow.log_param("dataset_version", DATASET_VERSION)

        if metrics.get("drift_alert"):
            log.warning("DRIFT ALERT — consider retraining!")
            mlflow.set_tag("alert", "DRIFT_DETECTED")

    log.info(f"Metrics pushed to MLflow: {MLFLOW_URI}")

if __name__ == "__main__":
    log.info("=== Navidrome Drift Monitor ===")

    training_stats = load_training_stats()
    prod_df        = load_production_sessions()

    if prod_df is None or len(prod_df) == 0:
        log.warning("No production data yet — skipping drift check")
        exit(0)

    metrics = compute_drift_metrics(training_stats, prod_df)
    push_to_mlflow(metrics)

    log.info("=== Drift Monitor Complete ===")
    if metrics.get("drift_alert"):
        log.warning("ACTION REQUIRED: Drift detected — trigger retraining")
        exit(1)
    exit(0)

"""
Integration test for finetune_gru4rec.py.

Pulls real data and model artifacts from MinIO fully autonomously:
  - Model:   latest under artifacts/finetune/ (falls back to pretrain/)
  - Dataset: latest v{YYYYMMDD}-{HHMMSS}-live from navidrome-datasets,
             starting from today and walking backwards up to 7 days

Requires env vars:
    MINIO_URL       e.g. http://129.114.27.204:9000
    MINIO_USER      access key
    MINIO_PASSWORD  secret key

Optional:
    MLFLOW_TRACKING_URI   defaults to http://129.114.27.204:8000

Usage:
    cd train/
    python test_finetune.py
"""

import os
import sys
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_env():
    missing = [v for v in ("MINIO_URL", "MINIO_USER", "MINIO_PASSWORD") if not os.environ.get(v)]
    if missing:
        print(f"[SKIP] Missing env vars: {', '.join(missing)}")
        sys.exit(0)


def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"Running {label} ...")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"\n[FAIL] {label} exited with non-zero status")
        sys.exit(1)
    print(f"[OK] {label} completed.")


def main():
    check_env()

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.27.204:8000")
    tmpdir     = tempfile.mkdtemp(prefix="gru4rec_integ_")

    print(f"Temp dir: {tmpdir}")
    print(f"MLflow:   {mlflow_uri}")
    print(f"MinIO:    {os.environ['MINIO_URL']}")
    print("Model:    auto-discover latest under artifacts/finetune/ → pretrain/")
    print("Dataset:  auto-detect latest v{YYYYMMDD}-{HHMMSS}-live, today → up to 7 days back")

    run(
        [
            sys.executable, "finetune_gru4rec.py",
            # no model/data args: both fully auto-detected from MinIO
            "--epochs",     "3",
            "--batch-size", "512",
            "--lr",         "1e-4",
            "--top-n",      "20",
            "--patience",   "2",
            "--mlflow-uri", mlflow_uri,
            "--experiment", "integration-test-finetune",
            "--cache-dir",  tmpdir,
        ],
        label="finetune_gru4rec.py",
    )

    print("\n[PASS] Integration test completed successfully.")
    print(f"Artifacts cached in: {tmpdir}")


if __name__ == "__main__":
    main()

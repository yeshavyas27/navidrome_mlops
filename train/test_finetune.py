"""
Integration test for finetune_gru4rec.py and eval_gru4rec.py.

Pulls real data and model artifacts from MinIO, then runs the full
fine-tuning and eval pipeline end-to-end.

Requires env vars:
    MINIO_URL       e.g. http://129.114.27.204:9000
    MINIO_USER      access key
    MINIO_PASSWORD  secret key

Optional:
    MLFLOW_TRACKING_URI   defaults to http://129.114.27.204:8000
    FINETUNE_DATA_VERSION pin a specific dataset version (e.g. v20260420-001232-live);
                          auto-detects latest for today's UTC date if unset
    PRETRAIN_MODEL_KEY    pin a specific MinIO model key; auto-discovers latest pretrain otherwise

Usage:
    cd train/
    python test_finetune.py
"""

import os
import sys
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minio_store import get_client, get_latest_model_key


def check_env():
    missing = [v for v in ("MINIO_URL", "MINIO_USER", "MINIO_PASSWORD") if not os.environ.get(v)]
    if missing:
        print(f"[SKIP] Missing env vars: {', '.join(missing)}")
        sys.exit(0)


def discover_pretrain_keys(s3) -> tuple[str, str]:
    """Return (model_key, vocab_key) for the latest pretrained model in MinIO."""
    model_key = os.environ.get("PRETRAIN_MODEL_KEY") or get_latest_model_key(s3, run_type="pretrain")
    if not model_key:
        print("[FAIL] No pretrained model found in gru4rec-models/pretrain/")
        sys.exit(1)
    vocab_key = model_key.replace("model.pt", "vocab.pkl")
    print(f"  pretrain model → {model_key}")
    print(f"  pretrain vocab → {vocab_key}")
    return model_key, vocab_key


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

    mlflow_uri         = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.27.204:8000")
    dataset_version    = os.environ.get("FINETUNE_DATA_VERSION", "")  # empty = auto-detect
    tmpdir             = tempfile.mkdtemp(prefix="gru4rec_integ_")

    print(f"Temp dir: {tmpdir}")
    print(f"MLflow:   {mlflow_uri}")
    print(f"MinIO:    {os.environ['MINIO_URL']}")

    s3 = get_client()
    model_key, vocab_key = discover_pretrain_keys(s3)

    # Build --finetune-data-version args: flag with no value → auto-detect today's latest
    version_args = (
        ["--finetune-data-version", dataset_version]
        if dataset_version
        else ["--finetune-data-version"]   # nargs="?" with no value → auto-detect
    )

    run(
        [
            sys.executable, "finetune_gru4rec.py",
            "--pretrain-model-key", model_key,
            "--pretrain-vocab-key", vocab_key,
            *version_args,
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

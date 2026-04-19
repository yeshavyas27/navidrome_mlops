"""
Reload Redis vocab from MinIO after retraining.
Run after each GRU4Rec training run completes.

Usage:
    python3 pipeline/reload_vocab.py --run-id <mlflow_run_id> --date 2026-04-18
    python3 pipeline/reload_vocab.py --latest  # auto-find latest run
"""
import io, os, pickle, json, argparse, logging
import boto3, redis
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

MINIO_URL    = os.getenv("MINIO_URL",      "http://129.114.27.204:9000")
MINIO_USER   = os.getenv("MINIO_USER",     "minioadmin")
MINIO_PASS   = os.getenv("MINIO_PASSWORD", "navidrome2026")
MINIO_BUCKET = os.getenv("MINIO_BUCKET",   "gru4rec-models")
REDIS_HOST   = os.getenv("REDIS_HOST",     "redis.navidrome-platform.svc.cluster.local")
REDIS_PORT   = int(os.getenv("REDIS_PORT", "6379"))

def get_minio():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=MINIO_USER,
        aws_secret_access_key=MINIO_PASS,
        region_name="us-east-1",
    )

def get_redis():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def find_latest_vocab(s3):
    """Find most recent vocab.pkl in MinIO pretrain/ prefix."""
    log.info("Scanning MinIO for latest vocab.pkl...")
    resp = s3.list_objects_v2(Bucket=MINIO_BUCKET, Prefix="pretrain/")
    vocabs = [
        obj for obj in resp.get("Contents", [])
        if obj["Key"].endswith("vocab.pkl")
    ]
    if not vocabs:
        raise FileNotFoundError("No vocab.pkl found in MinIO gru4rec-models/pretrain/")
    latest = sorted(vocabs, key=lambda x: x["LastModified"], reverse=True)[0]
    log.info(f"Latest vocab: {latest['Key']} ({latest['LastModified']})")
    return latest["Key"]

def load_vocab_from_minio(s3, vocab_key):
    """Download and unpickle vocab from MinIO."""
    log.info(f"Downloading {vocab_key}...")
    buf = io.BytesIO()
    s3.download_fileobj(MINIO_BUCKET, vocab_key, buf)
    buf.seek(0)
    vocab = pickle.load(buf)
    log.info(f"Loaded vocab: {len(vocab):,} items")
    return vocab

def reload_redis(item2idx, version_key):
    """Push item2idx into Redis replacing old vocab."""
    r = get_redis()
    log.info(f"Reloading Redis vocab ({len(item2idx):,} items)...")

    pipe = r.pipeline()
    pipe.delete("vocab:item2idx")
    pipe.execute()

    # load in batches
    items = list(item2idx.items())
    for i in range(0, len(items), 10000):
        batch = {str(k): str(v) for k, v in items[i:i+10000]}
        r.hset("vocab:item2idx", mapping=batch)
        if (i // 10000 + 1) % 10 == 0:
            log.info(f"  loaded {i+len(batch):,}/{len(items):,}...")

    r.set("vocab:loaded",  "1")
    r.set("vocab:version", version_key)
    log.info(f"Redis vocab reloaded. Version: {version_key}")

def trigger_feedback_api_reload():
    """Tell feedback API to reload vocab from Redis."""
    import requests
    try:
        url = os.getenv("FEEDBACK_API_URL",
                        "http://feedback-api-proj05.navidrome-platform.svc.cluster.local:8000")
        r = requests.post(f"{url}/api/reload-vocab", timeout=30)
        log.info(f"Feedback API reload: {r.status_code}")
    except Exception as e:
        log.warning(f"Feedback API reload failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id",  help="MLflow run ID")
    parser.add_argument("--date",    help="Date YYYY-MM-DD")
    parser.add_argument("--latest",  action="store_true", help="Auto-find latest")
    args = parser.parse_args()

    s3 = get_minio()

    if args.latest or (not args.run_id):
        vocab_key = find_latest_vocab(s3)
    else:
        date      = args.date or __import__("datetime").datetime.now().strftime("%Y-%m-%d")
        vocab_key = f"pretrain/{date}/{args.run_id}/vocab.pkl"

    item2idx = load_vocab_from_minio(s3, vocab_key)
    reload_redis(item2idx, vocab_key)
    trigger_feedback_api_reload()

    log.info("Done. Redis vocab is now in sync with latest model.")

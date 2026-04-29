#!/usr/bin/env bash
# run_finetune.sh — builds the training image and runs finetune_gru4rec.py
# on the Chameleon AMD MI100 bare-metal server.
#
# Usage:
#   ./run_finetune.sh [finetune-data-version]
#
# Environment (set these or export before calling):
#   MINIO_USER        MinIO access key
#   MINIO_PASSWORD    MinIO secret key
#   MINIO_URL         (optional) defaults to http://129.114.27.204:9000
#   MINIO_BUCKET      (optional) defaults to artifacts
#   MLFLOW_URI        (optional) defaults to http://129.114.27.204:8000

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_VERSION="${1:-}"

# Load credentials from /home/cc/train.env if it exists and vars aren't already set.
# Create it on the MI100 server once:
#   cat > /home/cc/train.env <<EOF
#   MINIO_USER=minioadmin
#   MINIO_PASSWORD=navidrome2026
#   EOF
ENV_FILE="/home/cc/train.env"
if [ -f "$ENV_FILE" ]; then
  set -o allexport
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +o allexport
fi

MINIO_URL="${MINIO_URL:-http://129.114.27.204:9000}"
MINIO_BUCKET="${MINIO_BUCKET:-artifacts}"
MLFLOW_URI="${MLFLOW_URI:-http://129.114.27.204:8000}"

: "${MINIO_USER:?MINIO_USER must be set}"
: "${MINIO_PASSWORD:?MINIO_PASSWORD must be set}"

echo "[finetune] pulling latest code..."
git -C "$REPO_DIR" pull origin navidrome-custom

# echo "[finetune] building train image (cache will make this fast)..."
# docker build \
#   -t train:latest \
#   -f "$REPO_DIR/train/docker_amd/Dockerfile" \
#   "$REPO_DIR/train"

# echo "[finetune] starting finetune run..."
# docker run --rm \
#   --device=/dev/kfd \
#   --device=/dev/dri \
#   --group-add "$(stat -c "%g" /dev/kfd)" \
#   --group-add "$(stat -c "%g" /dev/dri/card0)" \
#   --shm-size=12g \
#   -e MINIO_URL="$MINIO_URL" \
#   -e MINIO_BUCKET="$MINIO_BUCKET" \
#   -e MINIO_USER="$MINIO_USER" \
#   -e MINIO_PASSWORD="$MINIO_PASSWORD" \
#   -e MLFLOW_TRACKING_URI="$MLFLOW_URI" \
#   train:latest \
#   python3 finetune_gru4rec.py

# source /home/cc/train.env && docker run -d -p 8000:8000 -v /home/cc/navidrome_mlops:/home/appuser/work --device=/dev/kfd --device=/dev/dri --group-add $(stat -c "%g" /dev/kfd) --group-add $(stat -c "%g" /dev/dri/card0) --shm-size=12g -e MINIO_USER="$MINIO_USER" -e MINIO_PASSWORD="$MINIO_PASSWORD" -e MLFLOW_TRACKING_URI="http://129.114.27.204:8000" -e MINIO_URL="http://129.114.27.204:9000" -e MINIO_BUCKET="artifacts" -e PROMETHEUS_PUSHGATEWAY_URL="http://129.114.27.204:9090" --name train train:latest sleep infinity

docker exec -i train pip install prometheus_client

docker exec -i train python3 train/finetune_gru4rec.py

echo "[finetune] done."

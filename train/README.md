# Training — Helpful Commands

## Fine-tuning on the Chameleon MI100 Server

### One-time setup on the MI100 server

Create the credentials file:
```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@129.114.108.202 "cat > /home/cc/train.env <<EOF
MINIO_USER=minioadmin
MINIO_PASSWORD=navidrome2026
EOF"
```

### Trigger a finetune run via Argo (from the cluster node)

```bash
cd /tmp/navidrome_mlops && git pull && git submodule update --remote navidrome-iac
kubectl apply -f navidrome-iac/workflows/cron-finetune.yaml -n argo
argo submit --from cronworkflow/cron-finetune -n argo
argo watch -n argo @latest
```

### Run finetune manually on the MI100 server (no Argo)

```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@129.114.108.202
cd ~/navidrome_mlops && git pull origin navidrome-custom
bash train/run_finetune.sh v20260419-finetune-001
```

### Watch logs of a running Argo workflow

```bash
argo logs -n argo @latest --follow
```

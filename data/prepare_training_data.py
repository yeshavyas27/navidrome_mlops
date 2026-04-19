"""
Data preparation script for GRU4Rec + SessionKNN training.
Reads session parquet from Swift bucket, converts to interaction_df format
that gru4rec.py and session_knn.py expect.

Run: source ~/.chi_auth.sh && python data/prepare_training_data.py
"""
import os, subprocess, json, ast
import pandas as pd
import numpy as np

CONTAINER = "navidrome-bucket-proj05"
VERSION   = "v20260406-001"
SKIP_RATIO_THRESHOLD = 0.25

AUTH_ARGS = [
    "--os-auth-url", os.environ["OS_AUTH_URL"],
    "--os-auth-type", "v3applicationcredential",
    "--os-application-credential-id", os.environ["OS_APPLICATION_CREDENTIAL_ID"],
    "--os-application-credential-secret", os.environ["OS_APPLICATION_CREDENTIAL_SECRET"],
]

def swift_download(name, local):
    r = subprocess.run(["swift"] + AUTH_ARGS + [
        "download", "--output", local, CONTAINER, name
    ], capture_output=True)
    return r.returncode == 0

print(f"Downloading datasets/{VERSION}/ from Swift...")
swift_download(f"datasets/{VERSION}/train_sessions.parquet", "/tmp/train_sessions.parquet")
swift_download(f"datasets/{VERSION}/eval_sessions.parquet",  "/tmp/eval_sessions.parquet")
swift_download(f"datasets/{VERSION}/vocab.json",             "/tmp/vocab.json")

print("Loading sessions...")
train = pd.read_parquet("/tmp/train_sessions.parquet", engine="pyarrow")
eval_ = pd.read_parquet("/tmp/eval_sessions.parquet",  engine="pyarrow")

# add split label before merging
train["split"] = "train"
eval_["split"] = "eval"
sessions = pd.concat([train, eval_], ignore_index=True)
print(f"Total sessions: {len(sessions):,}")
print(f"Columns: {list(sessions.columns)}")

# explode lists into interaction rows
print("Building interaction_df...")
session_rows     = []
interaction_rows = []

for _, row in sessions.iterrows():
    track_ids   = row["track_ids"]
    play_ratios = row["play_ratios"]
    session_id  = str(row["session_id"])
    user_id     = row["user_id"]
    timestamp   = row.get("timestamp", 0)
    split       = row.get("split", "train")

    if isinstance(track_ids, str):
        try:
            track_ids = ast.literal_eval(track_ids)
        except:
            continue
    if isinstance(play_ratios, str):
        try:
            play_ratios = ast.literal_eval(play_ratios)
        except:
            play_ratios = [1.0] * len(track_ids)

    if not isinstance(track_ids, list) or len(track_ids) < 2:
        continue

    session_rows.append({
        "session_id": session_id,
        "user_id":    int(user_id),
        "timestamp":  int(pd.Timestamp(timestamp).timestamp()) if timestamp else 0,
        "num_tracks": len(track_ids),
        "split":      split,
    })

    for pos, (tid, pr) in enumerate(zip(track_ids, play_ratios)):
        pr = float(pr) if pr is not None else None
        skipped = pr is not None and pr <= SKIP_RATIO_THRESHOLD
        interaction_rows.append({
            "session_id": session_id,
            "user_id":    int(user_id),
            "position":   pos,
            "track_id":   int(tid),
            "playtime":   0,
            "playratio":  pr,
            "skipped":    skipped,
        })

session_df     = pd.DataFrame(session_rows)
interaction_df = pd.DataFrame(interaction_rows)

session_df["timestamp"]    = pd.to_numeric(session_df["timestamp"],    errors="coerce").astype("Int64")
session_df["user_id"]      = pd.to_numeric(session_df["user_id"],      errors="coerce").astype("Int64")
interaction_df["track_id"] = pd.to_numeric(interaction_df["track_id"], errors="coerce").astype("Int64")
interaction_df["user_id"]  = pd.to_numeric(interaction_df["user_id"],  errors="coerce").astype("Int64")

print(f"\ninteraction_df shape: {interaction_df.shape}")
print(f"session_df shape:     {session_df.shape}")
print(f"unique tracks:        {interaction_df['track_id'].nunique():,}")
print(f"unique users:         {interaction_df['user_id'].nunique():,}")
print(f"\nSample interaction_df:")
print(interaction_df.head(3).to_string())

os.makedirs("data/processed", exist_ok=True)
interaction_df.to_parquet("data/processed/interactions.parquet", index=False, engine="pyarrow")
session_df.to_parquet("data/processed/sessions.parquet",         index=False, engine="pyarrow")

with open("/tmp/vocab.json") as f:
    vocab = json.load(f)
with open("data/processed/vocab.json", "w") as f:
    json.dump(vocab, f)

print("\nSaved:")
print("  data/processed/interactions.parquet")
print("  data/processed/sessions.parquet")
print("  data/processed/vocab.json")
print(f"\nNext: update gru4rec.py cfg:")
print('  "data_format": "parquet"')
print('  "dataset_root": "data/processed/"')

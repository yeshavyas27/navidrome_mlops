"""
Data preparation script for GRU4Rec + SessionKNN training.
Reads session parquet from Swift, converts to interaction_df format
that gru4rec.py and session_knn.py expect.
Run: python data/prepare_training_data.py
"""
import os, subprocess, json
import pandas as pd
import numpy as np

CONTAINER = "navidrome-bucket-proj05"
VERSION   = "v20260406-001"
AUTH_ARGS = [
    "--os-auth-url", os.environ["OS_AUTH_URL"],
    "--os-auth-type", "v3applicationcredential",
    "--os-application-credential-id", os.environ["OS_APPLICATION_CREDENTIAL_ID"],
    "--os-application-credential-secret", os.environ["OS_APPLICATION_CREDENTIAL_SECRET"],
]

def swift_download(name, local):
    subprocess.run(["swift"] + AUTH_ARGS + [
        "download", "--output", local, CONTAINER, name
    ], capture_output=True)

print("Downloading train_sessions.parquet from Swift...")
swift_download(f"datasets/{VERSION}/train_sessions.parquet", "/tmp/train_sessions.parquet")
swift_download(f"datasets/{VERSION}/eval_sessions.parquet", "/tmp/eval_sessions.parquet")
swift_download(f"datasets/{VERSION}/vocab.json", "/tmp/vocab.json")

print("Loading sessions...")
train = pd.read_parquet("/tmp/train_sessions.parquet", engine="pyarrow")
eval_ = pd.read_parquet("/tmp/eval_sessions.parquet", engine="pyarrow")
sessions = pd.concat([train, eval_], ignore_index=True)
print(f"Total sessions: {len(sessions):,}")

# explode track_ids and play_ratios lists into rows
print("Exploding into interaction rows...")
rows = []
for _, row in sessions.iterrows():
    track_ids   = row["track_ids"]
    play_ratios = row["play_ratios"]
    session_id  = row["session_id"]
    user_id     = row["user_id"]

    if not isinstance(track_ids, list):
        continue

    for pos, (tid, pr) in enumerate(zip(track_ids, play_ratios)):
        rows.append({
            "session_id": session_id,
            "user_id":    int(user_id),
            "track_id":   int(tid),
            "playratio":  float(pr) if pr is not None else None,
            "position":   pos,
        })

interaction_df = pd.DataFrame(rows)
interaction_df["track_id"] = pd.to_numeric(interaction_df["track_id"], errors="coerce").astype("Int64")
interaction_df["user_id"]  = pd.to_numeric(interaction_df["user_id"],  errors="coerce").astype("Int64")

session_df = sessions[["session_id", "user_id", "session_len"]].rename(
    columns={"session_len": "num_tracks"}
)
session_df["user_id"] = pd.to_numeric(session_df["user_id"], errors="coerce").astype("Int64")

print(f"interaction_df shape: {interaction_df.shape}")
print(f"session_df shape:     {session_df.shape}")
print(f"unique tracks:        {interaction_df['track_id'].nunique():,}")
print(f"unique users:         {interaction_df['user_id'].nunique():,}")

# save to parquet for training scripts to read
os.makedirs("data/processed", exist_ok=True)
interaction_df.to_parquet("data/processed/interactions.parquet", index=False)
session_df.to_parquet("data/processed/sessions.parquet", index=False)

# save vocab
with open("/tmp/vocab.json") as f:
    vocab = json.load(f)

with open("data/processed/vocab.json", "w") as f:
    json.dump(vocab, f)

print("\nSaved to data/processed/:")
print("  interactions.parquet")
print("  sessions.parquet")
print("  vocab.json")
print("\nDone. Update gru4rec.py cfg to:")
print('  "dataset_root": "data/processed/"')
print('  "data_format": "parquet"')

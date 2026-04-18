"""
Navidrome Recommendation System - Ingest Pipeline
Uses Swift CLI directly for uploads.
Run: source ~/.chi_auth.sh && python3 pipeline/ingest.py
"""
import os, json, hashlib, zipfile, io, subprocess
import requests
import pandas as pd
from datetime import datetime, timezone

CONTAINER        = "navidrome-bucket-proj05"
CHUNK_SIZE       = 50 * 1024 * 1024
CHECKPOINT_FILE  = os.path.expanduser("~/ingest_checkpoint.json")
FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_SMALL_URL    = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
RUN_ID           = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

AUTH_ARGS = [
    "--os-auth-url", os.environ["OS_AUTH_URL"],
    "--os-auth-type", "v3applicationcredential",
    "--os-application-credential-id", os.environ["OS_APPLICATION_CREDENTIAL_ID"],
    "--os-application-credential-secret", os.environ["OS_APPLICATION_CREDENTIAL_SECRET"],
]

def swift_upload_file(local_path, object_name):
    cmd = ["swift"] + AUTH_ARGS + [
        "upload", "--object-name", object_name,
        CONTAINER, local_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  UPLOAD ERROR: {result.stderr[:200]}")
    else:
        size = os.path.getsize(local_path)
        print(f"  uploaded -> {object_name} ({size/1e6:.1f} MB)")

def swift_upload_bytes(data, object_name):
    tmp = f"/tmp/swift_tmp_{RUN_ID}.bin"
    with open(tmp, "wb") as f:
        f.write(data)
    swift_upload_file(tmp, object_name)
    os.remove(tmp)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": [], "fma_small_chunks": 0}

def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2)

def already_done(cp, key):
    return key in cp["completed"]

def mark_done(cp, key):
    if key not in cp["completed"]:
        cp["completed"].append(key)
    save_checkpoint(cp)

def ingest_fma_metadata(cp):
    key = "fma_metadata"
    if already_done(cp, key):
        print("[STEP 1] already done, skipping")
        return

    print("\n[STEP 1] Downloading FMA metadata (358MB)...")
    r = requests.get(FMA_METADATA_URL, stream=True, timeout=120)
    r.raise_for_status()

    tmp_zip = "/tmp/fma_metadata.zip"
    downloaded = 0
    with open(tmp_zip, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            print(f"  {downloaded/1e6:.0f} MB", end="\r")
    print(f"\n  done: {downloaded/1e6:.0f} MB")

    swift_upload_file(tmp_zip, "raw/fma/fma_metadata.zip")

    manifest = {
        "source": FMA_METADATA_URL,
        "sha256": sha256_file(tmp_zip),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "files": []
    }

    with zipfile.ZipFile(tmp_zip) as z:
        csv_files = [n for n in z.namelist() if n.endswith(".csv")]
        print(f"  found {len(csv_files)} CSV files")

        for name in csv_files:
            print(f"  processing {name}...")
            csv_bytes = z.read(name)

            tmp_csv = f"/tmp/fma_{os.path.basename(name)}"
            with open(tmp_csv, "wb") as f:
                f.write(csv_bytes)
            swift_upload_file(tmp_csv, f"raw/fma/{name}")

            try:
                try:
                    df = pd.read_csv(io.BytesIO(csv_bytes), index_col=0,
                                     header=[0,1], low_memory=False)
                    df.columns = ['_'.join(str(c) for c in col).strip()
                                  for col in df.columns]
                except Exception:
                    df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)

                before = len(df)
                df = df.dropna(how="all")
                after = len(df)
                print(f"    {name}: {before} -> {after} rows")

                tmp_pq = tmp_csv.replace(".csv", ".parquet")
                df_save = df.copy()
                for col in df_save.columns:
                    if df_save[col].dtype == object:
                        df_save[col] = df_save[col].astype(str)
                df_save.columns = [str(c) for c in df_save.columns]
                df_save.to_parquet(tmp_pq, index=True, engine="pyarrow")
                swift_upload_file(tmp_pq, f"processed/fma/{name.replace('.csv', '.parquet')}")

                manifest["files"].append({
                    "name": name,
                    "rows_raw": before,
                    "rows_clean": after,
                })
                if os.path.exists(tmp_pq):
                    os.remove(tmp_pq)
            except Exception as e:
                print(f"    error: {e}")
            finally:
                if os.path.exists(tmp_csv):
                    os.remove(tmp_csv)

    swift_upload_bytes(
        json.dumps(manifest, indent=2).encode(),
        "raw/fma/metadata_manifest.json"
    )
    if os.path.exists(tmp_zip):
        os.remove(tmp_zip)
    mark_done(cp, key)
    print("[STEP 1] FMA metadata complete")

def ingest_fma_small(cp):
    key = "fma_small"
    if already_done(cp, key):
        print("[STEP 2] already done, skipping")
        return

    print("\n[STEP 2] Streaming FMA small (7.2GB) in 50MB chunks...")
    head = requests.head(FMA_SMALL_URL, timeout=30)
    total = int(head.headers.get("content-length", 0))
    print(f"  total: {total/1e9:.2f} GB")

    start_chunk = cp.get("fma_small_chunks", 0)
    r = requests.get(FMA_SMALL_URL, stream=True, timeout=600)
    r.raise_for_status()

    chunk_num = 0
    uploaded  = 0
    buf       = io.BytesIO()
    buf_size  = 0

    for piece in r.iter_content(chunk_size=1024 * 1024):
        buf.write(piece)
        buf_size += len(piece)
        uploaded += len(piece)

        if buf_size >= CHUNK_SIZE:
            if chunk_num >= start_chunk:
                tmp_chunk = f"/tmp/chunk_{chunk_num:05d}.bin"
                buf.seek(0)
                with open(tmp_chunk, "wb") as f:
                    f.write(buf.read())
                swift_upload_file(tmp_chunk, f"raw/fma/fma_small/chunk_{chunk_num:05d}.bin")
                os.remove(tmp_chunk)
                cp["fma_small_chunks"] = chunk_num + 1
                save_checkpoint(cp)

            chunk_num += 1
            buf = io.BytesIO()
            buf_size = 0
            pct = uploaded / total * 100 if total else 0
            print(f"  {uploaded/1e9:.2f}/{total/1e9:.2f} GB ({pct:.1f}%)")

    if buf_size > 0:
        tmp_chunk = f"/tmp/chunk_{chunk_num:05d}.bin"
        buf.seek(0)
        with open(tmp_chunk, "wb") as f:
            f.write(buf.read())
        swift_upload_file(tmp_chunk, f"raw/fma/fma_small/chunk_{chunk_num:05d}.bin")
        os.remove(tmp_chunk)
        chunk_num += 1

    swift_upload_bytes(
        json.dumps({
            "source": FMA_SMALL_URL,
            "total_bytes": uploaded,
            "total_chunks": chunk_num,
            "chunk_size_mb": 50,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "run_id": RUN_ID
        }, indent=2).encode(),
        "raw/fma/fma_small_manifest.json"
    )
    mark_done(cp, key)
    print(f"[STEP 2] FMA small complete -- {chunk_num} chunks")

def compute_features(cp):
    key = "features"
    if already_done(cp, key):
        print("[STEP 3] already done, skipping")
        return

    print("\n[STEP 3] Computing audio features from echonest.csv...")

    csv_path = "/tmp/echonest.csv"
    cmd = ["swift"] + AUTH_ARGS + [
        "download", "--output", csv_path,
        CONTAINER, "raw/fma/fma_metadata/echonest.csv"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not os.path.exists(csv_path):
        print(f"  echonest.csv download failed: {result.stderr[:100]}")
        return

    df = pd.read_csv(csv_path, index_col=0, header=[0,1], low_memory=False)
    df.columns = ['_'.join(str(c) for c in col).strip() for col in df.columns]
    print(f"  loaded echonest: {df.shape}")

    # select only numeric columns that match audio keywords
    keywords = ["tempo", "loudness", "key", "mode", "energy",
                "danceability", "chroma", "mfcc", "spectral"]
    core = [c for c in df.columns
            if any(k in c.lower() for k in keywords)][:20]
    print(f"  selected {len(core)} features: {core[:5]}...")

    feat = df[core].copy()
    feat = feat.apply(pd.to_numeric, errors="coerce")
    before = len(feat)
    feat = feat.dropna(thresh=max(1, len(feat.columns)//2))
    print(f"  rows: {before} -> {len(feat)}")

    for col in feat.columns:
        mn, mx = feat[col].min(), feat[col].max()
        feat[col] = (feat[col] - mn) / (mx - mn) if mx > mn else 0.0

    version = datetime.now(timezone.utc).strftime("%Y%m%d")

    # save as CSV (avoids pyarrow type issues)
    tmp_csv = f"/tmp/embeddings_{version}.csv"
    feat.to_csv(tmp_csv, index=True)
    swift_upload_file(tmp_csv, f"features/song-audio/v{version}/embeddings.csv")
    os.remove(tmp_csv)

    swift_upload_bytes(
        json.dumps({
            "version": version,
            "n_songs": len(feat),
            "n_features": len(feat.columns),
            "features": list(feat.columns),
            "normalization": "min-max [0,1]",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": RUN_ID
        }, indent=2).encode(),
        f"features/song-audio/v{version}/feature_manifest.json"
    )
    os.remove(csv_path)
    mark_done(cp, key)
    print(f"[STEP 3] Features done -- {feat.shape}")

if __name__ == "__main__":
    print(f"=== Navidrome Ingest Pipeline | run {RUN_ID} ===")
    print(f"Container: {CONTAINER}")

    cp = load_checkpoint()
    print(f"Checkpoint: {cp['completed']}")

    ingest_fma_metadata(cp)
    ingest_fma_small(cp)
    compute_features(cp)

    swift_upload_bytes(
        json.dumps({
            "run_id": RUN_ID,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "steps": cp["completed"],
            "status": "success"
        }, indent=2).encode(),
        f"raw/run_manifest_{RUN_ID}.json"
    )
    print("\n=== INGEST COMPLETE ===")


def compute_features_v2(cp):
    """Fixed version - uses named audio features from echonest.csv"""
    key = "features_v2"
    if already_done(cp, key):
        print("[STEP 3b] already done, skipping")
        return

    print("\n[STEP 3b] Computing correct audio features...")

    csv_path = "/tmp/echonest_fix.csv"
    cmd = ["swift"] + AUTH_ARGS + [
        "download", "--output", csv_path,
        CONTAINER, "raw/fma/fma_metadata/echonest.csv"
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    df = pd.read_csv(csv_path, index_col=0, header=[0,1], low_memory=False)

    # extract the 8 named audio features
    audio = df['echonest'][['audio_features', 'audio_features.1', 'audio_features.2',
                             'audio_features.3', 'audio_features.4', 'audio_features.5',
                             'audio_features.6', 'audio_features.7']].copy()
    audio.columns = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                     'liveness', 'speechiness', 'tempo', 'valence']
    audio = audio.apply(pd.to_numeric, errors='coerce')

    before = len(audio)
    audio = audio.dropna(thresh=6)
    print(f"  rows: {before} -> {len(audio)}")

    # normalize to [0,1]
    for col in audio.columns:
        if col == 'tempo':
            audio[col] = (audio[col] - audio[col].min()) / (audio[col].max() - audio[col].min())
        # others already in [0,1]

    version = datetime.now(timezone.utc).strftime("%Y%m%d")
    tmp_csv = f"/tmp/embeddings_v2_{version}.csv"
    audio.to_csv(tmp_csv, index=True)
    swift_upload_file(tmp_csv, f"features/song-audio/v{version}/embeddings_audio_features.csv")
    os.remove(tmp_csv)
    os.remove(csv_path)

    swift_upload_bytes(
        json.dumps({
            "version": version,
            "n_songs": len(audio),
            "features": list(audio.columns),
            "description": "8 named Echo Nest audio features for BPR-kNN cold start",
            "normalization": "min-max [0,1], tempo normalized, others already in range",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": RUN_ID
        }, indent=2).encode(),
        f"features/song-audio/v{version}/audio_feature_manifest.json"
    )
    mark_done(cp, key)
    print(f"[STEP 3b] Correct features done -- {audio.shape}")
    print(f"  Features: {list(audio.columns)}")

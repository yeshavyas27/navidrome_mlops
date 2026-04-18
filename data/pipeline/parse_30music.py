"""
Navidrome - 30Music Parser (chunked version)
Parses idomaar files in 100K row batches to avoid OOM.
Run: source ~/.chi_auth.sh && python3 pipeline/parse_30music.py
"""
import os, json, io, tarfile, subprocess
import pandas as pd
from datetime import datetime, timezone

CONTAINER  = "navidrome-bucket-proj05"
TAR_LOCAL  = "/tmp/ThirtyMusic.tar.gz"
RUN_ID     = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
CHECKPOINT = os.path.expanduser("~/30music_checkpoint.json")
CHUNK_SIZE = 100000  # rows per batch

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

def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"completed": []}

def save_checkpoint(cp):
    with open(CHECKPOINT, "w") as f:
        json.dump(cp, f, indent=2)

def parse_idomaar_line(line):
    parts = line.strip().split("\t")
    if len(parts) < 4:
        return None
    try:
        props = json.loads(parts[3])
        props["id"] = int(parts[1])
        props["timestamp"] = int(parts[2]) if parts[2] != "-1" else None
        if len(parts) > 4:
            try:
                props["relations"] = json.loads(parts[4])
            except Exception:
                pass
        return props
    except Exception:
        return None

def parse_and_upload_chunked(tar, member, entity_type, object_prefix):
    """Parse idomaar file in chunks to avoid OOM."""
    print(f"  parsing {member.name} ({member.size/1e6:.1f} MB) in {CHUNK_SIZE:,}-row chunks...")
    f = tar.extractfile(member)
    if f is None:
        return 0, 0

    total_accepted = 0
    total_rejected = 0
    chunk_num = 0
    records = []

    for i, line in enumerate(io.TextIOWrapper(f, encoding="utf-8", errors="replace")):
        if not line.strip():
            continue
        row = parse_idomaar_line(line)
        if row is None:
            total_rejected += 1
            continue
        records.append(row)

        if len(records) >= CHUNK_SIZE:
            df = pd.DataFrame(records)
            df = df.dropna(axis=1, how="all")
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str)

            tmp_pq = f"/tmp/30music_{entity_type}_chunk{chunk_num:04d}.parquet"
            df.to_parquet(tmp_pq, index=False, engine="pyarrow")
            swift_upload_file(tmp_pq, f"processed/30music/chunks/{entity_type}/chunk_{chunk_num:04d}.parquet")
            os.remove(tmp_pq)

            total_accepted += len(records)
            chunk_num += 1
            records = []
            print(f"    chunk {chunk_num}: {total_accepted:,} rows processed so far...")

    # flush remaining
    if records:
        df = pd.DataFrame(records)
        df = df.dropna(axis=1, how="all")
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str)

        tmp_pq = f"/tmp/30music_{entity_type}_chunk{chunk_num:04d}.parquet"
        df.to_parquet(tmp_pq, index=False, engine="pyarrow")
        swift_upload_file(tmp_pq, f"processed/30music/chunks/{entity_type}/chunk_{chunk_num:04d}.parquet")
        os.remove(tmp_pq)
        total_accepted += len(records)
        chunk_num += 1

    print(f"    done: {total_accepted:,} accepted, {total_rejected:,} rejected in {chunk_num} chunks")
    return total_accepted, total_rejected

def run():
    cp = load_checkpoint()
    print(f"=== 30Music Parser | run {RUN_ID} ===")
    print(f"Checkpoint: {cp['completed']}")

    if not os.path.exists(TAR_LOCAL):
        print(f"Downloading {TAR_LOCAL} from Swift...")
        cmd = ["swift"] + AUTH_ARGS + [
            "download", "--output", TAR_LOCAL,
            CONTAINER, "raw/30music/ThirtyMusic.tar.gz"
        ]
        subprocess.run(cmd)

    targets = {
        "tracks":   ("tracks",   "entities/tracks.idomaar"),
        "users":    ("users",    "entities/users.idomaar"),
        "playlist": ("playlist", "entities/playlist.idomaar"),
    }

    manifest = {
        "source": "raw/30music/ThirtyMusic.tar.gz",
        "run_id": RUN_ID,
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        "entities": {}
    }

    with tarfile.open(TAR_LOCAL, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".idomaar"):
                continue
            if member.size == 0:
                continue

            entity_type = None
            for key, (etype, path_hint) in targets.items():
                if key in member.name.lower():
                    entity_type = etype
                    break

            if entity_type is None:
                continue
            if entity_type in cp["completed"]:
                print(f"  {entity_type} already done, skipping")
                continue

            accepted, rejected = parse_and_upload_chunked(
                tar, member, entity_type, f"processed/30music/{entity_type}"
            )

            manifest["entities"][entity_type] = {
                "accepted": accepted,
                "rejected": rejected,
                "source_file": member.name
            }

            cp["completed"].append(entity_type)
            save_checkpoint(cp)

    swift_upload_bytes(
        json.dumps(manifest, indent=2).encode(),
        "processed/30music/manifest.json"
    )

    print("\n=== 30Music parse complete ===")
    for k, v in manifest["entities"].items():
        print(f"  {k}: {v['accepted']:,} accepted")

if __name__ == "__main__":
    run()

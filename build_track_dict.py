"""
Build track lookup dictionary from 30Music track chunks.
Creates a dict: track_id -> {name, artist_name, artist_id}
Saves to Swift as: metadata/track_dict.json

Run: source ~/.chi_auth.sh && python3 pipeline/build_track_dict.py
"""
import os, json, subprocess, ast, urllib.parse
import pandas as pd
from datetime import datetime, timezone

CONTAINER = "navidrome-bucket-proj05"

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

def swift_upload_bytes(data, name):
    tmp = f"/tmp/track_dict_tmp.bin"
    with open(tmp, "wb") as f: f.write(data)
    swift_run(["upload", "--object-name", name, CONTAINER, tmp])
    os.remove(tmp)
    print(f"  uploaded -> {name}")

def list_objects(prefix):
    r = swift_run(["list", CONTAINER, "--prefix", prefix])
    return [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]

def decode_name(raw_name):
    """Decode URL-encoded name and split into artist/title."""
    try:
        decoded = urllib.parse.unquote_plus(str(raw_name))
        if "/_/" in decoded:
            parts  = decoded.split("/_/", 1)
            artist = parts[0].strip()
            title  = parts[1].strip()
        else:
            artist = ""
            title  = decoded.strip()
        return artist, title
    except Exception:
        return "", str(raw_name)

def parse_artist_id(relations):
    """Extract first artist ID from relations field."""
    try:
        if isinstance(relations, str):
            relations = ast.literal_eval(relations)
        if isinstance(relations, dict):
            artists = relations.get("artists", [])
            if artists:
                return artists[0].get("id", None)
    except Exception:
        pass
    return None

if __name__ == "__main__":
    print(f"=== Building Track Dictionary ===")

    # load all track chunks
    chunks = list_objects("processed/30music/chunks/tracks/")
    print(f"Found {len(chunks)} track chunks")

    track_dict = {}  # track_id -> {title, artist_name, artist_id}

    for i, chunk in enumerate(chunks):
        local = f"/tmp/track_chunk_{i}.parquet"
        swift_download(chunk, local)

        try:
            df = pd.read_parquet(local, engine="pyarrow")

            for _, row in df.iterrows():
                track_id = str(row["id"])
                raw_name = row.get("name", "")
                relations = row.get("relations", {})

                artist_name, title = decode_name(raw_name)
                artist_id = parse_artist_id(relations)

                track_dict[track_id] = {
                    "track_id":   track_id,
                    "title":      title,
                    "artist":     artist_name,
                    "artist_id":  str(artist_id) if artist_id is not None else None,
                }

        except Exception as e:
            print(f"  skip {chunk}: {e}")
        finally:
            if os.path.exists(local):
                os.remove(local)

        if (i+1) % 10 == 0:
            print(f"  processed {i+1}/{len(chunks)} chunks, {len(track_dict):,} tracks...")

    print(f"\nTotal tracks in dict: {len(track_dict):,}")

    # sample output
    sample_ids = list(track_dict.keys())[:3]
    for sid in sample_ids:
        print(f"  {sid}: {track_dict[sid]}")

    # upload to Swift
    print("\nUploading to Swift...")
    swift_upload_bytes(
        json.dumps(track_dict).encode(),
        "metadata/track_dict.json"
    )

    # also upload a manifest
    manifest = {
        "created_at":   datetime.now(timezone.utc).isoformat(),
        "total_tracks": len(track_dict),
        "fields":       ["track_id", "title", "artist", "artist_id"],
        "usage":        "lookup by track_id to get title and artist for YouTube search",
        "swift_url":    f"https://chi.uc.chameleoncloud.org:7480/swift/v1/AUTH_7c0a7a1952e44c94aa75cae1ff5dc9b4/{CONTAINER}/metadata/track_dict.json"
    }
    swift_upload_bytes(
        json.dumps(manifest, indent=2).encode(),
        "metadata/track_dict_manifest.json"
    )

    print(f"\nDone!")
    print(f"Public URL: {manifest['swift_url']}")
    print(f"\nUsage:")
    print(f"  track_dict['4698874'] -> {{title, artist, artist_id}}")

"""
Navidrome Recommendation System - Validation Pipeline
Reads processed parquet files from Swift, validates, rejects bad rows,
writes clean validated versions back to Swift.
Run: source ~/.chi_auth.sh && python3 pipeline/validate.py
"""
import os, json, io, subprocess
import pandas as pd
from datetime import datetime, timezone

CONTAINER  = "navidrome-bucket-proj05"
RUN_ID     = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

AUTH_ARGS = [
    "--os-auth-url", os.environ["OS_AUTH_URL"],
    "--os-auth-type", "v3applicationcredential",
    "--os-application-credential-id", os.environ["OS_APPLICATION_CREDENTIAL_ID"],
    "--os-application-credential-secret", os.environ["OS_APPLICATION_CREDENTIAL_SECRET"],
]

def swift_download(object_name, local_path):
    cmd = ["swift"] + AUTH_ARGS + [
        "download", "--output", local_path,
        CONTAINER, object_name
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

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
    tmp = f"/tmp/val_tmp_{RUN_ID}.bin"
    with open(tmp, "wb") as f:
        f.write(data)
    swift_upload_file(tmp, object_name)
    os.remove(tmp)

def run_checks(df, checks):
    """
    Run a list of validation checks on a dataframe.
    Each check is a dict with keys: name, mask (boolean series), description
    Returns clean df and report.
    """
    report = {
        "total_rows": len(df),
        "checks": [],
        "rejected": 0,
        "accepted": 0
    }
    mask = pd.Series([True] * len(df), index=df.index)

    for check in checks:
        bad = check["mask"]
        n_bad = int(bad.sum())
        report["checks"].append({
            "rule": check["name"],
            "description": check["description"],
            "rejected": n_bad,
            "pct": round(n_bad / len(df) * 100, 2)
        })
        mask &= ~bad
        if n_bad > 0:
            print(f"    [{check['name']}] rejected {n_bad} rows ({n_bad/len(df)*100:.1f}%)")

    df_clean = df[mask].copy()
    report["rejected"] = len(df) - len(df_clean)
    report["accepted"] = len(df_clean)
    return df_clean, report

# ══════════════════════════════════════════════════════════════
# VALIDATE FMA TRACKS
# ══════════════════════════════════════════════════════════════
def validate_fma_tracks():
    print("\n[1] Validating FMA tracks...")
    local = "/tmp/val_raw_tracks.parquet"

    if not swift_download("processed/fma/fma_metadata/raw_tracks_clean.parquet", local):
        print("  not found, skipping")
        return None

    df = pd.read_parquet(local, engine="pyarrow")
    df["track_listens"] = pd.to_numeric(df["track_listens"], errors="coerce")
    print(f"  loaded: {df.shape}")

    checks = [
        {
            "name": "track_id not null",
            "description": "Every track must have a valid ID",
            "mask": df["track_id"].isna()
        },
        {
            "name": "artist_name not null",
            "description": "Every track must have an artist",
            "mask": df["artist_name"].isna()
        },
        {
            "name": "track_listens >= 0",
            "description": "Listen count cannot be negative",
            "mask": df["track_listens"].notna() & (df["track_listens"] < 0)
        },
        {
            "name": "track_title not null",
            "description": "Every track must have a title",
            "mask": df["track_title"].isna()
        },
    ]

    df_clean, report = run_checks(df, checks)
    print(f"  result: {report['accepted']:,} accepted, {report['rejected']:,} rejected")

    tmp_out = "/tmp/val_tracks_clean.parquet"
    df_clean.to_parquet(tmp_out, index=False, engine="pyarrow")
    swift_upload_file(tmp_out, "validated/fma/tracks.parquet")
    os.remove(tmp_out)
    os.remove(local)
    return report

# ══════════════════════════════════════════════════════════════
# VALIDATE FMA AUDIO FEATURES
# ══════════════════════════════════════════════════════════════
def validate_audio_features():
    print("\n[2] Validating audio features...")
    local = "/tmp/val_embeddings.csv"

    if not swift_download(
        "features/song-audio/v20260406/embeddings_audio_features.csv", local
    ):
        print("  not found, skipping")
        return None

    df = pd.read_csv(local, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    print(f"  loaded: {df.shape}")

    checks = [
        {
            "name": "no all-null rows",
            "description": "Rows with all null features are useless",
            "mask": df.isna().all(axis=1)
        },
        {
            "name": "no all-zero feature vectors",
            "description": "All-zero vectors indicate failed feature extraction",
            "mask": (df == 0).all(axis=1)
        },
        {
            "name": "features in valid range",
            "description": "All features except tempo should be in [0,1]",
            "mask": df.drop(columns=["tempo"], errors="ignore")
                      .apply(lambda x: (x > 1.0) | (x < 0.0))
                      .any(axis=1)
        },
        {
            "name": "tempo in valid range",
            "description": "Normalized tempo should be in [0,1]",
            "mask": df["tempo"].notna() & ((df["tempo"] > 1.0) | (df["tempo"] < 0.0))
            if "tempo" in df.columns else pd.Series([False] * len(df), index=df.index)
        },
    ]

    df_clean, report = run_checks(df, checks)
    print(f"  result: {report['accepted']:,} accepted, {report['rejected']:,} rejected")

    tmp_out = "/tmp/val_embeddings_clean.csv"
    df_clean.to_csv(tmp_out, index=True)
    swift_upload_file(tmp_out, "validated/fma/embeddings_audio_features.csv")
    os.remove(tmp_out)
    os.remove(local)
    return report

# ══════════════════════════════════════════════════════════════
# VALIDATE 30MUSIC TRACKS
# ══════════════════════════════════════════════════════════════
def validate_30music_tracks():
    print("\n[3] Validating 30Music tracks...")
    local = "/tmp/val_30music_tracks.parquet"

    if not swift_download("processed/30music/tracks.parquet", local):
        print("  not found yet — run after parse_30music.py completes")
        return None

    df = pd.read_parquet(local, engine="pyarrow")
    df["playcount"] = pd.to_numeric(df["playcount"], errors="coerce")
    print(f"  loaded: {df.shape}")

    checks = [
        {
            "name": "id not null",
            "description": "Every track must have an ID",
            "mask": df["id"].isna() if "id" in df.columns
                    else pd.Series([False]*len(df), index=df.index)
        },
        {
            "name": "name not null",
            "description": "Every track must have a name",
            "mask": df["name"].isna() if "name" in df.columns
                    else pd.Series([False]*len(df), index=df.index)
        },
        {
            "name": "playcount >= 0",
            "description": "Play count cannot be negative",
            "mask": df["playcount"].notna() & (df["playcount"] < 0)
            if "playcount" in df.columns
            else pd.Series([False]*len(df), index=df.index)
        },
    ]

    df_clean, report = run_checks(df, checks)
    print(f"  result: {report['accepted']:,} accepted, {report['rejected']:,} rejected")

    tmp_out = "/tmp/val_30music_tracks_clean.parquet"
    df_clean.to_parquet(tmp_out, index=False, engine="pyarrow")
    swift_upload_file(tmp_out, "validated/30music/tracks.parquet")
    os.remove(tmp_out)
    os.remove(local)
    return report

# ══════════════════════════════════════════════════════════════
# VALIDATE 30MUSIC USERS
# ══════════════════════════════════════════════════════════════
def validate_30music_users():
    print("\n[4] Validating 30Music users...")
    local = "/tmp/val_30music_users.parquet"

    if not swift_download("processed/30music/users.parquet", local):
        print("  not found yet — run after parse_30music.py completes")
        return None

    df = pd.read_parquet(local, engine="pyarrow")
    df["playcount"] = pd.to_numeric(df["playcount"], errors="coerce")
    print(f"  loaded: {df.shape}")

    checks = [
        {
            "name": "id not null",
            "description": "Every user must have an ID",
            "mask": df["id"].isna() if "id" in df.columns
                    else pd.Series([False]*len(df), index=df.index)
        },
        {
            "name": "playcount > 0",
            "description": "Users with zero plays have no interaction signal",
            "mask": df["playcount"].notna() & (df["playcount"] <= 0)
            if "playcount" in df.columns
            else pd.Series([False]*len(df), index=df.index)
        },
    ]

    df_clean, report = run_checks(df, checks)
    print(f"  result: {report['accepted']:,} accepted, {report['rejected']:,} rejected")

    tmp_out = "/tmp/val_30music_users_clean.parquet"
    df_clean.to_parquet(tmp_out, index=False, engine="pyarrow")
    swift_upload_file(tmp_out, "validated/30music/users.parquet")
    os.remove(tmp_out)
    os.remove(local)
    return report

# ══════════════════════════════════════════════════════════════
# VALIDATE 30MUSIC PLAYLISTS
# ══════════════════════════════════════════════════════════════
def validate_30music_playlists():
    print("\n[5] Validating 30Music playlists...")
    local = "/tmp/val_30music_playlists.parquet"

    if not swift_download("processed/30music/playlists.parquet", local):
        print("  not found yet — run after parse_30music.py completes")
        return None

    df = pd.read_parquet(local, engine="pyarrow")
    print(f"  loaded: {df.shape}")

    checks = [
        {
            "name": "id not null",
            "description": "Every playlist must have an ID",
            "mask": df["id"].isna() if "id" in df.columns
                    else pd.Series([False]*len(df), index=df.index)
        },
        {
            "name": "numtracks > 0",
            "description": "Empty playlists have no training signal",
            "mask": pd.to_numeric(df["numtracks"], errors="coerce") <= 0
            if "numtracks" in df.columns
            else pd.Series([False]*len(df), index=df.index)
        },
        {
            "name": "relations not null",
            "description": "Playlists must have track relations",
            "mask": df["relations"].isna() if "relations" in df.columns
                    else pd.Series([False]*len(df), index=df.index)
        },
    ]

    df_clean, report = run_checks(df, checks)
    print(f"  result: {report['accepted']:,} accepted, {report['rejected']:,} rejected")

    tmp_out = "/tmp/val_30music_playlists_clean.parquet"
    df_clean.to_parquet(tmp_out, index=False, engine="pyarrow")
    swift_upload_file(tmp_out, "validated/30music/playlists.parquet")
    os.remove(tmp_out)
    os.remove(local)
    return report

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"=== Navidrome Validate Pipeline | run {RUN_ID} ===")

    full_report = {
        "run_id": RUN_ID,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {}
    }

    r = validate_fma_tracks()
    if r: full_report["datasets"]["fma_tracks"] = r

    r = validate_audio_features()
    if r: full_report["datasets"]["fma_audio_features"] = r

    r = validate_30music_tracks()
    if r: full_report["datasets"]["30music_tracks"] = r

    r = validate_30music_users()
    if r: full_report["datasets"]["30music_users"] = r

    r = validate_30music_playlists()
    if r: full_report["datasets"]["30music_playlists"] = r

    full_report["completed_at"] = datetime.now(timezone.utc).isoformat()
    full_report["status"] = "success"

    swift_upload_bytes(
        json.dumps(full_report, indent=2).encode(),
        f"validated/validation_report_{RUN_ID}.json"
    )

    print("\n=== VALIDATION COMPLETE ===")
    for k, v in full_report["datasets"].items():
        print(f"  {k}: {v.get('accepted',0):,} accepted, {v.get('rejected',0):,} rejected")

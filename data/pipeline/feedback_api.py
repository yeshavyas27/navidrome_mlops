"""
Navidrome Feedback API — Production Version v3
Receives live session events from Navidrome scrobbler
Writes to PostgreSQL permanently
Buffers to Swift as parquet batches
Uses Redis for instant vocab lookups during inference
Auto-detects latest dataset version from Swift

Internal K8S endpoints:
  PostgreSQL: postgres.navidrome-platform.svc.cluster.local:5432
  Redis:      redis.navidrome-platform.svc.cluster.local:6379
"""
import os, json, io, subprocess, logging
from datetime import datetime, timezone
from typing import List, Optional

import psycopg2
import psycopg2.pool
import redis
import requests
import pandas as pd
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Navidrome Feedback API", version="3.0")

# ============================================================
# CONFIG
# ============================================================
PG_HOST         = os.getenv("PG_HOST",     "postgres.navidrome-platform.svc.cluster.local")
PG_PORT         = int(os.getenv("PG_PORT", "5432"))
PG_DB           = os.getenv("PG_DB",       "navidrome")
PG_USER         = os.getenv("PG_USER",     "postgres")
PG_PASS         = os.getenv("PG_PASS",     "navidrome2026")
REDIS_HOST      = os.getenv("REDIS_HOST",  "redis.navidrome-platform.svc.cluster.local")
REDIS_PORT      = int(os.getenv("REDIS_PORT", "6379"))
SWIFT_CONTAINER = "navidrome-bucket-proj05"
SWIFT_BASE_URL  = f"https://chi.uc.chameleoncloud.org:7480/swift/v1/AUTH_7c0a7a1952e44c94aa75cae1ff5dc9b4/{SWIFT_CONTAINER}"
BUFFER_SIZE     = 100

AUTH_ARGS = [
    "--os-auth-url",   os.getenv("OS_AUTH_URL", ""),
    "--os-auth-type",  "v3applicationcredential",
    "--os-application-credential-id",     os.getenv("OS_APPLICATION_CREDENTIAL_ID", ""),
    "--os-application-credential-secret", os.getenv("OS_APPLICATION_CREDENTIAL_SECRET", ""),
]

# ============================================================
# DYNAMIC DATASET VERSION — always use latest
# ============================================================
def get_latest_dataset_version() -> str:
    """Find latest dataset version from Swift by listing datasets/ prefix."""
    try:
        r = subprocess.run(
            ["swift"] + AUTH_ARGS + ["list", SWIFT_CONTAINER, "--prefix", "datasets/"],
            capture_output=True, text=True
        )
        lines = [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]
        # lines look like: datasets/v20260418-001/interactions.parquet
        versions = set()
        for line in lines:
            parts = line.split("/")
            if len(parts) >= 2 and parts[1].startswith("v"):
                versions.add(parts[1])
        if versions:
            latest = sorted(versions)[-1]
            log.info(f"Latest dataset version: {latest}")
            return latest
    except Exception as e:
        log.warning(f"Could not detect dataset version: {e}")
    return "v20260418-001"  # fallback

DATASET_VERSION = os.getenv("DATASET_VERSION") or get_latest_dataset_version()

# ============================================================
# CONNECTIONS
# ============================================================
pg_pool      = None
redis_client = None
event_buffer = []

def get_pg():
    global pg_pool
    if pg_pool is None:
        pg_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,
            host=PG_HOST, port=PG_PORT,
            dbname=PG_DB, user=PG_USER, password=PG_PASS
        )
    return pg_pool.getconn()

def release_pg(conn):
    pg_pool.putconn(conn)

def get_redis():
    global redis_client
    if redis_client is None:
        redis_client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT,
            decode_responses=True, socket_timeout=2
        )
    return redis_client

# ============================================================
# STARTUP
# ============================================================
@app.on_event("startup")
async def startup():
    log.info(f"Starting Navidrome Feedback API v3 | dataset: {DATASET_VERSION}")
    await setup_postgres()
    await load_vocab_to_redis()

async def setup_postgres():
    try:
        conn = get_pg()
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id                SERIAL PRIMARY KEY,
                session_id        TEXT NOT NULL UNIQUE,
                user_id           TEXT,
                navidrome_user_id TEXT,
                track_ids         TEXT[],
                play_ratios       FLOAT[],
                play_times        FLOAT[],
                num_tracks        INT,
                timestamp         TIMESTAMPTZ DEFAULT NOW(),
                source            TEXT DEFAULT 'live',
                ingested_at       TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id   ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_sessions_source    ON sessions(source);
            CREATE INDEX IF NOT EXISTS idx_sessions_ingested  ON sessions(ingested_at);
        """)
        conn.commit()
        release_pg(conn)
        log.info("PostgreSQL sessions table ready")
    except Exception as e:
        log.error(f"PostgreSQL setup failed: {e}")

async def load_vocab_to_redis():
    try:
        r = get_redis()
        current_version = r.get("vocab:version")
        if current_version == DATASET_VERSION:
            log.info(f"Redis vocab already loaded for {DATASET_VERSION}")
            return
        # version changed or not loaded — reload
        log.info(f"Loading vocab from Swift dataset {DATASET_VERSION}...")
        for vocab_name, redis_key in [("item2idx", "vocab:item2idx"), ("user2idx", "vocab:user2idx")]:
            url  = f"{SWIFT_BASE_URL}/datasets/{DATASET_VERSION}/{vocab_name}.json"
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            vocab = resp.json()
            # clear old vocab
            r.delete(redis_key)
            # load in batches
            pipe  = r.pipeline()
            items = list(vocab.items())
            for i in range(0, len(items), 10000):
                pipe.hset(redis_key, mapping=dict(items[i:i+10000]))
            pipe.execute()
            log.info(f"  {vocab_name}: {len(vocab):,} entries loaded")
        r.set("vocab:loaded",   "1")
        r.set("vocab:version",  DATASET_VERSION)
        log.info(f"Redis vocab loaded for version {DATASET_VERSION}")
    except Exception as e:
        log.error(f"Redis vocab load failed: {e}")

# ============================================================
# SCHEMAS
# ============================================================
class SessionEvent(BaseModel):
    session_id:       str
    user_id:          str
    prefix_track_ids: List[str]
    playratios:       List[float]
    timestamp:        Optional[str] = None
    source:           Optional[str] = "live"

# ============================================================
# INFERENCE PREPROCESSING
# ============================================================
def preprocess_for_inference(session: SessionEvent) -> dict:
    import numpy as np
    r = get_redis()

    item_idxs    = []
    clean_ratios = []
    unknown      = []

    for tid, pr in zip(session.prefix_track_ids, session.playratios):
        idx = r.hget("vocab:item2idx", str(tid))
        if idx is None:
            unknown.append(tid)
            continue
        pr_clean = float(np.clip(pr, 0.0, 1.0))
        item_idxs.append(int(idx))
        clean_ratios.append(pr_clean)

    user_idx = r.hget("vocab:user2idx", str(session.user_id))
    user_idx = int(user_idx) if user_idx else 0

    return {
        "session_id":         session.session_id,
        "user_id":            session.user_id,
        "user_idx":           user_idx,
        "prefix_track_ids":   session.prefix_track_ids,
        "prefix_item_idxs":   item_idxs,
        "playratios":         clean_ratios,
        "num_known_tracks":   len(item_idxs),
        "num_unknown_tracks": len(unknown),
        "unknown_track_ids":  unknown,
        "vocab_version":      r.get("vocab:version"),
    }

# ============================================================
# WRITE TO POSTGRESQL
# ============================================================
def write_to_postgres(session: SessionEvent):
    try:
        conn = get_pg()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO sessions
                (session_id, user_id, track_ids, play_ratios,
                 num_tracks, timestamp, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO NOTHING
        """, (
            session.session_id,
            session.user_id,
            session.prefix_track_ids,
            session.playratios,
            len(session.prefix_track_ids),
            session.timestamp or datetime.now(timezone.utc).isoformat(),
            session.source,
        ))
        conn.commit()
        release_pg(conn)
        log.info(f"Written session {session.session_id} to PostgreSQL")
    except Exception as e:
        log.error(f"PostgreSQL write failed: {e}")

# ============================================================
# FLUSH TO SWIFT
# ============================================================
def flush_to_swift():
    global event_buffer
    if not event_buffer:
        return
    batch        = event_buffer.copy()
    event_buffer = []
    rows = [{
        "session_id":  s.session_id,
        "user_id":     s.user_id,
        "track_ids":   s.prefix_track_ids,
        "play_ratios": s.playratios,
        "num_tracks":  len(s.prefix_track_ids),
        "timestamp":   s.timestamp or datetime.now(timezone.utc).isoformat(),
        "source":      s.source,
    } for s in batch]

    df  = pd.DataFrame(rows)
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    tmp = f"/tmp/batch_{ts}.parquet"
    df.to_parquet(tmp, index=False, engine="pyarrow")
    name = f"production/sessions/sessions_{ts}_batch{len(batch):04d}.parquet"
    subprocess.run(["swift"] + AUTH_ARGS + [
        "upload", "--object-name", name, SWIFT_CONTAINER, tmp
    ], capture_output=True)
    os.remove(tmp)
    log.info(f"Flushed {len(batch)} events to Swift: {name}")

# ============================================================
# ENDPOINTS
# ============================================================
@app.post("/api/feedback")
async def receive_feedback(session: SessionEvent, background_tasks: BackgroundTasks):
    global event_buffer
    background_tasks.add_task(write_to_postgres, session)
    event_buffer.append(session)
    if len(event_buffer) >= BUFFER_SIZE:
        background_tasks.add_task(flush_to_swift)
    return {"status": "accepted", "session_id": session.session_id}

@app.get("/api/preprocess")
async def preprocess(session_id: str, user_id: str, track_ids: str, playratios: str):
    tids = track_ids.split(",")
    prs  = [float(x) for x in playratios.split(",")]
    session = SessionEvent(
        session_id=session_id, user_id=user_id,
        prefix_track_ids=tids, playratios=prs
    )
    return preprocess_for_inference(session)


@app.get("/api/session/latest")
async def get_latest_session(user_id: str, min_playratio: float = 0.0):
    """Return the most recent session for a user, optionally filtering tracks
    by playratio. Used by recommendations.go (Navidrome Go backend) to seed
    the GRU4Rec input with real session data — chronological order, only
    tracks the user actually listened to (playratio >= threshold).

    Returns 404 if the user has no sessions yet (caller should fall back to
    cold-start or annotation-table-derived input).
    """
    try:
        conn = get_pg()
        cur  = conn.cursor()
        cur.execute(
            """
            SELECT session_id, user_id, track_ids, play_ratios, timestamp
              FROM sessions
             WHERE user_id = %s
             ORDER BY timestamp DESC
             LIMIT 1
            """,
            (user_id,),
        )
        row = cur.fetchone()
        release_pg(conn)
    except Exception as e:
        log.error(f"sessions query failed for user_id={user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"db error: {e}")

    if row is None:
        raise HTTPException(status_code=404, detail="no session for user")

    sid, uid, track_ids, play_ratios, ts = row
    track_ids   = list(track_ids   or [])
    play_ratios = list(play_ratios or [])

    # Filter tracks the user skipped / barely played. Keep arrays aligned by
    # zipping then unzipping so play_ratios[i] always corresponds to track_ids[i].
    if min_playratio > 0.0 and play_ratios:
        kept = [(t, p) for t, p in zip(track_ids, play_ratios) if p >= min_playratio]
        track_ids   = [t for t, _ in kept]
        play_ratios = [p for _, p in kept]

    return {
        "session_id":  sid,
        "user_id":     uid,
        "track_ids":   track_ids,
        "play_ratios": play_ratios,
        "timestamp":   ts.isoformat() if ts else None,
        "num_tracks":  len(track_ids),
    }

@app.post("/api/reload-vocab")
async def reload_vocab():
    """Force reload vocab — call this after retraining."""
    global DATASET_VERSION
    r = get_redis()
    r.delete("vocab:loaded")
    r.delete("vocab:version")
    DATASET_VERSION = get_latest_dataset_version()
    await load_vocab_to_redis()
    return {"status": "reloaded", "version": DATASET_VERSION}

@app.get("/health")
async def health():
    status = {
        "api":             "ok",
        "dataset_version": DATASET_VERSION,
    }
    try:
        conn = get_pg()
        conn.cursor().execute("SELECT 1")
        release_pg(conn)
        status["postgres"] = "ok"
    except Exception as e:
        status["postgres"] = str(e)
    try:
        get_redis().ping()
        status["redis"] = "ok"
        status["vocab_version"] = get_redis().get("vocab:version")
    except Exception as e:
        status["redis"] = str(e)
    return status

@app.get("/api/stats")
async def stats():
    try:
        conn = get_pg()
        cur  = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(*) as total_sessions,
                COUNT(DISTINCT user_id) as unique_users,
                MAX(ingested_at) as last_ingested,
                COUNT(*) FILTER (WHERE source = 'live') as live_sessions,
                COUNT(*) FILTER (WHERE source = 'navidrome_live') as navidrome_sessions
            FROM sessions
        """)
        row = cur.fetchone()
        release_pg(conn)
        return {
            "total_sessions":     row[0],
            "unique_users":       row[1],
            "last_ingested":      str(row[2]),
            "live_sessions":      row[3],
            "navidrome_sessions": row[4],
            "buffer_size":        len(event_buffer),
            "dataset_version":    DATASET_VERSION,
            "vocab_version":      get_redis().get("vocab:version"),
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

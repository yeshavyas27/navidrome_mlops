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

from rollup import ensure_schema, rollup_user, SESSION_ROLLUP_SIZE

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
        ensure_schema(conn)
        release_pg(conn)
        log.info(f"PostgreSQL schema ready (rollup size={SESSION_ROLLUP_SIZE})")
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
class Activity(BaseModel):
    track_id:   str
    play_ratio: float
    timestamp:  str

class ActivityBatch(BaseModel):
    user_id:    str
    activities: List[Activity]
    source:     Optional[str] = "navidrome_live"

class SessionEvent(BaseModel):
    """Legacy payload for /api/feedback. Kept for backwards compatibility."""
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
def write_activities(batch: ActivityBatch):
    """Insert activities into user_activity, then roll up if threshold hit."""
    try:
        conn = get_pg()
        cur  = conn.cursor()
        rows = [
            (batch.user_id, a.track_id, float(a.play_ratio), a.timestamp)
            for a in batch.activities
        ]
        cur.executemany("""
            INSERT INTO user_activity (user_id, track_id, play_ratio, timestamp)
            VALUES (%s, %s, %s, %s)
        """, rows)
        conn.commit()
        sessions_created = rollup_user(conn, batch.user_id, source=batch.source)
        release_pg(conn)
        log.info(
            f"Inserted {len(rows)} activities for user={batch.user_id}; "
            f"rollups created: {sessions_created}"
        )
    except Exception as e:
        log.error(f"Activity write failed: {e}")

def session_event_to_batch(session: SessionEvent) -> ActivityBatch:
    """Compat: explode the legacy session payload into per-track activities."""
    base = session.timestamp or datetime.now(timezone.utc).isoformat()
    activities = [
        Activity(track_id=tid, play_ratio=float(pr), timestamp=base)
        for tid, pr in zip(session.prefix_track_ids, session.playratios)
    ]
    return ActivityBatch(
        user_id=session.user_id,
        activities=activities,
        source=session.source or "navidrome_live",
    )

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
@app.post("/api/activity")
async def receive_activity(batch: ActivityBatch, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_activities, batch)
    return {"status": "accepted", "count": len(batch.activities)}

@app.post("/api/feedback")
async def receive_feedback(session: SessionEvent, background_tasks: BackgroundTasks):
    """Legacy session-shaped payload. Routed through the activity pipeline."""
    global event_buffer
    batch = session_event_to_batch(session)
    background_tasks.add_task(write_activities, batch)
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
async def get_latest_session(
    user_id: str,
    min_playratio: float = 0.0,
    max_tracks: int = 200,
    within_minutes: int = 30,
):
    """Build a recent-activity snapshot for inference AND persist it as a
    sessions row (source='inference') for audit/replay.

    Trigger semantics: every call to this endpoint represents the user
    clicking 'recommend'. We materialise a session so we can later audit
    which prefix produced which recommendation. The 50-track rollup
    (run from /api/activity) handles the orthogonal "user accumulated
    enough plays for training data" trigger and writes
    source='navidrome_live' rows.

    Window:    last `within_minutes` of user_activity (default 30).
    Safety:    cap at `max_tracks` rows (default 200).
    Filter:    drop tracks with play_ratio < min_playratio.

    Activities are NOT marked with the snapshot's session_id, so the
    rollup still consumes them when its 50-track threshold trips.
    The 'inference' source label keeps these snapshots out of the
    training pipeline (build_dataset_live filters source='navidrome_live').

    Returns 404 if the user has no qualifying activity in the window.
    """
    conn = None
    try:
        conn = get_pg()
        cur  = conn.cursor()
        cur.execute(
            """
            SELECT track_id, play_ratio, timestamp
              FROM user_activity
             WHERE user_id = %s
               AND timestamp >= NOW() - (%s::int || ' minutes')::interval
             ORDER BY timestamp DESC, id DESC
             LIMIT %s
            """,
            (user_id, within_minutes, max_tracks),
        )
        rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404,
                                detail=f"no activity in last {within_minutes} minutes")

        rows = list(reversed(rows))  # chronological
        track_ids   = [r[0] for r in rows]
        play_ratios = [float(r[1]) for r in rows]
        timestamps  = [r[2] for r in rows]

        if min_playratio > 0.0:
            kept = [(t, p, ts) for t, p, ts in zip(track_ids, play_ratios, timestamps)
                    if p >= min_playratio]
            track_ids   = [t for t, _, _ in kept]
            play_ratios = [p for _, p, _ in kept]
            timestamps  = [ts for _, _, ts in kept]

        if not track_ids:
            raise HTTPException(status_code=404,
                                detail=f"all activity below min_playratio={min_playratio}")

        first_ts = timestamps[0]
        last_ts  = timestamps[-1]

        # Debounce: if we already wrote a snapshot for this user within
        # the last 60 seconds, return that one instead of writing again.
        # Catches rapid recommend-clicks / UI refreshes without losing audit.
        cur.execute(
            """
            SELECT session_id, num_tracks
              FROM sessions
             WHERE user_id = %s AND source = 'inference'
               AND ingested_at >= NOW() - INTERVAL '60 seconds'
             ORDER BY id DESC LIMIT 1
            """,
            (user_id,),
        )
        recent = cur.fetchone()
        if recent and recent[1] == len(track_ids):
            snap_id = recent[0]
        else:
            snap_id = f"{user_id}_snap_{int(datetime.now(timezone.utc).timestamp())}"
            try:
                cur.execute(
                    """
                    INSERT INTO sessions
                        (session_id, user_id, track_ids, play_ratios,
                         num_tracks, timestamp, end_ts, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id) DO NOTHING
                    """,
                    (snap_id, user_id, track_ids, play_ratios,
                     len(track_ids), first_ts, last_ts, "inference"),
                )
                conn.commit()
            except Exception as e:
                log.warning(f"snapshot session insert failed (non-fatal): {e}")

        return {
            "session_id":  snap_id,
            "user_id":     user_id,
            "track_ids":   track_ids,
            "play_ratios": play_ratios,
            "timestamp":   first_ts.isoformat(),
            "num_tracks":  len(track_ids),
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"user_activity query failed for user_id={user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"db error: {e}")
    finally:
        if conn is not None:
            release_pg(conn)

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
        srow = cur.fetchone()
        cur.execute("""
            SELECT
                COUNT(*) as total_activities,
                COUNT(DISTINCT user_id) as activity_users,
                COUNT(*) FILTER (WHERE session_id IS NULL) as unassigned,
                AVG(play_ratio)::float as avg_play_ratio
            FROM user_activity
        """)
        arow = cur.fetchone()
        release_pg(conn)
        return {
            "total_sessions":     srow[0],
            "unique_users":       srow[1],
            "last_ingested":      str(srow[2]),
            "live_sessions":      srow[3],
            "navidrome_sessions": srow[4],
            "total_activities":   arow[0],
            "activity_users":     arow[1],
            "unassigned_activities": arow[2],
            "avg_play_ratio":     round(arow[3], 3) if arow[3] is not None else None,
            "rollup_size":        SESSION_ROLLUP_SIZE,
            "buffer_size":        len(event_buffer),
            "dataset_version":    DATASET_VERSION,
            "vocab_version":      get_redis().get("vocab:version"),
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

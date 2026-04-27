"""
Shared rollup logic for user_activity → sessions.

A user_activity row records a single song play. Once a user accumulates
SESSION_ROLLUP_SIZE unassigned activities, the oldest N are bundled into
a sessions row and stamped with that session_id.
"""
import os

SESSION_ROLLUP_SIZE = int(os.getenv("SESSION_ROLLUP_SIZE", "50"))


def ensure_schema(conn):
    """Create user_activity + sessions tables and indexes if missing.

    Safe to call repeatedly. Idempotent on existing deployments —
    legacy columns (play_times, navidrome_user_id) are left alone if
    they already exist.
    """
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          SERIAL PRIMARY KEY,
            session_id  TEXT NOT NULL UNIQUE,
            user_id     TEXT NOT NULL,
            track_ids   TEXT[],
            play_ratios FLOAT[],
            num_tracks  INT,
            timestamp   TIMESTAMPTZ DEFAULT NOW(),
            end_ts      TIMESTAMPTZ,
            source      TEXT DEFAULT 'navidrome_live',
            ingested_at TIMESTAMPTZ DEFAULT NOW()
        );
        ALTER TABLE sessions ADD COLUMN IF NOT EXISTS end_ts TIMESTAMPTZ;
        CREATE INDEX IF NOT EXISTS idx_sessions_user_id   ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp);
        CREATE INDEX IF NOT EXISTS idx_sessions_source    ON sessions(source);
        CREATE INDEX IF NOT EXISTS idx_sessions_ingested  ON sessions(ingested_at);

        CREATE TABLE IF NOT EXISTS user_activity (
            id          BIGSERIAL PRIMARY KEY,
            user_id     TEXT NOT NULL,
            track_id    TEXT NOT NULL,
            play_ratio  FLOAT NOT NULL,
            timestamp   TIMESTAMPTZ NOT NULL,
            session_id  TEXT,
            ingested_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_ua_user_unassigned
          ON user_activity(user_id, timestamp) WHERE session_id IS NULL;
        CREATE INDEX IF NOT EXISTS idx_ua_user_ts
          ON user_activity(user_id, timestamp);
    """)
    conn.commit()


def rollup_user(conn, user_id, threshold=None, source="navidrome_live"):
    """Roll up unassigned activities for one user into sessions.

    Loops so a backlog of >2*threshold drains in one call.
    Returns the number of sessions created.
    """
    if threshold is None:
        threshold = SESSION_ROLLUP_SIZE
    cur = conn.cursor()
    sessions_created = 0
    while True:
        cur.execute("""
            SELECT id, track_id, play_ratio, timestamp
            FROM user_activity
            WHERE user_id = %s AND session_id IS NULL
            ORDER BY timestamp ASC, id ASC
            LIMIT %s
        """, (user_id, threshold))
        rows = cur.fetchall()
        if len(rows) < threshold:
            break

        ids         = [r[0] for r in rows]
        track_ids   = [r[1] for r in rows]
        play_ratios = [float(r[2]) for r in rows]
        first_ts    = rows[0][3]
        last_ts     = rows[-1][3]

        # Deterministic + collision-resistant: include first activity id.
        session_id = (
            f"{user_id}_{int(first_ts.timestamp())}"
            f"_{int(last_ts.timestamp())}_{ids[0]}"
        )

        cur.execute("""
            INSERT INTO sessions
                (session_id, user_id, track_ids, play_ratios,
                 num_tracks, timestamp, end_ts, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO NOTHING
        """, (session_id, user_id, track_ids, play_ratios,
              len(rows), first_ts, last_ts, source))

        cur.execute("""
            UPDATE user_activity
            SET session_id = %s
            WHERE id = ANY(%s)
        """, (session_id, ids))

        conn.commit()
        sessions_created += 1
    return sessions_created


def rollup_all_users(conn, threshold=None, source="navidrome_live"):
    """Run rollup for every user with at least one unassigned activity."""
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT user_id FROM user_activity WHERE session_id IS NULL
    """)
    users = [r[0] for r in cur.fetchall()]
    total = 0
    for u in users:
        total += rollup_user(conn, u, threshold=threshold, source=source)
    return total

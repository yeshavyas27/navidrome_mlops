"""
Generate realistic dummy session data and insert into PostgreSQL.
Simulates 7 days of user listening behavior on Navidrome.

Run from K8S node or anywhere with PostgreSQL access:
  python3 generate_dummy_data.py

Or with tunnel:
  python3 generate_dummy_data.py --host localhost --port 5432
"""
import psycopg2
import random
import json
import argparse
from datetime import datetime, timezone, timedelta

# ============================================================
# CONFIG
# ============================================================
DAYS          = 7
USERS         = 500          # realistic active user base
SESSIONS_PER_DAY = 2000      # ~4 sessions per user per day
MIN_TRACKS    = 3
MAX_TRACKS    = 25
SKIP_RATE     = 0.20         # 20% of tracks get skipped
REPLAY_RATE   = 0.05         # 5% replayed (playratio > 1.0)

# realistic track IDs from 30Music dataset
TRACK_IDS = [
    "4698874", "838286", "2588097", "455834", "2460503",
    "1234567", "9876543", "3456789", "7654321", "2345678",
    "8765432", "4567890", "6543210", "3210987", "5678901",
    "1357924", "2468013", "9753108", "8642097", "7531086",
    "6420975", "5319864", "4208753", "3197642", "2086531",
    "1975420", "9864309", "8753198", "7642087", "6530976",
    "5419865", "4308754", "3197643", "2086532", "1975421",
    "9864310", "8753199", "7642088", "6530977", "5419866",
    "4308755", "3197644", "2086533", "1975422", "9864311",
    "8753200", "7642089", "6530978", "5419867", "4308756",
]

# user IDs — mix of known and new users
USER_IDS = [str(i) for i in range(10000, 10500)]

# music genres influence session patterns
GENRE_PROFILES = {
    "binge_listener":   {"session_len": (10, 25), "skip_rate": 0.10, "sessions_per_day": 4},
    "casual_listener":  {"session_len": (3,  8),  "skip_rate": 0.25, "sessions_per_day": 2},
    "explorer":         {"session_len": (5,  15), "skip_rate": 0.30, "sessions_per_day": 3},
    "focused_listener": {"session_len": (8,  20), "skip_rate": 0.05, "sessions_per_day": 2},
}

def get_user_profile(user_id):
    """Assign consistent profile to each user."""
    profiles = list(GENRE_PROFILES.keys())
    idx = int(user_id) % len(profiles)
    return profiles[idx], GENRE_PROFILES[profiles[idx]]

def generate_playratio(skipped=False, replayed=False):
    """Generate realistic playratio."""
    if skipped:
        return round(random.uniform(0.05, 0.24), 2)
    elif replayed:
        return round(random.uniform(1.01, 1.5), 2)
    else:
        return round(random.uniform(0.75, 1.0), 2)

def generate_session(user_id, session_id, timestamp, profile_name, profile):
    """Generate one realistic listening session."""
    n_tracks = random.randint(*profile["session_len"])
    
    # users tend to listen to clusters of related tracks
    anchor_tracks = random.sample(TRACK_IDS, min(5, len(TRACK_IDS)))
    track_pool    = anchor_tracks + random.sample(TRACK_IDS, min(n_tracks, len(TRACK_IDS)))
    
    track_ids   = random.choices(track_pool, k=n_tracks)
    play_ratios = []
    
    for i, tid in enumerate(track_ids):
        is_skipped  = random.random() < profile["skip_rate"]
        is_replayed = not is_skipped and random.random() < REPLAY_RATE
        pr = generate_playratio(skipped=is_skipped, replayed=is_replayed)
        play_ratios.append(pr)
    
    return {
        "session_id":  session_id,
        "user_id":     user_id,
        "track_ids":   track_ids,
        "play_ratios": play_ratios,
        "num_tracks":  n_tracks,
        "timestamp":   timestamp,
        "source":      "navidrome_live",
    }

def generate_all_sessions():
    """Generate 7 days of realistic session data."""
    sessions = []
    now      = datetime.now(timezone.utc)
    start    = now - timedelta(days=DAYS)
    
    session_counter = 0
    
    for day in range(DAYS):
        day_start = start + timedelta(days=day)
        
        # more sessions on weekends
        is_weekend = day_start.weekday() >= 5
        day_sessions = int(SESSIONS_PER_DAY * (1.4 if is_weekend else 1.0))
        
        for _ in range(day_sessions):
            user_id = random.choice(USER_IDS)
            profile_name, profile = get_user_profile(user_id)
            
            # realistic listening hours — peak at evening
            hour_weights = [
                0.1, 0.1, 0.1, 0.1, 0.1, 0.2,   # 0-5am  low
                0.5, 0.8, 1.0, 1.0, 0.9, 0.9,   # 6-11am rising
                1.0, 0.9, 0.8, 0.8, 0.9, 1.2,   # 12-5pm midday
                1.5, 1.8, 2.0, 1.8, 1.5, 1.0,   # 6-11pm peak evening
            ]
            hour   = random.choices(range(24), weights=hour_weights)[0]
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            ts = day_start.replace(
                hour=hour, minute=minute, second=second
            ).isoformat()
            
            session_id = f"dummy_{day}_{session_counter:06d}"
            session    = generate_session(
                user_id, session_id, ts, profile_name, profile
            )
            sessions.append(session)
            session_counter += 1
    
    return sessions

def insert_to_postgres(sessions, host, port, dbname, user, password):
    """Insert all sessions into PostgreSQL."""
    conn = psycopg2.connect(
        host=host, port=port,
        dbname=dbname, user=user, password=password
    )
    cur = conn.cursor()
    
    inserted = 0
    skipped  = 0
    batch_size = 500
    
    for i in range(0, len(sessions), batch_size):
        batch = sessions[i:i+batch_size]
        for s in batch:
            try:
                cur.execute("""
                    INSERT INTO sessions
                        (session_id, user_id, track_ids, play_ratios,
                         num_tracks, timestamp, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id) DO NOTHING
                """, (
                    s["session_id"],
                    s["user_id"],
                    s["track_ids"],
                    s["play_ratios"],
                    s["num_tracks"],
                    s["timestamp"],
                    s["source"],
                ))
                inserted += 1
            except Exception as e:
                skipped += 1

        conn.commit()
        print(f"  inserted {min(i+batch_size, len(sessions))}/{len(sessions)} sessions...")
    
    conn.close()
    return inserted, skipped

def verify(host, port, dbname, user, password):
    """Verify data was inserted correctly."""
    conn = psycopg2.connect(
        host=host, port=port,
        dbname=dbname, user=user, password=password
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(DISTINCT user_id) as unique_users,
            AVG(num_tracks) as avg_tracks,
            MIN(timestamp) as earliest,
            MAX(timestamp) as latest,
            COUNT(*) FILTER (WHERE source = 'navidrome_live') as live_sessions
        FROM sessions
    """)
    row = cur.fetchone()
    conn.close()
    return {
        "total_sessions":  row[0],
        "unique_users":    row[1],
        "avg_tracks":      round(float(row[2] or 0), 2),
        "earliest":        str(row[3]),
        "latest":          str(row[4]),
        "live_sessions":   row[5],
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",     default="129.114.27.204")
    parser.add_argument("--port",     default=5432, type=int)
    parser.add_argument("--dbname",   default="navidrome")
    parser.add_argument("--user",     default="postgres")
    parser.add_argument("--password", default="navidrome2026")
    args = parser.parse_args()

    print(f"=== Generating {DAYS} days of dummy session data ===")
    print(f"  Users: {USERS}")
    print(f"  Sessions per day: {SESSIONS_PER_DAY}")
    print(f"  Total sessions: ~{DAYS * SESSIONS_PER_DAY:,}")

    print("\nGenerating sessions...")
    sessions = generate_all_sessions()
    print(f"Generated {len(sessions):,} sessions")

    print(f"\nInserting to PostgreSQL at {args.host}:{args.port}...")
    inserted, skipped = insert_to_postgres(
        sessions, args.host, args.port,
        args.dbname, args.user, args.password
    )
    print(f"Inserted: {inserted:,} | Skipped: {skipped:,}")

    print("\nVerifying...")
    stats = verify(args.host, args.port, args.dbname, args.user, args.password)
    print(json.dumps(stats, indent=2))
    print("\nDone!")

"""
simulate.py — generate realistic listening activity for synthetic users.

Flow:
  1. Optionally wipe user_activity + sessions tables (--wipe).
  2. Login as Navidrome admin and create user1, user2, user3 if missing
     (idempotent: GETs existing users by username if POST returns conflict).
  3. For each user:
       a. Cold-start: ask the recommender for COLD_START_SEED tracks with
          empty history (recommender's cold_start_alpha handles it).
       b. Loop until len(history) >= TARGET_PLAYS_PER_USER:
            - POST /recommend-by-tracks with current history, top_n = TOP_N
            - Pick PICKS_PER_LOOP recommendations at random
            - POST /api/activity for those tracks, randomized play_ratio
              in [0.25, 1.0] so they all clear the >0.2 dataset filter
            - Append picks to history (repeats allowed)

Run from inside the cluster (e.g. `cc@node1-mlops-proj05`) so the K8s
service DNS resolves. Required env: ADMIN_PASS. Everything else has a
sensible default.

Useful invocations:

  # Quick smoke test, 50 plays/user, fresh start
  ADMIN_PASS=... python3 simulate.py --wipe --target 50

  # Full run, 20k plays/user, no wipe (continue accumulating)
  ADMIN_PASS=... python3 simulate.py

  # Just wipe and exit
  ADMIN_PASS=... python3 simulate.py --wipe-only
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import psycopg2
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("simulate")


# ============================================================
# CONFIG
# ============================================================
NAVIDROME_URL = os.getenv("NAVIDROME_URL", "http://navidrome.navidrome-platform.svc.cluster.local:4533")
SERVE_URL     = os.getenv("SERVE_URL",     "http://navidrome-serve.navidrome-platform.svc.cluster.local:8080")
FEEDBACK_URL  = os.getenv("FEEDBACK_URL",  "http://feedback-api-proj05.navidrome-platform.svc.cluster.local:8000")

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "")

PG_HOST = os.getenv("PG_HOST", "postgres.navidrome-platform.svc.cluster.local")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB   = os.getenv("PG_DB",   "navidrome")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "navidrome2026")

COLD_START_SEED = int(os.getenv("COLD_START_SEED", "5"))
PICKS_PER_LOOP  = int(os.getenv("PICKS_PER_LOOP",  "5"))
TOP_N           = int(os.getenv("TOP_N",           "20"))

USERS = [
    {"username": "user1", "email": "user1@gmail.com", "password": "user123", "name": "User One"},
    {"username": "user2", "email": "user2@gmail.com", "password": "user123", "name": "User Two"},
    {"username": "user3", "email": "user3@gmail.com", "password": "user123", "name": "User Three"},
]

# Cold-start seeds — predefined genre clusters per user so each simulated
# user has a recognisable taste profile. The recommender adapts from there.
COLD_START_SEEDS = {
    "user1": ["1542241", "2019543", "886294", "97143", "2288509"],                            # 90s rock
    "user2": ["10008", "153052", "3542207", "3542211", "22341",
              "529621", "1216639", "1678409"],                                                # indie / alternative
    "user3": ["376525", "375370", "88996", "59999", "329288",
              "548294", "490015", "3666651", "7806"],                                         # pop / dance / r&b
}


# ============================================================
# DB WIPE
# ============================================================
def wipe_tables():
    log.info("Wiping user_activity and sessions tables...")
    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS)
    cur = conn.cursor()
    cur.execute("TRUNCATE user_activity, sessions RESTART IDENTITY")
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM user_activity; SELECT COUNT(*) FROM sessions;")
    conn.close()
    log.info("  done — both tables empty")


# ============================================================
# NAVIDROME ADMIN API
# ============================================================
def admin_login() -> str:
    log.info(f"Logging in as admin '{ADMIN_USER}' at {NAVIDROME_URL}...")
    r = requests.post(
        f"{NAVIDROME_URL}/auth/login",
        json={"username": ADMIN_USER, "password": ADMIN_PASS},
        timeout=10,
    )
    r.raise_for_status()
    body = r.json()
    if "token" not in body:
        raise RuntimeError(f"login response missing 'token': {body}")
    log.info("  authenticated")
    return body["token"]


def _auth_headers(token: str) -> dict:
    return {"X-ND-Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def find_user_by_username(token: str, username: str) -> Optional[dict]:
    """Find an existing Navidrome user by username via the native API list."""
    r = requests.get(
        f"{NAVIDROME_URL}/api/user",
        headers=_auth_headers(token),
        params={"_filter": json.dumps({"userName": username})},
        timeout=10,
    )
    if r.status_code != 200:
        return None
    data = r.json()
    rows = data if isinstance(data, list) else data.get("data", [])
    for row in rows:
        if row.get("userName", "").lower() == username.lower():
            return row
    return None


def create_or_get_user(token: str, u: dict) -> str:
    """Create user, or return existing user's id if already present."""
    existing = find_user_by_username(token, u["username"])
    if existing:
        log.info(f"  {u['username']} already exists -> {existing['id']}")
        return existing["id"]

    r = requests.post(
        f"{NAVIDROME_URL}/api/user",
        headers=_auth_headers(token),
        json={
            "userName": u["username"],
            "name":     u["name"],
            "email":    u["email"],
            "password": u["password"],
            "isAdmin":  False,
        },
        timeout=10,
    )
    if r.status_code in (200, 201):
        body = r.json()
        log.info(f"  created {u['username']} -> {body['id']}")
        return body["id"]

    if r.status_code == 409 or "exists" in r.text.lower():
        existing = find_user_by_username(token, u["username"])
        if existing:
            log.info(f"  {u['username']} already exists (post-conflict) -> {existing['id']}")
            return existing["id"]

    raise RuntimeError(f"user create failed for {u['username']}: HTTP {r.status_code} {r.text}")


def create_users(token: str) -> Dict[str, str]:
    log.info("Creating / fetching test users...")
    return {u["username"]: create_or_get_user(token, u) for u in USERS}


# ============================================================
# RECOMMENDER + ACTIVITY
# ============================================================
def get_recommendations(user_id: str, history: List[str], top_n: int) -> List[str]:
    """Hit /recommend-by-tracks; return list of recommended track_ids."""
    r = requests.post(
        f"{SERVE_URL}/recommend-by-tracks",
        json={
            "session_id":        f"sim-{user_id}-{int(time.time())}",
            "user_id":           user_id,
            # Trim to the most recent 50 — anything older is mostly noise
            # for next-track prediction and bigger payloads risk timeout.
            "track_ids":         history[-50:],
            "exclude_track_ids": [],   # repeats are fine for simulation
            "top_n":             top_n,
        },
        timeout=30,
    )
    r.raise_for_status()
    return [rec["track_id"] for rec in r.json().get("recommendations", [])]


def post_activities(user_id: str, track_ids: List[str], base_ts: datetime = None) -> None:
    """Record N plays for a user via /api/activity, ratios uniform in [0.25, 1.0].

    Timestamps anchor on wall-clock NOW (with 1-second spacing inside the batch)
    so simulated rows interleave naturally with real plays in time-series views.
    The base_ts arg is ignored — kept for backward compatibility with callers.
    """
    now = datetime.now(timezone.utc)
    activities = []
    for i, tid in enumerate(track_ids):
        ts = (now + timedelta(seconds=i)).isoformat()
        activities.append({
            "track_id":   str(tid),
            "play_ratio": round(random.uniform(0.25, 1.0), 3),
            "timestamp":  ts,
        })
    r = requests.post(
        f"{FEEDBACK_URL}/api/activity",
        json={"user_id": user_id, "activities": activities,
              "source": os.getenv("SIM_SOURCE", "navidrome_live")},
        timeout=15,
    )
    r.raise_for_status()


# ============================================================
# SIMULATION LOOP
# ============================================================
def simulate_user(username: str, user_id: str, target_plays: int, start_ts: datetime):
    log.info(f"=== Simulating {username} ({user_id}) (target: {target_plays:,} plays) ===")

    # 1. Cold start — use predefined genre seeds when available, otherwise
    # fall back to the recommender's cold-start path.
    seeds: List[str] = COLD_START_SEEDS.get(username, [])
    if seeds:
        log.info(f"  seeding from predefined cluster ({len(seeds)} tracks)")
    else:
        log.info(f"  no predefined seeds for {username}; asking recommender...")
        for attempt in range(3):
            try:
                seeds = get_recommendations(user_id, [], top_n=COLD_START_SEED)
                break
            except Exception as e:
                log.warning(f"  cold-start attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        if not seeds:
            log.error(f"  cold-start failed for {user_id}; skipping")
            return

    post_activities(user_id, seeds, start_ts)
    history = list(seeds)
    log.info(f"  seeded {len(history)} plays")

    # 2. Recommendation loop
    iter_ts = start_ts + timedelta(seconds=len(history) * 180)
    last_log = time.time()
    while len(history) < target_plays:
        try:
            recs = get_recommendations(user_id, history, top_n=TOP_N)
        except Exception as e:
            log.warning(f"  recs call failed for {user_id} (history={len(history)}): {e}")
            time.sleep(2)
            continue
        if not recs:
            log.warning(f"  empty recommendations at history={len(history)}, stopping early")
            break

        chosen = random.sample(recs, min(PICKS_PER_LOOP, len(recs)))
        try:
            post_activities(user_id, chosen, iter_ts)
        except Exception as e:
            log.warning(f"  activity POST failed: {e}")
            time.sleep(2)
            continue

        history.extend(chosen)
        iter_ts += timedelta(seconds=len(chosen) * 180)

        if time.time() - last_log > 5:
            log.info(f"  user {user_id}: {len(history):,}/{target_plays:,}")
            last_log = time.time()

    log.info(f"  done: {len(history):,} plays for {user_id}")


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=int(os.getenv("TARGET_PLAYS", "20000")),
                    help="Target plays per user (default 20000)")
    ap.add_argument("--wipe", action="store_true", help="Truncate user_activity + sessions before simulating")
    ap.add_argument("--wipe-only", action="store_true", help="Truncate tables and exit")
    ap.add_argument("--skip-create", action="store_true", help="Skip Navidrome user create (assume already exist)")
    args = ap.parse_args()

    if args.wipe or args.wipe_only:
        wipe_tables()
    if args.wipe_only:
        return

    if not ADMIN_PASS:
        log.error("ADMIN_PASS env var required (Navidrome admin password)")
        sys.exit(1)

    if args.skip_create:
        # Re-fetch existing user IDs from a quick login + GET
        token = admin_login()
        user_ids = {u["username"]: find_user_by_username(token, u["username"])["id"] for u in USERS}
    else:
        token = admin_login()
        user_ids = create_users(token)

    log.info(f"Users: {json.dumps(user_ids, indent=2)}")

    start = datetime.now(timezone.utc) - timedelta(hours=2)
    for username, user_id in user_ids.items():
        simulate_user(username, user_id, args.target, start)

    log.info("=== Simulation complete ===")
    log.info(f"Per-user totals: target={args.target:,} plays each")


if __name__ == "__main__":
    main()

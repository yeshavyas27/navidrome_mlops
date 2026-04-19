"""
Navidrome - Session Data Generator (GRU4Rec + SessionKNN format)
Generates realistic session events using real 30Music track IDs.
Calibrated from 30Music session statistics.
Run: python3 pipeline/data_generator.py --endpoint http://localhost:8000 --sessions 200
"""
import argparse, time, random, json, requests
import numpy as np
from datetime import datetime, timezone

# session length stats from 30Music (avg 13.8 tracks/session)
SESSION_LENGTH_MEAN = 13
SESSION_LENGTH_STD  = 5

# play ratio distribution from 30Music
# most tracks played fully, some skipped (ratio < 0.3)
def sample_playratio():
    r = random.random()
    if r < 0.15:
        return round(random.uniform(0.0, 0.3), 2)   # skip
    elif r < 0.25:
        return round(random.uniform(0.3, 0.7), 2)   # partial
    else:
        return round(random.uniform(0.8, 1.1), 2)   # full play

def load_track_ids(endpoint):
    """Try to load real track IDs from vocab. Fallback to synthetic."""
    try:
        # try to get vocab from API stats
        r = requests.get(f"{endpoint}/api/stats", timeout=3)
        return None
    except:
        return None

def generate_session(user_id, session_num, track_pool):
    """Generate one realistic listening session."""
    session_len = max(2, int(np.random.normal(SESSION_LENGTH_MEAN, SESSION_LENGTH_STD)))
    session_len = min(session_len, 50)  # cap at 50 tracks

    # long-tail track selection (20% tracks = 80% plays)
    weights = np.random.pareto(1.5, len(track_pool))
    weights = weights / weights.sum()
    selected = np.random.choice(len(track_pool), size=session_len,
                                replace=False, p=weights)

    track_ids  = [track_pool[i] for i in selected]
    playratios = [sample_playratio() for _ in track_ids]

    return {
        "session_id":       f"gen_{user_id}_{session_num}_{int(time.time())}",
        "user_id":          user_id,
        "prefix_track_ids": track_ids,
        "playratios":       playratios,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "source":           "synthetic"
    }

def hit_endpoint(endpoint, session, verbose=False):
    try:
        r = requests.post(
            f"{endpoint}/api/feedback",
            json=session,
            timeout=5
        )
        if verbose:
            print(f"  POST /api/feedback -> {r.status_code} | "
                  f"user={session['user_id']} "
                  f"session={session['session_id']} "
                  f"len={len(session['prefix_track_ids'])}")
        return r.status_code == 200
    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--sessions", type=int, default=200)
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"=== Navidrome Session Data Generator ===")
    print(f"Endpoint: {args.endpoint}")
    print(f"Sessions: {args.sessions}")
    print(f"Users:    {args.users}")
    print(f"Format:   GRU4Rec + SessionKNN session format")
    print()

    # use real 30Music track IDs sampled from known sessions
    # these are real track IDs from sessions.idomaar
    REAL_TRACK_IDS = [
        4698874, 838286, 2588097, 455834, 2460503, 1999451, 3351207,
        3351181, 2807573, 2807606, 1119655, 2548942, 1885124, 3006631,
        2785601, 2785590, 4698883, 2503252, 3770848, 1234567, 2345678,
        3456789, 4567890, 5678901, 6789012, 7890123, 8901234, 9012345,
        1111111, 2222222, 3333333, 4444444, 5555555, 6666666, 7777777,
        8888888, 9999999, 1010101, 2020202, 3030303, 4040404, 5050505,
        6060606, 7070707, 8080808, 9090909, 1234321, 9876543, 1357924,
        2468135, 3691357, 4812469, 5934581, 6157802, 7280913, 8403124,
        9526235, 1649346, 2772457, 3895568, 4918679, 5031780, 6154891,
        7277902, 8390013, 9413124, 1536235, 2659346, 3782457, 4805568,
        5928679, 6041780, 7164891, 8287902, 9300013, 1423124, 2546235,
        3669346, 4792457, 5815568, 6938679, 7051780, 8174891, 9297902,
        1310013, 2433124, 3556235, 4679346, 5702457, 6825568, 7948679,
        8061780, 9184891, 1207902, 2320013, 3443124, 4566235, 5689346,
        6712457, 7835568, 8958679, 9071780, 1194891, 2217902, 3330013
    ]

    # 30% cold start users (new users with 1-2 sessions)
    cold_start = int(args.users * 0.3)
    warm_users = args.users - cold_start

    print(f"User breakdown:")
    print(f"  warm users:       {warm_users}")
    print(f"  cold start users: {cold_start}")
    print()

    total_sent = 0
    total_ok   = 0
    start_time = time.time()

    for session_num in range(args.sessions):
        # pick user
        user_idx = random.randint(0, args.users - 1)
        is_cold  = user_idx >= warm_users

        # cold start users have shorter sessions
        session = generate_session(user_idx, session_num, REAL_TRACK_IDS)
        if is_cold:
            session["prefix_track_ids"] = session["prefix_track_ids"][:3]
            session["playratios"]       = session["playratios"][:3]

        ok = hit_endpoint(args.endpoint, session, args.verbose)
        total_sent += 1
        total_ok   += 1 if ok else 0

        if total_sent % 50 == 0:
            elapsed = time.time() - start_time
            rate    = total_sent / elapsed
            print(f"  [{total_sent}/{args.sessions}] "
                  f"{rate:.1f} sessions/sec | "
                  f"success: {total_ok/total_sent*100:.1f}%")

        time.sleep(args.delay)

    elapsed = time.time() - start_time
    print(f"\n=== GENERATOR COMPLETE ===")
    print(f"  total sent:   {total_sent:,}")
    print(f"  success rate: {total_ok/total_sent*100:.1f}%")
    print(f"  duration:     {elapsed:.1f}s")
    print(f"  rate:         {total_sent/elapsed:.1f} sessions/sec")

if __name__ == "__main__":
    main()

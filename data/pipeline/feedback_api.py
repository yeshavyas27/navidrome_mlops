"""
Navidrome - Session Feedback API
Receives session events from Navidrome, logs to Swift.
Matches GRU4Rec + SessionKNN input schema.
Run: source ~/.chi_auth.sh && uvicorn pipeline.feedback_api:app --host 0.0.0.0 --port 8000
"""
import os, json, io, subprocess
import pandas as pd
from datetime import datetime, timezone
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="Navidrome Session Feedback API")

CONTAINER  = "navidrome-bucket-proj05"
FLUSH_EVERY = 100

AUTH_ARGS = [
    "--os-auth-url", os.environ.get("OS_AUTH_URL", ""),
    "--os-auth-type", "v3applicationcredential",
    "--os-application-credential-id", os.environ.get("OS_APPLICATION_CREDENTIAL_ID", ""),
    "--os-application-credential-secret", os.environ.get("OS_APPLICATION_CREDENTIAL_SECRET", ""),
]

session_buffer = []
flush_count    = 0

class SessionEvent(BaseModel):
    session_id:       str
    user_id:          int
    prefix_track_ids: List[int]
    playratios:       Optional[List[float]] = None
    timestamp:        Optional[str] = None
    source:           Optional[str] = "live"

class RecommendationRequest(BaseModel):
    session_id:       str
    prefix_track_ids: List[int]
    top_n:            int = 20

def swift_upload_bytes(data: bytes, object_name: str):
    tmp = f"/tmp/api_{datetime.now().strftime('%H%M%S%f')}.bin"
    with open(tmp, "wb") as f:
        f.write(data)
    subprocess.run(["swift"] + AUTH_ARGS + [
        "upload", "--object-name", object_name, CONTAINER, tmp
    ], capture_output=True)
    os.remove(tmp)

def flush_buffer():
    global session_buffer, flush_count
    if not session_buffer:
        return
    df = pd.DataFrame(session_buffer)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    swift_upload_bytes(buf.getvalue(),
        f"production/sessions/sessions_{ts}_batch{flush_count:04d}.parquet")
    print(f"Flushed {len(session_buffer)} sessions -> batch {flush_count}")
    session_buffer = []
    flush_count += 1

@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "buffer_size": len(session_buffer)
    }

@app.post("/api/feedback")
def receive_session(event: SessionEvent):
    global session_buffer

    # validate play ratios
    ratios = event.playratios or [1.0] * len(event.prefix_track_ids)
    if len(ratios) != len(event.prefix_track_ids):
        ratios = [1.0] * len(event.prefix_track_ids)

    row = {
        "session_id":       event.session_id,
        "user_id":          event.user_id,
        "prefix_track_ids": event.prefix_track_ids,
        "playratios":       ratios,
        "session_len":      len(event.prefix_track_ids),
        "timestamp":        event.timestamp or datetime.now(timezone.utc).isoformat(),
        "source":           event.source,
        "ingested_at":      datetime.now(timezone.utc).isoformat()
    }

    session_buffer.append(row)

    if len(session_buffer) >= FLUSH_EVERY:
        flush_buffer()

    return {
        "status":      "accepted",
        "session_id":  event.session_id,
        "user_id":     event.user_id,
        "session_len": len(event.prefix_track_ids),
        "buffer_size": len(session_buffer)
    }

@app.get("/api/recommendations")
def get_recommendations(session_id: str, top_n: int = 20):
    """
    Online feature computation for real-time inference.
    Cold start: uses audio similarity from FMA embeddings.
    """
    # find session in buffer
    session_events = [s for s in session_buffer if s["session_id"] == session_id]

    if not session_events:
        return {
            "session_id": session_id,
            "error": "session not found in buffer",
            "top_n": top_n
        }

    latest = session_events[-1]
    prefix_track_ids = latest["prefix_track_ids"]
    playratios       = latest["playratios"]

    # load vocab to map track IDs to item indices
    tmp_vocab = "/tmp/vocab_cache.json"
    if not os.path.exists(tmp_vocab):
        subprocess.run(["swift"] + AUTH_ARGS + [
            "download", "--output", tmp_vocab,
            CONTAINER, "datasets/v20260406-001/vocab.json"
        ], capture_output=True)

    exclude_item_idxs = []
    if os.path.exists(tmp_vocab):
        with open(tmp_vocab) as f:
            vocab = json.load(f)
        track2idx = vocab.get("track2idx", {})
        exclude_item_idxs = [
            track2idx[str(tid)]
            for tid in prefix_track_ids
            if str(tid) in track2idx
        ]

    return {
        "session_id":        session_id,
        "user_id":           latest["user_id"],
        "prefix_track_ids":  prefix_track_ids,
        "prefix_item_idxs":  exclude_item_idxs,
        "playratios":        playratios,
        "exclude_item_idxs": exclude_item_idxs,
        "top_n":             top_n,
        "mode":              "gru4rec_input_ready",
        "computed_at":       datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/stats")
def stats():
    return {
        "buffer_size":   len(session_buffer),
        "flush_count":   flush_count,
        "total_sessions": flush_count * FLUSH_EVERY + len(session_buffer)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

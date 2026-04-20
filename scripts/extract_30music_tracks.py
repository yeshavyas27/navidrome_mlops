#!/usr/bin/env python3
"""
Extract artist - song pairs from 30Music dataset entities file.

Usage:
    python extract_30music_tracks.py --entities-dir /path/to/30music/entities --output tracks.csv

The 30Music dataset stores track entities in .idomaar format (one JSON per line).
Each line contains track metadata including id, title, and artist.

If you have the vocab.pkl from training, you can also filter to only include
tracks that are in the model vocabulary:
    python extract_30music_tracks.py --vocab /path/to/vocabs.pkl --output tracks.csv
"""

import argparse
import csv
import json
import os
import pickle
import sys


def parse_idomaar_entities(entities_dir: str):
    """Parse .idomaar entity files to extract track metadata."""
    tracks = []
    
    for fname in os.listdir(entities_dir):
        if not fname.endswith(".idomaar"):
            continue
        if "track" not in fname.lower():
            continue
        
        filepath = os.path.join(entities_dir, fname)
        print(f"Parsing {filepath}...")
        
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # idomaar format: tab-separated fields, JSON in later columns
                    parts = line.split("\t")
                    
                    # Try to find JSON object with track metadata
                    for part in parts:
                        part = part.strip()
                        if part.startswith("{"):
                            try:
                                data = json.loads(part)
                                track_id = data.get("id", "")
                                title = data.get("title", data.get("name", ""))
                                artist = data.get("artist", data.get("creator", ""))
                                album = data.get("album", "")
                                
                                if track_id and (title or artist):
                                    tracks.append({
                                        "track_id": str(track_id),
                                        "artist": artist,
                                        "title": title,
                                        "album": album,
                                    })
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    if line_num <= 5:
                        print(f"  Warning line {line_num}: {e}")
                    continue
        
        print(f"  Found {len(tracks)} tracks so far")
    
    return tracks


def parse_vocab_tracks(vocab_path: str):
    """Extract track IDs from vocabs.pkl and create placeholder entries."""
    with open(vocab_path, "rb") as f:
        data = pickle.load(f)
    
    if isinstance(data, tuple):
        item2idx = data[0]
    elif isinstance(data, dict):
        item2idx = data.get("item2idx", data)
    else:
        print(f"Unknown vocab format: {type(data)}")
        return []
    
    tracks = []
    for track_id in item2idx.keys():
        tracks.append({
            "track_id": str(track_id),
            "artist": "",
            "title": "",
            "album": "",
        })
    
    print(f"Found {len(tracks)} tracks in vocab")
    return tracks


def main():
    parser = argparse.ArgumentParser(description="Extract 30Music track metadata to CSV")
    parser.add_argument("--entities-dir", help="Path to 30Music entities directory")
    parser.add_argument("--vocab", help="Path to vocabs.pkl (alternative: extract IDs from vocab)")
    parser.add_argument("--output", default="30music_tracks.csv", help="Output CSV file")
    parser.add_argument("--filter-vocab", help="Path to vocabs.pkl to filter entities to only vocab tracks")
    args = parser.parse_args()
    
    if not args.entities_dir and not args.vocab:
        print("Error: provide --entities-dir or --vocab")
        sys.exit(1)
    
    tracks = []
    
    # Parse entities if provided
    if args.entities_dir:
        tracks = parse_idomaar_entities(args.entities_dir)
    
    # Or extract from vocab
    if args.vocab and not tracks:
        tracks = parse_vocab_tracks(args.vocab)
    
    # Filter to vocab if requested
    if args.filter_vocab and tracks:
        with open(args.filter_vocab, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, tuple):
            vocab_ids = set(str(k) for k in data[0].keys())
        elif isinstance(data, dict):
            vocab_ids = set(str(k) for k in data.get("item2idx", data).keys())
        
        before = len(tracks)
        tracks = [t for t in tracks if t["track_id"] in vocab_ids]
        print(f"Filtered {before} -> {len(tracks)} tracks (matching vocab)")
    
    # Remove duplicates
    seen = set()
    unique_tracks = []
    for t in tracks:
        if t["track_id"] not in seen:
            seen.add(t["track_id"])
            unique_tracks.append(t)
    tracks = unique_tracks
    
    # Write CSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "artist", "title", "album"])
        writer.writeheader()
        writer.writerows(tracks)
    
    print(f"Wrote {len(tracks)} tracks to {args.output}")


if __name__ == "__main__":
    main()

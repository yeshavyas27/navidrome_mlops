// Package navidrome_feedback implements a custom scrobbler that sends
// play events to our feedback API for ML training data collection.
package navidrome_feedback

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/navidrome/navidrome/core/scrobbler"
	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/model"
)

const scraperName = "navidrome_feedback"

var sessionTimeout = 30 * time.Minute

type sessionEntry struct {
	UserID     string
	TrackIDs   []string
	PlayRatios []float64
	StartTime  time.Time
	LastPlay   time.Time
}

type sessionBuffer struct {
	mu       sync.Mutex
	sessions map[string]*sessionEntry
}

type feedbackPayload struct {
	SessionID      string    `json:"session_id"`
	UserID         string    `json:"user_id"`
	PrefixTrackIDs []string  `json:"prefix_track_ids"`
	PlayRatios     []float64 `json:"playratios"`
	Timestamp      string    `json:"timestamp"`
	Source         string    `json:"source"`
}

var buf = &sessionBuffer{sessions: make(map[string]*sessionEntry)}

type feedbackScrobbler struct {
	ds          model.DataStore
	feedbackURL string
	httpClient  *http.Client
	posMu       sync.Mutex
	lastPos     map[string]int
}

func newFeedbackScrobbler(ds model.DataStore) scrobbler.Scrobbler {
	url := os.Getenv("FEEDBACK_API_URL")
	if url == "" {
		url = "http://feedback-api-proj05.navidrome-platform.svc.cluster.local:8000"
	}
	log.Info("Navidrome feedback scrobbler initialized", "feedback_url", url)
	return &feedbackScrobbler{
		ds:          ds,
		feedbackURL: url,
		httpClient:  &http.Client{Timeout: 5 * time.Second},
		lastPos:     make(map[string]int),
	}
}

func (f *feedbackScrobbler) IsAuthorized(_ context.Context, _ string) bool {
	return true
}

func (f *feedbackScrobbler) NowPlaying(_ context.Context, userID string, t *model.MediaFile, position int) error {
	f.posMu.Lock()
	f.lastPos[userID+"_"+t.ID] = position
	f.posMu.Unlock()
	return nil
}

func (f *feedbackScrobbler) Scrobble(ctx context.Context, userID string, s scrobbler.Scrobble) error {
	buf.mu.Lock()
	defer buf.mu.Unlock()

	entry, exists := buf.sessions[userID]
	now := time.Now()

	if !exists || now.Sub(entry.LastPlay) > sessionTimeout {
		entry = &sessionEntry{
			UserID:    userID,
			StartTime: now,
			LastPlay:  now,
		}
		buf.sessions[userID] = entry
		log.Debug(ctx, "New ML session started", "user", userID)
	}

	// ID mapping: extract the 30Music integer track_id from the filename.
	// Files are named like "audio_complete/3012335.mp3" where 3012335 is the
	// 30Music track ID. Fall back to MbzRecordingID, then Navidrome UUID.
	trackID := s.MediaFile.MbzRecordingID
	if trackID == "" {
		base := filepath.Base(s.MediaFile.Path)
		ext := filepath.Ext(base)
		trackID = strings.TrimSuffix(base, ext)
	}
	if trackID == "" {
		trackID = s.MediaFile.ID
	}

	// Compute playratio from the last NowPlaying position; fall back to 1.0
	// when we have no position (e.g. scrobbler fires on a track we never saw
	// NowPlaying for, or duration is missing).
	posKey := userID + "_" + s.MediaFile.ID
	f.posMu.Lock()
	pos := f.lastPos[posKey]
	delete(f.lastPos, posKey)
	f.posMu.Unlock()

	ratio := 1.0
	if s.MediaFile.Duration > 0 && pos > 0 {
		ratio = float64(pos) / float64(s.MediaFile.Duration)
		if ratio > 1.0 {
			ratio = 1.0
		}
	}

	entry.TrackIDs   = append(entry.TrackIDs, trackID)
	entry.PlayRatios = append(entry.PlayRatios, ratio)
	entry.LastPlay   = now

	if len(entry.TrackIDs) >= 3 {
		entryCopy := *entry
		go f.sendSession(ctx, userID, &entryCopy)
		delete(buf.sessions, userID)
	}

	return nil
}

func (f *feedbackScrobbler) sendSession(ctx context.Context, userID string, entry *sessionEntry) {
	sessionID := fmt.Sprintf("%s_%d", userID, entry.StartTime.Unix())
	payload := feedbackPayload{
		SessionID:      sessionID,
		UserID:         userID,
		PrefixTrackIDs: entry.TrackIDs,
		PlayRatios:     entry.PlayRatios,
		Timestamp:      entry.StartTime.UTC().Format(time.RFC3339),
		Source:         "navidrome_live",
	}

	body, err := json.Marshal(payload)
	if err != nil {
		log.Error(ctx, "Failed to marshal ML session", "user", userID, err)
		return
	}

	resp, err := f.httpClient.Post(
		f.feedbackURL+"/api/feedback",
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil {
		log.Error(ctx, "Failed to send ML session", "user", userID, err)
		return
	}
	defer resp.Body.Close()

	log.Info(ctx, "ML session sent",
		"user", userID,
		"tracks", len(entry.TrackIDs),
		"status", resp.StatusCode,
	)
}

func init() {
	scrobbler.Register(scraperName, newFeedbackScrobbler)
}

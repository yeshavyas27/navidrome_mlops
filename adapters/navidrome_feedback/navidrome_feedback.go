// Package navidrome_feedback implements a custom scrobbler that sends
// play events to our feedback API for ML training data collection.
package navidrome_feedback

import (
	"bytes"
	"context"
	"encoding/json"
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

const (
	scraperName       = "navidrome_feedback"
	flushTrackCount   = 1
	scrobbleThreshold = 0.5
	// maxActiveGap is the longest gap between two NowPlaying ticks (or
	// between the last tick and a track-change/scrobble) that we still
	// treat as "actively listening." Anything longer is assumed to be a
	// pause and excluded from ActivePlayTime.
	maxActiveGap = 60 * time.Second
)

var sessionTimeout = 30 * time.Minute

type playbackState struct {
	NavidromeID    string
	TrackID        string
	Duration       float64
	MaxPosition    float64
	Scrobbled      bool
	StartedAt      time.Time
	LastTickAt     time.Time
	ActivePlayTime float64
}

type sessionEntry struct {
	UserID         string
	TrackIDs       []string
	PlayRatios     []float64
	PlayTimestamps []time.Time
	StartTime      time.Time
	LastPlay       time.Time
	InFlight       *playbackState
	LastFinalized  string
}

type sessionBuffer struct {
	mu       sync.Mutex
	sessions map[string]*sessionEntry
}

type activityItem struct {
	TrackID   string  `json:"track_id"`
	PlayRatio float64 `json:"play_ratio"`
	Timestamp string  `json:"timestamp"`
}

type activityPayload struct {
	UserID     string         `json:"user_id"`
	Activities []activityItem `json:"activities"`
	Source     string         `json:"source"`
}

var buf = &sessionBuffer{sessions: make(map[string]*sessionEntry)}

type feedbackScrobbler struct {
	ds          model.DataStore
	feedbackURL string
	httpClient  *http.Client
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
	}
}

// resolveTrackID extracts the 30Music integer ID from the filename
// (audio_complete/3012335.mp3 -> "3012335"), falling back to MbzRecordingID,
// then Navidrome's UUID.
func resolveTrackID(m *model.MediaFile) string {
	if m.MbzRecordingID != "" {
		return m.MbzRecordingID
	}
	base := filepath.Base(m.Path)
	ext := filepath.Ext(base)
	if id := strings.TrimSuffix(base, ext); id != "" {
		return id
	}
	return m.ID
}

func (f *feedbackScrobbler) IsAuthorized(_ context.Context, _ string) bool {
	return true
}

func (f *feedbackScrobbler) NowPlaying(ctx context.Context, userID string, track *model.MediaFile, position int) error {
	if track == nil {
		return nil
	}
	buf.mu.Lock()
	defer buf.mu.Unlock()

	entry := getOrStartSession(ctx, userID)
	pos := float64(position)

	now := time.Now()
	if entry.InFlight == nil || entry.InFlight.NavidromeID != track.ID {
		if entry.InFlight != nil {
			finalizeInFlight(entry)
		}
		entry.InFlight = &playbackState{
			NavidromeID: track.ID,
			TrackID:     resolveTrackID(track),
			Duration:    float64(track.Duration),
			MaxPosition: pos,
			StartedAt:   now,
			LastTickAt:  now,
		}
	} else {
		// Same track — accumulate the gap since the last tick if it looks
		// like uninterrupted play (gap <= maxActiveGap). Larger gaps are
		// treated as a pause and excluded.
		if gap := now.Sub(entry.InFlight.LastTickAt); gap > 0 && gap <= maxActiveGap {
			entry.InFlight.ActivePlayTime += gap.Seconds()
		}
		entry.InFlight.LastTickAt = now
		if pos > entry.InFlight.MaxPosition {
			entry.InFlight.MaxPosition = pos
		}
	}
	entry.LastPlay = now

	f.maybeFlush(ctx, userID, entry)
	return nil
}

func (f *feedbackScrobbler) Scrobble(ctx context.Context, userID string, s scrobbler.Scrobble) error {
	buf.mu.Lock()
	defer buf.mu.Unlock()

	entry := getOrStartSession(ctx, userID)
	navidromeID := s.MediaFile.ID

	switch {
	case entry.InFlight != nil && entry.InFlight.NavidromeID == navidromeID:
		entry.InFlight.Scrobbled = true
	case entry.LastFinalized == navidromeID:
		// Already counted via track-change finalize; Scrobble is a duplicate signal.
	default:
		if entry.InFlight != nil {
			finalizeInFlight(entry)
		}
		entry.TrackIDs = append(entry.TrackIDs, resolveTrackID(&s.MediaFile))
		entry.PlayRatios = append(entry.PlayRatios, 1.0)
		entry.PlayTimestamps = append(entry.PlayTimestamps, time.Now())
		entry.LastFinalized = navidromeID
	}
	entry.LastPlay = time.Now()

	f.maybeFlush(ctx, userID, entry)
	return nil
}

// must hold buf.mu
func getOrStartSession(ctx context.Context, userID string) *sessionEntry {
	now := time.Now()
	entry, exists := buf.sessions[userID]
	if !exists || now.Sub(entry.LastPlay) > sessionTimeout {
		entry = &sessionEntry{
			UserID:    userID,
			StartTime: now,
			LastPlay:  now,
		}
		buf.sessions[userID] = entry
		log.Debug(ctx, "New ML session started", "user", userID)
	}
	return entry
}

// must hold buf.mu
func finalizeInFlight(entry *sessionEntry) {
	p := entry.InFlight
	if p == nil {
		return
	}
	ratio := computeRatio(p)
	entry.TrackIDs = append(entry.TrackIDs, p.TrackID)
	entry.PlayRatios = append(entry.PlayRatios, ratio)
	entry.PlayTimestamps = append(entry.PlayTimestamps, time.Now())
	entry.LastFinalized = p.NavidromeID
	entry.InFlight = nil
}

func computeRatio(p *playbackState) float64 {
	if p.Duration <= 0 {
		if p.Scrobbled {
			return 1.0
		}
		return 0.0
	}
	// Prefer ActivePlayTime — sum of NowPlaying-tick gaps that look like
	// real listening (each gap <= maxActiveGap; longer gaps treated as
	// pauses and skipped). Add the trailing gap from the last tick to
	// finalize if also within threshold.
	played := p.ActivePlayTime
	if !p.LastTickAt.IsZero() {
		if trailing := time.Since(p.LastTickAt).Seconds(); trailing >= 0 && trailing <= maxActiveGap.Seconds() {
			played += trailing
		}
	}
	// If the client never sent a second NowPlaying tick, ActivePlayTime
	// stays 0 and the trailing gap won't help if it's > maxActiveGap.
	// Fall back to wall-clock elapsed so we still produce a useful ratio.
	if played <= 0 && !p.StartedAt.IsZero() {
		played = time.Since(p.StartedAt).Seconds()
	}
	// MaxPosition (when the client sends progressive positions) is the
	// most authoritative signal — never go below it.
	if p.MaxPosition > played {
		played = p.MaxPosition
	}
	r := played / p.Duration
	if p.Scrobbled && r < scrobbleThreshold {
		r = scrobbleThreshold
	}
	if r > 1.0 {
		r = 1.0
	}
	if r < 0.0 {
		r = 0.0
	}
	return r
}

// must hold buf.mu
func (f *feedbackScrobbler) maybeFlush(ctx context.Context, userID string, entry *sessionEntry) {
	if len(entry.TrackIDs) < flushTrackCount {
		return
	}
	flushed := &sessionEntry{
		UserID:         entry.UserID,
		TrackIDs:       append([]string(nil), entry.TrackIDs...),
		PlayRatios:     append([]float64(nil), entry.PlayRatios...),
		PlayTimestamps: append([]time.Time(nil), entry.PlayTimestamps...),
		StartTime:      entry.StartTime,
	}
	go f.sendSession(ctx, userID, flushed)

	// Preserve in-flight track and dedupe state across the flush so the
	// currently-playing track keeps accumulating position data.
	now := time.Now()
	buf.sessions[userID] = &sessionEntry{
		UserID:        userID,
		StartTime:     now,
		LastPlay:      entry.LastPlay,
		InFlight:      entry.InFlight,
		LastFinalized: entry.LastFinalized,
	}
}

func (f *feedbackScrobbler) sendSession(ctx context.Context, userID string, entry *sessionEntry) {
	activities := make([]activityItem, len(entry.TrackIDs))
	for i := range entry.TrackIDs {
		ts := entry.PlayTimestamps[i]
		if ts.IsZero() {
			ts = time.Now()
		}
		activities[i] = activityItem{
			TrackID:   entry.TrackIDs[i],
			PlayRatio: entry.PlayRatios[i],
			Timestamp: ts.UTC().Format(time.RFC3339),
		}
	}
	payload := activityPayload{
		UserID:     userID,
		Activities: activities,
		Source:     "navidrome_live",
	}

	body, err := json.Marshal(payload)
	if err != nil {
		log.Error(ctx, "Failed to marshal activity batch", "user", userID, err)
		return
	}

	resp, err := f.httpClient.Post(
		f.feedbackURL+"/api/activity",
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil {
		log.Error(ctx, "Failed to send activity batch", "user", userID, err)
		return
	}
	defer resp.Body.Close()

	log.Info(ctx, "Activity batch sent",
		"user", userID,
		"activities", len(activities),
		"status", resp.StatusCode,
	)
}

func init() {
	scrobbler.Register(scraperName, newFeedbackScrobbler)
}

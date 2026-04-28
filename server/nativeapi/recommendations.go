package nativeapi

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Masterminds/squirrel"
	"github.com/go-chi/chi/v5"
	"github.com/navidrome/navidrome/conf"
	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/model/request"
)

// Default cluster-internal DNS for the feedback API. Overridable via
// FEEDBACK_API_URL env var on the navidrome deployment.
const defaultFeedbackAPIURL = "http://feedback-api.navidrome-platform.svc.cluster.local:8080"

// Tracks below this playratio are treated as skips and excluded from
// the inference input. Set to 0.0 (no filter) while the new activity-
// based scrobbler's playratio computation stabilises — it currently
// returns 0 even for tracks that scrobbled because NowPlaying position
// isn't being updated continuously, just at track start. Bump back
// toward 0.5 once playratio numbers are accurate.
const minPlayratioThreshold = 0.0

// latestSessionResponse mirrors feedback_api.py's GET /api/session/latest payload.
type latestSessionResponse struct {
	SessionID  string    `json:"session_id"`
	UserID     string    `json:"user_id"`
	TrackIDs   []string  `json:"track_ids"`
	PlayRatios []float64 `json:"play_ratios"`
	Timestamp  string    `json:"timestamp"`
	NumTracks  int       `json:"num_tracks"`
}

// fetchLatestSession returns the most recent session's track IDs for the
// given user from the feedback API, with playratio-based filtering applied
// server-side. Returns (nil, nil) when the user has no session yet (404)
// — caller should fall back to a different signal in that case.
func fetchLatestSession(ctx context.Context, userID string, minPR float64) ([]string, error) {
	base := os.Getenv("FEEDBACK_API_URL")
	if base == "" {
		base = defaultFeedbackAPIURL
	}
	u := fmt.Sprintf("%s/api/session/latest?user_id=%s&min_playratio=%.2f",
		base, url.QueryEscape(userID), minPR)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, nil // user has no sessions yet → caller falls back
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("feedback API status %d: %s", resp.StatusCode, string(body))
	}
	var sess latestSessionResponse
	if err := json.NewDecoder(resp.Body).Decode(&sess); err != nil {
		return nil, err
	}
	return sess.TrackIDs, nil
}

// RecommendationItem represents a single recommendation returned to the UI
type RecommendationItem struct {
	ID      string  `json:"id"`
	Title   string  `json:"title"`
	Artist  string  `json:"artist"`
	Album   string  `json:"album"`
	Score   float64 `json:"score"`
	Rank    int     `json:"rank,omitempty"`
	TrackID string  `json:"track_id,omitempty"`
}

// RecommendationResponse is the response sent to the UI
type RecommendationResponse struct {
	UserID          string               `json:"userId"`
	Recommendations []RecommendationItem `json:"recommendations"`
	ModelVersion    string               `json:"modelVersion,omitempty"`
	GeneratedAt     string               `json:"generatedAt,omitempty"`
	// 0.0 = pure popularity (cold start), 1.0 = pure GRU4Rec.
	// UI uses this to switch the subtitle between "based on popularity"
	// and "based on your listening history".
	ColdStartAlpha float64 `json:"coldStartAlpha"`
}

// serveRecommendRequest matches the serving container's /recommend-by-tracks schema
type serveRecommendRequest struct {
	SessionID       string   `json:"session_id"`
	UserID          string   `json:"user_id"`
	TrackIDs        []string `json:"track_ids"`
	ExcludeTrackIDs []string `json:"exclude_track_ids"`
	TopN            int      `json:"top_n"`
}

// serveRecommendResponse matches the serving container's response
type serveRecommendResponse struct {
	Recommendations []struct {
		Rank    int     `json:"rank"`
		ItemIdx int     `json:"item_idx"`
		TrackID string  `json:"track_id"`
		Title   string  `json:"title"`
		Artist  string  `json:"artist"`
		Score   float64 `json:"score"`
	} `json:"recommendations"`
	ModelVersion       string  `json:"model_version"`
	GeneratedAt        string  `json:"generated_at"`
	InferenceLatencyMs float64 `json:"inference_latency_ms"`
	ColdStartAlpha     float64 `json:"cold_start_alpha"`
}

func (api *Router) addRecommendationRoute(r chi.Router) {
	r.Route("/recommendation", func(r chi.Router) {
		r.Get("/", api.getRecommendations())
		// Note: /play/{trackId} is registered as a public route in native_api.go
		// because HTML5 <audio> tags cannot send auth headers.
	})
}

func (api *Router) getRecommendations() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		user, _ := request.UserFrom(ctx)

		if !conf.Server.EnableRecommendations {
			http.Error(w, "Recommendations are disabled", http.StatusNotFound)
			return
		}

		serviceURL := conf.Server.RecommendationServiceURL
		if serviceURL == "" {
			respondWithEmptyRecommendations(w, user.UserName)
			return
		}

		// mfRepo is used both for the annotation-table fallback below AND for
		// per-recommendation library enrichment further down, so we hoist it.
		mfRepo := api.ds.MediaFile(ctx)

		// Source priority for the GRU4Rec input prefix:
		//   1. Feedback API's latest session for this user — already chronological,
		//      already grouped into a real session, with skipped tracks (playratio
		//      below threshold) filtered out server-side.
		//   2. Fall back to Navidrome's annotation table when the user has no
		//      session yet (404) or the feedback API is unreachable.
		// We query by user.ID (the internal Navidrome UUID), not user.UserName,
		// because the scrobbler's Submit hook receives the UUID as its userID
		// arg and that's what gets persisted to the sessions.user_id column.
		var trackIDs []string
		feedbackTrackIDs, feedbackErr := fetchLatestSession(ctx, user.ID, minPlayratioThreshold)
		if feedbackErr != nil {
			log.Warn(ctx, "feedback API session fetch failed; falling back to annotation table",
				"user", user.UserName, "err", feedbackErr)
		}
		if len(feedbackTrackIDs) > 0 {
			trackIDs = feedbackTrackIDs
		} else {
			// Fallback: annotation table — last 50 plays, sorted DESC by play_date.
			// We reverse the slice so the sequence runs oldest → newest, matching
			// what GRU4Rec expects (its prediction is "what comes after the LAST
			// item in the input").
			songs, err := mfRepo.GetAll(model.QueryOptions{
				Max:     50,
				Sort:    "play_date",
				Order:   "desc",
				Filters: squirrel.Expr("COALESCE(annotation.play_count, 0) > 0"),
			})

			if err == nil {
				for i, j := 0, len(songs)-1; i < j; i, j = i+1, j-1 {
					songs[i], songs[j] = songs[j], songs[i]
				}
				// Extract 30Music track IDs from the filename path.
				// Files are named like "audio_complete/3012335.mp3" where 3012335
				// is the 30Music track ID. Falls back to MbzRecordingID.
				for _, mf := range songs {
					base := filepath.Base(mf.Path)
					ext := filepath.Ext(base)
					nameWithoutExt := strings.TrimSuffix(base, ext)
					if nameWithoutExt != "" && nameWithoutExt != base {
						trackIDs = append(trackIDs, nameWithoutExt)
					} else if mf.MbzRecordingID != "" {
						trackIDs = append(trackIDs, mf.MbzRecordingID)
					}
				}
			}
		}
		if trackIDs == nil {
			trackIDs = []string{}
		}

		// UserID is the Navidrome internal nanoid (matches what the
		// scrobbler writes into user_activity.user_id), so the model's
		// user-embedding lookup keys consistently across train and serve.
		reqBody := serveRecommendRequest{
			SessionID:       "navidrome-ui-" + user.ID,
			UserID:          user.ID,
			TrackIDs:        trackIDs,
			ExcludeTrackIDs: []string{},
			TopN:            10,
		}
		jsonBody, _ := json.Marshal(reqBody)

		client := &http.Client{Timeout: 10 * time.Second}
		resp, err := client.Post(
			serviceURL+"/recommend-by-tracks",
			"application/json",
			bytes.NewReader(jsonBody),
		)
		if err != nil {
			log.Error(ctx, "Failed to call recommendation service", err)
			respondWithEmptyRecommendations(w, user.UserName)
			return
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)

		if resp.StatusCode != http.StatusOK {
			log.Warn(ctx, "Recommendation service error", "status", resp.StatusCode, "body", string(body))
			respondWithEmptyRecommendations(w, user.UserName)
			return
		}

		// Parse serving response and convert to UI format
		var serveResp serveRecommendResponse
		if err := json.Unmarshal(body, &serveResp); err != nil {
			log.Error(ctx, "Failed to parse recommendation response", err)
			respondWithEmptyRecommendations(w, user.UserName)
			return
		}

		// Build recommendation items from the serving response.
		// For each track, try to enrich with local library metadata (Navidrome ID,
		// album). If the track isn't in our library we still include it — the
		// frontend's "audio not available" popup handles that case gracefully.
		var recs []RecommendationItem
		for i, rec := range serveResp.Recommendations {
			item := RecommendationItem{
				TrackID: rec.TrackID,
				Score:   rec.Score,
				Rank:    i + 1,
				Title:   rec.Title,
				Artist:  rec.Artist,
			}

			// Try to enrich from Navidrome's library by matching filename.
			pathPattern := "%" + rec.TrackID + ".mp3"
			matches, err := mfRepo.GetAll(model.QueryOptions{
				Max:     1,
				Filters: squirrel.Like{"media_file.path": pathPattern},
			})
			if err == nil && len(matches) > 0 {
				mf := matches[0]
				item.ID = mf.ID
				item.Title = mf.Title
				item.Artist = mf.Artist
				item.Album = mf.Album
			}

			recs = append(recs, item)
		}

		uiResp := RecommendationResponse{
			UserID:          user.UserName,
			Recommendations: recs,
			ModelVersion:    serveResp.ModelVersion,
			GeneratedAt:     serveResp.GeneratedAt,
			ColdStartAlpha:  serveResp.ColdStartAlpha,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(uiResp) //nolint:errcheck
	}
}

// playRecommendedTrack proxies to the serving container's /play/{track_id}
// endpoint, which returns a 302 redirect to a presigned Swift URL. We forward
// that redirect to the browser so the HTML5 <audio> tag streams directly
// from Swift (the Navidrome Go process is not in the audio byte path).
func (api *Router) playRecommendedTrack() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()

		if !conf.Server.EnableRecommendations {
			http.Error(w, "Recommendations are disabled", http.StatusNotFound)
			return
		}
		serviceURL := conf.Server.RecommendationServiceURL
		if serviceURL == "" {
			http.Error(w, "Recommendation service not configured", http.StatusServiceUnavailable)
			return
		}

		trackID := chi.URLParam(r, "trackId")
		if trackID == "" {
			http.Error(w, "missing trackId", http.StatusBadRequest)
			return
		}

		// Don't follow the redirect — we want to forward it to the browser.
		client := &http.Client{
			Timeout: 130 * time.Second, // > serving's per-track yt-dlp timeout
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse
			},
		}
		resp, err := client.Get(serviceURL + "/play/" + trackID)
		if err != nil {
			log.Error(ctx, "play proxy failed", "track_id", trackID, err)
			http.Error(w, "play proxy failed", http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()

		if loc := resp.Header.Get("Location"); loc != "" {
			w.Header().Set("Location", loc)
		}
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
	}
}

func respondWithEmptyRecommendations(w http.ResponseWriter, userID string) {
	resp := RecommendationResponse{
		UserID:          userID,
		Recommendations: []RecommendationItem{},
		ModelVersion:    "none",
		GeneratedAt:     time.Now().UTC().Format(time.RFC3339),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp) //nolint:errcheck
}

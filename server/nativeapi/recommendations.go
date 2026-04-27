package nativeapi

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
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

		// Fetch the user's play history from Postgres (via the annotation table).
		// The annotation table is joined per-user inside selectMediaFile/withAnnotation,
		// so filtering by annotation.play_count scopes results to the current user only.
		// We pull the 50 most recent plays (DESC), then reverse to chronological
		// order before sending to GRU4Rec — the model is a session RNN whose
		// prediction is "what comes after the LAST item in the input sequence,"
		// so the sequence must run oldest → newest.
		mfRepo := api.ds.MediaFile(ctx)
		songs, err := mfRepo.GetAll(model.QueryOptions{
			Max:     50,
			Sort:    "play_date",
			Order:   "desc",
			Filters: squirrel.Expr("COALESCE(annotation.play_count, 0) > 0"),
		})

		// Reverse so the slice runs oldest → newest (chronological order for the model).
		if err == nil {
			for i, j := 0, len(songs)-1; i < j; i, j = i+1, j-1 {
				songs[i], songs[j] = songs[j], songs[i]
			}
		}

		// Extract 30Music track IDs from the filename path.
		// Files are named like "audio_complete/3012335.mp3" where 3012335 is the
		// 30Music track ID that matches the model vocabulary.
		// Falls back to MbzRecordingID if available.
		var trackIDs []string
		if err == nil {
			for _, mf := range songs {
				// Try to extract integer ID from filename (e.g. "3012335" from "audio_complete/3012335.mp3")
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
		if trackIDs == nil {
			trackIDs = []string{}
		}

		reqBody := serveRecommendRequest{
			SessionID:       "navidrome-ui-" + user.UserName,
			UserID:          user.UserName,
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

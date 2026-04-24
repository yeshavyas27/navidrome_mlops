import React, { useState, useEffect } from 'react'
import { useTranslate, Title, useDataProvider } from 'react-admin'
import { useDispatch } from 'react-redux'
import {
  Card,
  CardContent,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Avatar,
  Typography,
  CircularProgress,
  Box,
  Chip,
  IconButton,
  Snackbar,
} from '@material-ui/core'
import MuiAlert from '@material-ui/lab/Alert'
import { makeStyles } from '@material-ui/core/styles'
import MusicNoteIcon from '@material-ui/icons/MusicNote'
import PlayArrowIcon from '@material-ui/icons/PlayArrow'
import { playTracks } from '../actions'

const useStyles = makeStyles((theme) => ({
  root: {
    padding: theme.spacing(2),
    maxWidth: 800,
    margin: '0 auto',
  },
  header: {
    marginBottom: theme.spacing(2),
  },
  listItem: {
    borderBottom: `1px solid ${theme.palette.divider}`,
    '&:last-child': {
      borderBottom: 'none',
    },
  },
  score: {
    marginLeft: theme.spacing(1),
  },
  avatar: {
    backgroundColor: theme.palette.primary.main,
  },
  emptyState: {
    textAlign: 'center',
    padding: theme.spacing(4),
    color: theme.palette.text.secondary,
  },
  modelInfo: {
    marginTop: theme.spacing(2),
    color: theme.palette.text.secondary,
    fontSize: '0.75rem',
  },
}))

const RecommendationList = () => {
  const classes = useStyles()
  const translate = useTranslate()
  const dispatch = useDispatch()
  const dataProvider = useDataProvider()
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [modelVersion, setModelVersion] = useState('')
  const [generatedAt, setGeneratedAt] = useState('')
  const [unavailableTrack, setUnavailableTrack] = useState(null)

  useEffect(() => {
    const fetchRecommendations = async () => {
      try {
        setLoading(true)
        const response = await fetch('/api/recommendation', {
          headers: {
            'x-nd-authorization': `Bearer ${localStorage.getItem('token')}`,
          },
        })
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }
        const data = await response.json()
        setRecommendations(data.recommendations || [])
        setModelVersion(data.modelVersion || '')
        setGeneratedAt(data.generatedAt || '')
      } catch (err) {
        console.error('Failed to fetch recommendations:', err)
        setError(err.message)
        setRecommendations([])
      } finally {
        setLoading(false)
      }
    }

    fetchRecommendations()
  }, [])

  const handlePlay = async (rec) => {
    try {
      // Fetch the full song record from Navidrome's API
      const { data } = await dataProvider.getOne('song', { id: rec.id })
      // Build the data object that playTracks expects
      const songData = { [data.id]: data }
      dispatch(playTracks(songData, [data.id], data.id))
    } catch (e) {
      console.error('Failed to play track:', e)
      setUnavailableTrack(rec.title || rec.track_id || 'this track')
    }
  }

  const handleCloseUnavailable = () => setUnavailableTrack(null)

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    )
  }

  return (
    <>
      <Title title="Recommendations" />
      <Card className={classes.root}>
        <CardContent>
          <Typography variant="h5" className={classes.header}>
            Recommended For You
          </Typography>

          {error && (
            <Typography color="error" gutterBottom>
              Could not load recommendations. The recommendation service may be
              starting up.
            </Typography>
          )}

          {!error && recommendations.length === 0 && (
            <div className={classes.emptyState}>
              <MusicNoteIcon style={{ fontSize: 48, opacity: 0.5 }} />
              <Typography variant="body1">
                No recommendations yet. Listen to some music and check back
                later!
              </Typography>
            </div>
          )}

          {recommendations.length > 0 && (
            <>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Top {Math.min(recommendations.length, 10)} recommendations based on your listening history
              </Typography>
              <List>
                {recommendations.slice(0, 10).map((rec, index) => (
                  <ListItem key={rec.id || rec.track_id || index} className={classes.listItem}>
                    <ListItemAvatar>
                      <Avatar className={classes.avatar}>
                        {index + 1}
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary={rec.title || `Track ${rec.track_id || rec.id}`}
                      secondary={rec.artist || 'Unknown Artist'}
                    />
                    {rec.score != null && (
                      <Chip
                        label={`Score: ${rec.score.toFixed(1)}`}
                        size="small"
                        color="primary"
                        variant="outlined"
                        className={classes.score}
                      />
                    )}
                    <IconButton
                      aria-label="play"
                      onClick={() => handlePlay(rec)}
                    >
                      <PlayArrowIcon />
                    </IconButton>
                  </ListItem>
                ))}
              </List>
            </>
          )}

          {modelVersion && (
            <Typography className={classes.modelInfo}>
              Model: {modelVersion}
              {generatedAt && ` · Generated: ${new Date(generatedAt).toLocaleString()}`}
            </Typography>
          )}
        </CardContent>
      </Card>

      <Snackbar
        open={unavailableTrack !== null}
        autoHideDuration={5000}
        onClose={handleCloseUnavailable}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <MuiAlert onClose={handleCloseUnavailable} severity="warning" elevation={6} variant="filled">
          Audio file not available for &quot;{unavailableTrack}&quot;
        </MuiAlert>
      </Snackbar>
    </>
  )
}

export default RecommendationList

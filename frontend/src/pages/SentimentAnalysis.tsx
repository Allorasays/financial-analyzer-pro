import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Alert,
  CircularProgress,
  Chip,
  Paper,
  LinearProgress,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  SentimentSatisfied,
  SentimentDissatisfied,
  SentimentNeutral,
  TrendingUp,
  TrendingDown,
  Psychology,
  Refresh,
  Info,
  Warning,
  CheckCircle
} from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';
import axios from 'axios';
import { API_CONFIG } from '../config/api';

interface SymbolSentiment {
  symbol: string;
  overall_sentiment: string;
  sentiment_score: number;
  confidence: number;
  news_sentiment: {
    sentiment: string;
    score: number;
    confidence: number;
    article_count: number;
  };
  social_sentiment: {
    sentiment: string;
    score: number;
    confidence: number;
    source: string;
  };
  sentiment_trend: string;
  key_insights: string[];
  timestamp: string;
}

interface MarketSentiment {
  market_sentiment: string;
  sentiment_score: number;
  confidence: number;
  news_sentiment: {
    sentiment: string;
    score: number;
    confidence: number;
    article_count: number;
  };
  vix_sentiment: {
    sentiment: string;
    score: number;
    confidence: number;
    vix_level: number;
    avg_vix: number;
  };
  sentiment_trend: string;
  key_insights: string[];
  timestamp: string;
}

const SentimentAnalysis: React.FC = () => {
  const { user } = useAuth();
  const [symbol, setSymbol] = useState('AAPL');
  const [newsText, setNewsText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Sentiment data
  const [symbolSentiment, setSymbolSentiment] = useState<SymbolSentiment | null>(null);
  const [marketSentiment, setMarketSentiment] = useState<MarketSentiment | null>(null);
  const [newsSentiment, setNewsSentiment] = useState<any>(null);
  
  // Loading states
  const [symbolLoading, setSymbolLoading] = useState(false);
  const [marketLoading, setMarketLoading] = useState(false);
  const [newsLoading, setNewsLoading] = useState(false);

  useEffect(() => {
    // Load market sentiment on component mount
    loadMarketSentiment();
  }, []);

  const loadMarketSentiment = async () => {
    try {
      setMarketLoading(true);
      const response = await axios.get(`${API_CONFIG.BASE_URL}/api/sentiment/market`);
      setMarketSentiment(response.data);
    } catch (err) {
      setError('Failed to load market sentiment');
      console.error('Market sentiment error:', err);
    } finally {
      setMarketLoading(false);
    }
  };

  const analyzeSymbolSentiment = async () => {
    if (!symbol.trim()) return;

    try {
      setSymbolLoading(true);
      setError(null);
      const response = await axios.get(`${API_CONFIG.BASE_URL}/api/sentiment/symbol/${symbol}`);
      setSymbolSentiment(response.data);
    } catch (err) {
      setError('Failed to analyze symbol sentiment');
      console.error('Symbol sentiment error:', err);
    } finally {
      setSymbolLoading(false);
    }
  };

  const analyzeNewsSentiment = async () => {
    if (!newsText.trim()) return;

    try {
      setNewsLoading(true);
      setError(null);
      const response = await axios.post(`${API_CONFIG.BASE_URL}/api/sentiment/analyze-news`, {
        news_text: newsText
      });
      setNewsSentiment(response.data);
    } catch (err) {
      setError('Failed to analyze news sentiment');
      console.error('News sentiment error:', err);
    } finally {
      setNewsLoading(false);
    }
  };

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case 'positive':
      case 'bullish':
        return <SentimentSatisfied color="success" />;
      case 'negative':
      case 'bearish':
        return <SentimentDissatisfied color="error" />;
      default:
        return <SentimentNeutral color="warning" />;
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case 'positive':
      case 'bullish':
        return 'success';
      case 'negative':
      case 'bearish':
        return 'error';
      default:
        return 'warning';
    }
  };

  const getSentimentScoreColor = (score: number) => {
    if (score > 0.2) return 'success';
    if (score < -0.2) return 'error';
    return 'warning';
  };

  const formatSentimentScore = (score: number) => {
    return `${score >= 0 ? '+' : ''}${(score * 100).toFixed(1)}%`;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">
          🧠 Sentiment Analysis
        </Typography>
        <Button
          variant="outlined"
          onClick={loadMarketSentiment}
          disabled={marketLoading}
          startIcon={<Refresh />}
        >
          Refresh Market Sentiment
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Market Sentiment Overview */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            📊 Market Sentiment Overview
          </Typography>
          
          {marketLoading ? (
            <Box display="flex" justifyContent="center" py={2}>
              <CircularProgress />
            </Box>
          ) : marketSentiment ? (
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Box display="flex" alignItems="center" gap={2}>
                  {getSentimentIcon(marketSentiment.market_sentiment)}
                  <Box>
                    <Typography variant="h5" fontWeight="bold">
                      {marketSentiment.market_sentiment}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Overall Market Mood
                    </Typography>
                  </Box>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Sentiment Score
                  </Typography>
                  <Typography
                    variant="h5"
                    fontWeight="bold"
                    color={`${getSentimentScoreColor(marketSentiment.sentiment_score)}.main`}
                  >
                    {formatSentimentScore(marketSentiment.sentiment_score)}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Confidence
                  </Typography>
                  <Chip
                    label={`${(marketSentiment.confidence * 100).toFixed(1)}%`}
                    color={getConfidenceColor(marketSentiment.confidence)}
                    size="small"
                  />
                </Box>
              </Grid>
            </Grid>
          ) : (
            <Typography variant="body2" color="text.secondary">
              Loading market sentiment...
            </Typography>
          )}
        </CardContent>
      </Card>

      <Grid container spacing={3}>
        {/* Symbol Sentiment Analysis */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📈 Symbol Sentiment Analysis
              </Typography>
              
              <Box display="flex" gap={2} mb={3}>
                <TextField
                  label="Symbol"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                  placeholder="e.g., AAPL"
                  size="small"
                  sx={{ flexGrow: 1 }}
                />
                <Button
                  variant="contained"
                  onClick={analyzeSymbolSentiment}
                  disabled={symbolLoading || !symbol.trim()}
                  startIcon={<Psychology />}
                >
                  Analyze
                </Button>
              </Box>

              {symbolLoading ? (
                <Box display="flex" justifyContent="center" py={2}>
                  <CircularProgress />
                </Box>
              ) : symbolSentiment ? (
                <Box>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <Box display="flex" alignItems="center" gap={2} mb={2}>
                        {getSentimentIcon(symbolSentiment.overall_sentiment)}
                        <Box>
                          <Typography variant="h6" fontWeight="bold">
                            {symbolSentiment.overall_sentiment}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Overall Sentiment
                          </Typography>
                        </Box>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={6}>
                      <Box>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Sentiment Score
                        </Typography>
                        <Typography
                          variant="h6"
                          fontWeight="bold"
                          color={`${getSentimentScoreColor(symbolSentiment.sentiment_score)}.main`}
                        >
                          {formatSentimentScore(symbolSentiment.sentiment_score)}
                        </Typography>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={6}>
                      <Box>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Confidence
                        </Typography>
                        <Chip
                          label={`${(symbolSentiment.confidence * 100).toFixed(1)}%`}
                          color={getConfidenceColor(symbolSentiment.confidence)}
                          size="small"
                        />
                      </Box>
                    </Grid>
                  </Grid>

                  <Divider sx={{ my: 2 }} />

                  <Typography variant="subtitle2" gutterBottom>
                    News Sentiment
                  </Typography>
                  <Box display="flex" alignItems="center" gap={2} mb={2}>
                    <Chip
                      label={symbolSentiment.news_sentiment.sentiment}
                      color={getSentimentColor(symbolSentiment.news_sentiment.sentiment)}
                      size="small"
                    />
                    <Typography variant="body2">
                      {formatSentimentScore(symbolSentiment.news_sentiment.score)} 
                      ({symbolSentiment.news_sentiment.article_count} articles)
                    </Typography>
                  </Box>

                  <Typography variant="subtitle2" gutterBottom>
                    Key Insights
                  </Typography>
                  <List dense>
                    {symbolSentiment.key_insights.map((insight, index) => (
                      <ListItem key={index} sx={{ py: 0 }}>
                        <ListItemIcon sx={{ minWidth: 32 }}>
                          <CheckCircle color="primary" fontSize="small" />
                        </ListItemIcon>
                        <ListItemText
                          primary={insight}
                          primaryTypographyProps={{ variant: 'body2' }}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary" textAlign="center">
                  Enter a symbol and click "Analyze" to get sentiment analysis
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* News Sentiment Analysis */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📰 News Sentiment Analysis
              </Typography>
              
              <TextField
                fullWidth
                multiline
                rows={4}
                label="News Text"
                value={newsText}
                onChange={(e) => setNewsText(e.target.value)}
                placeholder="Paste news text here for sentiment analysis..."
                sx={{ mb: 2 }}
              />
              
              <Button
                variant="contained"
                onClick={analyzeNewsSentiment}
                disabled={newsLoading || !newsText.trim()}
                startIcon={<Psychology />}
                fullWidth
              >
                Analyze News
              </Button>

              {newsLoading ? (
                <Box display="flex" justifyContent="center" py={2}>
                  <CircularProgress />
                </Box>
              ) : newsSentiment ? (
                <Box mt={2}>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <Box display="flex" alignItems="center" gap={2} mb={2}>
                        {getSentimentIcon(newsSentiment.combined_sentiment || 'neutral')}
                        <Box>
                          <Typography variant="h6" fontWeight="bold">
                            {newsSentiment.combined_sentiment || 'Neutral'}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Overall Sentiment
                          </Typography>
                        </Box>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={6}>
                      <Box>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          TextBlob Polarity
                        </Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {newsSentiment.textblob_polarity?.toFixed(3) || 'N/A'}
                        </Typography>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={6}>
                      <Box>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Combined Score
                        </Typography>
                        <Typography
                          variant="body2"
                          fontWeight="bold"
                          color={`${getSentimentScoreColor(newsSentiment.combined_score || 0)}.main`}
                        >
                          {formatSentimentScore(newsSentiment.combined_score || 0)}
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>

                  <Divider sx={{ my: 2 }} />

                  <Typography variant="subtitle2" gutterBottom>
                    Analysis Details
                  </Typography>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      TextBlob Subjectivity: {newsSentiment.textblob_subjectivity?.toFixed(3) || 'N/A'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Keyword Sentiment: {formatSentimentScore(newsSentiment.keyword_sentiment?.score || 0)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Confidence: {((newsSentiment.combined_confidence || 0) * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary" textAlign="center" mt={2}>
                  Enter news text and click "Analyze News" to get sentiment analysis
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* VIX Analysis */}
      {marketSentiment?.vix_sentiment && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              😰 Fear Gauge (VIX) Analysis
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Box display="flex" alignItems="center" gap={2}>
                  {getSentimentIcon(marketSentiment.vix_sentiment.sentiment)}
                  <Box>
                    <Typography variant="h6" fontWeight="bold">
                      {marketSentiment.vix_sentiment.sentiment}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Market Fear Level
                    </Typography>
                  </Box>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Current VIX
                  </Typography>
                  <Typography variant="h6" fontWeight="bold">
                    {marketSentiment.vix_sentiment.vix_level.toFixed(2)}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Average VIX (5-day)
                  </Typography>
                  <Typography variant="h6" fontWeight="bold">
                    {marketSentiment.vix_sentiment.avg_vix.toFixed(2)}
                  </Typography>
                </Box>
              </Grid>
            </Grid>

            <Box mt={2}>
              <Typography variant="body2" color="text.secondary">
                VIX Interpretation: {marketSentiment.vix_sentiment.sentiment === 'Fearful' ? 
                  'High fear levels may indicate potential buying opportunities' :
                  marketSentiment.vix_sentiment.sentiment === 'Complacent' ?
                  'Low fear levels may indicate market overconfidence' :
                  'Normal fear levels suggest balanced market sentiment'}
              </Typography>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Market Insights */}
      {marketSentiment?.key_insights && marketSentiment.key_insights.length > 0 && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              💡 Market Insights
            </Typography>
            
            <List>
              {marketSentiment.key_insights.map((insight, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    <CheckCircle color="primary" />
                  </ListItemIcon>
                  <ListItemText primary={insight} />
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default SentimentAnalysis;



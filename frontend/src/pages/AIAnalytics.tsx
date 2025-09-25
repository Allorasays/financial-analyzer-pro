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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tabs,
  Tab,
  LinearProgress,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Psychology,
  Assessment,
  ShowChart,
  Refresh,
  Info,
  Warning,
  CheckCircle
} from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';
import axios from 'axios';
import { API_CONFIG } from '../config/api';

interface PredictionResult {
  symbol: string;
  current_price: number;
  predicted_price: number;
  price_change: number;
  price_change_percent: number;
  confidence: number;
  days_ahead: number;
  timestamp: string;
}

interface TrendAnalysis {
  symbol: string;
  trend_direction: string;
  trend_strength: number;
  trend_confidence: number;
  support_level: number;
  resistance_level: number;
  signals: Array<{
    type: string;
    signal: string;
    strength: string;
    description: string;
  }>;
  timestamp: string;
}

interface RiskAssessment {
  symbol: string;
  risk_level: string;
  risk_score: number;
  volatility: number;
  beta: number;
  sharpe_ratio: number;
  max_drawdown: number;
  var_95: number;
  recommendations: string[];
  timestamp: string;
}

const AIAnalytics: React.FC = () => {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState(0);
  const [symbol, setSymbol] = useState('AAPL');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Prediction data
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  
  // Trend analysis data
  const [trendAnalysis, setTrendAnalysis] = useState<TrendAnalysis | null>(null);
  const [trendLoading, setTrendLoading] = useState(false);
  
  // Risk assessment data
  const [riskAssessment, setRiskAssessment] = useState<RiskAssessment | null>(null);
  const [riskLoading, setRiskLoading] = useState(false);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const analyzeSymbol = async () => {
    if (!symbol.trim()) return;

    setError(null);
    await Promise.all([
      predictPrice(),
      analyzeTrend(),
      assessRisk()
    ]);
  };

  const predictPrice = async () => {
    try {
      setPredictionLoading(true);
      const response = await axios.get(`${API_CONFIG.BASE_URL}/api/ai/predict-price/${symbol}`);
      
      if (response.data.error) {
        setError(response.data.error);
        return;
      }
      
      setPrediction(response.data);
    } catch (err) {
      setError('Failed to get price prediction');
      console.error('Prediction error:', err);
    } finally {
      setPredictionLoading(false);
    }
  };

  const analyzeTrend = async () => {
    try {
      setTrendLoading(true);
      const response = await axios.get(`${API_CONFIG.BASE_URL}/api/ai/analyze-trend/${symbol}`);
      
      if (response.data.error) {
        setError(response.data.error);
        return;
      }
      
      setTrendAnalysis(response.data);
    } catch (err) {
      setError('Failed to analyze trend');
      console.error('Trend analysis error:', err);
    } finally {
      setTrendLoading(false);
    }
  };

  const assessRisk = async () => {
    try {
      setRiskLoading(true);
      const response = await axios.get(`${API_CONFIG.BASE_URL}/api/ai/assess-risk/${symbol}`);
      
      if (response.data.error) {
        setError(response.data.error);
        return;
      }
      
      setRiskAssessment(response.data);
    } catch (err) {
      setError('Failed to assess risk');
      console.error('Risk assessment error:', err);
    } finally {
      setRiskLoading(false);
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(amount);
  };

  const formatPercent = (percent: number) => {
    return `${percent >= 0 ? '+' : ''}${percent.toFixed(2)}%`;
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel.toLowerCase()) {
      case 'very low': return 'success';
      case 'low': return 'success';
      case 'medium': return 'warning';
      case 'high': return 'error';
      case 'very high': return 'error';
      default: return 'default';
    }
  };

  const getTrendColor = (direction: string) => {
    switch (direction.toLowerCase()) {
      case 'bullish': return 'success';
      case 'bearish': return 'error';
      case 'sideways': return 'warning';
      default: return 'default';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Medium';
    return 'Low';
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">
          🤖 AI Analytics
        </Typography>
        <Box display="flex" gap={2}>
          <TextField
            label="Symbol"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="e.g., AAPL"
            size="small"
            sx={{ width: 120 }}
          />
          <Button
            variant="contained"
            onClick={analyzeSymbol}
            disabled={loading || !symbol.trim()}
            startIcon={<Psychology />}
          >
            Analyze
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Card>
        <Tabs value={activeTab} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tab label="Price Prediction" icon={<ShowChart />} />
          <Tab label="Trend Analysis" icon={<TrendingUp />} />
          <Tab label="Risk Assessment" icon={<Assessment />} />
        </Tabs>

        <Box sx={{ p: 3 }}>
          {/* Price Prediction Tab */}
          {activeTab === 0 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                📈 Price Prediction
              </Typography>
              
              {predictionLoading ? (
                <Box display="flex" justifyContent="center" py={4}>
                  <CircularProgress />
                </Box>
              ) : prediction ? (
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Current vs Predicted Price
                        </Typography>
                        <Box display="flex" justifyContent="space-between" mb={2}>
                          <Typography variant="body2" color="text.secondary">
                            Current Price:
                          </Typography>
                          <Typography variant="h6" fontWeight="bold">
                            {formatCurrency(prediction.current_price)}
                          </Typography>
                        </Box>
                        <Box display="flex" justifyContent="space-between" mb={2}>
                          <Typography variant="body2" color="text.secondary">
                            Predicted Price ({prediction.days_ahead} days):
                          </Typography>
                          <Typography variant="h6" fontWeight="bold">
                            {formatCurrency(prediction.predicted_price)}
                          </Typography>
                        </Box>
                        <Box display="flex" justifyContent="space-between" mb={2}>
                          <Typography variant="body2" color="text.secondary">
                            Expected Change:
                          </Typography>
                          <Typography
                            variant="h6"
                            fontWeight="bold"
                            color={prediction.price_change >= 0 ? 'success.main' : 'error.main'}
                          >
                            {formatCurrency(prediction.price_change)} ({formatPercent(prediction.price_change_percent)})
                          </Typography>
                        </Box>
                        <Box display="flex" justifyContent="space-between" alignItems="center">
                          <Typography variant="body2" color="text.secondary">
                            Confidence:
                          </Typography>
                          <Box display="flex" alignItems="center" gap={1}>
                            <Chip
                              label={`${getConfidenceLabel(prediction.confidence)} (${(prediction.confidence * 100).toFixed(1)}%)`}
                              color={getConfidenceColor(prediction.confidence)}
                              size="small"
                            />
                            <Tooltip title="Confidence based on historical accuracy">
                              <IconButton size="small">
                                <Info fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Prediction Details
                        </Typography>
                        <Box mb={2}>
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            Confidence Level
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={prediction.confidence * 100}
                            color={getConfidenceColor(prediction.confidence)}
                            sx={{ height: 8, borderRadius: 4 }}
                          />
                        </Box>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Analysis Method: Machine Learning (Random Forest)
                        </Typography>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Data Period: 2 years historical data
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Last Updated: {new Date(prediction.timestamp).toLocaleString()}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              ) : (
                <Box textAlign="center" py={4}>
                  <Typography variant="h6" color="text.secondary">
                    Enter a symbol and click "Analyze" to get AI predictions
                  </Typography>
                </Box>
              )}
            </Box>
          )}

          {/* Trend Analysis Tab */}
          {activeTab === 1 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                📊 Trend Analysis
              </Typography>
              
              {trendLoading ? (
                <Box display="flex" justifyContent="center" py={4}>
                  <CircularProgress />
                </Box>
              ) : trendAnalysis ? (
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Trend Overview
                        </Typography>
                        <Box display="flex" alignItems="center" gap={2} mb={2}>
                          <Typography variant="body2" color="text.secondary">
                            Direction:
                          </Typography>
                          <Chip
                            label={trendAnalysis.trend_direction}
                            color={getTrendColor(trendAnalysis.trend_direction)}
                            icon={trendAnalysis.trend_direction === 'Bullish' ? <TrendingUp /> : <TrendingDown />}
                          />
                        </Box>
                        <Box display="flex" justifyContent="space-between" mb={2}>
                          <Typography variant="body2" color="text.secondary">
                            Strength:
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {trendAnalysis.trend_strength.toFixed(1)}%
                          </Typography>
                        </Box>
                        <Box display="flex" justifyContent="space-between" mb={2}>
                          <Typography variant="body2" color="text.secondary">
                            Confidence:
                          </Typography>
                          <Chip
                            label={`${getConfidenceLabel(trendAnalysis.trend_confidence)} (${(trendAnalysis.trend_confidence * 100).toFixed(1)}%)`}
                            color={getConfidenceColor(trendAnalysis.trend_confidence)}
                            size="small"
                          />
                        </Box>
                        <Box display="flex" justifyContent="space-between" mb={1}>
                          <Typography variant="body2" color="text.secondary">
                            Support Level:
                          </Typography>
                          <Typography variant="body2" fontWeight="bold" color="success.main">
                            {formatCurrency(trendAnalysis.support_level)}
                          </Typography>
                        </Box>
                        <Box display="flex" justifyContent="space-between">
                          <Typography variant="body2" color="text.secondary">
                            Resistance Level:
                          </Typography>
                          <Typography variant="body2" fontWeight="bold" color="error.main">
                            {formatCurrency(trendAnalysis.resistance_level)}
                          </Typography>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Technical Signals
                        </Typography>
                        {trendAnalysis.signals.length > 0 ? (
                          <Box>
                            {trendAnalysis.signals.map((signal, index) => (
                              <Box key={index} display="flex" alignItems="center" gap={2} mb={2}>
                                <Chip
                                  label={signal.signal}
                                  color={signal.signal === 'BUY' ? 'success' : 'error'}
                                  size="small"
                                />
                                <Box>
                                  <Typography variant="body2" fontWeight="bold">
                                    {signal.type}
                                  </Typography>
                                  <Typography variant="caption" color="text.secondary">
                                    {signal.description}
                                  </Typography>
                                </Box>
                                <Chip
                                  label={signal.strength}
                                  size="small"
                                  variant="outlined"
                                />
                              </Box>
                            ))}
                          </Box>
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            No strong signals detected
                          </Typography>
                        )}
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              ) : (
                <Box textAlign="center" py={4}>
                  <Typography variant="h6" color="text.secondary">
                    Enter a symbol and click "Analyze" to get trend analysis
                  </Typography>
                </Box>
              )}
            </Box>
          )}

          {/* Risk Assessment Tab */}
          {activeTab === 2 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                ⚠️ Risk Assessment
              </Typography>
              
              {riskLoading ? (
                <Box display="flex" justifyContent="center" py={4}>
                  <CircularProgress />
                </Box>
              ) : riskAssessment ? (
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Risk Overview
                        </Typography>
                        <Box display="flex" alignItems="center" gap={2} mb={2}>
                          <Typography variant="body2" color="text.secondary">
                            Risk Level:
                          </Typography>
                          <Chip
                            label={riskAssessment.risk_level}
                            color={getRiskColor(riskAssessment.risk_level)}
                            icon={riskAssessment.risk_level.includes('High') ? <Warning /> : <CheckCircle />}
                          />
                        </Box>
                        <Box display="flex" justifyContent="space-between" mb={2}>
                          <Typography variant="body2" color="text.secondary">
                            Risk Score:
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {riskAssessment.risk_score.toFixed(1)}/100
                          </Typography>
                        </Box>
                        <Box display="flex" justifyContent="space-between" mb={2}>
                          <Typography variant="body2" color="text.secondary">
                            Volatility:
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {(riskAssessment.volatility * 100).toFixed(2)}%
                          </Typography>
                        </Box>
                        <Box display="flex" justifyContent="space-between" mb={2}>
                          <Typography variant="body2" color="text.secondary">
                            Beta:
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {riskAssessment.beta.toFixed(2)}
                          </Typography>
                        </Box>
                        <Box display="flex" justifyContent="space-between" mb={2}>
                          <Typography variant="body2" color="text.secondary">
                            Sharpe Ratio:
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {riskAssessment.sharpe_ratio.toFixed(2)}
                          </Typography>
                        </Box>
                        <Box display="flex" justifyContent="space-between" mb={2}>
                          <Typography variant="body2" color="text.secondary">
                            Max Drawdown:
                          </Typography>
                          <Typography variant="body2" fontWeight="bold" color="error.main">
                            {(riskAssessment.max_drawdown * 100).toFixed(2)}%
                          </Typography>
                        </Box>
                        <Box display="flex" justifyContent="space-between">
                          <Typography variant="body2" color="text.secondary">
                            VaR (95%):
                          </Typography>
                          <Typography variant="body2" fontWeight="bold" color="error.main">
                            {(riskAssessment.var_95 * 100).toFixed(2)}%
                          </Typography>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Risk Recommendations
                        </Typography>
                        {riskAssessment.recommendations.length > 0 ? (
                          <Box>
                            {riskAssessment.recommendations.map((recommendation, index) => (
                              <Box key={index} display="flex" alignItems="flex-start" gap={2} mb={2}>
                                <CheckCircle color="primary" fontSize="small" />
                                <Typography variant="body2">
                                  {recommendation}
                                </Typography>
                              </Box>
                            ))}
                          </Box>
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            No specific recommendations available
                          </Typography>
                        )}
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              ) : (
                <Box textAlign="center" py={4}>
                  <Typography variant="h6" color="text.secondary">
                    Enter a symbol and click "Analyze" to get risk assessment
                  </Typography>
                </Box>
              )}
            </Box>
          )}
        </Box>
      </Card>
    </Box>
  );
};

export default AIAnalytics;


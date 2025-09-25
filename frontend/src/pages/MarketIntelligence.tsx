import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
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
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Assessment,
  BarChart,
  PieChart,
  Refresh,
  CheckCircle,
  Warning,
  Info,
  ShowChart
} from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';
import axios from 'axios';
import { API_CONFIG } from '../config/api';

interface SectorRotation {
  sector_performance: Record<string, {
    total_return: number;
    volatility: number;
    sharpe_ratio: number;
    performance_rank: number;
  }>;
  rotation_patterns: {
    rotation_pattern: string;
    cyclical_performance: number;
    defensive_performance: number;
    rotation_strength: number;
  };
  leading_sectors: {
    top_performers: Array<[string, any]>;
    bottom_performers: Array<[string, any]>;
    leading_sector: string;
    lagging_sector: string;
  };
  rotation_insights: string[];
  timestamp: string;
}

interface VolatilityPatterns {
  symbol: string;
  volatility_metrics: {
    historical_volatility: number;
    current_volatility: number;
    average_volatility: number;
    volatility_of_volatility: number;
    volatility_percentile: number;
  };
  volatility_patterns: {
    volatility_clustering: number;
    low_volatility_periods: number;
    volatility_trend: string;
    volatility_regime: string;
  };
  volatility_regimes: {
    current_regime: string;
    regime_threshold_high: number;
    regime_threshold_low: number;
    regime_duration: number;
  };
  volatility_insights: string[];
  timestamp: string;
}

const MarketIntelligence: React.FC = () => {
  const { user } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Market data
  const [sectorRotation, setSectorRotation] = useState<SectorRotation | null>(null);
  const [volatilityPatterns, setVolatilityPatterns] = useState<VolatilityPatterns | null>(null);
  const [volatilitySymbol, setVolatilitySymbol] = useState('SPY');
  
  // Loading states
  const [sectorLoading, setSectorLoading] = useState(false);
  const [volatilityLoading, setVolatilityLoading] = useState(false);

  useEffect(() => {
    // Load sector rotation on component mount
    loadSectorRotation();
  }, []);

  const loadSectorRotation = async () => {
    try {
      setSectorLoading(true);
      setError(null);
      const response = await axios.get(`${API_CONFIG.BASE_URL}/api/analytics/sector-rotation`);
      setSectorRotation(response.data);
    } catch (err) {
      setError('Failed to load sector rotation data');
      console.error('Sector rotation error:', err);
    } finally {
      setSectorLoading(false);
    }
  };

  const analyzeVolatilityPatterns = async () => {
    if (!volatilitySymbol.trim()) return;

    try {
      setVolatilityLoading(true);
      setError(null);
      const response = await axios.get(`${API_CONFIG.BASE_URL}/api/analytics/volatility-patterns/${volatilitySymbol}`);
      setVolatilityPatterns(response.data);
    } catch (err) {
      setError('Failed to analyze volatility patterns');
      console.error('Volatility patterns error:', err);
    } finally {
      setVolatilityLoading(false);
    }
  };

  const formatPercent = (percent: number) => {
    return `${percent >= 0 ? '+' : ''}${percent.toFixed(2)}%`;
  };

  const getPerformanceColor = (value: number) => {
    if (value > 5) return 'success.main';
    if (value > 0) return 'success.main';
    if (value > -5) return 'warning.main';
    return 'error.main';
  };

  const getRotationColor = (pattern: string) => {
    switch (pattern.toLowerCase()) {
      case 'risk-on': return 'success';
      case 'risk-off': return 'error';
      default: return 'warning';
    }
  };

  const getVolatilityColor = (volatility: number) => {
    if (volatility > 30) return 'error';
    if (volatility > 20) return 'warning';
    return 'success';
  };

  const getRegimeColor = (regime: string) => {
    switch (regime.toLowerCase()) {
      case 'high volatility': return 'error';
      case 'low volatility': return 'success';
      default: return 'warning';
    }
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">
          🧠 Market Intelligence
        </Typography>
        <Box display="flex" gap={2}>
          <Button
            variant="outlined"
            onClick={loadSectorRotation}
            disabled={sectorLoading}
            startIcon={<Refresh />}
          >
            Refresh Sector Data
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Sector Rotation Analysis */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🔄 Sector Rotation Analysis
          </Typography>
          
          {sectorLoading ? (
            <Box display="flex" justifyContent="center" py={4}>
              <CircularProgress />
            </Box>
          ) : sectorRotation ? (
            <Box>
              {/* Rotation Pattern Overview */}
              <Grid container spacing={3} sx={{ mb: 3 }}>
                <Grid item xs={12} md={4}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Rotation Pattern
                      </Typography>
                      <Chip
                        label={sectorRotation.rotation_patterns.rotation_pattern}
                        color={getRotationColor(sectorRotation.rotation_patterns.rotation_pattern)}
                        icon={sectorRotation.rotation_patterns.rotation_pattern === 'Risk-On' ? <TrendingUp /> : <TrendingDown />}
                      />
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Cyclical Performance
                      </Typography>
                      <Typography
                        variant="h6"
                        fontWeight="bold"
                        color={getPerformanceColor(sectorRotation.rotation_patterns.cyclical_performance)}
                      >
                        {formatPercent(sectorRotation.rotation_patterns.cyclical_performance)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Defensive Performance
                      </Typography>
                      <Typography
                        variant="h6"
                        fontWeight="bold"
                        color={getPerformanceColor(sectorRotation.rotation_patterns.defensive_performance)}
                      >
                        {formatPercent(sectorRotation.rotation_patterns.defensive_performance)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>

              {/* Leading Sectors */}
              <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                Leading Sectors
              </Typography>
              
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} md={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                        Top Performers
                      </Typography>
                      {sectorRotation.leading_sectors.top_performers.slice(0, 3).map(([sector, data], index) => (
                        <Box key={sector} display="flex" justifyContent="space-between" mb={1}>
                          <Typography variant="body2">
                            {index + 1}. {sector}
                          </Typography>
                          <Typography
                            variant="body2"
                            fontWeight="bold"
                            color={getPerformanceColor(data.total_return)}
                          >
                            {formatPercent(data.total_return)}
                          </Typography>
                        </Box>
                      ))}
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                        Bottom Performers
                      </Typography>
                      {sectorRotation.leading_sectors.bottom_performers.slice(0, 3).map(([sector, data], index) => (
                        <Box key={sector} display="flex" justifyContent="space-between" mb={1}>
                          <Typography variant="body2">
                            {index + 1}. {sector}
                          </Typography>
                          <Typography
                            variant="body2"
                            fontWeight="bold"
                            color={getPerformanceColor(data.total_return)}
                          >
                            {formatPercent(data.total_return)}
                          </Typography>
                        </Box>
                      ))}
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>

              {/* Sector Performance Table */}
              <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                Sector Performance (3 months)
              </Typography>
              
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Sector</TableCell>
                      <TableCell align="right">Return</TableCell>
                      <TableCell align="right">Volatility</TableCell>
                      <TableCell align="right">Sharpe Ratio</TableCell>
                      <TableCell align="right">Rank</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(sectorRotation.sector_performance)
                      .sort(([, a], [, b]) => b.performance_rank - a.performance_rank)
                      .map(([sector, data]) => (
                        <TableRow key={sector}>
                          <TableCell>
                            <Typography variant="subtitle2" fontWeight="bold">
                              {sector}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography
                              color={getPerformanceColor(data.total_return)}
                              fontWeight="bold"
                            >
                              {formatPercent(data.total_return)}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            {data.volatility.toFixed(2)}%
                          </TableCell>
                          <TableCell align="right">
                            {data.sharpe_ratio.toFixed(2)}
                          </TableCell>
                          <TableCell align="right">
                            <Chip
                              label={`#${data.performance_rank}`}
                              size="small"
                              color={data.performance_rank <= 3 ? 'success' : data.performance_rank <= 6 ? 'warning' : 'default'}
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </TableContainer>

              <Divider sx={{ my: 3 }} />

              {/* Rotation Insights */}
              <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                Rotation Insights
              </Typography>
              
              <List>
                {sectorRotation.rotation_insights.map((insight, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <Info color="primary" />
                    </ListItemIcon>
                    <ListItemText primary={insight} />
                  </ListItem>
                ))}
              </List>
            </Box>
          ) : (
            <Typography variant="body2" color="text.secondary" textAlign="center">
              Loading sector rotation data...
            </Typography>
          )}
        </CardContent>
      </Card>

      {/* Volatility Patterns Analysis */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            📊 Volatility Patterns Analysis
          </Typography>
          
          <Box display="flex" gap={2} mb={3}>
            <TextField
              label="Symbol"
              value={volatilitySymbol}
              onChange={(e) => setVolatilitySymbol(e.target.value.toUpperCase())}
              placeholder="e.g., SPY, QQQ"
              size="small"
              sx={{ width: 120 }}
            />
            <Button
              variant="contained"
              onClick={analyzeVolatilityPatterns}
              disabled={volatilityLoading || !volatilitySymbol.trim()}
              startIcon={<ShowChart />}
            >
              Analyze Volatility
            </Button>
          </Box>

          {volatilityLoading ? (
            <Box display="flex" justifyContent="center" py={4}>
              <CircularProgress />
            </Box>
          ) : volatilityPatterns ? (
            <Box>
              {/* Volatility Overview */}
              <Grid container spacing={3} sx={{ mb: 3 }}>
                <Grid item xs={12} md={3}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Current Regime
                      </Typography>
                      <Chip
                        label={volatilityPatterns.volatility_regimes.current_regime}
                        color={getRegimeColor(volatilityPatterns.volatility_regimes.current_regime)}
                      />
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Current Volatility
                      </Typography>
                      <Typography
                        variant="h6"
                        fontWeight="bold"
                        color={`${getVolatilityColor(volatilityPatterns.volatility_metrics.current_volatility * 100)}.main`}
                      >
                        {(volatilityPatterns.volatility_metrics.current_volatility * 100).toFixed(2)}%
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Average Volatility
                      </Typography>
                      <Typography variant="h6" fontWeight="bold">
                        {(volatilityPatterns.volatility_metrics.average_volatility * 100).toFixed(2)}%
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        Volatility Percentile
                      </Typography>
                      <Typography variant="h6" fontWeight="bold">
                        {volatilityPatterns.volatility_metrics.volatility_percentile.toFixed(1)}%
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>

              {/* Volatility Metrics */}
              <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                Volatility Metrics
              </Typography>
              
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={6} md={3}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Historical Volatility
                    </Typography>
                    <Typography variant="h6" fontWeight="bold">
                      {(volatilityPatterns.volatility_metrics.historical_volatility * 100).toFixed(2)}%
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Volatility of Volatility
                    </Typography>
                    <Typography variant="h6" fontWeight="bold">
                      {(volatilityPatterns.volatility_metrics.volatility_of_volatility * 100).toFixed(2)}%
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Volatility Trend
                    </Typography>
                    <Typography variant="h6" fontWeight="bold">
                      {volatilityPatterns.volatility_patterns.volatility_trend}
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Clustering Events
                    </Typography>
                    <Typography variant="h6" fontWeight="bold">
                      {volatilityPatterns.volatility_patterns.volatility_clustering}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>

              {/* Volatility Insights */}
              <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                Volatility Insights
              </Typography>
              
              <List>
                {volatilityPatterns.volatility_insights.map((insight, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <Info color="primary" />
                    </ListItemIcon>
                    <ListItemText primary={insight} />
                  </ListItem>
                ))}
              </List>
            </Box>
          ) : (
            <Typography variant="body2" color="text.secondary" textAlign="center">
              Enter a symbol and click "Analyze Volatility" to get volatility patterns
            </Typography>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default MarketIntelligence;



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
  Info
} from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';
import axios from 'axios';
import { API_CONFIG } from '../config/api';

interface PortfolioPerformance {
  performance_metrics: {
    total_value: number;
    total_cost: number;
    total_gain_loss: number;
    total_gain_loss_pct: number;
    portfolio_return: number;
    portfolio_volatility: number;
    sharpe_ratio: number;
    max_drawdown: number;
    positions_count: number;
  };
  benchmark_comparison: {
    benchmark_performance: Record<string, {
      symbol: string;
      return: number;
      outperformance: number;
    }>;
    best_benchmark: [string, any];
  };
  risk_analysis: {
    concentration_risk: string;
    hhi_index: number;
    portfolio_beta: number;
    portfolio_var: number;
    risk_level: string;
  };
  attribution_analysis: {
    attribution: Array<{
      symbol: string;
      contribution: number;
      contribution_pct: number;
      weight: number;
    }>;
    top_contributors: Array<any>;
    bottom_contributors: Array<any>;
    total_attribution: number;
  };
  recommendations: string[];
  timestamp: string;
}

interface MarketCorrelation {
  correlation_matrix: Record<string, Record<string, number>>;
  high_correlations: Array<{
    symbol1: string;
    symbol2: string;
    correlation: number;
    strength: string;
  }>;
  correlation_clusters: {
    clusters: Record<string, string[]>;
    cluster_count: number;
  };
  insights: string[];
  symbols: string[];
  timestamp: string;
}

const PortfolioAnalytics: React.FC = () => {
  const { user } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Portfolio data
  const [portfolioPerformance, setPortfolioPerformance] = useState<PortfolioPerformance | null>(null);
  const [marketCorrelation, setMarketCorrelation] = useState<MarketCorrelation | null>(null);
  const [portfolioData, setPortfolioData] = useState<any>(null);
  
  // Loading states
  const [performanceLoading, setPerformanceLoading] = useState(false);
  const [correlationLoading, setCorrelationLoading] = useState(false);

  useEffect(() => {
    if (user) {
      loadPortfolioData();
    }
  }, [user]);

  const loadPortfolioData = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_CONFIG.BASE_URL}/api/portfolio/${user?.id}`);
      setPortfolioData(response.data);
    } catch (err) {
      setError('Failed to load portfolio data');
      console.error('Portfolio data error:', err);
    } finally {
      setLoading(false);
    }
  };

  const analyzePortfolioPerformance = async () => {
    if (!portfolioData) return;

    try {
      setPerformanceLoading(true);
      setError(null);
      const response = await axios.post(`${API_CONFIG.BASE_URL}/api/analytics/portfolio-performance`, portfolioData);
      setPortfolioPerformance(response.data);
    } catch (err) {
      setError('Failed to analyze portfolio performance');
      console.error('Portfolio performance error:', err);
    } finally {
      setPerformanceLoading(false);
    }
  };

  const analyzeMarketCorrelation = async () => {
    if (!portfolioData?.positions) return;

    const symbols = portfolioData.positions.map((pos: any) => pos.symbol);
    if (symbols.length < 2) {
      setError('Need at least 2 positions for correlation analysis');
      return;
    }

    try {
      setCorrelationLoading(true);
      setError(null);
      const response = await axios.post(`${API_CONFIG.BASE_URL}/api/analytics/market-correlation`, symbols);
      setMarketCorrelation(response.data);
    } catch (err) {
      setError('Failed to analyze market correlation');
      console.error('Market correlation error:', err);
    } finally {
      setCorrelationLoading(false);
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

  const getPerformanceColor = (value: number) => {
    return value >= 0 ? 'success.main' : 'error.main';
  };

  const getRiskColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'low': return 'success';
      case 'medium': return 'warning';
      case 'high': return 'error';
      default: return 'default';
    }
  };

  const getCorrelationColor = (correlation: number) => {
    if (Math.abs(correlation) > 0.8) return 'error';
    if (Math.abs(correlation) > 0.6) return 'warning';
    return 'success';
  };

  const getCorrelationStrength = (correlation: number) => {
    const abs = Math.abs(correlation);
    if (abs > 0.8) return 'Strong';
    if (abs > 0.6) return 'Moderate';
    if (abs > 0.3) return 'Weak';
    return 'Very Weak';
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">
          📊 Portfolio Analytics
        </Typography>
        <Box display="flex" gap={2}>
          <Button
            variant="outlined"
            onClick={loadPortfolioData}
            startIcon={<Refresh />}
          >
            Refresh Portfolio
          </Button>
          <Button
            variant="contained"
            onClick={analyzePortfolioPerformance}
            disabled={performanceLoading || !portfolioData}
            startIcon={<Assessment />}
          >
            Analyze Performance
          </Button>
          <Button
            variant="contained"
            onClick={analyzeMarketCorrelation}
            disabled={correlationLoading || !portfolioData?.positions || portfolioData?.positions?.length < 2}
            startIcon={<BarChart />}
          >
            Analyze Correlation
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Portfolio Performance Analysis */}
      {portfolioPerformance && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              📈 Portfolio Performance Analysis
            </Typography>
            
            <Grid container spacing={3}>
              {/* Performance Metrics */}
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                  Performance Metrics
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Total Value
                      </Typography>
                      <Typography variant="h6" fontWeight="bold">
                        {formatCurrency(portfolioPerformance.performance_metrics.total_value)}
                      </Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Total Gain/Loss
                      </Typography>
                      <Typography
                        variant="h6"
                        fontWeight="bold"
                        color={getPerformanceColor(portfolioPerformance.performance_metrics.total_gain_loss)}
                      >
                        {formatCurrency(portfolioPerformance.performance_metrics.total_gain_loss)}
                      </Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Return %
                      </Typography>
                      <Typography
                        variant="h6"
                        fontWeight="bold"
                        color={getPerformanceColor(portfolioPerformance.performance_metrics.total_gain_loss_pct)}
                      >
                        {formatPercent(portfolioPerformance.performance_metrics.total_gain_loss_pct)}
                      </Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Sharpe Ratio
                      </Typography>
                      <Typography variant="h6" fontWeight="bold">
                        {portfolioPerformance.performance_metrics.sharpe_ratio.toFixed(2)}
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Grid>

              {/* Risk Analysis */}
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                  Risk Analysis
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Concentration Risk
                      </Typography>
                      <Chip
                        label={portfolioPerformance.risk_analysis.concentration_risk}
                        color={getRiskColor(portfolioPerformance.risk_analysis.concentration_risk)}
                        size="small"
                      />
                    </Box>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Portfolio Beta
                      </Typography>
                      <Typography variant="h6" fontWeight="bold">
                        {portfolioPerformance.risk_analysis.portfolio_beta.toFixed(2)}
                      </Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Volatility
                      </Typography>
                      <Typography variant="h6" fontWeight="bold">
                        {(portfolioPerformance.performance_metrics.portfolio_volatility * 100).toFixed(2)}%
                      </Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Max Drawdown
                      </Typography>
                      <Typography variant="h6" fontWeight="bold" color="error.main">
                        {(portfolioPerformance.performance_metrics.max_drawdown * 100).toFixed(2)}%
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Grid>
            </Grid>

            <Divider sx={{ my: 3 }} />

            {/* Benchmark Comparison */}
            <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
              Benchmark Comparison
            </Typography>
            
            <Grid container spacing={2}>
              {Object.entries(portfolioPerformance.benchmark_comparison.benchmark_performance).map(([benchmark, data]) => (
                <Grid item xs={12} sm={6} md={3} key={benchmark}>
                  <Card variant="outlined">
                    <CardContent sx={{ p: 2 }}>
                      <Typography variant="subtitle2" fontWeight="bold">
                        {benchmark}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {data.symbol}
                      </Typography>
                      <Typography variant="h6" fontWeight="bold">
                        {formatPercent(data.return)}
                      </Typography>
                      <Typography
                        variant="body2"
                        color={getPerformanceColor(data.outperformance)}
                      >
                        {formatPercent(data.outperformance)} vs Portfolio
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>

            <Divider sx={{ my: 3 }} />

            {/* Performance Attribution */}
            <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
              Performance Attribution
            </Typography>
            
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Symbol</TableCell>
                    <TableCell align="right">Weight</TableCell>
                    <TableCell align="right">Contribution</TableCell>
                    <TableCell align="right">Contribution %</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {portfolioPerformance.attribution_analysis.attribution.map((item) => (
                    <TableRow key={item.symbol}>
                      <TableCell>
                        <Typography variant="subtitle2" fontWeight="bold">
                          {item.symbol}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        {formatPercent(item.weight)}
                      </TableCell>
                      <TableCell align="right">
                        <Typography
                          color={getPerformanceColor(item.contribution)}
                        >
                          {formatCurrency(item.contribution)}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography
                          color={getPerformanceColor(item.contribution_pct)}
                        >
                          {formatPercent(item.contribution_pct)}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>

            <Divider sx={{ my: 3 }} />

            {/* Recommendations */}
            <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
              Recommendations
            </Typography>
            
            <List>
              {portfolioPerformance.recommendations.map((recommendation, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    <CheckCircle color="primary" />
                  </ListItemIcon>
                  <ListItemText primary={recommendation} />
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      )}

      {/* Market Correlation Analysis */}
      {marketCorrelation && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              🔗 Market Correlation Analysis
            </Typography>
            
            {/* High Correlations */}
            {marketCorrelation.high_correlations.length > 0 && (
              <Box mb={3}>
                <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                  High Correlations
                </Typography>
                
                <Grid container spacing={2}>
                  {marketCorrelation.high_correlations.slice(0, 6).map((correlation, index) => (
                    <Grid item xs={12} sm={6} md={4} key={index}>
                      <Card variant="outlined">
                        <CardContent sx={{ p: 2 }}>
                          <Typography variant="subtitle2" fontWeight="bold">
                            {correlation.symbol1} ↔ {correlation.symbol2}
                          </Typography>
                          <Typography
                            variant="h6"
                            fontWeight="bold"
                            color={`${getCorrelationColor(correlation.correlation)}.main`}
                          >
                            {correlation.correlation.toFixed(3)}
                          </Typography>
                          <Chip
                            label={correlation.strength}
                            color={getCorrelationColor(correlation.correlation)}
                            size="small"
                          />
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            )}

            {/* Correlation Clusters */}
            {marketCorrelation.correlation_clusters.cluster_count > 0 && (
              <Box mb={3}>
                <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                  Correlation Clusters
                </Typography>
                
                <Grid container spacing={2}>
                  {Object.entries(marketCorrelation.correlation_clusters.clusters).map(([clusterId, symbols]) => (
                    <Grid item xs={12} sm={6} md={4} key={clusterId}>
                      <Card variant="outlined">
                        <CardContent sx={{ p: 2 }}>
                          <Typography variant="subtitle2" fontWeight="bold">
                            Cluster {clusterId}
                          </Typography>
                          <Typography variant="body2">
                            {symbols.join(', ')}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {symbols.length} symbols
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            )}

            {/* Insights */}
            {marketCorrelation.insights.length > 0 && (
              <Box>
                <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                  Insights
                </Typography>
                
                <List>
                  {marketCorrelation.insights.map((insight, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <Info color="primary" />
                      </ListItemIcon>
                      <ListItemText primary={insight} />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
          </CardContent>
        </Card>
      )}

      {/* No Portfolio Data */}
      {!portfolioData && !loading && (
        <Card>
          <CardContent>
            <Box textAlign="center" py={4}>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No portfolio data available
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Add positions to your portfolio to enable analytics
              </Typography>
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default PortfolioAnalytics;



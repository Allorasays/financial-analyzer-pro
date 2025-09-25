import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Alert,
  CircularProgress,
  Chip
} from '@mui/material';
import { useWebSocket } from '../contexts/WebSocketContext';
import axios from 'axios';
import { API_CONFIG } from '../config/api';

interface MarketOverview {
  [key: string]: {
    price: number;
    change: number;
    change_percent: number;
    name: string;
  };
}

interface GlobalMarket {
  name: string;
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
}

const Dashboard: React.FC = () => {
  const { isConnected, lastMessage } = useWebSocket();
  const [marketOverview, setMarketOverview] = useState<MarketOverview>({});
  const [globalMarkets, setGlobalMarkets] = useState<GlobalMarket[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [symbol, setSymbol] = useState('AAPL');
  const [stockData, setStockData] = useState<any>(null);
  const [analyzing, setAnalyzing] = useState(false);

  useEffect(() => {
    fetchMarketData();
  }, []);

  useEffect(() => {
    if (lastMessage?.type === 'market_data') {
      setMarketOverview(lastMessage.data);
    }
  }, [lastMessage]);

  const fetchMarketData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch market overview
      const overviewResponse = await axios.get(`${API_CONFIG.BASE_URL}/api/market-overview`);
      setMarketOverview(overviewResponse.data);

      // Fetch global markets
      const globalResponse = await axios.get(`${API_CONFIG.BASE_URL}/api/global-markets`);
      setGlobalMarkets(globalResponse.data);

    } catch (err) {
      setError('Failed to fetch market data');
      console.error('Market data fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const analyzeStock = async () => {
    if (!symbol.trim()) return;

    try {
      setAnalyzing(true);
      setError(null);

      const response = await axios.get(`${API_CONFIG.BASE_URL}/api/market-data/${symbol}`);
      setStockData(response.data);

    } catch (err) {
      setError(`Failed to analyze ${symbol}`);
      console.error('Stock analysis error:', err);
    } finally {
      setAnalyzing(false);
    }
  };

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(price);
  };

  const formatChange = (change: number, changePercent: number) => {
    const sign = change >= 0 ? '+' : '';
    return `${sign}${formatPrice(change)} (${sign}${changePercent.toFixed(2)}%)`;
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
      <Typography variant="h4" gutterBottom>
        📊 Financial Dashboard
      </Typography>

      {/* Connection Status */}
      <Box mb={3}>
        <Chip
          label={isConnected ? '🟢 Real-time Connected' : '🔴 Disconnected'}
          color={isConnected ? 'success' : 'error'}
          variant="outlined"
        />
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Market Overview */}
      <Typography variant="h5" gutterBottom>
        📈 Market Overview
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {Object.entries(marketOverview).map(([symbol, data]) => (
          <Grid item xs={12} sm={6} md={3} key={symbol}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="primary">
                  {data.name}
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                  {formatPrice(data.price)}
                </Typography>
                <Typography
                  variant="body2"
                  color={data.change >= 0 ? 'success.main' : 'error.main'}
                >
                  {formatChange(data.change, data.change_percent)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Global Markets */}
      <Typography variant="h5" gutterBottom>
        🌍 Global Markets
      </Typography>
      <Grid container spacing={2} sx={{ mb: 4 }}>
        {globalMarkets.slice(0, 8).map((market) => (
          <Grid item xs={12} sm={6} md={3} key={market.symbol}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle1" fontWeight="bold">
                  {market.name}
                </Typography>
                <Typography variant="h6">
                  {market.price.toFixed(2)}
                </Typography>
                <Typography
                  variant="body2"
                  color={market.change >= 0 ? 'success.main' : 'error.main'}
                >
                  {formatChange(market.change, market.change_percent)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Stock Analysis */}
      <Typography variant="h5" gutterBottom>
        🔍 Stock Analysis
      </Typography>
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" gap={2} alignItems="center" mb={2}>
            <TextField
              label="Stock Symbol"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="e.g., AAPL, MSFT, GOOGL"
              size="small"
            />
            <Button
              variant="contained"
              onClick={analyzeStock}
              disabled={analyzing || !symbol.trim()}
            >
              {analyzing ? <CircularProgress size={20} /> : 'Analyze'}
            </Button>
          </Box>

          {stockData && (
            <Box>
              <Typography variant="h6" gutterBottom>
                {symbol} Analysis Results
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Data points: {stockData.data.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Last updated: {new Date(stockData.last_updated).toLocaleString()}
              </Typography>
              
              {stockData.data.length > 0 && (
                <Box mt={2}>
                  <Typography variant="subtitle2">Recent Data:</Typography>
                  <Box component="pre" sx={{ fontSize: '0.8rem', overflow: 'auto' }}>
                    {JSON.stringify(stockData.data.slice(-3), null, 2)}
                  </Box>
                </Box>
              )}
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Real-time Updates Info */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🔄 Real-time Updates
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This dashboard receives real-time market data updates via WebSocket connections.
            Market overview data refreshes every 30 seconds automatically.
          </Typography>
          {lastMessage && (
            <Box mt={2}>
              <Typography variant="subtitle2">Last Update:</Typography>
              <Typography variant="body2" color="text.secondary">
                {new Date(lastMessage.timestamp * 1000).toLocaleString()}
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default Dashboard;


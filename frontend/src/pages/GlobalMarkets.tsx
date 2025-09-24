import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
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
  Button
} from '@mui/material';
import { Refresh, TrendingUp, TrendingDown } from '@mui/icons-material';
import { useWebSocket } from '../contexts/WebSocketContext';
import axios from 'axios';
import { API_CONFIG } from '../config/api';

interface GlobalMarket {
  name: string;
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
}

const GlobalMarkets: React.FC = () => {
  const { isConnected, lastMessage } = useWebSocket();
  const [globalMarkets, setGlobalMarkets] = useState<GlobalMarket[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchGlobalMarkets();
  }, []);

  useEffect(() => {
    if (lastMessage?.type === 'market_data') {
      // Update global markets if market data includes global indices
      const marketData = lastMessage.data;
      if (marketData) {
        // This would be enhanced to update global markets data
        console.log('Market data update received:', marketData);
      }
    }
  }, [lastMessage]);

  const fetchGlobalMarkets = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await axios.get(`${API_CONFIG.BASE_URL}/api/global-markets`);
      setGlobalMarkets(response.data);

    } catch (err) {
      setError('Failed to fetch global market data');
      console.error('Global markets fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(price);
  };

  const formatChange = (change: number, changePercent: number) => {
    const sign = change >= 0 ? '+' : '';
    return `${sign}${change.toFixed(2)} (${sign}${changePercent.toFixed(2)}%)`;
  };

  const getChangeColor = (change: number) => {
    return change >= 0 ? 'success.main' : 'error.main';
  };

  const getChangeIcon = (change: number) => {
    return change >= 0 ? <TrendingUp color="success" /> : <TrendingDown color="error" />;
  };

  const getRegionColor = (symbol: string) => {
    if (symbol.includes('^FTSE') || symbol.includes('^GDAXI') || symbol.includes('^FCHI')) {
      return 'primary';
    } else if (symbol.includes('^N225') || symbol.includes('^HSI') || symbol.includes('000001.SS')) {
      return 'secondary';
    } else if (symbol.includes('^BSESN') || symbol.includes('^AXJO') || symbol.includes('^KS11')) {
      return 'success';
    }
    return 'default';
  };

  const getRegionName = (symbol: string) => {
    if (symbol.includes('^FTSE') || symbol.includes('^GDAXI') || symbol.includes('^FCHI')) {
      return 'Europe';
    } else if (symbol.includes('^N225') || symbol.includes('^HSI') || symbol.includes('000001.SS')) {
      return 'Asia';
    } else if (symbol.includes('^BSESN') || symbol.includes('^AXJO') || symbol.includes('^KS11')) {
      return 'Emerging Markets';
    }
    return 'Other';
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
          🌍 Global Markets
        </Typography>
        <Box display="flex" gap={2}>
          <Chip
            label={isConnected ? '🟢 Live Updates' : '🔴 Disconnected'}
            color={isConnected ? 'success' : 'error'}
            variant="outlined"
          />
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={fetchGlobalMarkets}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Market Overview Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {globalMarkets.slice(0, 8).map((market) => (
          <Grid item xs={12} sm={6} md={3} key={market.symbol}>
            <Card>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="subtitle1" fontWeight="bold">
                    {market.name}
                  </Typography>
                  <Chip
                    label={getRegionName(market.symbol)}
                    color={getRegionColor(market.symbol)}
                    size="small"
                  />
                </Box>
                <Typography variant="h5" sx={{ fontWeight: 'bold', mb: 1 }}>
                  {formatPrice(market.price)}
                </Typography>
                <Box display="flex" alignItems="center">
                  {getChangeIcon(market.change)}
                  <Typography
                    variant="body2"
                    color={getChangeColor(market.change)}
                    sx={{ ml: 1 }}
                  >
                    {formatChange(market.change, market.change_percent)}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Detailed Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            📊 Global Markets Overview
          </Typography>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Index</TableCell>
                  <TableCell>Region</TableCell>
                  <TableCell align="right">Price</TableCell>
                  <TableCell align="right">Change</TableCell>
                  <TableCell align="right">Change %</TableCell>
                  <TableCell align="center">Trend</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {globalMarkets.map((market) => (
                  <TableRow key={market.symbol}>
                    <TableCell>
                      <Typography variant="subtitle2" fontWeight="bold">
                        {market.name}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {market.symbol}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={getRegionName(market.symbol)}
                        color={getRegionColor(market.symbol)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="h6" fontWeight="bold">
                        {formatPrice(market.price)}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography
                        variant="body2"
                        color={getChangeColor(market.change)}
                      >
                        {market.change >= 0 ? '+' : ''}{market.change.toFixed(2)}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography
                        variant="body2"
                        color={getChangeColor(market.change_percent)}
                      >
                        {market.change_percent >= 0 ? '+' : ''}{market.change_percent.toFixed(2)}%
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      {getChangeIcon(market.change)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Market Summary */}
      <Grid container spacing={3} sx={{ mt: 4 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📈 Market Performance Summary
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Total Markets Tracked: {globalMarkets.length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Markets Up: {globalMarkets.filter(m => m.change >= 0).length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Markets Down: {globalMarkets.filter(m => m.change < 0).length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Average Change: {globalMarkets.length > 0 ? 
                    (globalMarkets.reduce((sum, m) => sum + m.change_percent, 0) / globalMarkets.length).toFixed(2) : 0}%
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🌐 Regional Distribution
              </Typography>
              <Box sx={{ mt: 2 }}>
                {['Europe', 'Asia', 'Emerging Markets'].map(region => {
                  const count = globalMarkets.filter(m => getRegionName(m.symbol) === region).length;
                  return (
                    <Box key={region} display="flex" justifyContent="space-between" mb={1}>
                      <Typography variant="body2">{region}</Typography>
                      <Typography variant="body2" fontWeight="bold">{count}</Typography>
                    </Box>
                  );
                })}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default GlobalMarkets;

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
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress
} from '@mui/material';
import { Add, Delete, TrendingUp, TrendingDown, Refresh, Visibility, VisibilityOff } from '@mui/icons-material';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useAuth } from '../contexts/AuthContext';
import axios from 'axios';
import { API_CONFIG } from '../config/api';

interface SymbolData {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  timestamp: number;
}

interface MarketData {
  symbol: string;
  period: string;
  data: any[];
  last_updated: string;
}

const MarketData: React.FC = () => {
  const { user } = useAuth();
  const { isConnected, lastMessage, sendMessage } = useWebSocket();
  const [subscribedSymbols, setSubscribedSymbols] = useState<Set<string>>(new Set());
  const [symbolData, setSymbolData] = useState<Map<string, SymbolData>>(new Map());
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [addSymbolOpen, setAddSymbolOpen] = useState(false);
  const [newSymbol, setNewSymbol] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState('');

  useEffect(() => {
    if (user) {
      // Load any existing subscriptions from localStorage
      const savedSubscriptions = localStorage.getItem('subscribedSymbols');
      if (savedSubscriptions) {
        const symbols = JSON.parse(savedSubscriptions);
        setSubscribedSymbols(new Set(symbols));
        symbols.forEach((symbol: string) => {
          sendMessage({ type: 'subscribe_symbol', symbol });
        });
      }
    }
  }, [user]);

  useEffect(() => {
    if (lastMessage?.type === 'symbol_update') {
      const data = lastMessage.data;
      setSymbolData(prev => new Map(prev.set(data.symbol, data)));
    }
  }, [lastMessage]);

  const handleSubscribeSymbol = async (symbol: string) => {
    try {
      setError(null);
      
      // Add to subscribed symbols
      const newSubscriptions = new Set(subscribedSymbols);
      newSubscriptions.add(symbol);
      setSubscribedSymbols(newSubscriptions);
      
      // Save to localStorage
      localStorage.setItem('subscribedSymbols', JSON.stringify([...newSubscriptions]));
      
      // Send WebSocket subscription
      sendMessage({ type: 'subscribe_symbol', symbol });
      
      // Fetch initial data
      await fetchSymbolData(symbol);
      
    } catch (err) {
      setError(`Failed to subscribe to ${symbol}`);
      console.error('Subscribe error:', err);
    }
  };

  const handleUnsubscribeSymbol = (symbol: string) => {
    try {
      // Remove from subscribed symbols
      const newSubscriptions = new Set(subscribedSymbols);
      newSubscriptions.delete(symbol);
      setSubscribedSymbols(newSubscriptions);
      
      // Save to localStorage
      localStorage.setItem('subscribedSymbols', JSON.stringify([...newSubscriptions]));
      
      // Send WebSocket unsubscription
      sendMessage({ type: 'unsubscribe_symbol', symbol });
      
      // Remove from local data
      setSymbolData(prev => {
        const newMap = new Map(prev);
        newMap.delete(symbol);
        return newMap;
      });
      
    } catch (err) {
      setError(`Failed to unsubscribe from ${symbol}`);
      console.error('Unsubscribe error:', err);
    }
  };

  const fetchSymbolData = async (symbol: string) => {
    try {
      setLoading(true);
      setError(null);

      const response = await axios.get(`${API_CONFIG.BASE_URL}/api/market-data/${symbol}`);
      setMarketData(response.data);

    } catch (err) {
      setError(`Failed to fetch data for ${symbol}`);
      console.error('Symbol data fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAddSymbol = async () => {
    if (newSymbol.trim()) {
      await handleSubscribeSymbol(newSymbol.trim().toUpperCase());
      setNewSymbol('');
      setAddSymbolOpen(false);
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

  const formatVolume = (volume: number) => {
    if (volume >= 1e9) {
      return `${(volume / 1e9).toFixed(1)}B`;
    } else if (volume >= 1e6) {
      return `${(volume / 1e6).toFixed(1)}M`;
    } else if (volume >= 1e3) {
      return `${(volume / 1e3).toFixed(1)}K`;
    }
    return volume.toString();
  };

  const getChangeColor = (change: number) => {
    return change >= 0 ? 'success.main' : 'error.main';
  };

  const getChangeIcon = (change: number) => {
    return change >= 0 ? <TrendingUp color="success" /> : <TrendingDown color="error" />;
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">
          📊 Real-Time Market Data
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
            onClick={() => {
              subscribedSymbols.forEach(symbol => {
                sendMessage({ type: 'subscribe_symbol', symbol });
              });
            }}
          >
            Refresh All
          </Button>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setAddSymbolOpen(true)}
          >
            Add Symbol
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Subscription Statistics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary">
                Subscribed Symbols
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                {subscribedSymbols.size}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary">
                Live Updates
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                {symbolData.size}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary">
                Connection Status
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                {isConnected ? '🟢' : '🔴'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary">
                Update Frequency
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                30s
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Real-Time Symbol Data */}
      {subscribedSymbols.size > 0 ? (
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              🔴 Live Symbol Updates
            </Typography>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Symbol</TableCell>
                    <TableCell align="right">Price</TableCell>
                    <TableCell align="right">Change</TableCell>
                    <TableCell align="right">Change %</TableCell>
                    <TableCell align="right">Volume</TableCell>
                    <TableCell align="right">Last Update</TableCell>
                    <TableCell align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Array.from(subscribedSymbols).map((symbol) => {
                    const data = symbolData.get(symbol);
                    return (
                      <TableRow key={symbol}>
                        <TableCell>
                          <Typography variant="subtitle2" fontWeight="bold">
                            {symbol}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="h6" fontWeight="bold">
                            {data ? formatCurrency(data.price) : 'Loading...'}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          {data ? (
                            <Box display="flex" alignItems="center" justifyContent="flex-end">
                              {getChangeIcon(data.change)}
                              <Typography
                                variant="body2"
                                color={getChangeColor(data.change)}
                                sx={{ ml: 1 }}
                              >
                                {formatCurrency(data.change)}
                              </Typography>
                            </Box>
                          ) : (
                            <LinearProgress sx={{ width: 100 }} />
                          )}
                        </TableCell>
                        <TableCell align="right">
                          {data ? (
                            <Typography
                              variant="body2"
                              color={getChangeColor(data.change_percent)}
                            >
                              {formatPercent(data.change_percent)}
                            </Typography>
                          ) : (
                            <LinearProgress sx={{ width: 100 }} />
                          )}
                        </TableCell>
                        <TableCell align="right">
                          {data ? formatVolume(data.volume) : 'Loading...'}
                        </TableCell>
                        <TableCell align="right">
                          {data ? new Date(data.timestamp * 1000).toLocaleTimeString() : 'Waiting...'}
                        </TableCell>
                        <TableCell align="center">
                          <IconButton
                            color="primary"
                            onClick={() => fetchSymbolData(symbol)}
                            sx={{ mr: 1 }}
                          >
                            <Refresh />
                          </IconButton>
                          <IconButton
                            color="error"
                            onClick={() => handleUnsubscribeSymbol(symbol)}
                          >
                            <Delete />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      ) : (
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Box textAlign="center" py={4}>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No symbols subscribed yet
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Subscribe to symbols to receive real-time price updates
              </Typography>
              <Button
                variant="contained"
                startIcon={<Add />}
                onClick={() => setAddSymbolOpen(true)}
              >
                Add Symbol
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Market Data Details */}
      {marketData && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              📈 Market Data Details - {marketData.symbol}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Period: {marketData.period} | Last Updated: {new Date(marketData.last_updated).toLocaleString()}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Data Points: {marketData.data.length}
            </Typography>
          </CardContent>
        </Card>
      )}

      {/* Add Symbol Dialog */}
      <Dialog open={addSymbolOpen} onClose={() => setAddSymbolOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Subscribe to Symbol</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <TextField
              fullWidth
              label="Symbol"
              value={newSymbol}
              onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
              placeholder="e.g., AAPL, MSFT, GOOGL"
              helperText="Enter a stock symbol to subscribe to real-time updates"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddSymbolOpen(false)}>Cancel</Button>
          <Button onClick={handleAddSymbol} variant="contained">
            Subscribe
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default MarketData;


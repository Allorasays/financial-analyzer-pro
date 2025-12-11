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
  MenuItem
} from '@mui/material';
import { Add, Delete, TrendingUp, TrendingDown, Refresh } from '@mui/icons-material';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useAuth } from '../contexts/AuthContext';
import axios from 'axios';
import { API_CONFIG } from '../config/api';

interface Position {
  id: string;
  symbol: string;
  quantity: number;
  purchase_price: number;
  current_price: number;
  cost_basis: number;
  current_value: number;
  pnl: number;
  pnl_percent: number;
  weight: number;
}

interface PortfolioSummary {
  portfolio_id: string;
  positions: Position[];
  total_value: number;
  total_cost: number;
  total_gain_loss: number;
  total_gain_loss_pct: number;
  diversification: any;
  performance_metrics: any;
}

const Portfolio: React.FC = () => {
  const { user } = useAuth();
  const { isConnected, lastMessage, sendMessage } = useWebSocket();
  const [portfolio, setPortfolio] = useState<PortfolioSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [addPositionOpen, setAddPositionOpen] = useState(false);
  const [newPosition, setNewPosition] = useState({
    symbol: '',
    shares: '',
    price: '',
    purchase_date: '',
    transaction_type: 'BUY',
    fees: '0',
    notes: ''
  });

  useEffect(() => {
    if (user) {
      fetchPortfolio();
      // Subscribe to portfolio updates
      sendMessage({ type: 'subscribe_portfolio' });
    }
  }, [user]);

  useEffect(() => {
    if (lastMessage?.type === 'portfolio_update') {
      setPortfolio(prev => ({
        ...prev,
        ...lastMessage.data
      }));
    }
  }, [lastMessage]);

  const fetchPortfolio = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await axios.get(`${API_CONFIG.BASE_URL}/api/portfolio/${user?.id}`);
      setPortfolio(response.data);

    } catch (err) {
      setError('Failed to fetch portfolio data');
      console.error('Portfolio fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAddPosition = async () => {
    try {
      setError(null);

      const response = await axios.post(`${API_CONFIG.BASE_URL}/api/portfolio/${user?.id}/positions`, {
        symbol: newPosition.symbol,
        shares: parseFloat(newPosition.shares),
        price: parseFloat(newPosition.price),
        purchase_date: newPosition.purchase_date,
        transaction_type: newPosition.transaction_type,
        fees: parseFloat(newPosition.fees),
        notes: newPosition.notes
      });

      if (response.data.message) {
        setAddPositionOpen(false);
        setNewPosition({
          symbol: '',
          shares: '',
          price: '',
          purchase_date: '',
          transaction_type: 'BUY',
          fees: '0',
          notes: ''
        });
        fetchPortfolio();
      }

    } catch (err) {
      setError('Failed to add position');
      console.error('Add position error:', err);
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
          💼 Portfolio Management
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
            onClick={fetchPortfolio}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setAddPositionOpen(true)}
          >
            Add Position
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Portfolio Summary */}
      {portfolio && (
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="primary">
                  Total Value
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                  {formatCurrency(portfolio.total_value)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="primary">
                  Total Cost
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                  {formatCurrency(portfolio.total_cost)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="primary">
                  Gain/Loss
                </Typography>
                <Typography 
                  variant="h4" 
                  sx={{ 
                    fontWeight: 'bold',
                    color: portfolio.total_gain_loss >= 0 ? 'success.main' : 'error.main'
                  }}
                >
                  {formatCurrency(portfolio.total_gain_loss)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="primary">
                  Gain/Loss %
                </Typography>
                <Typography 
                  variant="h4" 
                  sx={{ 
                    fontWeight: 'bold',
                    color: portfolio.total_gain_loss >= 0 ? 'success.main' : 'error.main'
                  }}
                >
                  {formatPercent(portfolio.total_gain_loss_pct)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Positions Table */}
      {portfolio && portfolio.positions && portfolio.positions.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              📈 Positions ({portfolio.positions.length})
            </Typography>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Symbol</TableCell>
                    <TableCell align="right">Quantity</TableCell>
                    <TableCell align="right">Purchase Price</TableCell>
                    <TableCell align="right">Current Price</TableCell>
                    <TableCell align="right">Current Value</TableCell>
                    <TableCell align="right">P&L</TableCell>
                    <TableCell align="right">P&L %</TableCell>
                    <TableCell align="right">Weight</TableCell>
                    <TableCell align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {portfolio.positions.map((position) => (
                    <TableRow key={position.id}>
                      <TableCell>
                        <Typography variant="subtitle2" fontWeight="bold">
                          {position.symbol}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        {position.quantity.toFixed(2)}
                      </TableCell>
                      <TableCell align="right">
                        {formatCurrency(position.purchase_price)}
                      </TableCell>
                      <TableCell align="right">
                        {formatCurrency(position.current_price)}
                      </TableCell>
                      <TableCell align="right">
                        {formatCurrency(position.current_value)}
                      </TableCell>
                      <TableCell align="right">
                        <Box display="flex" alignItems="center" justifyContent="flex-end">
                          {position.pnl >= 0 ? (
                            <TrendingUp color="success" />
                          ) : (
                            <TrendingDown color="error" />
                          )}
                          <Typography
                            variant="body2"
                            color={position.pnl >= 0 ? 'success.main' : 'error.main'}
                            sx={{ ml: 1 }}
                          >
                            {formatCurrency(position.pnl)}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell align="right">
                        <Typography
                          variant="body2"
                          color={position.pnl_percent >= 0 ? 'success.main' : 'error.main'}
                        >
                          {formatPercent(position.pnl_percent)}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        {formatPercent(position.weight)}
                      </TableCell>
                      <TableCell align="center">
                        <IconButton
                          color="error"
                          size="small"
                          onClick={() => {
                            // TODO: Implement delete position
                            console.log('Delete position:', position.id);
                          }}
                        >
                          <Delete />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {/* Add Position Dialog */}
      <Dialog open={addPositionOpen} onClose={() => setAddPositionOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add New Position</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Symbol"
                value={newPosition.symbol}
                onChange={(e) => setNewPosition({ ...newPosition, symbol: e.target.value.toUpperCase() })}
                placeholder="e.g., AAPL"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Shares"
                type="number"
                value={newPosition.shares}
                onChange={(e) => setNewPosition({ ...newPosition, shares: e.target.value })}
                placeholder="e.g., 10"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Purchase Price"
                type="number"
                value={newPosition.price}
                onChange={(e) => setNewPosition({ ...newPosition, price: e.target.value })}
                placeholder="e.g., 150.00"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Purchase Date"
                type="date"
                value={newPosition.purchase_date}
                onChange={(e) => setNewPosition({ ...newPosition, purchase_date: e.target.value })}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Transaction Type</InputLabel>
                <Select
                  value={newPosition.transaction_type}
                  onChange={(e) => setNewPosition({ ...newPosition, transaction_type: e.target.value })}
                >
                  <MenuItem value="BUY">Buy</MenuItem>
                  <MenuItem value="SELL">Sell</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Fees"
                type="number"
                value={newPosition.fees}
                onChange={(e) => setNewPosition({ ...newPosition, fees: e.target.value })}
                placeholder="e.g., 9.99"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Notes"
                multiline
                rows={2}
                value={newPosition.notes}
                onChange={(e) => setNewPosition({ ...newPosition, notes: e.target.value })}
                placeholder="Optional notes about this position"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddPositionOpen(false)}>Cancel</Button>
          <Button onClick={handleAddPosition} variant="contained">
            Add Position
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Portfolio;



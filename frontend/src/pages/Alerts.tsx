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
  Switch,
  FormControlLabel
} from '@mui/material';
import { Add, Delete, Notifications, NotificationsOff, TrendingUp, TrendingDown } from '@mui/icons-material';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useAuth } from '../contexts/AuthContext';
import axios from 'axios';
import { API_CONFIG } from '../config/api';

interface Alert {
  id: string;
  user_id: number;
  symbol?: string;
  alert_type: string;
  target_price?: number;
  target_value?: number;
  created_at: string;
  is_active: boolean;
}

const Alerts: React.FC = () => {
  const { user } = useAuth();
  const { isConnected, lastMessage } = useWebSocket();
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [addAlertOpen, setAddAlertOpen] = useState(false);
  const [alertType, setAlertType] = useState<'price' | 'portfolio'>('price');
  const [newAlert, setNewAlert] = useState({
    symbol: '',
    alert_type: 'PRICE_ABOVE',
    target_price: '',
    target_value: '',
    is_active: true
  });

  useEffect(() => {
    if (user) {
      fetchAlerts();
    }
  }, [user]);

  useEffect(() => {
    if (lastMessage?.type === 'price_alert' || lastMessage?.type === 'portfolio_alert') {
      // Show notification for triggered alert
      showNotification(lastMessage);
    }
  }, [lastMessage]);

  const fetchAlerts = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await axios.get(`${API_CONFIG.BASE_URL}/api/alerts`);
      setAlerts(response.data.alerts || []);

    } catch (err) {
      setError('Failed to fetch alerts');
      console.error('Alerts fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateAlert = async () => {
    try {
      setError(null);

      if (alertType === 'price') {
        const response = await axios.post(`${API_CONFIG.BASE_URL}/api/alerts/price`, {
          symbol: newAlert.symbol,
          alert_type: newAlert.alert_type,
          target_price: parseFloat(newAlert.target_price)
        });

        if (response.data.message) {
          setAddAlertOpen(false);
          resetNewAlert();
          fetchAlerts();
        }
      } else {
        const response = await axios.post(`${API_CONFIG.BASE_URL}/api/alerts/portfolio`, {
          alert_type: newAlert.alert_type,
          target_value: parseFloat(newAlert.target_value)
        });

        if (response.data.message) {
          setAddAlertOpen(false);
          resetNewAlert();
          fetchAlerts();
        }
      }

    } catch (err) {
      setError('Failed to create alert');
      console.error('Create alert error:', err);
    }
  };

  const handleDeleteAlert = async (alertId: string) => {
    try {
      const response = await axios.delete(`${API_CONFIG.BASE_URL}/api/alerts/${alertId}`);
      
      if (response.data.message) {
        fetchAlerts();
      }
    } catch (err) {
      setError('Failed to delete alert');
      console.error('Delete alert error:', err);
    }
  };

  const resetNewAlert = () => {
    setNewAlert({
      symbol: '',
      alert_type: 'PRICE_ABOVE',
      target_price: '',
      target_value: '',
      is_active: true
    });
  };

  const showNotification = (message: any) => {
    if ('Notification' in window && Notification.permission === 'granted') {
      const notification = new Notification(
        message.data?.title || 'Alert Triggered',
        {
          body: message.data?.message || 'An alert has been triggered',
          icon: '/favicon.ico',
          tag: message.data?.symbol || 'alert'
        }
      );
      
      notification.onclick = () => {
        window.focus();
        notification.close();
      };
    }
  };

  const requestNotificationPermission = async () => {
    if ('Notification' in window && Notification.permission === 'default') {
      const permission = await Notification.requestPermission();
      if (permission === 'granted') {
        console.log('Notification permission granted');
      }
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

  const getAlertTypeLabel = (type: string) => {
    const labels: { [key: string]: string } = {
      'PRICE_ABOVE': 'Price Above',
      'PRICE_BELOW': 'Price Below',
      'PRICE_CHANGE_UP': 'Price Increase',
      'PRICE_CHANGE_DOWN': 'Price Decrease',
      'PORTFOLIO_VALUE_ABOVE': 'Portfolio Value Above',
      'PORTFOLIO_VALUE_BELOW': 'Portfolio Value Below',
      'PORTFOLIO_GAIN_ABOVE': 'Portfolio Gain Above',
      'PORTFOLIO_GAIN_BELOW': 'Portfolio Gain Below',
      'PORTFOLIO_GAIN_PCT_ABOVE': 'Portfolio Gain % Above',
      'PORTFOLIO_GAIN_PCT_BELOW': 'Portfolio Gain % Below'
    };
    return labels[type] || type;
  };

  const getAlertIcon = (type: string) => {
    if (type.includes('ABOVE') || type.includes('UP')) {
      return <TrendingUp color="success" />;
    } else if (type.includes('BELOW') || type.includes('DOWN')) {
      return <TrendingDown color="error" />;
    }
    return <Notifications />;
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
          🔔 Alert Management
        </Typography>
        <Box display="flex" gap={2}>
          <Chip
            label={isConnected ? '🟢 Live Alerts' : '🔴 Disconnected'}
            color={isConnected ? 'success' : 'error'}
            variant="outlined"
          />
          <Button
            variant="outlined"
            onClick={requestNotificationPermission}
            startIcon={<Notifications />}
          >
            Enable Notifications
          </Button>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setAddAlertOpen(true)}
          >
            Create Alert
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Alert Statistics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary">
                Total Alerts
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                {alerts.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary">
                Active Alerts
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                {alerts.filter(alert => alert.is_active).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary">
                Price Alerts
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                {alerts.filter(alert => alert.symbol).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary">
                Portfolio Alerts
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                {alerts.filter(alert => !alert.symbol).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Alerts Table */}
      {alerts.length > 0 ? (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              📋 Active Alerts
            </Typography>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Type</TableCell>
                    <TableCell>Symbol</TableCell>
                    <TableCell align="right">Target</TableCell>
                    <TableCell align="center">Status</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {alerts.map((alert) => (
                    <TableRow key={alert.id}>
                      <TableCell>
                        <Box display="flex" alignItems="center">
                          {getAlertIcon(alert.alert_type)}
                          <Typography variant="body2" sx={{ ml: 1 }}>
                            {getAlertTypeLabel(alert.alert_type)}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Typography variant="subtitle2" fontWeight="bold">
                          {alert.symbol || 'Portfolio'}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        {alert.target_price ? formatCurrency(alert.target_price) : 
                         alert.target_value ? formatCurrency(alert.target_value) : 'N/A'}
                      </TableCell>
                      <TableCell align="center">
                        <Chip
                          label={alert.is_active ? 'Active' : 'Inactive'}
                          color={alert.is_active ? 'success' : 'default'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        {new Date(alert.created_at).toLocaleDateString()}
                      </TableCell>
                      <TableCell align="center">
                        <IconButton
                          color="error"
                          size="small"
                          onClick={() => handleDeleteAlert(alert.id)}
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
      ) : (
        <Card>
          <CardContent>
            <Box textAlign="center" py={4}>
              <NotificationsOff sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No alerts created yet
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Create your first alert to get notified about price changes and portfolio updates
              </Typography>
              <Button
                variant="contained"
                startIcon={<Add />}
                onClick={() => setAddAlertOpen(true)}
              >
                Create Alert
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Create Alert Dialog */}
      <Dialog open={addAlertOpen} onClose={() => setAddAlertOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Alert</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Alert Type</InputLabel>
              <Select
                value={alertType}
                onChange={(e) => setAlertType(e.target.value as 'price' | 'portfolio')}
              >
                <MenuItem value="price">Price Alert</MenuItem>
                <MenuItem value="portfolio">Portfolio Alert</MenuItem>
              </Select>
            </FormControl>

            {alertType === 'price' && (
              <>
                <TextField
                  fullWidth
                  label="Symbol"
                  value={newAlert.symbol}
                  onChange={(e) => setNewAlert({ ...newAlert, symbol: e.target.value.toUpperCase() })}
                  placeholder="e.g., AAPL"
                  sx={{ mb: 2 }}
                />
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Alert Condition</InputLabel>
                  <Select
                    value={newAlert.alert_type}
                    onChange={(e) => setNewAlert({ ...newAlert, alert_type: e.target.value })}
                  >
                    <MenuItem value="PRICE_ABOVE">Price Above</MenuItem>
                    <MenuItem value="PRICE_BELOW">Price Below</MenuItem>
                    <MenuItem value="PRICE_CHANGE_UP">Price Increase</MenuItem>
                    <MenuItem value="PRICE_CHANGE_DOWN">Price Decrease</MenuItem>
                  </Select>
                </FormControl>
                <TextField
                  fullWidth
                  label="Target Price"
                  type="number"
                  value={newAlert.target_price}
                  onChange={(e) => setNewAlert({ ...newAlert, target_price: e.target.value })}
                  placeholder="e.g., 150.00"
                />
              </>
            )}

            {alertType === 'portfolio' && (
              <>
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Alert Condition</InputLabel>
                  <Select
                    value={newAlert.alert_type}
                    onChange={(e) => setNewAlert({ ...newAlert, alert_type: e.target.value })}
                  >
                    <MenuItem value="PORTFOLIO_VALUE_ABOVE">Portfolio Value Above</MenuItem>
                    <MenuItem value="PORTFOLIO_VALUE_BELOW">Portfolio Value Below</MenuItem>
                    <MenuItem value="PORTFOLIO_GAIN_ABOVE">Portfolio Gain Above</MenuItem>
                    <MenuItem value="PORTFOLIO_GAIN_BELOW">Portfolio Gain Below</MenuItem>
                    <MenuItem value="PORTFOLIO_GAIN_PCT_ABOVE">Portfolio Gain % Above</MenuItem>
                    <MenuItem value="PORTFOLIO_GAIN_PCT_BELOW">Portfolio Gain % Below</MenuItem>
                  </Select>
                </FormControl>
                <TextField
                  fullWidth
                  label="Target Value"
                  type="number"
                  value={newAlert.target_value}
                  onChange={(e) => setNewAlert({ ...newAlert, target_value: e.target.value })}
                  placeholder="e.g., 10000.00"
                />
              </>
            )}

            <FormControlLabel
              control={
                <Switch
                  checked={newAlert.is_active}
                  onChange={(e) => setNewAlert({ ...newAlert, is_active: e.target.checked })}
                />
              }
              label="Active"
              sx={{ mt: 2 }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddAlertOpen(false)}>Cancel</Button>
          <Button onClick={handleCreateAlert} variant="contained">
            Create Alert
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Alerts;

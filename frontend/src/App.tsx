import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box, AppBar, Toolbar, Typography, IconButton, Switch } from '@mui/material';
import { Brightness4, Brightness7 } from '@mui/icons-material';
import Dashboard from './pages/Dashboard';
import Portfolio from './pages/Portfolio';
import MarketData from './pages/MarketData';
import GlobalMarkets from './pages/GlobalMarkets';
import Alerts from './pages/Alerts';
import AIAnalytics from './pages/AIAnalytics';
import SentimentAnalysis from './pages/SentimentAnalysis';
import PortfolioAnalytics from './pages/PortfolioAnalytics';
import MarketIntelligence from './pages/MarketIntelligence';
import Login from './pages/Login';
import Register from './pages/Register';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { WebSocketProvider } from './contexts/WebSocketContext';
import './App.css';

// Create light and dark themes
const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#667eea',
    },
    secondary: {
      main: '#764ba2',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
});

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#7c3aed',
    },
    secondary: {
      main: '#a855f7',
    },
    background: {
      default: '#1a1a1a',
      paper: '#2d2d2d',
    },
  },
});

function AppContent() {
  const { user, loading } = useAuth();
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });

  const theme = darkMode ? darkTheme : lightTheme;

  const handleThemeToggle = () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    localStorage.setItem('darkMode', JSON.stringify(newMode));
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <Typography>Loading...</Typography>
      </Box>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ flexGrow: 1 }}>
          <AppBar position="static">
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                📊 Financial Analyzer Pro
              </Typography>
              <IconButton color="inherit" onClick={handleThemeToggle}>
                {darkMode ? <Brightness7 /> : <Brightness4 />}
              </IconButton>
            </Toolbar>
          </AppBar>
          
          <Box sx={{ p: 3 }}>
            <Routes>
              {user ? (
                <>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/portfolio" element={<Portfolio />} />
                  <Route path="/market-data" element={<MarketData />} />
                  <Route path="/global-markets" element={<GlobalMarkets />} />
                  <Route path="/alerts" element={<Alerts />} />
                  <Route path="/ai-analytics" element={<AIAnalytics />} />
                  <Route path="/sentiment-analysis" element={<SentimentAnalysis />} />
                  <Route path="/portfolio-analytics" element={<PortfolioAnalytics />} />
                  <Route path="/market-intelligence" element={<MarketIntelligence />} />
                  <Route path="/login" element={<Navigate to="/" replace />} />
                  <Route path="/register" element={<Navigate to="/" replace />} />
                </>
              ) : (
                <>
                  <Route path="/login" element={<Login />} />
                  <Route path="/register" element={<Register />} />
                  <Route path="*" element={<Navigate to="/login" replace />} />
                </>
              )}
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

function App() {
  return (
    <AuthProvider>
      <WebSocketProvider>
        <AppContent />
      </WebSocketProvider>
    </AuthProvider>
  );
}

export default App;

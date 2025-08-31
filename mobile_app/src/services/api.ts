import axios from 'axios';

// Create axios instance with base configuration
export const api = axios.create({
  baseURL: 'http://localhost:8000', // Change this to your backend URL
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    // Token will be added by AuthContext
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      console.log('Unauthorized access, redirecting to login');
    }
    return Promise.reject(error);
  }
);

// API endpoints
export const endpoints = {
  // Authentication
  auth: {
    login: '/api/auth/login',
    register: '/api/auth/register',
  },
  
  // Market data
  market: {
    realtime: (ticker: string) => `/api/market/realtime/${ticker}`,
    overview: '/api/market/overview',
  },
  
  // Technical analysis
  technical: (ticker: string, period: string = '1y') => 
    `/api/technical/${ticker}?period=${period}`,
  
  // Machine learning
  ml: {
    predictions: (ticker: string, days: number = 30) => 
      `/api/ml/predictions/${ticker}?days=${days}`,
  },
  
  // Portfolio
  portfolio: {
    get: '/api/portfolio',
    add: '/api/portfolio/add',
  },
  
  // Watchlist
  watchlist: {
    get: '/api/watchlist',
    add: '/api/watchlist/add',
  },
};

// API functions
export const apiService = {
  // Market data
  getRealtimeData: async (ticker: string) => {
    const response = await api.get(endpoints.market.realtime(ticker));
    return response.data;
  },
  
  getMarketOverview: async () => {
    const response = await api.get(endpoints.market.overview);
    return response.data;
  },
  
  // Technical analysis
  getTechnicalAnalysis: async (ticker: string, period: string = '1y') => {
    const response = await api.get(endpoints.technical(ticker, period));
    return response.data;
  },
  
  // ML predictions
  getMLPredictions: async (ticker: string, days: number = 30) => {
    const response = await api.get(endpoints.ml.predictions(ticker, days));
    return response.data;
  },
  
  // Portfolio
  getPortfolio: async () => {
    const response = await api.get(endpoints.portfolio.get);
    return response.data;
  },
  
  addToPortfolio: async (ticker: string, shares: number, avgPrice: number) => {
    const response = await api.post(endpoints.portfolio.add, {
      ticker,
      shares,
      avg_price: avgPrice,
    });
    return response.data;
  },
  
  // Watchlist
  getWatchlist: async () => {
    const response = await api.get(endpoints.watchlist.get);
    return response.data;
  },
  
  addToWatchlist: async (ticker: string) => {
    const response = await api.post(endpoints.watchlist.add, { ticker });
    return response.data;
  },
};

import axios from 'axios';
import { getApiUrl } from '../config/api';
import { mockApiService } from './mockApi';

// Create axios instance with base configuration
export const api = axios.create({
  baseURL: getApiUrl(), // Dynamic API URL based on environment
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

// API functions - Using mock data for now
export const apiService = {
  // Market data
  getRealtimeData: async (ticker: string) => {
    // For now, use mock data. Later we'll connect to real API
    return await mockApiService.getRealtimeData(ticker);
  },
  
  getMarketOverview: async () => {
    return await mockApiService.getMarketOverview();
  },
  
  // Technical analysis
  getTechnicalAnalysis: async (ticker: string, period: string = '1y') => {
    return await mockApiService.getTechnicalAnalysis(ticker, period);
  },
  
  // ML predictions
  getMLPredictions: async (ticker: string, days: number = 30) => {
    return await mockApiService.getMLPredictions(ticker, days);
  },
  
  // Portfolio
  getPortfolio: async () => {
    return await mockApiService.getPortfolio();
  },
  
  addToPortfolio: async (ticker: string, shares: number, avgPrice: number) => {
    return await mockApiService.addToPortfolio(ticker, shares, avgPrice);
  },
  
  // Watchlist
  getWatchlist: async () => {
    return await mockApiService.getWatchlist();
  },
  
  addToWatchlist: async (ticker: string) => {
    return await mockApiService.addToWatchlist(ticker);
  },
};

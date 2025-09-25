export const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  WS_URL: process.env.REACT_APP_WS_URL || 'ws://localhost:8000',
  TIMEOUT: 10000,
};

export const ENDPOINTS = {
  AUTH: {
    LOGIN: '/auth/login',
    REGISTER: '/auth/register',
  },
  MARKET_DATA: '/api/market-data',
  MARKET_OVERVIEW: '/api/market-overview',
  GLOBAL_MARKETS: '/api/global-markets',
  PORTFOLIO: '/api/portfolio',
  WEBSOCKET: '/ws',
};

export const WEBSOCKET_MESSAGE_TYPES = {
  PING: 'ping',
  PONG: 'pong',
  MARKET_DATA: 'market_data',
  PORTFOLIO_UPDATE: 'portfolio_update',
  PRICE_ALERT: 'price_alert',
  SYMBOL_UPDATE: 'symbol_update',
  SUBSCRIBE: 'subscribe',
  SUBSCRIPTION_CONFIRMED: 'subscription_confirmed',
};


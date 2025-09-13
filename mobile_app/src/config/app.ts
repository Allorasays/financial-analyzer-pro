// App configuration
export const APP_CONFIG = {
  // Set to true to use mock data instead of real API
  USE_MOCK_DATA: true,
  
  // Set to false to disable authentication (for testing)
  ENABLE_AUTH: false,
  
  // App version
  VERSION: '1.0.0',
  
  // App name
  NAME: 'Financial Analyzer Pro',
  
  // Default refresh intervals (in milliseconds)
  REFRESH_INTERVALS: {
    MARKET_DATA: 30000, // 30 seconds
    PORTFOLIO: 60000,   // 1 minute
    PREDICTIONS: 300000 // 5 minutes
  }
};

// Helper function to check if mock data is enabled
export const isMockMode = () => APP_CONFIG.USE_MOCK_DATA;

// Helper function to check if auth is enabled
export const isAuthEnabled = () => APP_CONFIG.ENABLE_AUTH;







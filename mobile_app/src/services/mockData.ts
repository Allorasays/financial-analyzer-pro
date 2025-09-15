// Mock data for testing the app without a backend
export const mockData = {
  // Mock market data
  marketOverview: {
    indices: [
      {
        symbol: 'SPX',
        name: 'S&P 500',
        value: 4567.89,
        change: 23.45,
        change_pct: 0.52
      },
      {
        symbol: 'IXIC',
        name: 'NASDAQ',
        value: 14234.56,
        change: -12.34,
        change_pct: -0.09
      },
      {
        symbol: 'DJI',
        name: 'DOW JONES',
        value: 34567.89,
        change: 45.67,
        change_pct: 0.13
      }
    ]
  },

  // Mock trending stocks
  trendingStocks: [
    {
      ticker: 'AAPL',
      price: 175.43,
      change: 2.34,
      change_pct: 1.35,
      volume: 45678900
    },
    {
      ticker: 'GOOGL',
      price: 142.67,
      change: -1.23,
      change_pct: -0.85,
      volume: 23456700
    },
    {
      ticker: 'MSFT',
      price: 378.91,
      change: 4.56,
      change_pct: 1.22,
      volume: 34567800
    },
    {
      ticker: 'TSLA',
      price: 234.56,
      change: -5.67,
      change_pct: -2.36,
      volume: 67890100
    }
  ],

  // Mock portfolio data
  portfolio: {
    items: [
      {
        ticker: 'AAPL',
        shares: 10,
        avg_price: 170.00,
        current_price: 175.43,
        total_value: 1754.30,
        total_cost: 1700.00,
        gain_loss: 54.30,
        gain_loss_pct: 3.19,
        added_at: '2024-01-15'
      },
      {
        ticker: 'GOOGL',
        shares: 5,
        avg_price: 145.00,
        current_price: 142.67,
        total_value: 713.35,
        total_cost: 725.00,
        gain_loss: -11.65,
        gain_loss_pct: -1.61,
        added_at: '2024-01-20'
      }
    ],
    summary: {
      total_value: 2467.65,
      total_cost: 2425.00,
      total_gain_loss: 42.65,
      total_gain_loss_pct: 1.76,
      num_positions: 2
    }
  },

  // Mock ML predictions
  mlPredictions: {
    ticker: 'AAPL',
    current_price: 175.43,
    predicted_price_1d: 177.89,
    confidence_score: 0.85,
    model_accuracy: 0.78,
    future_predictions: [
      { day: 1, predicted_price: 177.89, date: '2024-01-22' },
      { day: 7, predicted_price: 182.34, date: '2024-01-28' },
      { day: 30, predicted_price: 195.67, date: '2024-02-20' }
    ],
    timestamp: new Date().toISOString()
  }
};

// Mock API service for testing
export const mockApiService = {
  getMarketOverview: async () => {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    return mockData.marketOverview;
  },

  getTrendingStocks: async () => {
    await new Promise(resolve => setTimeout(resolve, 800));
    return mockData.trendingStocks;
  },

  getPortfolio: async () => {
    await new Promise(resolve => setTimeout(resolve, 1200));
    return mockData.portfolio;
  },

  getMLPredictions: async (ticker: string, days: number = 30) => {
    await new Promise(resolve => setTimeout(resolve, 2000));
    return {
      ...mockData.mlPredictions,
      ticker: ticker.toUpperCase()
    };
  },

  login: async (username: string, password: string) => {
    await new Promise(resolve => setTimeout(resolve, 1500));
    if (username === 'demo' && password === 'demo') {
      return {
        token: 'mock-jwt-token-12345',
        user: {
          username: 'demo',
          email: 'demo@example.com'
        }
      };
    }
    throw new Error('Invalid credentials');
  }
};









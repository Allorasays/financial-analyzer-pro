# Financial Analyzer Pro - Mobile App

A React Native mobile application that provides advanced financial analysis, real-time market data, portfolio management, and machine learning predictions.

## Features

### 🔐 User Authentication
- Secure user registration and login
- JWT token-based authentication
- Persistent login sessions

### 📊 Real-time Market Data
- Live stock prices and market data
- Major market indices (S&P 500, NASDAQ, DOW, Russell 2000)
- Trending stocks with real-time updates
- Market overview and sentiment analysis

### 💼 Portfolio Management
- Track your stock investments
- Add/remove stocks with share quantities and average prices
- Real-time portfolio value calculations
- Gain/loss tracking with percentage changes
- Portfolio performance summary

### ⭐ Watchlist
- Monitor stocks of interest
- Real-time price updates
- Easy addition and removal of stocks

### 📈 Advanced Technical Analysis
- Moving averages (SMA 20, 50, 200)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic oscillators
- Volume analysis
- ATR (Average True Range)

### 🤖 Machine Learning Predictions
- AI-powered price forecasting
- Random Forest algorithm
- Confidence scores and accuracy metrics
- Short-term and long-term predictions
- Historical data analysis

## Tech Stack

- **Frontend**: React Native with Expo
- **Navigation**: React Navigation (Stack, Tab, Drawer)
- **UI Components**: React Native Paper, Custom components
- **Charts**: React Native Chart Kit
- **Icons**: Expo Vector Icons
- **HTTP Client**: Axios
- **Storage**: AsyncStorage
- **Authentication**: JWT tokens with bcrypt
- **Styling**: StyleSheet with modern design

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Expo CLI (`npm install -g @expo/cli`)
- iOS Simulator (for iOS development)
- Android Studio (for Android development)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mobile_app
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Start the development server**
   ```bash
   npm start
   # or
   expo start
   ```

4. **Run on device/simulator**
   - Press `i` for iOS simulator
   - Press `a` for Android emulator
   - Scan QR code with Expo Go app on your phone

## Configuration

### Backend API
Update the API base URL in `src/services/api.ts`:
```typescript
export const api = axios.create({
  baseURL: 'http://your-backend-url:8000', // Change this
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});
```

### Environment Variables
Create a `.env` file in the root directory:
```env
EXPO_PUBLIC_API_URL=http://your-backend-url:8000
```

## Project Structure

```
mobile_app/
├── src/
│   ├── contexts/
│   │   └── AuthContext.tsx          # Authentication context
│   ├── screens/
│   │   ├── LoginScreen.tsx          # User login
│   │   ├── RegisterScreen.tsx       # User registration
│   │   ├── DashboardScreen.tsx      # Main dashboard
│   │   ├── PortfolioScreen.tsx      # Portfolio management
│   │   ├── WatchlistScreen.tsx      # Stock watchlist
│   │   ├── MarketScreen.tsx         # Market overview
│   │   ├── StockDetailScreen.tsx    # Stock details
│   │   ├── TechnicalAnalysisScreen.tsx # Technical indicators
│   │   ├── MLPredictionsScreen.tsx  # ML predictions
│   │   └── SettingsScreen.tsx       # App settings
│   └── services/
│       └── api.ts                   # API service layer
├── App.tsx                          # Main app component
├── app.json                         # Expo configuration
├── package.json                     # Dependencies
└── README.md                        # This file
```

## API Endpoints

The mobile app communicates with the backend API for:

- **Authentication**: `/api/auth/login`, `/api/auth/register`
- **Market Data**: `/api/market/realtime/{ticker}`, `/api/market/overview`
- **Technical Analysis**: `/api/technical/{ticker}`
- **ML Predictions**: `/api/ml/predictions/{ticker}`
- **Portfolio**: `/api/portfolio`, `/api/portfolio/add`
- **Watchlist**: `/api/watchlist`, `/api/watchlist/add`

## Development

### Adding New Screens
1. Create a new screen component in `src/screens/`
2. Add navigation route in `App.tsx`
3. Update the navigation structure as needed

### Adding New Features
1. Create new API endpoints in `src/services/api.ts`
2. Add new context providers if needed
3. Update the UI components accordingly

### Styling
- Use the existing color scheme: `#667eea` (primary), `#764ba2` (secondary)
- Follow the established component patterns
- Use consistent spacing and typography

## Building for Production

### iOS
```bash
expo build:ios
```

### Android
```bash
expo build:android
```

### Web
```bash
expo build:web
```

## Testing

### Unit Tests
```bash
npm test
```

### E2E Tests
```bash
npm run test:e2e
```

## Troubleshooting

### Common Issues

1. **Metro bundler issues**
   ```bash
   expo start --clear
   ```

2. **iOS build errors**
   - Clean Xcode build folder
   - Reset iOS simulator

3. **Android build errors**
   - Clean Android build folder
   - Reset Android emulator

4. **API connection issues**
   - Check backend server is running
   - Verify API URL configuration
   - Check network connectivity

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

## Roadmap

- [ ] Push notifications for price alerts
- [ ] Advanced charting with TradingView integration
- [ ] Social trading features
- [ ] News sentiment analysis
- [ ] Cryptocurrency support
- [ ] Dark mode theme
- [ ] Offline mode support
- [ ] Multi-language support

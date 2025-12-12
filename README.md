# 📊 Financial Analyzer Pro

A comprehensive web-based financial analysis application built with Streamlit and FastAPI, providing advanced financial insights, market analysis, and peer comparisons.

## ✨ Features

### 🎯 Core Functionality
- **Stock Analysis**: Comprehensive financial data analysis for any stock ticker
- **Financial Metrics**: Revenue, Net Income, EBITDA, Free Cash Flow, and more
- **Growth Analysis**: Year-over-year growth rates and trend analysis
- **Peer Comparison**: Industry peer analysis with key metrics
- **Market Overview**: Real-time market indices and trending stocks
- **Industry Analysis**: Sector-wide performance metrics

### 📈 Data Visualization
- Interactive charts using Plotly
- Financial trend analysis
- Growth rate comparisons
- Peer benchmarking charts
- Industry scatter plots

### 🏗️ Architecture
- **Frontend**: Streamlit web application
- **Backend**: FastAPI REST API
- **Data**: Realistic financial data generation
- **Styling**: Modern, responsive UI with custom CSS

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd financial_analyzer_web_latest
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI backend**
   ```bash
   python proxy.py
   ```
   The API will be available at `http://localhost:8000`

4. **Start the Streamlit frontend**
   ```bash
   streamlit run app.py
   ```
   The web app will open at `http://localhost:8501`

## 📁 Project Structure

```
financial_analyzer_web_latest/
├── app.py              # Streamlit frontend application
├── proxy.py            # FastAPI backend server
├── requirements.txt    # Python dependencies
├── render.yaml         # Deployment configuration
└── README.md          # Project documentation
```

## 🔧 API Endpoints

### Financial Data
- `GET /api/financials/{ticker}` - Get financial data for a specific ticker
- `GET /api/peers/{ticker}` - Get peer comparison data
- `GET /api/market` - Get market overview data
- `GET /api/industries` - Get industry analysis data

### Utility
- `GET /` - API information and available endpoints
- `GET /health` - Health check endpoint

## 💡 Usage Examples

### Analyzing a Stock
1. Enter a stock ticker (e.g., AAPL, TSLA, MSFT)
2. Select analysis type
3. View comprehensive financial metrics
4. Explore interactive charts and trends

### Market Overview
1. Click "📊 Market Overview" in the sidebar
2. View major market indices
3. Check trending stocks
4. Monitor market sentiment

### Industry Analysis
1. Click "🏭 Industry Analysis" in the sidebar
2. Compare industry P/E ratios
3. Analyze growth patterns
4. Identify sector trends

## 🎨 Customization

### Chart Themes
- Choose from multiple Plotly themes
- Customize colors and styling
- Responsive design for all devices

### Data Sources
- Currently uses generated data for demonstration
- Easy to integrate with real financial APIs
- Extensible architecture for additional data sources

## 🔒 Security Features

- CORS middleware for cross-origin requests
- Input validation and sanitization
- Error handling and logging
- Rate limiting ready

## 🚀 Deployment

### Local Development
```bash
# Terminal 1 - Backend
python proxy.py

# Terminal 2 - Frontend
streamlit run app.py
```

### Production Deployment
The application includes a `render.yaml` file for easy deployment on Render.com:

1. Connect your GitHub repository
2. Render will automatically detect the configuration
3. Deploy both frontend and backend services

### Environment Variables
- `PORT`: Backend server port (default: 8000)
- `HOST`: Backend server host (default: 0.0.0.0)

## 🧪 Testing

### API Testing
```bash
# Test the API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/financials/AAPL
```

### Frontend Testing
- Open the Streamlit app in your browser
- Test different stock tickers
- Verify chart interactions
- Check responsive design

## 🔮 Future Enhancements

- [ ] Real-time market data integration
- [ ] User authentication and portfolios
- [ ] Advanced technical analysis
- [ ] Machine learning predictions
- [ ] Mobile app version
- [ ] API rate limiting
- [ ] Database integration
- [ ] Export functionality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This application is for educational and demonstration purposes only. The financial data is generated and should not be used for actual investment decisions. Always consult with qualified financial professionals before making investment decisions.

## 🆘 Support

If you encounter any issues:

1. Check the console logs for error messages
2. Verify both backend and frontend are running
3. Ensure all dependencies are installed
4. Check the API endpoints are accessible

## 📊 Screenshots

*Screenshots will be added here showing the application interface*

---

**Built with ❤️ using Streamlit and FastAPI**

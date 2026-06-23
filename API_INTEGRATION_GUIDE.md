# 🔗 Financial Analyzer Pro - AI Integration Guide

## 📋 **Overview**

This guide shows you how to integrate your Financial Analyzer Pro with any AI program using REST API endpoints. Both programs remain completely separate and independent.

## 🚀 **Quick Start**

### **1. Start Financial Analyzer Pro**
```bash
# Terminal 1 - Start the API server
python proxy.py
# API will be available at http://localhost:8000
```

### **2. Use the Client Library in Your AI Program**
```python
from financial_analyzer_client import FinancialAnalyzerClient

# Initialize client
client = FinancialAnalyzerClient(base_url="http://localhost:8000")

# Get market data
market_data = client.get_market_data("AAPL", period="1y")
print(f"Got data for {market_data.ticker}")
```

## 📊 **Available API Endpoints**

### **Market Data Endpoints**

#### `GET /api/ai/market-data/{ticker}`
Get comprehensive market data for AI analysis.

**Parameters:**
- `ticker` (path): Stock symbol (e.g., AAPL, MSFT)
- `period` (query): Time period (1mo, 3mo, 6mo, 1y, 2y)
- `include_indicators` (query): Include technical indicators (true/false)
- `include_risk` (query): Include risk metrics (true/false)

**Response:**
```json
{
  "ticker": "AAPL",
  "period": "1y",
  "timestamp": "2024-01-15T10:30:00",
  "data_points": 252,
  "price_data": {
    "dates": ["2023-01-15", "2023-01-16", ...],
    "open": [150.0, 151.0, ...],
    "high": [152.0, 153.0, ...],
    "low": [149.0, 150.0, ...],
    "close": [151.0, 152.0, ...],
    "volume": [1000000, 1200000, ...]
  },
  "technical_indicators": {
    "sma_20": [150.5, 151.0, ...],
    "sma_50": [149.0, 149.5, ...],
    "rsi": [65.2, 67.1, ...],
    "macd": [0.5, 0.8, ...],
    "bb_upper": [155.0, 156.0, ...],
    "bb_lower": [145.0, 146.0, ...]
  },
  "risk_metrics": {
    "Volatility (Annualized)": "25.4%",
    "Sharpe Ratio": "1.2",
    "Max Drawdown": "-12.5%",
    "VaR (95%)": "-2.1%"
  }
}
```

#### `GET /api/ai/market-overview`
Get market overview data.

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "indices": {
    "^GSPC": {
      "price": 4500.0,
      "change": 25.5,
      "change_percent": 0.57
    },
    "^IXIC": {
      "price": 14000.0,
      "change": -50.0,
      "change_percent": -0.36
    }
  }
}
```

#### `GET /api/ai/global-markets`
Get global markets data.

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "markets": [
    {
      "name": "S&P 500",
      "symbol": "^GSPC",
      "price": 4500.0,
      "change": 25.5,
      "change_percent": 0.57
    }
  ]
}
```

### **Technical Analysis Endpoints**

#### `GET /api/ai/technical-analysis/{ticker}`
Get technical analysis data.

**Parameters:**
- `ticker` (path): Stock symbol
- `period` (query): Time period
- `indicators` (query): Comma-separated indicators (sma,ema,rsi,macd,bb,all)

**Response:**
```json
{
  "ticker": "AAPL",
  "period": "1y",
  "timestamp": "2024-01-15T10:30:00",
  "indicators": {
    "sma": {
      "sma_20": [150.5, 151.0, ...],
      "sma_50": [149.0, 149.5, ...]
    },
    "rsi": [65.2, 67.1, ...],
    "macd": {
      "macd": [0.5, 0.8, ...],
      "signal": [0.3, 0.6, ...]
    }
  }
}
```

### **Risk Analysis Endpoints**

#### `GET /api/ai/risk-analysis/{ticker}`
Get risk analysis data.

**Response:**
```json
{
  "ticker": "AAPL",
  "period": "1y",
  "timestamp": "2024-01-15T10:30:00",
  "risk_metrics": {
    "Volatility (Annualized)": "25.4%",
    "Sharpe Ratio": "1.2",
    "Max Drawdown": "-12.5%",
    "VaR (95%)": "-2.1%",
    "VaR (99%)": "-3.2%"
  },
  "additional_metrics": {
    "daily_returns": [0.01, -0.02, ...],
    "volatility_daily": 0.016,
    "skewness": -0.2,
    "kurtosis": 2.1,
    "sharpe_ratio_annual": 1.2
  }
}
```

### **ML Predictions Endpoints**

#### `GET /api/ai/predictions/{ticker}`
Get ML predictions.

**Parameters:**
- `ticker` (path): Stock symbol
- `prediction_days` (query): Days to predict (default: 5)
- `model_type` (query): Model type (linear, forest, boosting, ensemble)

**Response:**
```json
{
  "ticker": "AAPL",
  "prediction_days": 5,
  "model_type": "ensemble",
  "timestamp": "2024-01-15T10:30:00",
  "predictions": {
    "price_forecast": [155.0, 156.5, 158.0, 157.5, 159.0],
    "confidence_scores": [0.85, 0.82, 0.78, 0.75, 0.70],
    "model_accuracy": 0.75,
    "risk_assessment": "medium"
  },
  "model_metadata": {
    "training_data_points": 1000,
    "last_training_date": "2024-01-14",
    "model_version": "1.0"
  }
}
```

### **Batch Processing Endpoints**

#### `POST /api/ai/batch-market-data`
Get market data for multiple tickers.

**Request Body:**
```json
["AAPL", "MSFT", "GOOGL", "TSLA"]
```

**Query Parameters:**
- `period`: Time period
- `include_indicators`: Include technical indicators

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "period": "1y",
  "tickers": {
    "AAPL": {
      "price_data": {
        "dates": ["2023-01-15", ...],
        "close": [151.0, ...],
        "volume": [1000000, ...]
      },
      "technical_indicators": {
        "rsi": [65.2, ...],
        "sma_20": [150.5, ...]
      }
    }
  },
  "summary": {
    "total_tickers": 4,
    "successful_requests": 4,
    "failed_requests": 0
  }
}
```

## 🔧 **Client Library Usage**

### **Basic Usage**

```python
from financial_analyzer_client import FinancialAnalyzerClient

# Initialize client
client = FinancialAnalyzerClient(base_url="http://localhost:8000")

# Get market data
market_data = client.get_market_data("AAPL", period="1y")

# Convert to DataFrame for analysis
df = create_dataframe_from_financial_data(market_data)

# Calculate returns
returns = calculate_returns_from_dataframe(df)

# Get technical signals
signals = get_technical_signals(df)
```

### **Advanced Usage**

```python
# Get multiple tickers at once
batch_data = client.get_batch_market_data(["AAPL", "MSFT", "GOOGL"])

# Get technical analysis
tech_analysis = client.get_technical_analysis("AAPL", indicators="rsi,macd")

# Get risk analysis
risk_data = client.get_risk_analysis("AAPL")

# Get predictions
predictions = client.get_predictions("AAPL", prediction_days=10)

# Analyze market sentiment
overview = client.get_market_overview()
sentiment = analyze_market_sentiment(overview)
```

## 📊 **Data Schema Reference**

### **Price Data Schema**
```json
{
  "dates": ["YYYY-MM-DD", ...],
  "open": [float, ...],
  "high": [float, ...],
  "low": [float, ...],
  "close": [float, ...],
  "volume": [integer, ...]
}
```

### **Technical Indicators Schema**
```json
{
  "sma_20": [float, ...],
  "sma_50": [float, ...],
  "ema_12": [float, ...],
  "ema_26": [float, ...],
  "rsi": [float, ...],
  "macd": [float, ...],
  "macd_signal": [float, ...],
  "bb_upper": [float, ...],
  "bb_middle": [float, ...],
  "bb_lower": [float, ...]
}
```

### **Risk Metrics Schema**
```json
{
  "Volatility (Annualized)": "XX.XX%",
  "Sharpe Ratio": "X.XX",
  "Max Drawdown": "-XX.XX%",
  "VaR (95%)": "-X.XX%",
  "VaR (99%)": "-X.XX%"
}
```

## ⚠️ **Error Handling**

### **Common Error Responses**

#### **404 Not Found**
```json
{
  "detail": "No data available for INVALID_TICKER"
}
```

#### **500 Internal Server Error**
```json
{
  "detail": "Error processing market data: [error message]"
}
```

#### **429 Too Many Requests**
```json
{
  "detail": "Rate limit exceeded",
  "retry_after": 3600
}
```

### **Error Handling in Client**

```python
try:
    market_data = client.get_market_data("AAPL")
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        print("Ticker not found")
    elif e.response.status_code == 429:
        print("Rate limit exceeded")
    else:
        print(f"API error: {e}")
except requests.exceptions.RequestException as e:
    print(f"Network error: {e}")
```

## 🔐 **Authentication (Optional)**

If you implement authentication in your Financial Analyzer Pro:

```python
# Initialize client with API key
client = FinancialAnalyzerClient(
    base_url="http://localhost:8000",
    api_key="your-api-key-here"
)
```

## 📈 **Performance Tips**

### **1. Use Batch Endpoints**
```python
# Instead of multiple individual requests
tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
batch_data = client.get_batch_market_data(tickers)
```

### **2. Cache Data**
```python
import time
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_market_data(ticker, period):
    return client.get_market_data(ticker, period)
```

### **3. Use Appropriate Time Periods**
- Use shorter periods (1mo, 3mo) for recent analysis
- Use longer periods (1y, 2y) for historical patterns

## 🚀 **Integration Examples**

### **Example 1: Simple Data Retrieval**
```python
from financial_analyzer_client import FinancialAnalyzerClient

client = FinancialAnalyzerClient()

# Get AAPL data
data = client.get_market_data("AAPL")
print(f"Got {len(data.price_data['close'])} data points for {data.ticker}")
```

### **Example 2: Technical Analysis**
```python
# Get technical analysis
tech_data = client.get_technical_analysis("AAPL", indicators="rsi,macd")

# Extract RSI values
rsi_values = tech_data.indicators['rsi']
print(f"Current RSI: {rsi_values[-1]:.2f}")
```

### **Example 3: Risk Assessment**
```python
# Get risk analysis
risk_data = client.get_risk_analysis("AAPL")

# Extract volatility
volatility = risk_data['risk_metrics']['Volatility (Annualized)']
print(f"Annualized volatility: {volatility}")
```

### **Example 4: Market Sentiment**
```python
# Get market overview
overview = client.get_market_overview()

# Analyze sentiment
sentiment = analyze_market_sentiment(overview)
print(f"Market sentiment: {sentiment['overall_sentiment']}")
```

## 🔧 **Setup Instructions**

### **1. Add API Endpoints to Financial Analyzer Pro**

Add this to your `proxy.py` file:

```python
# Add at the top
from ai_integration_api import ai_router

# Add to your FastAPI app
app.include_router(ai_router)
```

### **2. Install Client Library in AI Program**

```bash
# Copy the client library to your AI program directory
cp financial_analyzer_client.py /path/to/your/ai/program/

# Install required dependencies
pip install requests pandas numpy
```

### **3. Test Integration**

```python
# Test script
from financial_analyzer_client import FinancialAnalyzerClient

client = FinancialAnalyzerClient()
status = client.get_status()
print(f"API Status: {status['status']}")
```

## 📞 **Support**

If you encounter any issues:

1. Check that Financial Analyzer Pro is running on `http://localhost:8000`
2. Verify the API endpoints are accessible
3. Check the logs for error messages
4. Ensure all required dependencies are installed

## 🎯 **Next Steps**

1. **Start Financial Analyzer Pro**: `python proxy.py`
2. **Copy client library** to your AI program
3. **Test basic connection**: Use the example code above
4. **Integrate data** into your AI analysis pipeline
5. **Scale up**: Use batch endpoints for multiple tickers

This integration allows you to leverage all the financial analysis capabilities of your Financial Analyzer Pro in any AI program without modifying either system!












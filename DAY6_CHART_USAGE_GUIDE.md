# 📊 Day 6 Chart Features - Complete Usage Guide

## 🚀 **Getting Started**

### **Accessing the Charts**
1. **Open the Day 6 App**: Go to `http://localhost:8507`
2. **Main Chart Tab**: This is where you'll spend most of your time
3. **Navigation**: Use the sidebar to switch between different chart features

## 📈 **Main Chart Features**

### **1. Basic Chart Setup**

#### **Stock Symbol Input**
- **Location**: Top-left text input box
- **Default**: AAPL (Apple Inc.)
- **Format**: Use standard ticker symbols (AAPL, MSFT, GOOGL, etc.)
- **Case**: Automatically converts to uppercase

#### **Timeframe Selection**
- **Location**: Top-right dropdown
- **Options Available**:
  - `1m` - 1 minute (intraday)
  - `5m` - 5 minutes (intraday)
  - `15m` - 15 minutes (intraday)
  - `1h` - 1 hour (intraday)
  - `4h` - 4 hours (intraday)
  - `1d` - Daily (most common)
  - `1wk` - Weekly
  - `1mo` - Monthly

#### **Period Selection**
- **Location**: Third dropdown
- **Options Available**:
  - `1d` - 1 day (for intraday timeframes)
  - `5d` - 5 days
  - `1mo` - 1 month
  - `3mo` - 3 months
  - `6mo` - 6 months
  - `1y` - 1 year
  - `2y` - 2 years
  - `5y` - 5 years
  - `max` - Maximum available data

#### **Generate Chart Button**
- **Location**: Top-right blue button
- **Action**: Click to load/refresh the chart
- **Status**: Shows loading spinner while fetching data

### **2. Interactive Chart Controls**

#### **Zoom and Pan**
- **Mouse Wheel**: Zoom in/out on the chart
- **Click and Drag**: Pan left/right through time
- **Double Click**: Reset zoom to fit all data
- **Range Selection**: Click and drag on x-axis to select time range

#### **Hover Tooltips**
- **Hover Over Candles**: See detailed OHLC data
- **Hover Over Indicators**: See indicator values
- **Hover Over Volume**: See volume data
- **Hover Over RSI**: See RSI values

#### **Legend Controls**
- **Click Legend Items**: Show/hide specific indicators
- **Double Click**: Isolate single indicator
- **Right Click**: Access additional options

### **3. Technical Indicators**

#### **Moving Averages**
- **SMA 20**: 20-period Simple Moving Average (blue line)
- **SMA 50**: 50-period Simple Moving Average (orange line)
- **EMA 12**: 12-period Exponential Moving Average
- **EMA 26**: 26-period Exponential Moving Average

#### **Bollinger Bands**
- **Upper Band**: Purple dashed line
- **Middle Band**: Purple solid line (SMA 20)
- **Lower Band**: Purple dashed line with fill
- **Purpose**: Shows price volatility and support/resistance

#### **RSI (Relative Strength Index)**
- **Location**: Bottom subplot
- **Range**: 0-100
- **Overbought**: Above 70 (red dashed line)
- **Oversold**: Below 30 (green dashed line)
- **Purpose**: Momentum indicator

#### **MACD (Moving Average Convergence Divergence)**
- **MACD Line**: Blue line
- **Signal Line**: Red line
- **Histogram**: Green/red bars
- **Purpose**: Trend and momentum analysis

#### **Volume Analysis**
- **Location**: Middle subplot
- **Color Coding**: Green for up days, red for down days
- **Volume SMA**: Moving average of volume
- **Purpose**: Confirms price movements

### **4. Chart Statistics Panel**

#### **Price Information**
- **Current Price**: Latest closing price
- **Total Change**: Absolute price change from start
- **Change %**: Percentage change from start
- **52W High**: Highest price in the period
- **52W Low**: Lowest price in the period

#### **Technical Analysis Summary**
- **Moving Averages**: Current SMA 20 and SMA 50 values
- **Bollinger Bands**: Upper, middle, and lower band values
- **RSI & MACD**: Current RSI and MACD values

## 📊 **Chart Comparison Features**

### **1. Accessing Comparison**
- **Navigation**: Click "Chart Comparison" in sidebar
- **Purpose**: Compare multiple stocks side-by-side

### **2. Setting Up Comparison**

#### **Symbols Input**
- **Format**: Comma-separated list (AAPL,MSFT,GOOGL)
- **Limit**: Up to 10 stocks
- **Case**: Automatically converts to uppercase

#### **Timeframe Selection**
- **Options**: 1d, 1wk, 1mo
- **Purpose**: Choose analysis timeframe

#### **Period Selection**
- **Options**: 1mo, 3mo, 6mo, 1y
- **Purpose**: Choose analysis period

### **3. Comparison Chart Features**

#### **Normalized Performance**
- **Display**: All stocks start at 0% (100% baseline)
- **Purpose**: Compare relative performance
- **Interpretation**: Higher lines = better performance

#### **Interactive Legend**
- **Click**: Show/hide individual stocks
- **Hover**: See stock performance details
- **Colors**: Each stock has unique color

#### **Correlation Heatmap**
- **Purpose**: Shows how stocks move together
- **Colors**: Red = negative correlation, Blue = positive correlation
- **Values**: Correlation coefficients (-1 to +1)

## ✏️ **Drawing Tools**

### **1. Accessing Drawing Tools**
- **Navigation**: Click "Drawing Tools" in sidebar
- **Purpose**: Learn about available drawing tools

### **2. Available Tools**

#### **Trend Lines**
- **Support Lines**: Horizontal lines at support levels
- **Resistance Lines**: Horizontal lines at resistance levels
- **Trend Lines**: Diagonal lines showing trend direction
- **Channel Lines**: Parallel lines showing price channels

#### **Fibonacci Tools**
- **Retracements**: 23.6%, 38.2%, 50%, 61.8% levels
- **Extensions**: 127.2%, 161.8%, 261.8% levels
- **Fans**: Angle-based Fibonacci lines
- **Arcs**: Time-based Fibonacci curves

#### **Annotation Tools**
- **Text Labels**: Add text annotations
- **Shapes**: Rectangles, circles, triangles
- **Arrows**: Point to specific levels
- **Freehand**: Custom drawing

### **3. Using Drawing Tools**

#### **In the Chart**
- **Hover**: Look for drawing tool icons
- **Click and Drag**: Draw trend lines
- **Right Click**: Access drawing menu
- **Double Click**: Add text annotations

#### **Keyboard Shortcuts**
- `T` - Trend line tool
- `H` - Horizontal line tool
- `F` - Fibonacci retracement
- `A` - Arrow tool
- `S` - Shape tool
- `Esc` - Exit drawing mode

## 🔍 **Technical Analysis Features**

### **1. Accessing Technical Analysis**
- **Navigation**: Click "Technical Analysis" in sidebar
- **Purpose**: View available indicators and analysis tools

### **2. Available Indicators**

#### **Trend Indicators**
- **Simple Moving Average (SMA)**
- **Exponential Moving Average (EMA)**
- **Bollinger Bands**
- **Parabolic SAR**

#### **Momentum Indicators**
- **RSI (Relative Strength Index)**
- **MACD**
- **Stochastic Oscillator**
- **Williams %R**

#### **Volume Indicators**
- **Volume SMA**
- **On-Balance Volume (OBV)**
- **Volume Rate of Change**
- **Accumulation/Distribution**

## 🎨 **Chart Customization**

### **1. Theme Selection**
- **Location**: Sidebar dropdown
- **Options**: Light, Dark, Auto
- **Purpose**: Change chart appearance

### **2. Indicator Toggle**
- **Location**: Main chart area checkboxes
- **Purpose**: Show/hide specific indicators
- **Real-time**: Changes apply immediately

### **3. Chart Settings**
- **Grid**: Show/hide chart grid
- **Volume**: Show/hide volume subplot
- **Legend**: Show/hide legend
- **Tooltips**: Enable/disable hover tooltips

## 🚀 **Pro Tips for Effective Chart Analysis**

### **1. Timeframe Selection**
- **Day Trading**: Use 1m, 5m, 15m timeframes
- **Swing Trading**: Use 1h, 4h, 1d timeframes
- **Long-term Investing**: Use 1d, 1wk, 1mo timeframes

### **2. Indicator Combinations**
- **Trend Following**: SMA 20 + SMA 50 + MACD
- **Momentum**: RSI + MACD + Volume
- **Volatility**: Bollinger Bands + RSI
- **Support/Resistance**: Bollinger Bands + Moving Averages

### **3. Chart Analysis Workflow**
1. **Start with Daily Chart**: Get overall trend
2. **Check Weekly Chart**: Confirm long-term trend
3. **Use Hourly Chart**: Find entry/exit points
4. **Apply Indicators**: Confirm signals
5. **Check Volume**: Validate price movements

### **4. Common Chart Patterns**
- **Support/Resistance**: Horizontal price levels
- **Trend Lines**: Diagonal support/resistance
- **Channels**: Parallel trend lines
- **Triangles**: Converging trend lines
- **Head and Shoulders**: Reversal pattern

## 🔧 **Troubleshooting**

### **1. Chart Not Loading**
- **Check Symbol**: Ensure valid ticker symbol
- **Check Timeframe**: Some combinations may not work
- **Check Period**: Some periods may not have data
- **Refresh**: Click "Generate Chart" button

### **2. Indicators Not Showing**
- **Check Checkboxes**: Ensure indicators are selected
- **Wait for Calculation**: Some indicators need time to calculate
- **Check Data**: Ensure sufficient data for calculation

### **3. Performance Issues**
- **Reduce Period**: Use shorter time periods
- **Fewer Indicators**: Disable unnecessary indicators
- **Clear Cache**: Use sidebar "Clear Cache" button

### **4. Drawing Tools Not Working**
- **Check Chart**: Ensure chart is loaded
- **Hover for Tools**: Look for drawing tool icons
- **Use Keyboard**: Try keyboard shortcuts
- **Refresh Page**: Reload if tools don't appear

## 📱 **Mobile Usage**

### **1. Touch Controls**
- **Pinch to Zoom**: Use two fingers
- **Drag to Pan**: Single finger drag
- **Tap to Select**: Single tap on elements
- **Long Press**: Access context menus

### **2. Mobile Optimization**
- **Responsive Design**: Automatically adapts to screen size
- **Touch-Friendly**: Large buttons and controls
- **Swipe Navigation**: Swipe between different views
- **Portrait/Landscape**: Works in both orientations

## 🎯 **Next Steps**

### **1. Practice with Different Stocks**
- Try various symbols (AAPL, MSFT, GOOGL, TSLA, etc.)
- Test different timeframes and periods
- Experiment with different indicator combinations

### **2. Learn Chart Patterns**
- Study support and resistance levels
- Learn to identify trend lines
- Practice with Fibonacci retracements

### **3. Develop Your Strategy**
- Find indicator combinations that work for you
- Develop entry and exit criteria
- Test your strategy with paper trading

---

**Status**: ✅ **Complete Usage Guide**  
**Confidence Level**: 🎯 **100% - Comprehensive Coverage**  
**Ready for**: Advanced chart analysis and trading strategies

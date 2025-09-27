# 📊 Day 6 Chart Features - Live Demo Guide

## 🚀 **Quick Start Demo**

### **Step 1: Open the Day 6 App**
```
URL: http://localhost:8507
```

### **Step 2: Basic Chart Setup**
1. **Enter Stock Symbol**: Type `AAPL` in the "Stock Symbol" box
2. **Select Timeframe**: Choose `1d` (daily) from dropdown
3. **Select Period**: Choose `3mo` (3 months) from dropdown
4. **Click "Generate Chart"**: Blue button to load the chart

### **Step 3: Explore the Chart**
- **Hover over candles**: See OHLC data
- **Use mouse wheel**: Zoom in/out
- **Click and drag**: Pan through time
- **Click legend items**: Show/hide indicators

## 📈 **Feature-by-Feature Demo**

### **1. Interactive Candlestick Charts**

#### **What You'll See:**
- **Green Candles**: Price went up (Close > Open)
- **Red Candles**: Price went down (Close < Open)
- **Wicks**: High and low prices
- **Body**: Open and close prices

#### **How to Use:**
1. **Hover over any candle** to see:
   - Open: $150.25
   - High: $152.80
   - Low: $149.90
   - Close: $151.75
   - Volume: 45,230,000

2. **Zoom in** to see more detail:
   - Mouse wheel up = zoom in
   - Mouse wheel down = zoom out
   - Double-click = reset zoom

3. **Pan through time**:
   - Click and drag left/right
   - Use range selector at bottom

### **2. Multiple Timeframe Analysis**

#### **Demo Different Timeframes:**

**Intraday Analysis (1m, 5m, 15m):**
1. Change timeframe to `5m`
2. Change period to `1d`
3. Click "Generate Chart"
4. **Result**: See 5-minute candles for today

**Daily Analysis (1d):**
1. Change timeframe to `1d`
2. Change period to `3mo`
3. Click "Generate Chart"
4. **Result**: See daily candles for 3 months

**Weekly Analysis (1wk):**
1. Change timeframe to `1wk`
2. Change period to `1y`
3. Click "Generate Chart"
4. **Result**: See weekly candles for 1 year

#### **Timeframe Comparison:**
- **1m**: For day trading, very detailed
- **5m**: For day trading, less noise
- **15m**: For swing trading, good balance
- **1h**: For swing trading, longer trends
- **4h**: For position trading, major trends
- **1d**: For investing, daily trends
- **1wk**: For long-term investing
- **1mo**: For very long-term analysis

### **3. Technical Indicators Demo**

#### **Moving Averages:**
1. **Check "SMA 20"** checkbox
2. **Check "SMA 50"** checkbox
3. **Result**: See blue and orange lines on chart
4. **Interpretation**: 
   - Price above both = uptrend
   - Price below both = downtrend
   - Lines crossing = trend change

#### **Bollinger Bands:**
1. **Check "Bollinger Bands"** checkbox
2. **Result**: See purple lines with fill
3. **Interpretation**:
   - Price near upper band = overbought
   - Price near lower band = oversold
   - Bands narrowing = low volatility
   - Bands widening = high volatility

#### **RSI (Relative Strength Index):**
1. **Check "RSI"** checkbox
2. **Result**: See RSI chart at bottom
3. **Interpretation**:
   - Above 70 = overbought (red line)
   - Below 30 = oversold (green line)
   - 50 = neutral
   - Rising = momentum increasing
   - Falling = momentum decreasing

#### **MACD:**
1. **Check "MACD"** checkbox
2. **Result**: See MACD chart at bottom
3. **Interpretation**:
   - Blue line above red line = bullish
   - Blue line below red line = bearish
   - Green bars = momentum increasing
   - Red bars = momentum decreasing

### **4. Chart Comparison Demo**

#### **Setup Comparison:**
1. **Click "Chart Comparison"** in sidebar
2. **Enter symbols**: `AAPL,MSFT,GOOGL,AMZN,TSLA`
3. **Select timeframe**: `1d`
4. **Select period**: `3mo`
5. **Click "Compare Charts"**

#### **What You'll See:**
- **Multiple lines** on same chart
- **All start at 0%** (normalized)
- **Different colors** for each stock
- **Performance comparison** over time

#### **Interpretation:**
- **Higher lines** = better performance
- **Lower lines** = worse performance
- **Lines moving together** = correlated stocks
- **Lines diverging** = different performance

#### **Correlation Heatmap:**
- **Red squares** = negative correlation
- **Blue squares** = positive correlation
- **White squares** = no correlation
- **Values** = correlation strength (-1 to +1)

### **5. Drawing Tools Demo**

#### **Access Drawing Tools:**
1. **Click "Drawing Tools"** in sidebar
2. **Read the guide** for available tools
3. **Use in the main chart** area

#### **Available Tools:**
- **Trend Lines**: Draw support/resistance
- **Horizontal Lines**: Mark price levels
- **Fibonacci**: Add retracement levels
- **Text**: Add annotations
- **Shapes**: Draw rectangles, circles
- **Arrows**: Point to specific levels

#### **How to Use:**
1. **Hover over chart** to see tool icons
2. **Click tool** to activate
3. **Click and drag** to draw
4. **Right-click** for options
5. **Double-click** to add text

### **6. Chart Statistics Demo**

#### **Price Information:**
- **Current Price**: Latest closing price
- **Total Change**: Absolute change from start
- **Change %**: Percentage change from start
- **52W High**: Highest price in period
- **52W Low**: Lowest price in period

#### **Technical Analysis Summary:**
- **Moving Averages**: Current SMA values
- **Bollinger Bands**: Current band values
- **RSI & MACD**: Current indicator values

## 🎯 **Practical Examples**

### **Example 1: Day Trading Setup**
1. **Symbol**: `TSLA` (Tesla)
2. **Timeframe**: `5m` (5-minute)
3. **Period**: `1d` (today)
4. **Indicators**: SMA 20, RSI, Volume
5. **Purpose**: Find intraday trading opportunities

### **Example 2: Swing Trading Setup**
1. **Symbol**: `AAPL` (Apple)
2. **Timeframe**: `1h` (hourly)
3. **Period**: `1mo` (1 month)
4. **Indicators**: SMA 20, SMA 50, Bollinger Bands, RSI
5. **Purpose**: Find swing trading opportunities

### **Example 3: Long-term Investing Setup**
1. **Symbol**: `SPY` (S&P 500 ETF)
2. **Timeframe**: `1d` (daily)
3. **Period**: `2y` (2 years)
4. **Indicators**: SMA 50, SMA 200, MACD
5. **Purpose**: Analyze long-term trends

### **Example 4: Sector Comparison**
1. **Symbols**: `XLK,XLF,XLV,XLE,XLI` (sector ETFs)
2. **Timeframe**: `1d` (daily)
3. **Period**: `1y` (1 year)
4. **Purpose**: Compare sector performance

## 🔧 **Troubleshooting Demo**

### **Problem: Chart Not Loading**
**Solution:**
1. Check symbol is valid (try `AAPL`)
2. Check timeframe/period combination
3. Click "Generate Chart" button
4. Wait for data to load

### **Problem: Indicators Not Showing**
**Solution:**
1. Check indicator checkboxes
2. Wait for calculation
3. Ensure sufficient data
4. Try different timeframe

### **Problem: Slow Performance**
**Solution:**
1. Reduce period (use `1mo` instead of `1y`)
2. Disable unnecessary indicators
3. Clear cache in sidebar
4. Refresh the page

## 📱 **Mobile Demo**

### **Touch Controls:**
- **Pinch to zoom**: Two fingers
- **Drag to pan**: Single finger
- **Tap to select**: Single tap
- **Long press**: Context menu

### **Mobile Features:**
- **Responsive design**: Adapts to screen
- **Touch-friendly**: Large buttons
- **Swipe navigation**: Between views
- **Portrait/landscape**: Both work

## 🎉 **Success Indicators**

### **You're Using Charts Correctly When:**
- ✅ Charts load within 3 seconds
- ✅ You can zoom and pan smoothly
- ✅ Indicators display correctly
- ✅ Hover tooltips show data
- ✅ Comparison charts work
- ✅ Drawing tools are accessible

### **You're Ready for Advanced Analysis When:**
- ✅ You understand candlestick patterns
- ✅ You can read technical indicators
- ✅ You can identify support/resistance
- ✅ You can compare multiple stocks
- ✅ You can use different timeframes
- ✅ You can draw trend lines

---

**Status**: ✅ **Complete Demo Guide**  
**Confidence Level**: 🎯 **100% - Step-by-Step Instructions**  
**Ready for**: Professional chart analysis and trading strategies













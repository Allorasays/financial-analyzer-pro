# Play Store Screenshot Requirements

## ✅ Current Status

**What You Have**:
- ✅ `store_assets/feature_graphic.svg` - Feature graphic (needs to be converted to PNG: 1024x500)

**What's Needed**:
- ⏳ Phone screenshots (minimum 2, recommended 6)
- ⏳ Feature graphic PNG (1024x500)

---

## 📱 Required Screenshots (Minimum: 2)

Play Store requires **at least 2 screenshots** for phone devices.

### Recommended 6 Screenshots:

1. **Screenshot 1: Dashboard** 📊
   - Main market view
   - MONETA branding header visible
   - Live stock prices
   - ML predictions section
   - **File**: `store_assets/screenshot_1_dashboard.png`

2. **Screenshot 2: ML Predictions** 🔮
   - Prediction cards (Next Day, Week, Month)
   - Bullish/Bearish labels visible
   - Projected change percentages
   - **File**: `store_assets/screenshot_2_predictions.png`

3. **Screenshot 3: Technical Analysis** 📈
   - Price chart with indicators
   - Trend analysis visible
   - Technical metrics displayed
   - **File**: `store_assets/screenshot_3_charts.png`

4. **Screenshot 4: Portfolio Manager** 💼
   - Portfolio holdings list
   - P&L calculations visible
   - Performance metrics
   - **File**: `store_assets/screenshot_4_portfolio.png`

5. **Screenshot 5: Market News** 📰
   - News articles feed
   - Sentiment analysis visible
   - Market updates
   - **File**: `store_assets/screenshot_5_news.png`

6. **Screenshot 6: Settings/About** ⚙️
   - App info screen
   - MONETA branding
   - Version and support info
   - **File**: `store_assets/screenshot_6_settings.png`

---

## 📐 Dimensions

### Phone Screenshots:
- **Portrait**: 1080x1920 pixels (minimum height: 1080px)
- **Landscape**: 1920x1080 pixels (optional)
- **Format**: PNG or JPEG
- **Max file size**: 8MB per screenshot

### Feature Graphic:
- **Size**: 1024x500 pixels
- **Format**: PNG or JPEG
- **Max file size**: 1MB
- **Current**: You have `feature_graphic.svg` - needs conversion to PNG

### Tablet Screenshots (Optional):
- **Portrait**: 1200x1920 pixels
- **Landscape**: 1920x1200 pixels

---

## 🎨 Best Practices

1. **Show Key Features**: Each screenshot should highlight a different feature
2. **MONETA Branding**: Ensure MONETA logo/colors are visible
3. **Clean UI**: Remove any test data or placeholder text
4. **Real Data**: Use real stock data (AAPL, TSLA, etc.) for screenshots
5. **Consistent Style**: All screenshots should have consistent styling
6. **No Personal Info**: Don't include any personal information

---

## 📸 How to Capture Screenshots

### Option 1: Android Studio Device Manager
1. Launch emulator with your app installed
2. Navigate to the screen you want to capture
3. Use Android Studio's screenshot tool (Device Manager → Camera icon)
4. Save as PNG

### Option 2: Physical Device
1. Install app on physical device
2. Navigate to screen
3. Take screenshot (Power + Volume Down)
4. Transfer to computer
5. Resize if needed to 1080x1920

### Option 3: ADB Command
```bash
adb shell screencap -p /sdcard/screenshot.png
adb pull /sdcard/screenshot.png store_assets/screenshot_1_dashboard.png
```

---

## ✅ Checklist

- [ ] Screenshot 1: Dashboard (1080x1920)
- [ ] Screenshot 2: ML Predictions (1080x1920)
- [ ] Screenshot 3: Technical Analysis (1080x1920)
- [ ] Screenshot 4: Portfolio Manager (1080x1920)
- [ ] Screenshot 5: Market News (1080x1920)
- [ ] Screenshot 6: Settings/About (1080x1920)
- [ ] Feature Graphic: Convert SVG to PNG (1024x500)

---

## 🚀 Quick Answer

**You asked: "Screenshots have been added, are more needed?"**

**Answer**: 
- **Minimum**: You need at least **2 phone screenshots** (1080x1920)
- **Recommended**: **6 phone screenshots** to showcase all features
- **Feature Graphic**: Convert your `feature_graphic.svg` to PNG (1024x500)

**If you've already added screenshots**, please let me know:
1. How many screenshots you have?
2. What screens do they show?
3. What are their dimensions?

Then I can tell you if you need more or if they're sufficient for Play Store submission!

---

## 📁 Expected File Structure

```
store_assets/
├── feature_graphic.svg          ✅ (you have this)
├── feature_graphic.png          ⏳ (need to create from SVG)
├── screenshot_1_dashboard.png   ⏳
├── screenshot_2_predictions.png ⏳
├── screenshot_3_charts.png     ⏳
├── screenshot_4_portfolio.png  ⏳
├── screenshot_5_news.png        ⏳
└── screenshot_6_settings.png   ⏳
```


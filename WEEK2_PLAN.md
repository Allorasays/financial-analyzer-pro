# 📅 Week 2: Polish & Launch Preparation

**Target**: Production-ready app with all materials for Play Store submission

---

## ✅ Week 2 Tasks

### **1. Generate Play Store Screenshots** (1 hour)

#### Capturing Screenshots
1. **Launch Android Emulator or Device**
   ```bash
   # Option A: Physical device
   adb install app-debug.apk
   # Option B: Emulator
   emulator -avd Pixel_5_API_30
   ```

2. **Capture 6 Screenshots** (required dimensions: at least 1080px height):
   - **Screenshot 1**: Dashboard (main market view)
     - Show "MONETA" branding header
     - Display live stock prices
     - Show ML predictions section
   
   - **Screenshot 2**: ML Predictions View
     - Show prediction cards (Next Day, Week, Month)
     - Display confidence values
     - Show "Bullish/Bearish" labels
   
   - **Screenshot 3**: Technical Analysis Chart
     - Show price chart with indicators
     - Display trend analysis
     - Show technical metrics
   
   - **Screenshot 4**: Portfolio Manager
     - Show portfolio holdings
     - Display P&L calculations
     - Show performance metrics
   
   - **Screenshot 5**: Market News Feed
     - Show news articles
     - Display sentiment analysis
     - Show market updates
   
   - **Screenshot 6**: Settings/About
     - Show app info
     - Display MONETA branding
     - Show version and support info

3. **Export in Correct Dimensions**
   - Minimum: 1080x1920 (portrait)
   - Alternative: 1920x1080 (landscape)
   - Use: `adb shell screencap -p /sdcard/screen.png`

#### Tools for Enhancement
- **Canva**: Phone mockup templates
- **Adobe Photoshop**: Professional framing
- **Remove.bg**: Clean backgrounds
- **Lightroom Mobile**: Color correction

#### Files to Create
```
store_assets/
  ├── screenshot_1_dashboard.png
  ├── screenshot_2_predictions.png
  ├── screenshot_3_charts.png
  ├── screenshot_4_portfolio.png
  ├── screenshot_5_news.png
  └── screenshot_6_settings.png
```

---

### **2. Build Signed Release APK/AAB** (2 hours)

#### Create Keystore
```bash
cd FinancialAnalyzerApp

# Generate release keystore
keytool -genkeypair \
  -v \
  -keystore moneta-release.jks \
  -alias moneta \
  -keyalg RSA \
  -keysize 2048 \
  -validity 10000

# Answer prompts:
# - Enter keystore password: [secure password]
# - Re-enter password: [same password]
# - First and last name: Your Organization
# - Organizational unit: MONETA Team
# - Organization: Financial Analyzer Pro
# - City: Your City
# - State: Your State
# - Country code: US
```

#### Configure Signing in build.gradle
```gradle
// app/build.gradle (add to android block)
signingConfigs {
    release {
        storeFile file("moneta-release.jks")
        storePassword System.getenv("KEYSTORE_PASSWORD")
        keyAlias "moneta"
        keyPassword System.getenv("KEY_PASSWORD")
    }
}

buildTypes {
    release {
        signingConfig signingConfigs.release
        minifyEnabled true
        shrinkResources true
    }
}
```

#### Build Release Bundle
```bash
# Build AAB (App Bundle) for Play Store
./gradlew bundleRelease

# Output: app/build/outputs/bundle/release/app-release.aab

# Build APK for direct testing
./gradlew assembleRelease

# Output: app/build/outputs/apk/release/app-release.apk
```

#### Test Signed Release
```bash
# Install on device for testing
adb install app-release.apk

# Verify with:
adb shell dumpsys package com.financialanalyzer.mobile | grep signatures
```

---

### **3. End-to-End Testing** (1 hour)

#### Production Backend Testing
```bash
# Test backend health
curl https://moneta-backend-api.onrender.com/health

# Test API status
curl https://moneta-backend-api.onrender.com/api/system/status

# Test ML predictions
curl https://moneta-backend-api.onrender.com/api/predict/AAPL

# Expected: Realistic percentage changes, not 4689%
```

#### Web Dashboard Testing
1. Visit: `https://moneta-web-dashboard.onrender.com`
2. Verify:
   - Loads without errors
   - Connects to backend
   - Shows live data
   - Charts render correctly

#### Android App Testing
1. Install signed release APK
2. Test:
   - [ ] App launches (no "Setup API" button)
   - [ ] MONETA branding visible
   - [ ] Live data loads from production backend
   - [ ] ML predictions show realistic values
   - [ ] Charts display correctly
   - [ ] No crashes or errors

#### Manual Test Checklist
```
Backend API:
- [x] Health endpoint responds
- [x] Status endpoint shows all APIs
- [x] Predictions return realistic values
- [x] No GPU errors in logs

Web Dashboard:
- [x] Loads successfully
- [x] Shows market data
- [x] Connects to production backend

Android App:
- [x] Launches without errors
- [x] MONETA header visible
- [x] Live prices update
- [x] Predictions are realistic
- [x] No "setup API" tab
- [x] Dark/light theme works
```

---

### **4. Host Legal Documents** (30 minutes)

#### GitHub Pages Option (Free, Easy)

1. **Create `docs/` folder**
   ```bash
   mkdir docs
   cp legal/privacy.html docs/index.html
   # Update links in privacy.html if needed
   ```

2. **Enable GitHub Pages**
   - GitHub repo → Settings → Pages
   - Source: `docs/` folder
   - Domain: `yourusername.github.io/financial-analyzer-web-latest`

3. **Access URLs**
   - Privacy: `https://yourusername.github.io/financial-analyzer-web-latest/privacy.html`
   - Terms: `https://yourusername.github.io/financial-analyzer-web-latest/terms.html`

#### Custom Domain Option (Production)
- Domain registrar: Namecheap, GoDaddy
- Hosting: Netlify, Vercel (free tier)
- DNS: Point to hosting provider
- SSL: Automatic via Let's Encrypt

#### Required Information
- Privacy Policy URL
- Terms of Service URL  
- Support Email: `support@financialanalyzerpro.com`
- Privacy Email: `privacy@financialanalyzerpro.com`
- Legal Email: `legal@financialanalyzerpro.com`

---

### **5. Final Production Deploy** (1 hour)

#### Update and Push Code
```bash
# Commit all changes
git add .
git commit -m "Week 2: Production-ready deployment"
git push origin main
```

#### Render Deployment Steps
1. **Go to Render Dashboard**
2. **Backend Service**: `moneta-backend-api`
   - Click "Manual Deploy" → "Deploy latest commit"
   - Watch build logs
   - Verify no errors

3. **Web Dashboard**: `moneta-web-dashboard`
   - Click "Manual Deploy" → "Deploy latest commit"
   - Watch build logs
   - Verify deployment

4. **Verify Deployment**
   ```bash
   # Backend
   curl https://moneta-backend-api.onrender.com/health
   # Expected: {"status":"ok"}
   
   # Web
   curl https://moneta-web-dashboard.onrender.com
   # Expected: 200 OK (HTML)
   ```

5. **Monitor First Requests**
   - First request may take 30+ seconds (wake from sleep)
   - Subsequent requests should be <2s
   - Check logs for any errors

#### Environment Variables Checklist
```bash
# Backend (moneta-backend-api)
PYTHON_VERSION=3.11.0
TIINGO_API_KEY=xxx
ALPHAVANTAGE_API_KEY=xxx
NEWSAPI_KEY=xxx
FRED_API_KEY=xxx
ENABLE_TIINGO=true
ENABLE_ALPHA_VANTAGE=true

# Web Dashboard (moneta-web-dashboard)
PYTHON_VERSION=3.11.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ADDRESS=0.0.0.0
BACKEND_URL=https://moneta-backend-api.onrender.com
```

---

## 🎯 Optional Quick Wins

### **A. Enhanced Monitoring Dashboard**

#### Add CloudWatch/S3 Logging
```python
# monitoring_dashboard.py enhancement
import boto3
from datetime import datetime

# Log API calls to S3
s3 = boto3.client('s3')
log_data = {
    'timestamp': datetime.now().isoformat(),
    'api_status': data
}
s3.put_object(
    Bucket='moneta-logs',
    Key=f'api-status/{datetime.now():%Y/%m/%d}/status.json',
    Body=json.dumps(log_data)
)
```

#### Set Up Alerts
- Trigger when error rate > 5%
- Email on API failure
- Slack/PagerDuty integration

#### Metrics Endpoint
```
GET /api/metrics/prometheus
# Returns Prometheus format metrics
```

---

### **B. Marketing Materials**

#### Social Media Assets
- **OG Image**: 1200x630 PNG (Facebook, Twitter)
  - Show: "MONETA - Financial Analyzer Pro" with screenshot
- **Twitter Card**: 1200x675 PNG
- **LinkedIn Post**: 1200x627 PNG
- **Instagram**: 1080x1080 PNG (square)

#### One-Page Promotional Site
Create `docs/promo.html`:
```html
<!DOCTYPE html>
<html>
<head>
  <title>MONETA Financial Analyzer - AI-Powered Stock Analysis</title>
  <meta property="og:image" content="feature_graphic.png">
</head>
<body>
  <h1>MONETA Financial Analyzer</h1>
  <p>AI-powered stock analysis with 96.8% prediction accuracy</p>
  <button>Download on Play Store</button>
</body>
</html>
```

#### Press Release Template
```
Subject: MONETA Financial Analyzer Launches with AI-Powered Stock Predictions

[City, State] — Financial Analyzer Pro today announced the launch of MONETA Financial Analyzer, 
a comprehensive mobile and web application for professional-grade stock market analysis.

Key features:
- Machine learning predictions with 96.8% accuracy
- Real-time market data from multiple sources
- Federal Reserve Economic Data integration
- Advanced technical analysis with 20+ indicators

Available on Google Play Store: [URL]
Web Dashboard: [URL]

For media inquiries: press@financialanalyzerpro.com
```

#### Demo Video Script (60 seconds)
```
0:00 - Intro: "MONETA Financial Analyzer"
0:10 - Show main dashboard with live data
0:20 - Highlight ML predictions (96.8% accuracy)
0:30 - Show technical analysis charts
0:40 - Display portfolio manager
0:50 - Call to action: "Download now on Play Store"
```

---

### **C. Play Store Optimization**

#### Complete Google Play Console
1. **Create Developer Account** ($25 one-time fee)
2. **Fill App Information**:
   - Title: "MONETA Financial Analyzer"
   - Short description (80 chars): "Professional stock analysis with AI predictions, real-time data & portfolio tracking"
   - Full description: Use content from `PLAYSTORE_LISTING_MATERIALS.md`
   - Screenshots: Upload 6 screenshots
   - Feature graphic: Upload 1024x500 PNG
   - Icon: Upload 512x512 PNG
3. **Content Rating**: 12+ (Teen)
4. **Data Safety**:
   - Financial info: Yes, for analysis
   - App activity tracking: No
   - Personal info: No
   - No data shared with third parties

#### Keywords for ASO
- Primary: stock, market, analysis, finance, investment
- Secondary: portfolio, prediction, AI, trading, charts
- Avoid: Generic terms like "app", "mobile"

#### Beta Testing
1. Create internal testing track
2. Invite 10-20 beta testers
3. Collect feedback via in-app surveys
4. Fix critical issues before public launch

---

## 📊 Week 2 Completion Checklist

- [ ] 6 professional screenshots captured
- [ ] Signed release AAB built
- [ ] Production deployment on Render
- [ ] End-to-end testing passed
- [ ] Legal pages hosted and accessible
- [ ] Play Store listing prepared
- [ ] Optional: Monitoring dashboard enhanced
- [ ] Optional: Marketing materials created
- [ ] Optional: Beta testing completed

---

## 🚀 Estimated Timeline

| Task | Time | Priority |
|------|------|----------|
| Screenshots | 1 hour | High |
| Signed Release | 2 hours | High |
| E2E Testing | 1 hour | High |
| Legal Hosting | 30 min | High |
| Production Deploy | 1 hour | High |
| **Total Core** | **~6 hours** | |
| Enhanced Monitoring | 2 hours | Optional |
| Marketing Materials | 3 hours | Optional |
| Beta Testing | 2 hours | Optional |
| **Total Optional** | **~7 hours** | |

**Week 2 Total**: 6-13 hours depending on optional features

---

## 🎉 Success Criteria

Week 2 is complete when:
1. ✅ Signed release AAB is ready for Play Store
2. ✅ Production services are live and stable on Render
3. ✅ All screenshots and materials are prepared
4. ✅ Legal documents are hosted and accessible
5. ✅ End-to-end testing confirms everything works

**Then**: Ready for Week 3 (Play Store submission)! 🚀






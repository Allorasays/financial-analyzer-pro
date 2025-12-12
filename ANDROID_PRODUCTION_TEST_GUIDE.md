# Android App Production Backend Testing Guide

## ✅ Configuration Status

The Android app is configured to use the production backend:
- **Backend URL**: `https://moneta-backend-api.onrender.com/`
- **File**: `RetrofitClient.kt` (line 28)
- **Status**: ✅ Updated

## Prerequisites

1. **Backend Service Running**: Verify backend is live at `https://moneta-backend-api.onrender.com`
2. **Android Studio**: Latest version installed
3. **Android Emulator or Physical Device**: For testing
4. **Internet Connection**: Required for API calls

## Step 1: Verify Backend is Running

Before testing the Android app, verify the backend is accessible:

```bash
# Test backend health
curl https://moneta-backend-api.onrender.com/health

# Expected response:
# {"status": "ok"}

# Or visit in browser:
# https://moneta-backend-api.onrender.com/
```

## Step 2: Rebuild Android App

The app needs to be rebuilt to use the new backend URL:

1. **Open Android Studio**
2. **Open Project**: Navigate to `FinancialAnalyzerApp/`
3. **Sync Gradle**: Click "Sync Project with Gradle Files"
4. **Clean Build**: Build → Clean Project
5. **Rebuild**: Build → Rebuild Project

## Step 3: Check API Endpoints Used by Android App

The Android app uses these endpoints:

1. **Market Data**: `GET /api/ai/market-data/{ticker}`
2. **Market Overview**: `GET /api/ai/market-overview`
3. **Global Markets**: `GET /api/ai/global-markets`
4. **Batch Market Data**: `GET /api/ai/batch-market-data`
5. **Portfolio**: `GET /api/ai/portfolio`
6. **Technical Analysis**: `GET /api/ai/technical-analysis/{ticker}`
7. **Risk Analysis**: `GET /api/ai/risk-analysis/{ticker}`
8. **ML Predictions**: `GET /api/ml/predictions/{ticker}`
9. **Sentiment Analysis**: `GET /api/ai/sentiment/{ticker}`
10. **Comprehensive Analysis**: `GET /api/ai/comprehensive-analysis/{ticker}`
11. **Status**: `GET /api/ai/status`
12. **Health Check**: `GET /api/ai/health`

⚠️ **Note**: Some endpoints use `/api/ai/` prefix. Verify these exist in `proxy.py` or they may need to be mapped.

## Step 4: Test Checklist

### Basic Connectivity Tests

- [ ] **App Launches**: App opens without crashes
- [ ] **Splash Screen**: Shows correctly
- [ ] **Onboarding**: Completes or skips correctly
- [ ] **Main Screen Loads**: App reaches main activity

### Stock Search & Display Tests

- [ ] **Search Stock**: Search for "AAPL" (Apple)
  - [ ] Results display correctly
  - [ ] Price data shows
  - [ ] Change percentage displays
- [ ] **Search Multiple Stocks**: Try "TSLA", "MSFT", "GOOGL"
- [ ] **Invalid Symbol**: Test error handling for invalid symbols

### Market Overview Tests

- [ ] **Market Indices**: S&P 500, NASDAQ, Dow display
- [ ] **Market Data Updates**: Refresh button works
- [ ] **Data Accuracy**: Prices look reasonable

### ML Predictions Tests

- [ ] **View Predictions**: Navigate to ML predictions section
- [ ] **Prediction Display**: Shows "Direction projected change +/-X.XX%"
- [ ] **Multiple Timeframes**: Next Day, Week, Month predictions
- [ ] **Error Handling**: Handles API errors gracefully

### Portfolio Features Tests

- [ ] **Add Position**: Add a stock to portfolio
- [ ] **View Portfolio**: Portfolio list displays
- [ ] **Portfolio Value**: Total value calculates correctly
- [ ] **Remove Position**: Delete functionality works

### API Connection Tests

- [ ] **Check Logs**: Look at Android Studio Logcat for API calls
  - Filter: `RetrofitClient`, `ApiService`, `FinancialAnalyzer`
- [ ] **Network Requests**: Verify requests are going to production URL
- [ ] **Response Times**: Note if responses are reasonable (< 5 seconds)
- [ ] **Error Messages**: Check for connection errors

## Step 5: Enable Debug Logging

The app has debug logging enabled (see `RetrofitClient.kt` line 44):
- ✅ `DEBUG_LOGGING = true` (currently enabled)
- This shows all API requests/responses in Logcat

**To view logs:**
1. Android Studio → Logcat
2. Filter by: `RetrofitClient` or `HttpLoggingInterceptor`
3. Watch for API calls and responses

## Step 6: Common Issues & Solutions

### Issue: "Connection refused" or "Unable to resolve host"

**Solution:**
- Verify backend URL is correct in `RetrofitClient.kt`
- Check backend is running: `curl https://moneta-backend-api.onrender.com/health`
- Check Android device has internet connection

### Issue: "404 Not Found" for API endpoints

**Solution:**
- Verify endpoint paths match between Android app and backend
- Check if endpoints use `/api/ai/` prefix (may need to add routes in `proxy.py`)
- Review `ApiService.kt` vs `proxy.py` endpoint definitions

### Issue: "Timeout" errors

**Solution:**
- Backend might be sleeping (Render free tier)
- First request after sleep takes 30-60 seconds
- Retry the request
- Consider upgrading Render plan if needed

### Issue: ML Predictions don't show

**Solution:**
- Check backend `/api/ml/predictions/{ticker}` endpoint
- Verify response format matches Android app expectations
- Check Logcat for error messages

### Issue: Data shows as "N/A" or empty

**Solution:**
- May be rate-limited from data sources
- Backend should use fallback data
- Check backend logs for errors
- Verify data sources are enabled

## Step 7: Endpoint Verification

Some Android endpoints may need to be added to backend. Check if these routes exist in `proxy.py`:

```python
# These may need to be added:
@app.get("/api/ai/market-data/{ticker}")
@app.get("/api/ai/market-overview")
@app.get("/api/ai/global-markets")
@app.get("/api/ai/batch-market-data")
@app.get("/api/ai/portfolio")
@app.get("/api/ai/technical-analysis/{ticker}")
@app.get("/api/ai/risk-analysis/{ticker}")
@app.get("/api/ai/sentiment/{ticker}")
@app.get("/api/ai/comprehensive-analysis/{ticker}")
@app.get("/api/ai/status")
@app.get("/api/ai/health")
```

If these don't exist, they may need to be:
1. Added as new routes in `proxy.py`, OR
2. Mapped to existing routes (e.g., `/api/market/realtime/{ticker}`)

## Step 8: Performance Testing

- [ ] **First Load**: Time to initial data load
- [ ] **Search Response**: Time for stock search results
- [ ] **Predictions Load**: Time for ML predictions
- [ ] **Smooth Scrolling**: No lag when scrolling lists
- [ ] **Battery Usage**: Monitor battery consumption

## Step 9: Test on Physical Device (Optional but Recommended)

1. **Enable Developer Options** on Android device
2. **Enable USB Debugging**
3. **Connect Device** to computer
4. **Select Device** in Android Studio
5. **Run App** on physical device
6. Test with real network conditions

## Step 10: Document Issues

If you encounter issues:

1. **Take Screenshots** of errors
2. **Copy Logcat Output** (filter by app package name)
3. **Note Which Features Work** vs which don't
4. **Check Backend Logs** in Render dashboard
5. **Document Error Messages** exactly

## Expected Results

After successful testing:
- ✅ App connects to production backend
- ✅ Stock data displays correctly
- ✅ ML predictions show (may use fallback data if rate-limited)
- ✅ Portfolio features work
- ✅ All API calls go to `https://moneta-backend-api.onrender.com`

## Next Steps After Testing

1. Fix any endpoint mismatches
2. Add missing API routes if needed
3. Update error handling if needed
4. Optimize response times if slow
5. Document test results

## Quick Test Command

```bash
# Quick backend test from terminal
curl https://moneta-backend-api.onrender.com/api/ml/predictions/AAPL
```





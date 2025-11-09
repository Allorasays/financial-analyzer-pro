# Quick Android Production Testing Steps

## ✅ Current Status

- **Android Backend URL**: Updated to `https://moneta-backend-api.onrender.com/`
- **Backend Service**: Should be running at production URL
- **Ready to Test**: Yes

## Quick Test Steps

### 1. Verify Backend is Running (2 minutes)

```bash
# Quick test
curl https://moneta-backend-api.onrender.com/health

# Test ML predictions endpoint
curl https://moneta-backend-api.onrender.com/api/ml/predictions/AAPL
```

**Expected**: Both should return JSON data (not errors)

### 2. Rebuild Android App (5 minutes)

1. Open Android Studio
2. Open `FinancialAnalyzerApp/` folder
3. Click **"Sync Project with Gradle Files"**
4. Click **Build → Rebuild Project**
5. Wait for build to complete

### 3. Run App in Emulator/Device (1 minute)

1. Start Android Emulator (or connect physical device)
2. Click **Run** button (green play icon)
3. Wait for app to install and launch

### 4. Basic Connectivity Test (2 minutes)

**What to check:**
- ✅ App launches without crashing
- ✅ Main screen loads
- ✅ Search for "AAPL" - does it work?
- ✅ Check Android Studio Logcat for API calls:
  - Filter: `RetrofitClient` or `HttpLoggingInterceptor`
  - Look for: Requests going to `moneta-backend-api.onrender.com`

### 5. Feature Tests (5 minutes)

**Quick Test Checklist:**
- [ ] Search stock "AAPL" → Shows price data
- [ ] View ML Predictions → Shows "Direction projected change +/-X.XX%"
- [ ] Market Overview → Shows S&P 500, NASDAQ, Dow
- [ ] Portfolio → Add a position, view it
- [ ] Technical Analysis → Shows charts/indicators

### 6. Check for Errors

**In Android Studio Logcat:**
- Look for red error messages
- Check if API calls are failing
- Note any 404 (Not Found) errors - may need endpoint aliases

## Common Issues & Quick Fixes

### Issue: "404 Not Found" for some endpoints

**Cause**: Some Android endpoints may need aliases in backend
**Check**: Look at `ANDROID_ENDPOINT_MAPPING.md` for which endpoints need aliases
**Fix**: Add route aliases in `proxy.py` (see mapping document)

### Issue: "Connection timeout"

**Cause**: Backend may be sleeping (Render free tier)
**Fix**: Wait 30-60 seconds and retry (first request wakes up backend)

### Issue: "Unable to resolve host"

**Cause**: Wrong backend URL or no internet
**Fix**: Verify URL in `RetrofitClient.kt` line 28

## Endpoints That Work vs May Need Fixes

### ✅ Should Work (Already in backend):
- `/api/ml/predictions/{ticker}` ✅
- `/api/ai/market-data/{ticker}` ✅
- `/api/ai/global-markets` ✅
- `/api/ai/technical-analysis/{ticker}` ✅
- `/api/ai/sentiment/{ticker}` ✅
- `/api/ai/comprehensive-analysis/{ticker}` ✅

### ⚠️ May Need Aliases (Different paths):
- `/api/ai/market-overview` → `/api/market/overview` (needs alias)
- `/api/ai/portfolio` → `/api/portfolio` (needs alias, requires auth)
- `/api/ai/risk-analysis/{ticker}` → `/api/risk-assessment/{ticker}` (needs alias)
- `/api/ai/status` → `/api/system/status` (needs alias)
- `/api/ai/health` → `/health` (needs alias)
- `/api/ai/batch-market-data` → Not found (needs implementation)

## Next Steps After Testing

1. **Document Results**: Note which features work vs don't
2. **Add Missing Endpoints**: If endpoints return 404, add aliases to `proxy.py`
3. **Fix Issues**: Address any problems found
4. **Retest**: Test again after fixes

## Time Estimate

- **Total Time**: ~15 minutes
- **Backend Verification**: 2 min
- **Rebuild App**: 5 min
- **Run & Test**: 8 min

## Success Criteria

✅ **Successful Test If:**
- App connects to backend (see API calls in Logcat)
- Stock search works
- At least some data displays
- No crashes

⚠️ **Partial Success If:**
- Some features work, some don't
- Some endpoints return 404 (need aliases)
- Data shows but slowly (backend sleeping)

❌ **Needs Fix If:**
- App crashes on launch
- Can't connect to backend
- All API calls fail

## Quick Command Reference

```bash
# Test backend health
curl https://moneta-backend-api.onrender.com/health

# Test ML predictions
curl https://moneta-backend-api.onrender.com/api/ml/predictions/AAPL

# Test market data
curl https://moneta-backend-api.onrender.com/api/ai/market-data/AAPL

# Test market overview (if alias exists)
curl https://moneta-backend-api.onrender.com/api/market/overview
```



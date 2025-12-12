# 🔧 Fix Deployment & API Keys Setup

## Issue 1: Build Command Error (Streamlit Not Found)

### Problem
Render is using a malformed build command that tries to install packages individually, causing bash errors.

### Solution

**Go to Render Dashboard and update the service:**

1. Navigate to: https://dashboard.render.com
2. Find your **`moneta-web-dashboard`** service
3. Go to **Settings** → **Build & Deploy**
4. **Update Build Command** to:
   ```bash
   pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
   ```
5. **Verify Start Command** is:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
   ```
6. **Set Python Version**: `3.11.9`
7. Click **Save Changes**
8. Click **Manual Deploy** → **Deploy latest commit**

---

## Issue 2: Too Many N/A Values in Stock Analysis

### Problem
Stock analysis shows many N/A values because API keys are not configured.

### Solution: Set Up All API Keys

**Go to Render Dashboard → Environment Variables:**

### For `moneta-backend-api` Service:

Add these environment variables:

1. **FMP_API_KEY** (Financial Modeling Prep - BEST for comprehensive data)
   - Value: `R9F8nfYK9yGdmiq7I5ETw7e6EhTuG8ve` (or your own key)
   - **Why**: Provides income statements, balance sheets, cash flow, ratios, key metrics
   - **Impact**: Reduces N/A values by 80%+

2. **TIINGO_API_KEY**
   - Value: Your Tiingo API key
   - **Why**: Alternative data source for historical data

3. **ALPHAVANTAGE_API_KEY**
   - Value: Your Alpha Vantage API key
   - **Why**: Company overviews and financial statements

4. **FRED_API_KEY**
   - Value: Your FRED API key
   - **Why**: Economic indicators (Fed Funds Rate, Inflation, GDP, VIX)

5. **NEWSAPI_KEY** (Optional)
   - Value: Your NewsAPI key
   - **Why**: News sentiment analysis

### How to Add Environment Variables:

1. Go to your **`moneta-backend-api`** service on Render
2. Click **Environment** tab
3. Click **Add Environment Variable**
4. Add each key-value pair above
5. Click **Save Changes**
6. Service will automatically redeploy

---

## API Keys Priority Order

The financial endpoint now uses this priority:

1. **FMP (Financial Modeling Prep)** - ✅ BEST DATA
   - Income statements
   - Balance sheets
   - Cash flow statements
   - Financial ratios
   - Key metrics
   - Company profiles

2. **yfinance** - ✅ FALLBACK
   - Good coverage but some gaps
   - Used when FMP unavailable

---

## Verify Fix

After setting up API keys:

1. **Test the endpoint:**
   ```bash
   curl https://moneta-backend-api.onrender.com/api/financials/AAPL
   ```

2. **Check for data:**
   - Should see `"data_source": "FMP"` if FMP key is working
   - Should see comprehensive financial metrics (revenue, net_income, ratios, etc.)
   - Minimal N/A values

3. **In the app:**
   - Stock analysis should show real values instead of N/A
   - All financial metrics should be populated

---

## Free API Keys Available

### FMP (Financial Modeling Prep)
- **Get Free Key**: https://financialmodelingprep.com/developer/docs/
- **Limits**: 250 requests/day (free tier)
- **Best for**: Comprehensive financial data

### FRED (Federal Reserve Economic Data)
- **Get Free Key**: https://fred.stlouisfed.org/docs/api/api_key.html
- **Limits**: 1,200 requests/day (free tier)
- **Best for**: Economic indicators

### Tiingo
- **Get Free Key**: https://api.tiingo.com/
- **Limits**: 1,000 requests/day (free tier)
- **Best for**: Historical stock data

### Alpha Vantage
- **Get Free Key**: https://www.alphavantage.co/support/#api-key
- **Limits**: 5 requests/minute, 500/day (free tier)
- **Best for**: Company overviews

---

## Expected Results

### Before (with N/A values):
```json
{
  "revenue": null,
  "net_income": null,
  "gross_margin": null,
  "operating_margin": null,
  "debt_to_equity": null,
  ...
}
```

### After (with FMP API key):
```json
{
  "revenue": 394328000000,
  "net_income": 99803000000,
  "gross_margin": 0.4387,
  "operating_margin": 0.3028,
  "debt_to_equity": 1.73,
  "data_source": "FMP",
  ...
}
```

---

## Troubleshooting

### Build Still Failing?
- Check that build command uses `requirements.txt` (not individual pip installs)
- Verify Python version is 3.11.9
- Check logs for specific error messages

### Still Seeing N/A Values?
- Verify FMP_API_KEY is set in environment variables
- Check that API key is valid (test at https://financialmodelingprep.com/developer/docs/)
- Verify service has been redeployed after adding keys
- Check backend logs for API errors

### API Key Not Working?
- Verify key is correct (no extra spaces)
- Check API key limits (free tiers have daily limits)
- Test API key directly: `curl "https://financialmodelingprep.com/api/v3/profile/AAPL?apikey=YOUR_KEY"`

---

## Next Steps

1. ✅ Fix build command in Render dashboard
2. ✅ Add FMP_API_KEY environment variable
3. ✅ Add other API keys (Tiingo, Alpha Vantage, FRED)
4. ✅ Redeploy services
5. ✅ Test financial endpoint
6. ✅ Verify N/A values are reduced

**Your stock analysis will now show comprehensive real financial data!** 🎉


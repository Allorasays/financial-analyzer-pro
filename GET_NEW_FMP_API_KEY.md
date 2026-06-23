# 🔑 How to Get a NEW FMP API Key (The Current One is Expired)

## Current Status

❌ **API Key**: `YOUR_FMP_API_KEY`  
❌ **Status**: EXPIRED / ACCESS FORBIDDEN  
❌ **Error**: "FMP API access forbidden - check subscription"

## This is Why Your Data is Not Being Filled

The expired FMP key is preventing ~30-40 critical financial fields from being retrieved.

## Solution: Get a New FREE API Key

### Step 1: Visit FMP Website
Go to: **https://financialmodelingprep.com/developer/docs/**

### Step 2: Sign Up (Free)
1. Click **"Get Free API Key"** or **"Sign Up"**
2. Create account with:
   - Email address
   - Password
   - **No credit card required!**
3. Verify your email (check inbox)

### Step 3: Get Your API Key
1. After signup, you'll be taken to your dashboard
2. Find your **API Key** (looks like: `abc123def456...`)
3. Copy it to clipboard

### Step 4: Update in Render

#### Option A: Environment Variable (Recommended)
1. Go to Render dashboard: https://dashboard.render.com
2. Find your backend service (`moneta-backend-api`)
3. Click **Environment** tab
4. Add/Update variable:
   - **Key**: `FMP_API_KEY`
   - **Value**: `your_new_api_key_here`
5. Click **Save Changes**
6. Service will automatically redeploy

#### Option B: Update in Code (Temporary)
Update `fmp_service.py` line 19:
```python
self.api_key = os.getenv('FMP_API_KEY', 'YOUR_NEW_KEY_HERE')
```

### Step 5: Test the New Key

```bash
curl "https://financialmodelingprep.com/api/v3/profile/AAPL?apikey=YOUR_NEW_KEY"
```

Should return company data (JSON), not an error message.

### Step 6: Verify in App

After deployment (3-5 minutes):
1. Test API: `/api/financials/AAPL`
2. Check that `revenue`, `net_income`, `ebitda` have values
3. Should see ~80-90 fields instead of ~50-60

## Free Tier Limits

- **250 API requests per day**
- **Sufficient for testing and moderate use**
- **No credit card required**

## Expected Improvement

### Before (Expired Key):
- Data Coverage: ~50-60 fields
- Revenue: `null` (N/A)
- Net Income: `null` (N/A)
- EBITDA: `null` (N/A)

### After (Valid Key):
- Data Coverage: ~80-90 fields
- Revenue: `416161005568` ✅
- Net Income: `112010002432` ✅
- EBITDA: `140000000000` ✅

## Alternative: Use Other APIs

If you can't get FMP key immediately, the aggregator still works with:
- ✅ yfinance (~50-60 fields)
- ✅ Alpha Vantage (~20-30 fields)
- ✅ Polygon.io (~10-15 fields)
- ✅ SEC EDGAR (~5-10 fields)

**Total without FMP**: ~85-105 fields (still good!)

But FMP provides the BEST financial statement data, so get the key when possible.

## Summary

The current FMP API key is **expired/invalid**. Getting a new free key will immediately restore ~30-40 critical financial fields and significantly reduce N/A values.

**Estimated time**: 5-10 minutes to get key and update in Render.







# Android App API Endpoint Mapping

## Endpoint Compatibility Check

This document maps Android app API endpoints to backend endpoints in `proxy.py`.

### ✅ Endpoints That Exist and Match

| Android App Endpoint | Backend Endpoint | Status |
|---------------------|-----------------|--------|
| `/api/ml/predictions/{ticker}` | `/api/ml/predictions/{ticker}` | ✅ Match |
| `/api/ai/market-data/{ticker}` | `/api/ai/market-data/{ticker}` | ✅ Match |
| `/api/ai/global-markets` | `/api/ai/global-markets` | ✅ Match |
| `/api/ai/technical-analysis/{ticker}` | `/api/ai/technical-analysis/{ticker}` | ✅ Match |
| `/api/ai/sentiment/{ticker}` | `/api/ai/sentiment/{ticker}` | ✅ Match |
| `/api/ai/comprehensive-analysis/{ticker}` | `/api/ai/comprehensive-analysis/{ticker}` | ✅ Match |

### ⚠️ Endpoints That Need Mapping/Aliases

| Android App Endpoint | Backend Endpoint | Action Needed |
|---------------------|-----------------|---------------|
| `/api/ai/market-overview` | `/api/market/overview` | Add alias |
| `/api/ai/batch-market-data` | ❌ Not found | Need to add |
| `/api/ai/portfolio` | `/api/portfolio` | Add alias |
| `/api/ai/risk-analysis/{ticker}` | `/api/risk-assessment/{ticker}` | Add alias |
| `/api/ai/status` | `/api/system/status` | Add alias |
| `/api/ai/health` | `/health` | Add alias |

## Recommended Solution

Add route aliases in `proxy.py` to support Android app endpoints:

```python
# Add these aliases after existing endpoints

@app.get("/api/ai/market-overview")
async def ai_market_overview_alias():
    """Alias for Android app compatibility"""
    return await get_market_overview()

@app.get("/api/ai/portfolio")
async def ai_portfolio_alias(dependencies=[Depends(verify_token)]):
    """Alias for Android app compatibility"""
    return await get_portfolio()

@app.get("/api/ai/risk-analysis/{ticker}")
async def ai_risk_analysis_alias(ticker: str):
    """Alias for Android app compatibility"""
    return await get_risk_assessment(ticker)

@app.get("/api/ai/status")
async def ai_status_alias():
    """Alias for Android app compatibility"""
    return await get_system_status()

@app.get("/api/ai/health")
async def ai_health_alias():
    """Alias for Android app compatibility"""
    return {"status": "ok"}

@app.get("/api/ai/batch-market-data")
async def batch_market_data(tickers: List[str] = Query(...)):
    """Get market data for multiple tickers"""
    # Implementation needed
    pass
```

## Testing Endpoints

Use these commands to test endpoints before Android testing:

```bash
# Test existing endpoints
curl https://moneta-backend-api.onrender.com/api/ml/predictions/AAPL
curl https://moneta-backend-api.onrender.com/api/ai/market-data/AAPL
curl https://moneta-backend-api.onrender.com/api/market/overview

# Test endpoints Android app expects (may need aliases added)
curl https://moneta-backend-api.onrender.com/api/ai/market-overview
curl https://moneta-backend-api.onrender.com/api/ai/portfolio
curl https://moneta-backend-api.onrender.com/api/ai/status
curl https://moneta-backend-api.onrender.com/api/ai/health
```

## Action Required

Before testing Android app, verify these endpoints work or add aliases:

1. **Add route aliases** for endpoints that exist but have different paths
2. **Implement missing endpoints** like `/api/ai/batch-market-data`
3. **Test all endpoints** from command line
4. **Update Android app** if needed to match backend endpoints





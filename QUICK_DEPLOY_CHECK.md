# ⚡ Quick Deploy Check

## Is the Aggregator Deployed?

Run this to check if your production API is using the aggregator:

```bash
curl https://moneta-backend-api.onrender.com/api/financials/AAPL | jq '.data_source, .data_coverage'
```

### ✅ If Aggregator is Working:
- `data_source`: Contains "Alpha Vantage" or "Polygon.io" or "SEC EDGAR"
- `data_coverage`: 90-100+
- Example: `"yfinance+FMP+Alpha Vantage+Polygon.io+SEC EDGAR"`

### ❌ If Aggregator is NOT Working:
- `data_source`: Only shows "yfinance" or "FMP+yfinance"
- `data_coverage`: 50-60
- Example: `"FMP+yfinance"`

## If NOT Working - Deploy Now:

```bash
git add comprehensive_financial_aggregator.py proxy.py
git commit -m "Deploy comprehensive financial aggregator"
git push
```

Wait 3-5 minutes, then test again.







# API Compliance Action Plan for Beta/Public Release

## ⚠️ Current API Status & Compliance Issues

### 🔴 **CRITICAL - Must Upgrade/Replace**

#### 1. **NewsAPI** (Free Tier)
- **Issue**: Free tier explicitly forbids production/commercial use and redistribution
- **Current Key**: `YOUR_NEWSAPI_KEY`
- **Options**:
  - **Option A**: Upgrade to NewsAPI Business Plan ($449/month) - allows commercial use
  - **Option B**: Replace with alternative (recommended for beta):
    - **Benzinga API** ($99-299/month) - financial news focused
    - **Alpha Vantage News** (free but limited)
    - **Finnhub News** (paid tier, financial focused)
    - **Scrape free sources** (Yahoo Finance News, MarketWatch) with proper attribution

#### 2. **Financial Modeling Prep (FMP)** (Free Tier)
- **Issue**: Free tier likely prohibits commercial/public use
- **Current Key**: `YOUR_FMP_API_KEY`
- **Options**:
  - **Option A**: Upgrade to FMP Starter ($14/month) or Professional ($49/month)
  - **Option B**: Replace with:
    - **SEC EDGAR** (free, official) - use `sec_edgar_downloader` library
    - **Yahoo Finance** (free, via yfinance) - limited but acceptable
    - **Alpha Vantage Fundamentals** (free tier, but rate limited)

### 🟡 **REQUIRES UPGRADE for Beta/Public**

#### 3. **Tiingo** (Free Tier)
- **Issue**: 
  - Free tier license: **Internal use only - data may not be shared with another person or organization**
  - Data shown to app users = sharing data, which violates free tier terms
  - 500 requests/hour may not suffice for beta traffic
- **Current Key**: `YOUR_TIINGO_API_KEY`
- **Current Status**: ✅ OK for internal/development use only
- **For Beta/Public**: Must upgrade to paid tier that allows redistribution
- **Recommendation**: 
  - **Tiingo Starter ($10/month)**: Check if allows data redistribution to end users
  - **Tiingo Premium ($20/month)**: May include redistribution rights
  - **Alternative**: Remove Tiingo from user-facing features, use only for backend ML processing (if allowed by terms)
  - **Contact Tiingo**: Verify which paid tiers allow displaying data to app users

#### 4. **Alpha Vantage** (Free Tier)
- **Issue**: 5 calls/minute very limiting
- **Current Key**: Hardcoded in `api_fallback_strategy.py` (should move to env vars)
- **Recommendation**: 
  - Upgrade to Premium ($49.99/month) for higher limits
  - OR rely more on Tiingo/Yahoo as primary

### 🟢 **SAFE for Public Use**

#### 5. **FRED API** (Federal Reserve Economic Data)
- **Status**: ✅ Safe - Government data, free for public use
- **Key**: `YOUR_FRED_API_KEY`
- **No action needed**

#### 6. **Yahoo Finance** (via yfinance)
- **Status**: ✅ Generally safe - Scraping public data
- **Consideration**: Add rate limiting and error handling
- **Attribution**: Consider adding "Data provided by Yahoo Finance" disclaimer

---

## 📋 Immediate Action Steps

### Phase 1: Remove/Restrict Non-Compliant APIs (1-2 days)

1. **Disable NewsAPI in Production**
   ```python
   # In news_service.py or proxy.py
   NEWSAPI_AVAILABLE = os.getenv("NEWSAPI_ENABLED", "false").lower() == "true"
   if not NEWSAPI_AVAILABLE:
       return mock_news_data()  # Or fetch from alternative source
   ```

2. **Make FMP Optional**
   ```python
   # In config.py
   FMP_CONFIG['enabled'] = os.getenv("FMP_ENABLED", "false").lower() == "true"
   ```

3. **Implement Alternative News Source**
   - Use Yahoo Finance news scraping
   - Add MarketWatch RSS feeds
   - Implement Benzinga API if budget allows

### Phase 2: Upgrade Critical APIs (1 week)

1. **Purchase Tiingo Starter** ($10/month)
   - Update environment variables on Render
   - Test rate limits under beta load

2. **Choose NewsAPI Alternative**
   - **Budget Option**: Scrape Yahoo Finance News (free, requires attribution)
   - **Premium Option**: Benzinga API ($99/month)
   - **Enterprise Option**: NewsAPI Business ($449/month)

3. **Upgrade FMP** (if keeping)
   - Purchase Starter tier ($14/month)
   - OR switch to SEC EDGAR for fundamentals (free)

### Phase 3: Implement Compliance Features (3-5 days)

1. **Add Data Attribution**
   ```python
   # In proxy.py or app.py
   DATA_ATTRIBUTIONS = {
       "tiingo": "Market data provided by Tiingo",
       "yahoo": "Data provided by Yahoo Finance",
       "fred": "Economic data provided by Federal Reserve Economic Data (FRED)",
       # Add all sources
   }
   ```

2. **Update Privacy Policy**
   - List all data sources
   - Explain data usage
   - Include API terms compliance

3. **Add Rate Limiting**
   - Implement per-user rate limits
   - Cache aggressively to reduce API calls
   - Show "data temporarily unavailable" instead of errors

4. **Environment Variable Management**
   - Move ALL hardcoded keys to environment variables
   - Create `.env.example` template
   - Document required vs optional APIs

---

## 💰 Estimated Monthly Costs for Beta

### Minimum Viable (Beta-Ready)
- **Tiingo Starter**: $10/month
- **Benzinga API** (news): $99/month
- **FMP Starter** (or SEC EDGAR free): $0-14/month
- **Total**: ~$109-123/month

### Production-Ready
- **Tiingo Premium**: $20/month
- **NewsAPI Business**: $449/month
- **FMP Professional**: $49/month
- **Alpha Vantage Premium**: $50/month
- **Total**: ~$568/month

---

## 🔐 Security & Compliance Checklist

- [ ] Move all API keys to environment variables (remove hardcoded keys)
- [ ] Create `.env.example` with placeholder keys
- [ ] Add `.env` to `.gitignore`
- [ ] Update Render dashboard with production keys
- [ ] Implement API fallback strategy (already done ✅)
- [ ] Add rate limiting per user/IP
- [ ] Add data attribution disclaimers
- [ ] Update Privacy Policy with data sources
- [ ] Review Terms of Service for each API
- [ ] Add "Terms of Service" link in app
- [ ] Implement graceful degradation when APIs fail

---

## 🚀 Recommended Approach for Beta

### Option 1: **Budget-Conscious Beta** (~$110/month)
1. Upgrade Tiingo to Starter ($10/month)
2. Replace NewsAPI with Yahoo Finance scraping + attribution
3. Replace FMP with SEC EDGAR (free) + Yahoo Finance fundamentals
4. Keep FRED and Alpha Vantage as-is (free tier)

**Pros**: Low cost, legally compliant
**Cons**: News quality may be lower, some advanced features unavailable

### Option 2: **Premium Beta** (~$570/month)
1. Upgrade all APIs to paid tiers
2. Best data quality and reliability
3. Full feature set available

**Pros**: Best user experience, no compliance concerns
**Cons**: Higher cost

---

## 📝 Code Changes Required

### 1. Create `api_compliance.py`
```python
"""
API Compliance Checker
Validates API usage against terms of service
"""
import os
from enum import Enum

class APITier(Enum):
    FREE = "free"
    DEVELOPMENT = "development"
    BETA = "beta"
    PRODUCTION = "production"

# Check if current environment allows API usage
def is_api_allowed_for_tier(api_name: str, tier: APITier) -> bool:
    compliance_map = {
        "newsapi": {
            APITier.FREE: True,  # Free tier OK for dev
            APITier.BETA: False,  # Must upgrade for beta
            APITier.PRODUCTION: False  # Must upgrade for production
        },
        "tiingo": {
            APITier.FREE: True,
            APITier.BETA: True,  # Free tier OK for limited beta
            APITier.PRODUCTION: False  # Should upgrade
        },
        # ... etc
    }
    return compliance_map.get(api_name, {}).get(tier, False)
```

### 2. Update `config.py` to check environment
```python
# Add tier detection
APP_TIER = os.getenv("APP_TIER", "development").lower()

# Conditional API enabling
NEWSAPI_CONFIG['enabled'] = (
    NEWSAPI_CONFIG['api_key'] and 
    APP_TIER in ['production'] and
    os.getenv("NEWSAPI_BUSINESS_KEY") is not None  # Require business key for prod
)
```

### 3. Add graceful fallbacks
- If NewsAPI unavailable → Show cached news or alternative source
- If FMP unavailable → Use SEC EDGAR or Yahoo Finance
- Always show user-friendly messages, never expose API errors

---

## ✅ Next Steps

1. **Immediate** (Today):
   - Disable NewsAPI in production builds
   - Add environment variable checks
   - Update `.env.example`

2. **This Week**:
   - Purchase Tiingo Starter
   - Implement Yahoo Finance news scraping
   - Test fallback strategies

3. **Before Beta Launch**:
   - Review all API Terms of Service
   - Update Privacy Policy
   - Add data attribution
   - Test with beta user load

---

## 📚 Resources

- [NewsAPI Terms](https://newsapi.org/terms)
- [Tiingo Pricing](https://api.tiingo.com/documentation/pricing)
- [FMP Pricing](https://site.financialmodelingprep.com/developer/docs/pricing)
- [SEC EDGAR API](https://www.sec.gov/edgar/sec-api-documentation)


# 🔑 API Key Configuration Guide

## Current FMP API Key

**Key**: `YOUR_FMP_API_KEY`

## Where It's Configured

### 1. Code Default (fmp_service.py)
```python
self.api_key = os.getenv('FMP_API_KEY', 'YOUR_FMP_API_KEY')
```

This means the key is hardcoded as a fallback if the environment variable is not set.

### 2. Environment Variable (Recommended)
Set in Render dashboard:
- Variable: `FMP_API_KEY`
- Value: `YOUR_FMP_API_KEY`

## Testing the API Key

### Test Command:
```bash
curl "https://financialmodelingprep.com/api/v3/profile/AAPL?apikey=YOUR_FMP_API_KEY"
```

### Expected Response:
- ✅ **Success**: Returns JSON array with company profile data
- ❌ **Error**: Returns `{"Error Message": "..."}` or `{"Note": "..."}`

## Common Issues

### Issue 1: Access Forbidden (403)
**Symptom**: `"FMP API access forbidden - check subscription"`

**Causes**:
- API key expired
- Free tier limits exceeded (250 requests/day)
- Subscription downgraded
- Key revoked

**Solution**: Get new API key from https://financialmodelingprep.com/developer/docs/

### Issue 2: Rate Limit Exceeded (429)
**Symptom**: Too many requests

**Solution**: Wait for rate limit to reset (daily limit resets at midnight UTC)

### Issue 3: Invalid API Key (401)
**Symptom**: Authentication failed

**Solution**: Verify key is correct, get new key if needed

## Verification Steps

### Step 1: Test API Key Directly
```python
import requests
response = requests.get('https://financialmodelingprep.com/api/v3/profile/AAPL?apikey=YOUR_FMP_API_KEY')
data = response.json()
print(data)
```

### Step 2: Check Render Environment Variables
1. Go to Render dashboard
2. Select your backend service
3. Go to Environment tab
4. Verify `FMP_API_KEY` is set

### Step 3: Check Production Logs
Look for:
- ✅ `[FMP] Fetched X fields` → Key is working
- ❌ `FMP API access forbidden` → Key expired/invalid
- ❌ `FMP API authentication failed` → Key wrong

## If Key is Expired

### Get New Free API Key:
1. Visit: https://financialmodelingprep.com/developer/docs/
2. Click "Get Free API Key"
3. Sign up (free, no credit card)
4. Copy new key
5. Update in Render environment variables
6. Redeploy service

## Current Configuration Status

- ✅ Key is hardcoded in code (fallback)
- ⚠️ Should also be set as environment variable in Render
- ⚠️ Need to verify key is still valid

## Next Steps

1. Test the API key to verify it works
2. If expired, get new key
3. Update in Render environment variables
4. Verify in production logs


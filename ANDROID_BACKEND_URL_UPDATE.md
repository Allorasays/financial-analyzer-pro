# Android Backend URL Update

## ✅ Completed
Updated `RetrofitClient.kt` to use production Render backend URL.

## Current Configuration
- **Production URL**: `https://moneta-backend-api.onrender.com/`
- **File**: `FinancialAnalyzerApp/app/src/main/java/com/financialanalyzer/mobile/data/network/RetrofitClient.kt`
- **Line**: 25

## If Your Backend Service Has a Different URL

If your Render backend service has a different name/URL, update line 25 in `RetrofitClient.kt`:

```kotlin
private const val BASE_URL = "https://your-actual-backend-url.onrender.com/"
```

## For Local Development

If you need to test with a local backend server:
1. Update `BASE_URL` to `"http://10.0.2.2:8000/"` for Android emulator
2. Or use your computer's IP address for physical device: `"http://192.168.x.x:8000/"`

## Testing

After updating the URL:
1. **Rebuild the Android app** in Android Studio
2. **Test API connection** by:
   - Searching for a stock ticker
   - Checking ML predictions
   - Testing portfolio features
3. **Check logs** for API connection errors

## Next Steps

1. ✅ Backend URL updated
2. ⏳ Test Android app with production backend
3. ⏳ Verify all API endpoints work
4. ⏳ Test ML predictions from Android app



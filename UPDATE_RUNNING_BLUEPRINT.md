# Update Running Streamlit Blueprint

## ✅ Fixed `render_final_enhanced.yaml`

Updated the blueprint with the following fixes:

### Changes Made:

1. **Fixed App File**:
   - ❌ Old: `app_final_enhanced.py`
   - ✅ New: `app.py` (has all latest fixes including S&P 500 fallback)

2. **Fixed Build Command**:
   - ❌ Old: `pip install streamlit pandas plotly yfinance numpy requests scikit-learn scipy --no-cache-dir`
   - ✅ New: `python -m pip install --upgrade pip setuptools wheel && python -m pip install -r requirements.txt`
   - Now uses `requirements.txt` which includes all dependencies (`bcrypt`, `PyJWT`, `pytz`, `ta`, etc.)

3. **Fixed Start Command**:
   - ❌ Old: `streamlit run app_final_enhanced.py ... --server.enableCORS false --server.enableXsrfProtection false`
   - ✅ New: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - Simplified and uses correct app file

4. **Updated Python Version**:
   - ❌ Old: `3.11.0`
   - ✅ New: `3.11.9`

5. **Added Backend Connection**:
   - ✅ Added: `API_BASE_URL = https://moneta-backend-api.onrender.com`
   - Now dashboard can connect to backend for ML predictions and API features

6. **Removed Unnecessary Variables**:
   - Removed `STREAMLIT_SERVER_ENABLE_CORS` and `STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION` (not needed)

## Next Steps to Apply Update:

### Option 1: Auto-Sync (If Blueprint Auto-Syncs)

1. **Commit and push** the updated `render_final_enhanced.yaml` to GitHub
2. Render should auto-detect the change
3. Wait for deployment to complete

### Option 2: Manual Update in Render Dashboard

1. Go to Render Dashboard → Blueprints
2. Find `render_final_enhanced.yaml`
3. Click **"Sync"** or **"Apply"** to pull latest changes
4. OR manually update the service:
   - Go to the `financial-analyzer-pro-final` service
   - **Settings** → **Build Command**: Update to new command
   - **Settings** → **Start Command**: Update to new command
   - **Environment** → Add `API_BASE_URL` = `https://moneta-backend-api.onrender.com`
   - **Environment** → Update `PYTHON_VERSION` = `3.11.9`
   - Click **"Save Changes"**
   - Service will redeploy

## Verification After Update:

1. **Check Service URL**: Visit the dashboard URL
2. **Verify S&P 500 Displays**: Should show even when rate-limited
3. **Test Backend Connection**: 
   - Try searching for a stock
   - Check if ML predictions work
   - Verify API features connect to backend

## If Backend Service Has Different URL:

If your backend service isn't `moneta-backend-api`, update the `API_BASE_URL` value to your actual backend URL.

## Expected Results:

After update:
- ✅ Dashboard uses latest `app.py` with all fixes
- ✅ All dependencies installed correctly
- ✅ Dashboard connects to backend API
- ✅ S&P 500 always displays (with fallback data)
- ✅ ML predictions work via backend API
- ✅ Python 3.11.9 (latest stable)





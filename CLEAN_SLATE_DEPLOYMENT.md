# Clean Slate Deployment - Option 2

This guide will help you delete all existing services and deploy fresh using the `render_final.yaml` blueprint.

## Step 1: Delete All Existing Services

### Delete Active Services:

1. Go to Render Dashboard: https://dashboard.render.com
2. For each of these services, click on it, then:
   - Click **"Settings"** (left sidebar)
   - Scroll down to the bottom
   - Click **"Delete Service"** (red button)
   - Confirm deletion

**Delete these services:**
- ✅ `financial-analyzer-pro-simple-yzfr`
- ✅ `financial-analyzer-pro-simple-bt4h`
- ✅ `financial-analyzer-pro-simple`
- ✅ `financial-analyzer-pro-simple-z6jp`
- ✅ `financial-analyzer-pro-1` (if still deploying, cancel first)

**Already Canceled (can delete for cleanliness):**
- `financial-analyzer-pro-3znv`
- `financial-analyzer-pro-ad6y`
- `financial-analyzer-pro-bt4h`
- `financial-analyzer-pro-simple-ad6y`
- `financial-analyzer-pro`

### Wait for Deletion
- Services will take a minute to fully delete
- Refresh the dashboard to confirm they're gone

## Step 2: Deploy from Blueprint

### Using Render Blueprint (Recommended):

1. **Go to Render Dashboard**
   - Click **"New"** button (top right)
   - Select **"Blueprint"**

2. **Connect Repository**
   - Select your GitHub repository
   - Or connect via GitHub if not already connected

3. **Configure Blueprint**
   - Render should auto-detect `render_final.yaml`
   - If not, manually select it
   - **Blueprint File**: `render_final.yaml`
   - **Branch**: `main` (or your default branch)

4. **Review Services**
   - Render will show it will create:
     - `moneta-backend-api` (Backend API)
     - `moneta-web-dashboard` (Web Dashboard)
   - The dashboard will automatically have `API_BASE_URL` set to `https://moneta-backend-api.onrender.com`

5. **Click "Apply"**
   - Render will create both services
   - This may take 10-15 minutes

6. **Monitor Deployment**
   - Watch the deployment logs for both services
   - Backend should show: `Uvicorn running on...`
   - Dashboard should show: `You can now view your Streamlit app`

## Step 3: Verify Deployment

### Check Backend Service:

1. **Service**: `moneta-backend-api`
2. **Visit**: `https://moneta-backend-api.onrender.com`
3. **Expected**: JSON response like:
   ```json
   {
     "message": "Financial Analyzer Pro API v2.0",
     "version": "2.0.0",
     ...
   }
   ```
4. **Test Health**: `https://moneta-backend-api.onrender.com/health`
   - Should return: `{"status": "ok"}`

### Check Dashboard Service:

1. **Service**: `moneta-web-dashboard`
2. **Visit**: `https://moneta-web-dashboard.onrender.com`
3. **Expected**: Streamlit web interface (NOT JSON!)
   - Should see sidebar, search bars, charts
   - Should be able to search for stocks

### Test Connection:

1. Go to the dashboard
2. Try searching for a stock (e.g., "AAPL")
3. The dashboard should connect to the backend API
4. Check browser console (F12) for any connection errors

## Step 4: Configure Environment Variables (If Needed)

After deployment, verify these environment variables:

### Backend Service (`moneta-backend-api`):
- `PYTHON_VERSION` = `3.11.9`
- `ENABLE_TIINGO` = `true`
- `ENABLE_ALPHA_VANTAGE` = `true`
- (API keys are optional - set if you have them)

### Dashboard Service (`moneta-web-dashboard`):
- `PYTHON_VERSION` = `3.11.9`
- `STREAMLIT_SERVER_HEADLESS` = `true`
- `STREAMLIT_SERVER_ADDRESS` = `0.0.0.0`
- `API_BASE_URL` = `https://moneta-backend-api.onrender.com` ⚠️ **VERIFY THIS!**

**To Check/Update:**
1. Click on `moneta-web-dashboard` service
2. Go to **"Environment"** tab
3. Verify `API_BASE_URL` is set correctly
4. If not, click **"Add Environment Variable"** and set it

## Troubleshooting

### Backend won't start:
- Check build logs for dependency errors
- Verify `requirements.txt` includes `bcrypt`, `PyJWT`, `pytz`, `ta`
- Check Python version is `3.11.9`

### Dashboard won't start:
- Check build logs for Streamlit installation
- Verify `app.py` exists in root directory
- Check Python version is `3.11.9`

### Dashboard can't connect to backend:
- Verify `API_BASE_URL` environment variable in dashboard service
- Test backend URL directly in browser
- Check CORS settings in `proxy.py` (should allow all origins)

### Services take too long to deploy:
- Free tier services can take 10-15 minutes
- Check build logs for progress
- First deployment always takes longer (installing dependencies)

## Success Checklist

After deployment, you should have:

- ✅ 2 services in Render dashboard:
  - `moneta-backend-api` (backend)
  - `moneta-web-dashboard` (frontend)
- ✅ Backend shows JSON at its URL
- ✅ Dashboard shows web interface at its URL
- ✅ Dashboard can connect to backend
- ✅ No duplicate services
- ✅ All old services deleted

## Next Steps

Once both services are running:
1. Update Android app to use backend URL: `https://moneta-backend-api.onrender.com`
2. Test end-to-end: Android app → Backend → Dashboard
3. Share your dashboard URL with users!





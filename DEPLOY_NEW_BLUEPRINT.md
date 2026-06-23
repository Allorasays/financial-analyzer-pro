# Deploy New Blueprint with Both Services

## Current Situation
- ✅ You have `render_final.yaml` ready with both backend and dashboard
- ❌ `render_final.yaml` is not in your Render blueprints yet
- ⚠️ Existing blueprints only have single services (not what we need)

## Solution: Create New Blueprint from `render_final.yaml`

### Option A: Create New Blueprint (Recommended)

1. **In Render Dashboard:**
   - Go to **"Blueprints"** section
   - Click **"New Blueprint"**

2. **Configure Blueprint:**
   - **Repository**: `Allorasays / financial-analyzer-pro`
   - **Branch**: `main`
   - **Blueprint File Path**: `render_final.yaml` ⚠️ **Important: Type this exactly**
   - **Name**: `moneta-complete-stack` (or any name you prefer)

3. **Save Blueprint**
   - Click **"Save"** or **"Create Blueprint"**

4. **Deploy Blueprint:**
   - Click on the new blueprint
   - Click **"Apply"** or **"Deploy"**
   - Render will create both services:
     - `moneta-backend-api`
     - `moneta-web-dashboard`

### Option B: Manual Deployment (If Blueprint Doesn't Work)

If creating a blueprint doesn't work, deploy manually:

#### Step 1: Deploy Backend

1. **Render Dashboard → "New" → "Web Service"**
2. **Connect Repository**: `Allorasays / financial-analyzer-pro`
3. **Configure:**
   - **Name**: `moneta-backend-api`
   - **Region**: `oregon` (or closest)
   - **Branch**: `main`
   - **Root Directory**: `/` (or leave empty)
   - **Environment**: `Python 3`
   - **Build Command**:
     ```
     python -m pip install --upgrade pip setuptools wheel && python -m pip install -r requirements.txt
     ```
   - **Start Command**:
     ```
     python -m uvicorn proxy:app --host 0.0.0.0 --port $PORT
     ```
4. **Environment Variables:**
   - `PYTHON_VERSION` = `3.11.9`
   - `ENABLE_TIINGO` = `true`
   - `ENABLE_ALPHA_VANTAGE` = `true`
5. **Click "Create Web Service"**
6. **Wait for deployment** (copy the URL when ready, e.g., `https://moneta-backend-api.onrender.com`)

#### Step 2: Deploy Dashboard

1. **Render Dashboard → "New" → "Web Service"**
2. **Connect Repository**: Same repository
3. **Configure:**
   - **Name**: `moneta-web-dashboard`
   - **Region**: Same as backend
   - **Branch**: `main`
   - **Root Directory**: `/` (or leave empty)
   - **Environment**: `Python 3`
   - **Build Command**:
     ```
     python -m pip install --upgrade pip setuptools wheel && python -m pip install -r requirements.txt
     ```
   - **Start Command**:
     ```
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
     ```
4. **Environment Variables:**
   - `PYTHON_VERSION` = `3.11.9`
   - `STREAMLIT_SERVER_HEADLESS` = `true`
   - `STREAMLIT_SERVER_ADDRESS` = `0.0.0.0`
   - `API_BASE_URL` = `https://moneta-backend-api.onrender.com` ⚠️ **Use actual backend URL from Step 1**
5. **Click "Create Web Service"**
6. **Wait for deployment**

## Verification Steps

### After Deployment:

1. **Check Backend:**
   - URL: `https://moneta-backend-api.onrender.com`
   - Should show: JSON API information
   - Test: `https://moneta-backend-api.onrender.com/health` → `{"status": "ok"}`

2. **Check Dashboard:**
   - URL: `https://moneta-web-dashboard.onrender.com`
   - Should show: Streamlit web interface (NOT JSON!)
   - Test: Try searching for "AAPL" - should connect to backend

3. **Verify Connection:**
   - Go to dashboard service
   - **Settings** → **Environment**
   - Verify `API_BASE_URL` points to backend URL

## Troubleshooting

### Blueprint file not found:
- Verify `render_final.yaml` is in the root of your repository
- Check it's committed and pushed to `main` branch
- Try full path: `/render_final.yaml`

### Services fail to start:
- Check build logs for errors
- Verify `requirements.txt` has all dependencies
- Check Python version is `3.11.9`

### Dashboard can't connect:
- Verify `API_BASE_URL` environment variable
- Test backend URL directly in browser
- Check backend logs for CORS errors

## Success Criteria

✅ 2 new services created:
- `moneta-backend-api` (shows JSON)
- `moneta-web-dashboard` (shows web UI)

✅ Dashboard connects to backend
✅ Both services running and healthy
✅ Old duplicate services deleted










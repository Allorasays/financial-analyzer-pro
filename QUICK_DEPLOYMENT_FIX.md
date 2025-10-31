# Quick Fix: Deploy Web Dashboard

## Current Situation
- ✅ **Backend Service**: `financial-analyzer-pro-simple-yzf` (already deployed, showing JSON)
- ❌ **Dashboard Service**: Not deployed yet

## Solution: Deploy Dashboard Service

### Step 1: Get Your Backend URL
Your backend service URL is: `https://financial-analyzer-pro-simple-yzf.onrender.com`

**Verify it's working:**
- Visit: `https://financial-analyzer-pro-simple-yzf.onrender.com`
- Should see: JSON with API information
- Test: `https://financial-analyzer-pro-simple-yzf.onrender.com/health` → Should return `{"status": "ok"}`

### Step 2: Deploy Dashboard Service

**Option A: Manual Deployment (Fastest)**

1. Go to Render Dashboard → **"New"** → **"Web Service"**
2. Connect your GitHub repository
3. Configure:
   - **Name**: `moneta-web-dashboard`
   - **Region**: Same as backend (oregon)
   - **Branch**: `main` (or your default branch)
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
4. **Environment Variables** (click "Add Environment Variable" for each):
   - `PYTHON_VERSION` = `3.11.9`
   - `STREAMLIT_SERVER_HEADLESS` = `true`
   - `STREAMLIT_SERVER_ADDRESS` = `0.0.0.0`
   - `API_BASE_URL` = `https://financial-analyzer-pro-simple-yzf.onrender.com` ⚠️ **IMPORTANT: Use your actual backend URL**
5. Click **"Create Web Service"**
6. Wait for deployment (5-10 minutes)

**Option B: Blueprint Deployment**

1. In Render Dashboard → **"New"** → **"Blueprint"**
2. Connect GitHub repository
3. Render will auto-detect `render_final.yaml`
4. It will create both services (or just the dashboard if backend exists)
5. After deployment, update the `API_BASE_URL` environment variable in the dashboard service to point to your existing backend

### Step 3: Verify Dashboard

After deployment:
- Visit: `https://moneta-web-dashboard.onrender.com` (or whatever name you chose)
- Should see: **Streamlit web interface** (not JSON!)
- The dashboard will automatically connect to your backend at `https://financial-analyzer-pro-simple-yzf.onrender.com`

### Step 4: Test Connection

1. Go to the dashboard
2. Try to analyze a stock (e.g., "AAPL")
3. Check browser console (F12) for any API connection errors
4. If errors occur, verify `API_BASE_URL` environment variable is set correctly

## Troubleshooting

### Dashboard shows JSON instead of web UI
- Check the Start Command: Should be `streamlit run app.py...`
- Check logs: Look for Streamlit startup messages

### Dashboard can't connect to backend
- Verify `API_BASE_URL` environment variable matches backend URL exactly
- Test backend URL directly: Visit `https://financial-analyzer-pro-simple-yzf.onrender.com/health`
- Check backend logs for errors

### Build fails
- Check `requirements.txt` is in the root directory
- Verify Python version is `3.11.9` or compatible
- Check build logs for specific error messages


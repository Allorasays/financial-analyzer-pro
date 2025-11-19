# Service Cleanup and Configuration Guide

## Current Situation
You have **5 active services** deployed, which is too many. We need to identify which are:
- ✅ **Backend API** (shows JSON when visited)
- ✅ **Web Dashboard** (shows Streamlit web interface)
- ❌ **Duplicates** (to be deleted)

## Step 1: Identify Each Service Type

### How to Check Each Service:

1. **Visit each service URL** in your browser:
   - `https://financial-analyzer-pro-simple-yzfr.onrender.com`
   - `https://financial-analyzer-pro-simple-bt4h.onrender.com`
   - `https://financial-analyzer-pro-simple.onrender.com`
   - `https://financial-analyzer-pro-simple-z6jp.onrender.com`

2. **What you'll see:**
   - **Backend API**: JSON response like `{"message":"Financial Analyzer Pro API v2.0",...}`
   - **Web Dashboard**: Streamlit interface with sidebar, search bars, charts (no JSON)

3. **Or check in Render Dashboard:**
   - Click each service → **"Logs"** tab
   - Look for startup message:
     - Backend: `Uvicorn running on...` or `Started server process`
     - Dashboard: `You can now view your Streamlit app` or `Network URL:`

## Step 2: Recommended Cleanup Plan

### Keep Only 2 Services:
1. **One Backend API** - Choose the newest/most stable one
2. **One Web Dashboard** - Deploy fresh using `render_final.yaml` config

### Services to Delete (after confirming):

**Likely Duplicates (keep only 1 backend):**
- `financial-analyzer-pro-simple-yzfr` (27min old - newest)
- `financial-analyzer-pro-simple-bt4h` (30min old)
- `financial-analyzer-pro-simple` (8h old)
- `financial-analyzer-pro-simple-z6jp` (8h old)

**Delete:**
- `financial-analyzer-pro-1` (Docker, deploying - cancel if not needed)
- All canceled services (already inactive, but delete for cleanliness)

## Step 3: Set Up Proper Configuration

### Option A: Keep Existing Backend + Deploy New Dashboard

1. **Identify your best backend service:**
   - Visit each URL
   - Choose the one that shows JSON and works reliably
   - Note the URL (e.g., `https://financial-analyzer-pro-simple-yzfr.onrender.com`)

2. **Deploy Dashboard Service:**
   - Render Dashboard → **"New"** → **"Web Service"**
   - Use `render_final.yaml` configuration OR manually set:
     - **Name**: `moneta-web-dashboard`
     - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
     - **Environment Variable**: `API_BASE_URL` = `https://[your-backend-url].onrender.com`

3. **Delete duplicate services:**
   - Render Dashboard → Click each duplicate service
   - Click **"Settings"** → Scroll down → **"Delete Service"**

### Option B: Clean Slate (Recommended)

1. **Delete ALL existing services** (except maybe keep 1 backend for testing)

2. **Deploy from `render_final.yaml` Blueprint:**
   - Render Dashboard → **"New"** → **"Blueprint"**
   - Connect GitHub repository
   - Render will create:
     - `moneta-backend-api` (backend)
     - `moneta-web-dashboard` (dashboard, connects to backend automatically)

3. **Update `API_BASE_URL` in dashboard service:**
   - After deployment, go to `moneta-web-dashboard` service
   - **Settings** → **Environment** → Add `API_BASE_URL` = `https://moneta-backend-api.onrender.com`

## Step 4: Verification

After cleanup, you should have:

1. **Backend API** (`moneta-backend-api` or kept service):
   - URL: `https://[backend-url].onrender.com`
   - Visit: Shows JSON API info
   - Test: `https://[backend-url].onrender.com/health` → `{"status": "ok"}`

2. **Web Dashboard** (`moneta-web-dashboard`):
   - URL: `https://[dashboard-url].onrender.com`
   - Visit: Shows Streamlit web interface
   - Test: Try searching for a stock (e.g., "AAPL")

## Quick Action Items

**Immediate Actions:**
1. ✅ Visit each service URL to identify backend vs dashboard
2. ✅ Choose 1 backend service to keep
3. ✅ Delete duplicate services
4. ✅ Deploy dashboard using `render_final.yaml` OR manually
5. ✅ Configure `API_BASE_URL` environment variable

**Help Needed?**
- Share which services show JSON vs web UI
- I can help you identify and clean up specific services




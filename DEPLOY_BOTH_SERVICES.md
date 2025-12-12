# Deploying MONETA Financial Analyzer to Render

## Current Status
- Backend API service: **NOT DEPLOYED** ❌
- Web Dashboard service: **NOT DEPLOYED** ❌

## Deployment Steps

### Method 1: Blueprint Deployment (Recommended)

1. Go to Render Dashboard: https://dashboard.render.com
2. Click **"New"** → **"Blueprint"**
3. Connect your GitHub repository
4. Select **`render_final.yaml`** as the blueprint file
5. Render will automatically create both services:
   - `moneta-backend-api` (FastAPI backend)
   - `moneta-web-dashboard` (Streamlit frontend)

### Method 2: Manual Service Creation

#### Step 1: Deploy Backend API

1. Go to Render Dashboard → **"New"** → **"Web Service"**
2. Connect your GitHub repository
3. Configure:
   - **Name**: `moneta-backend-api`
   - **Region**: Choose closest to you
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty (or `/` if required)
   - **Environment**: `Python 3`
   - **Build Command**: 
     ```
     python -m pip install --upgrade pip setuptools wheel && python -m pip install -r requirements.txt
     ```
   - **Start Command**: 
     ```
     python -m uvicorn proxy:app --host 0.0.0.0 --port $PORT
     ```
4. **Environment Variables**:
   - `PYTHON_VERSION` = `3.11.9`
   - `ENABLE_TIINGO` = `true`
   - `ENABLE_ALPHA_VANTAGE` = `true`
   - `TIINGO_API_KEY` = (your key, if you have one)
   - `ALPHAVANTAGE_API_KEY` = (your key, if you have one)
   - `NEWSAPI_KEY` = (your key, if you have one)
   - `FRED_API_KEY` = (your key, if you have one)
5. Click **"Create Web Service"**
6. Wait for deployment to complete
7. **Copy the service URL** (e.g., `https://moneta-backend-api-xyz.onrender.com`)

#### Step 2: Deploy Web Dashboard

1. Go to Render Dashboard → **"New"** → **"Web Service"**
2. Connect the same GitHub repository
3. Configure:
   - **Name**: `moneta-web-dashboard`
   - **Region**: Same as backend
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty
   - **Environment**: `Python 3`
   - **Build Command**: 
     ```
     python -m pip install --upgrade pip setuptools wheel && python -m pip install -r requirements.txt
     ```
   - **Start Command**: 
     ```
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
     ```
4. **Environment Variables**:
   - `PYTHON_VERSION` = `3.11.9`
   - `STREAMLIT_SERVER_HEADLESS` = `true`
   - `STREAMLIT_SERVER_ADDRESS` = `0.0.0.0`
   - `API_BASE_URL` = `https://[YOUR-BACKEND-SERVICE-URL].onrender.com` ⚠️ **USE THE ACTUAL URL FROM STEP 1**
5. Click **"Create Web Service"**
6. Wait for deployment to complete

## Verification

After both services are deployed:

1. **Backend API**:
   - Visit: `https://[backend-url].onrender.com`
   - Should see: JSON with API information
   - Test: `https://[backend-url].onrender.com/health` → `{"status": "ok"}`

2. **Web Dashboard**:
   - Visit: `https://[dashboard-url].onrender.com`
   - Should see: Streamlit web interface (not JSON)
   - The dashboard will connect to the backend automatically using `API_BASE_URL`

## Troubleshooting

### Backend shows "No module named 'bcrypt'"
✅ **Fixed** - Added to `requirements.txt`

### Backend shows "uvicorn: command not found"
✅ **Fixed** - Using `python -m uvicorn` in start command

### Dashboard can't connect to backend
- Check `API_BASE_URL` environment variable matches backend URL exactly
- Verify backend service is running (check logs)
- Test backend URL directly in browser

### Services don't appear in Render
- Make sure you're using the correct GitHub repository
- Check that `render_final.yaml` is in the root directory
- Verify you have permission to create services in Render





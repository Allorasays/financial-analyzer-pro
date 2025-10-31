# 🚀 MONETA Production Deployment Instructions

## Week 1 Foundation - Deployment to Render.com

### ✅ Completed Steps 1-3
1. ✅ **Generated PNG assets** from SVG sources
2. ✅ **Built Android app** (debug APK created successfully)
3. ✅ **React Native setup** (dependencies installed, assets generated)

### 📋 Step 4: Deploy to Render.com

#### Option A: Using Render Dashboard (Recommended)

1. **Prepare GitHub Repository**
   - Ensure all code is committed and pushed
   - Create `.gitignore` if not present

2. **Login to Render.com**
   - Go to https://render.com
   - Sign up or login with GitHub

3. **Deploy Backend Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository
   - Configure:
     - **Name**: `moneta-backend-api`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn proxy:app --host 0.0.0.0 --port $PORT`
     - **Plan**: Free
   - Add Environment Variables:
     ```
     PYTHON_VERSION=3.11.0
     TIINGO_API_KEY=your_key_here
     ALPHAVANTAGE_API_KEY=your_key_here
     NEWSAPI_KEY=your_key_here
     FRED_API_KEY=your_key_here
     ENABLE_TIINGO=true
     ENABLE_ALPHA_VANTAGE=true
     ```
   - Click "Create Web Service"

4. **Deploy Web Dashboard**
   - Repeat "New +" → "Web Service"
   - Configure:
     - **Name**: `moneta-web-dashboard`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
     - **Plan**: Free
   - Add Environment Variables:
     ```
     PYTHON_VERSION=3.11.0
     STREAMLIT_SERVER_HEADLESS=true
     STREAMLIT_SERVER_ADDRESS=0.0.0.0
     BACKEND_URL=https://moneta-backend-api.onrender.com
     ```
   - Click "Create Web Service"

#### Option B: Using render_production.yaml (Blue/Green)

1. Use the existing Blueprint file:
   ```yaml
   # See: render_production.yaml
   ```
2. In Render Dashboard:
   - Click "New +" → "Blueprint"
   - Paste content from `render_production.yaml`
   - Connect your GitHub repository
   - **Add API keys** in the environment variables section
   - Click "Apply"

### 🔍 Post-Deployment Validation

1. **Check Backend Health**
   ```bash
   curl https://moneta-backend-api.onrender.com/health
   ```
   Expected: `{"status":"ok"}`

2. **Check API Status**
   ```bash
   curl https://moneta-backend-api.onrender.com/api/system/status
   ```
   Expected: JSON with service status

3. **Test Streamlit Dashboard**
   - Visit: `https://moneta-web-dashboard.onrender.com`
   - Verify it loads and connects to backend

### 📊 Monitoring

**Built-in Dashboard:**
- Deploy `monitoring_dashboard.py` as a third Render service
- Or run locally pointing to prod URL:
  ```bash
  export API_BASE=https://moneta-backend-api.onrender.com
  streamlit run monitoring_dashboard.py
  ```

### 🐛 Troubleshooting

**Build Fails:**
- Check Python version is 3.11.0
- Verify requirements.txt is complete
- Review build logs in Render dashboard

**Service Crashes:**
- Check logs in Render dashboard
- Verify environment variables are set
- Test API keys manually

**Timeout Issues:**
- Free tier has 750 hours/month limit
- Services sleep after 15 minutes inactivity
- First request after sleep takes 30+ seconds

### 🔐 Security Notes

- Never commit API keys to GitHub
- Use Render's encrypted environment variables
- Enable HTTPS (automatic on Render)
- Review CORS settings in `proxy.py`

### 📈 Next Steps After Deployment

1. Update Android app with production API URL
2. Update React Native app config
3. Test end-to-end from mobile apps
4. Generate release APK/AAB for Play Store
5. Prepare Play Store submission materials

---

**Status**: Week 1 Foundation complete! 🎉



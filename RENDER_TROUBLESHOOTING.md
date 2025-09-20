# 🔧 Render Deployment Troubleshooting Guide

## 🚨 Common Issues & Solutions

### Issue 1: App Won't Load/Start
**Symptoms:** Blank page, timeout, or "Service Unavailable"

**Solutions:**
1. **Use Simplified Version**
   - Deploy `app_render_simple.py` instead of the complex version
   - Use `requirements_render_simple.txt`
   - Use `render_simple.yaml` configuration

2. **Check Start Command**
   ```bash
   streamlit run app_render_simple.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --server.fileWatcherType none
   ```

3. **Verify Environment Variables**
   - `PYTHON_VERSION=3.11.0`
   - `STREAMLIT_SERVER_HEADLESS=true`
   - `STREAMLIT_SERVER_ADDRESS=0.0.0.0`

### Issue 2: Build Fails
**Symptoms:** Build process fails during dependency installation

**Solutions:**
1. **Use Minimal Requirements**
   ```txt
   streamlit>=1.28.0
   pandas>=2.0.0
   plotly>=5.0.0
   yfinance>=0.2.0
   numpy>=1.24.0
   scikit-learn>=1.3.0
   ```

2. **Check Build Command**
   ```bash
   pip install --upgrade pip && pip install -r requirements_render_simple.txt --no-cache-dir
   ```

3. **Remove Problematic Dependencies**
   - Remove `scipy` if causing issues
   - Remove `sqlite3` (built-in)
   - Remove `threading` (built-in)

### Issue 3: Memory Issues
**Symptoms:** App crashes or runs out of memory

**Solutions:**
1. **Use Simplified App**
   - `app_render_simple.py` uses less memory
   - Removed complex caching system
   - Simplified data processing

2. **Optimize Cache Size**
   - Reduced cache size to 50 items
   - Simple cache implementation

### Issue 4: Import Errors
**Symptoms:** ModuleNotFoundError or ImportError

**Solutions:**
1. **Test Imports Locally**
   ```bash
   python -c "import streamlit, pandas, plotly, yfinance, numpy, sklearn"
   ```

2. **Use Only Essential Imports**
   - Removed complex imports
   - Added graceful fallbacks
   - Simplified error handling

## 🚀 Recommended Deployment Steps

### Step 1: Use Simplified Version
1. **Files to Use:**
   - `app_render_simple.py` - Simplified app
   - `requirements_render_simple.txt` - Minimal dependencies
   - `render_simple.yaml` - Simple configuration
   - `Procfile_simple` - Simple process file

### Step 2: Deploy on Render
1. Go to [Render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Use these settings:

**Service Configuration:**
- **Name**: `financial-analyzer-pro-simple`
- **Environment**: `Python`
- **Build Command**: 
  ```bash
  pip install --upgrade pip && pip install -r requirements_render_simple.txt --no-cache-dir
  ```
- **Start Command**:
  ```bash
  streamlit run app_render_simple.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --server.fileWatcherType none
  ```

**Environment Variables:**
```
PYTHON_VERSION=3.11.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
```

### Step 3: Test Deployment
1. Wait for build to complete
2. Check if app loads at the provided URL
3. Test basic functionality (enter AAPL and click "Analyze Stock")

## 🔍 Debugging Steps

### 1. Check Render Logs
- Go to your service dashboard
- Click on "Logs" tab
- Look for error messages

### 2. Test Locally First
```bash
# Test the simplified app locally
streamlit run app_render_simple.py
```

### 3. Verify Dependencies
```bash
# Test all imports
python -c "import streamlit, pandas, plotly, yfinance, numpy, sklearn; print('All imports successful')"
```

### 4. Check File Structure
Ensure these files are in your repository:
- `app_render_simple.py`
- `requirements_render_simple.txt`
- `render_simple.yaml` (optional)
- `Procfile_simple` (rename to `Procfile`)

## ✅ Success Indicators

### Build Success
- ✅ Build completes without errors
- ✅ All dependencies installed
- ✅ No import errors

### App Success
- ✅ App starts successfully
- ✅ Health check passes
- ✅ App loads in browser
- ✅ Basic functionality works

### Feature Success
- ✅ Stock analysis works
- ✅ Charts display correctly
- ✅ ML predictions work
- ✅ Cache functions properly

## 🆘 Emergency Fallback

If the simplified version still doesn't work:

### Ultra-Minimal Version
1. **Create `app_minimal.py`:**
   ```python
   import streamlit as st
   import pandas as pd
   import yfinance as yf
   
   st.title("Financial Analyzer Pro")
   st.write("Enter a stock symbol to analyze")
   
   symbol = st.text_input("Stock Symbol", "AAPL")
   if st.button("Analyze"):
       try:
           ticker = yf.Ticker(symbol)
           data = ticker.history(period="1mo")
           if not data.empty:
               st.write(f"Current Price: ${data['Close'].iloc[-1]:.2f}")
               st.line_chart(data['Close'])
           else:
               st.error("No data found")
       except Exception as e:
           st.error(f"Error: {str(e)}")
   ```

2. **Create `requirements_minimal.txt`:**
   ```txt
   streamlit
   pandas
   yfinance
   ```

3. **Deploy with minimal configuration**

## 📞 Support

If you're still having issues:
1. Check Render logs for specific errors
2. Test locally first
3. Use the ultra-minimal version as fallback
4. Contact Render support if needed

## 🎉 Expected Result

Once deployed successfully, you should see:
- Financial Analyzer Pro interface
- Stock symbol input field
- Analysis button
- Working stock analysis with charts
- ML predictions (if sklearn is available)

Your app should be accessible at: `https://your-app-name.onrender.com`




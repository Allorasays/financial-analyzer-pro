# Step-by-Step Guide: Deploy Render Blueprint

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ Render account (free tier works)
- ✅ GitHub repository with your code
- ✅ `render_final.yaml` file in repository root
- ✅ All code committed and pushed to GitHub

---

## 🚀 Step-by-Step Deployment

### **Step 1: Prepare Your Repository**

1. **Verify files are in repository**:
   - `render_final.yaml` ✅ (in root directory)
   - `proxy.py` ✅ (backend code)
   - `app.py` ✅ (Streamlit dashboard)
   - `requirements.txt` ✅ (Python dependencies)
   - `api_documentation.html` ✅ (optional, for docs page)

2. **Commit and push to GitHub**:
   ```bash
   git add render_final.yaml proxy.py app.py requirements.txt
   git commit -m "Deploy MONETA Financial Analyzer blueprint"
   git push origin main
   ```

---

### **Step 2: Access Render Dashboard**

1. **Go to Render Dashboard**:
   - Visit: https://dashboard.render.com
   - Log in to your account

2. **Navigate to Blueprints**:
   - Click "Blueprints" in the left sidebar
   - Or go directly to: https://dashboard.render.com/blueprints

---

### **Step 3: Create New Blueprint**

1. **Click "New Blueprint"**:
   - Button is usually in the top right
   - Or click "New" → "Blueprint"

2. **Connect Repository**:
   - Select your GitHub repository
   - Choose the branch (usually `main` or `master`)
   - Render will detect `render_final.yaml` automatically

3. **Review Detected Services**:
   - Render should detect 2 services from the blueprint:
     - `moneta-backend-api` (FastAPI backend)
     - `moneta-web-dashboard` (Streamlit dashboard)
   - Verify both services are listed

4. **Click "Apply"** or "Deploy":
   - This will create both services
   - Deployment will start automatically

---

### **Step 4: Configure Environment Variables**

After services are created, configure environment variables:

#### **For `moneta-backend-api` Service**:

1. **Go to Service Settings**:
   - Click on `moneta-backend-api` service
   - Go to "Environment" tab

2. **Add/Verify Environment Variables**:
   - `PYTHON_VERSION` = `3.11.9` ✅ (should be auto-set)
   - `SECRET_KEY` = (auto-generated) ✅
   - `ENABLE_TIINGO` = `true` ✅
   - `ENABLE_ALPHA_VANTAGE` = `true` ✅
   - `TIINGO_API_KEY` = (your key, if you have one)
   - `ALPHAVANTAGE_API_KEY` = (your key, if you have one)
   - `NEWSAPI_KEY` = (your key, if you have one)
   - `FRED_API_KEY` = (your key, if you have one)

3. **Save Changes**

#### **For `moneta-web-dashboard` Service**:

1. **Go to Service Settings**:
   - Click on `moneta-web-dashboard` service
   - Go to "Environment" tab

2. **Add/Verify Environment Variables**:
   - `PYTHON_VERSION` = `3.11.9` ✅ (should be auto-set)
   - `STREAMLIT_SERVER_HEADLESS` = `true` ✅
   - `STREAMLIT_SERVER_ADDRESS` = `0.0.0.0` ✅
   - `API_BASE_URL` = `https://moneta-backend-api.onrender.com` ✅

3. **Save Changes**

---

### **Step 5: Monitor Deployment**

1. **Watch Build Logs**:
   - Click on each service
   - Go to "Logs" tab
   - Watch for build progress

2. **Expected Build Steps**:
   ```
   ==> Cloning from GitHub...
   ==> Building...
   ==> Installing dependencies...
   ==> Starting service...
   ```

3. **Look for Success Messages**:
   - ✅ "Build successful"
   - ✅ "Service is live"
   - ✅ "Application is running"

4. **Watch for Errors**:
   - ❌ If you see errors, check the logs
   - Common issues:
     - Missing dependencies (check `requirements.txt`)
     - Python version issues
     - Port binding errors

---

### **Step 6: Verify Deployment**

#### **Test Backend Service**:

1. **Get Service URL**:
   - Backend URL: `https://moneta-backend-api.onrender.com`
   - (Render will show the actual URL in dashboard)

2. **Test Health Endpoint**:
   ```bash
   curl https://moneta-backend-api.onrender.com/health
   ```
   Expected: `{"status":"ok"}`

3. **Test Root Endpoint**:
   - Visit: `https://moneta-backend-api.onrender.com/`
   - Should show JSON with API information

4. **Test Documentation**:
   - Visit: `https://moneta-backend-api.onrender.com/docs`
   - Should show FastAPI interactive docs

#### **Test Streamlit Dashboard**:

1. **Get Service URL**:
   - Dashboard URL: `https://moneta-web-dashboard.onrender.com`
   - (Render will show the actual URL in dashboard)

2. **Visit Dashboard**:
   - Open URL in browser
   - Should see "Financial Analyzer Pro" header
   - Should load without errors

3. **Test Backend Connection**:
   - Try using a feature (e.g., Stock Analysis)
   - Should connect to backend API
   - Should display data

---

### **Step 7: Verify Android App Compatibility**

1. **Test Android Endpoints**:
   ```bash
   # Test all 12 Android endpoints
   python test_all_android_endpoints.py
   ```

2. **Expected Results**:
   - All endpoints should return 200 OK
   - If 404 errors, service may be sleeping (wait 30-60 seconds)

3. **Update Android App** (if needed):
   - Verify `RetrofitClient.kt` points to: `https://moneta-backend-api.onrender.com/`
   - Rebuild Android app
   - Test with production backend

---

## 🔧 Troubleshooting

### **Issue: Service Not Starting**

**Symptoms**: Service shows "Failed" or "Crashed"

**Solutions**:
1. Check build logs for errors
2. Verify `requirements.txt` has all dependencies
3. Check Python version matches (3.11.9)
4. Verify start command is correct

### **Issue: 404 Errors**

**Symptoms**: Endpoints return 404

**Solutions**:
1. Service may be sleeping (free tier) - wait 30-60 seconds
2. Verify service is actually running (check status)
3. Check service URL is correct

### **Issue: Streamlit Can't Connect to Backend**

**Symptoms**: Streamlit shows connection errors

**Solutions**:
1. Verify `API_BASE_URL` environment variable is set
2. Check backend service is running
3. Verify backend URL is correct (no trailing slash issues)

### **Issue: Build Fails**

**Symptoms**: Build logs show errors

**Solutions**:
1. Check `requirements.txt` for missing packages
2. Verify Python version compatibility
3. Check for syntax errors in code
4. Review build logs for specific error messages

---

## ✅ Deployment Checklist

After deployment, verify:

- [ ] Backend service (`moneta-backend-api`) is running
- [ ] Backend health endpoint works: `/health`
- [ ] Backend root endpoint works: `/`
- [ ] Backend docs work: `/docs`
- [ ] Streamlit service (`moneta-web-dashboard`) is running
- [ ] Streamlit dashboard loads without errors
- [ ] Streamlit connects to backend
- [ ] All environment variables are set
- [ ] Android app endpoints work (test with script)
- [ ] No errors in service logs

---

## 📝 Post-Deployment

### **1. Update Android App** (if needed):
- Verify `RetrofitClient.kt` uses correct backend URL
- Test Android app with production backend

### **2. Monitor Services**:
- Check service logs regularly
- Monitor for errors
- Watch for rate limiting issues

### **3. Set Up Auto-Deploy** (optional):
- In Render dashboard, enable "Auto-Deploy"
- Services will update automatically on git push

---

## 🎯 Summary

**Deployment Steps**:
1. ✅ Prepare repository (commit and push)
2. ✅ Create blueprint in Render
3. ✅ Configure environment variables
4. ✅ Monitor deployment
5. ✅ Verify services are running
6. ✅ Test endpoints
7. ✅ Update Android app (if needed)

**Expected Result**:
- ✅ Backend API running at `https://moneta-backend-api.onrender.com`
- ✅ Streamlit dashboard running at `https://moneta-web-dashboard.onrender.com`
- ✅ All Android app endpoints working
- ✅ Services connected and communicating

---

## 🆘 Need Help?

If you encounter issues:
1. Check service logs in Render dashboard
2. Verify all environment variables are set
3. Test endpoints individually
4. Check for service sleeping (free tier limitation)

The blueprint is correctly configured - deployment should be straightforward!





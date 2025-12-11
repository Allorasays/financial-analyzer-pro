# 🚀 Quick Deployment Steps

## ✅ **Files Committed**

All key files for deployment have been committed:
- ✅ `render_final.yaml` - Render blueprint with cron job
- ✅ `proxy.py` - Backend with prediction tracking
- ✅ `requirements.txt` - Dependencies updated
- ✅ Prediction tracking system
- ✅ Daily prediction jobs
- ✅ All new ML feature modules

---

## 📋 **Deployment Steps**

### **1. Push to GitHub** (if you have a remote)
```bash
git push origin complete-app-restoration
```

### **2. Deploy on Render**

#### **Option A: Using Blueprint (Recommended)**

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Click**: "New +" → "Blueprint"
3. **Connect Repository**: Select your GitHub repo
4. **Blueprint File**: Select `render_final.yaml`
5. **Review Services**:
   - `moneta-backend-api` (Backend API)
   - `moneta-web-dashboard` (Web Dashboard)
   - `daily-predictions-validation` (Cron Job - NEW)
6. **Set Environment Variables** in Render dashboard:
   - `TIINGO_API_KEY` = (your key)
   - `ALPHAVANTAGE_API_KEY` = (your key)
   - `NEWSAPI_KEY` = (your key) - optional
   - `FRED_API_KEY` = (your key)
7. **Click**: "Apply" to deploy all services

#### **Option B: Manual Deploy (If services already exist)**

1. **Update Backend API**:
   - Go to your `moneta-backend-api` service
   - Click "Manual Deploy" → "Deploy latest commit"

2. **Update Web Dashboard**:
   - Go to your `moneta-web-dashboard` service
   - Click "Manual Deploy" → "Deploy latest commit"

3. **Create Cron Job** (NEW):
   - Click "New +" → "Cron Job"
   - Name: `daily-predictions-validation`
   - Schedule: `0 23 * * *` (11 PM UTC daily)
   - Build: `python -m pip install --upgrade pip setuptools wheel && python -m pip install -r requirements.txt`
   - Run: `python combined_daily_job.py`
   - Environment:
     - `PYTHON_VERSION` = `3.11.9`
     - `API_BASE_URL` = `https://moneta-backend-api.onrender.com`
     - `RENDER` = `true`

---

## ✅ **Post-Deployment Verification**

### **1. Check Backend Health**
```bash
curl https://moneta-backend-api.onrender.com/health
# Expected: {"status":"ok"}
```

### **2. Test ML Predictions**
```bash
curl https://moneta-backend-api.onrender.com/api/ml/predictions/AAPL
# Should return prediction data with all new features
```

### **3. Check Prediction Tracking**
```bash
# View pending predictions
curl https://moneta-backend-api.onrender.com/api/prediction-pending

# View accuracy metrics (after validations)
curl https://moneta-backend-api.onrender.com/api/prediction-accuracy
```

### **4. Verify Cron Job**
- Go to Render Dashboard → Cron Jobs
- Check `daily-predictions-validation` exists
- Wait for 11 PM UTC or trigger manually to test
- Check logs to see predictions being made

---

## 🎯 **What Happens After Deployment**

1. **Backend API** - Serves predictions with all 112 features
2. **Web Dashboard** - Streamlit interface accessible
3. **Cron Job** - Automatically runs daily at 11 PM UTC:
   - Makes 10 predictions per day
   - Validates pending predictions
   - Tracks accuracy

---

## ⚠️ **Important Notes**

- **API Keys**: Make sure all API keys are set in Render dashboard
- **First Cron Run**: Will happen at next 11 PM UTC
- **Predictions**: Start accumulating immediately
- **Validations**: Begin 1 day after first predictions
- **Accuracy Data**: Available after 1-2 days of validations

---

## 🔍 **Monitor Deployment**

1. **Check Logs**: Render Dashboard → Service → Logs
2. **Check Status**: All services should show "Live"
3. **Test Endpoints**: Use curl commands above
4. **Watch First Cron**: After 11 PM UTC, check cron job logs

---

## ✅ **Deployment Complete!**

Once deployed, the system will:
- ✅ Make 10 predictions every day automatically
- ✅ Track predictions against actual outcomes
- ✅ Calculate real-world accuracy metrics
- ✅ Provide accuracy data via API

**Your deployment is ready to go!** 🚀


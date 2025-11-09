# Quick Deployment Checklist

## Before Deployment

- [ ] Code committed to GitHub
- [ ] `render_final.yaml` in repository root
- [ ] `requirements.txt` has all dependencies
- [ ] `proxy.py` and `app.py` are up to date

## Deployment Steps

### 1. Render Dashboard
- [ ] Go to https://dashboard.render.com
- [ ] Click "Blueprints" → "New Blueprint"
- [ ] Select your GitHub repository
- [ ] Review detected services (should show 2)
- [ ] Click "Apply" to deploy

### 2. Environment Variables

**Backend (`moneta-backend-api`)**:
- [ ] `PYTHON_VERSION` = `3.11.9`
- [ ] `SECRET_KEY` = (auto-generated)
- [ ] `ENABLE_TIINGO` = `true`
- [ ] `ENABLE_ALPHA_VANTAGE` = `true`
- [ ] API keys (if you have them)

**Dashboard (`moneta-web-dashboard`)**:
- [ ] `PYTHON_VERSION` = `3.11.9`
- [ ] `STREAMLIT_SERVER_HEADLESS` = `true`
- [ ] `STREAMLIT_SERVER_ADDRESS` = `0.0.0.0`
- [ ] `API_BASE_URL` = `https://moneta-backend-api.onrender.com`

### 3. Verify Deployment

- [ ] Backend service shows "Live"
- [ ] Dashboard service shows "Live"
- [ ] Test: `https://moneta-backend-api.onrender.com/health`
- [ ] Test: Visit dashboard URL
- [ ] Check logs for errors

### 4. Test Endpoints

- [ ] Run: `python test_all_android_endpoints.py`
- [ ] All endpoints return 200 OK
- [ ] Android app can connect

## Done! ✅

Your services should now be running and ready for production use.



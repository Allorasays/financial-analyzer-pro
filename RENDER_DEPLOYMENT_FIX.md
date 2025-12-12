# 🔧 Render Deployment Fix - Build Command Error

## Problem
The deployment is failing because:
1. Render is using Poetry (detected automatically)
2. The build command has malformed pip install commands
3. Streamlit is not being installed

## Solution

### Option 1: Update Existing Service (Recommended)

1. Go to Render Dashboard: https://dashboard.render.com
2. Find your `moneta-web-dashboard` service
3. Go to **Settings** → **Build & Deploy**
4. Update the **Build Command** to:
   ```bash
   pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
   ```
5. Update **Start Command** to:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
   ```
6. Set **Python Version** to: `3.11.9`
7. Click **Save Changes**
8. Click **Manual Deploy** → **Deploy latest commit**

### Option 2: Recreate Service Using Blueprint

1. Delete the existing `moneta-web-dashboard` service
2. Go to **New +** → **Blueprint**
3. Connect your GitHub repo: `Allorasays/financial-analyzer-pro`
4. Select branch: `complete-app-restoration`
5. Blueprint file: `render_final.yaml`
6. Click **Apply**

### Option 3: Fix via Render CLI

If you have Render CLI installed:
```bash
render services update moneta-web-dashboard --build-command "pip install --upgrade pip setuptools wheel && pip install -r requirements.txt"
```

## Verify Build Command

The correct build command should be:
```bash
pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
```

**NOT:**
```bash
pip install streamlit>=1.28.0,<2.0.0
pip install pandas>=2.0.0,<3.0.0
...
```

## Verify Requirements.txt

Make sure `requirements.txt` exists and contains:
```
streamlit>=1.28.0,<2.0.0
pandas>=2.0.0,<3.0.0
plotly>=5.0.0,<6.0.0
yfinance>=0.2.0,<1.0.0
numpy>=1.24.0,<2.0.0
scipy>=1.11.0,<2.0.0
scikit-learn>=1.3.0,<2.0.0
requests>=2.31.0
feedparser>=6.0.10
python-dotenv>=1.0.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
bcrypt>=4.0.0
PyJWT>=2.8.0
pytz>=2023.3
ta>=0.11.0
sec-edgar-downloader>=5.0.3
vaderSentiment>=3.3.0
```

## After Fix

Once deployed, verify:
1. Service shows "Live" status
2. Logs show: "Successfully installed streamlit..."
3. App is accessible at: `https://moneta-web-dashboard.onrender.com`

## Troubleshooting

If still failing:
1. Check logs for specific error messages
2. Verify Python version is 3.11.9
3. Ensure `app.py` exists in root directory
4. Check that all imports in `app.py` are available in `requirements.txt`

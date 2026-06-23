# 🚨 URGENT: Fix Streamlit Not Found Error on Render

## Problem
Render service is still using old build command that doesn't install streamlit properly.

## ✅ IMMEDIATE FIX (Do This Now)

### Step 1: Go to Render Dashboard
1. Navigate to: https://dashboard.render.com
2. Find your **`moneta-web-dashboard`** service (or the service showing the error)

### Step 2: Update Build Command
1. Click on the service
2. Go to **Settings** tab
3. Scroll to **Build & Deploy** section
4. Find **Build Command** field
5. **DELETE** the current build command
6. **REPLACE** with this exact command:
   ```bash
   pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
   ```
7. Click **Save Changes**

### Step 3: Verify Start Command
Make sure **Start Command** is:
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

### Step 4: Verify Python Version
Make sure **Python Version** is set to: `3.11.9`

### Step 5: Manual Deploy
1. Click **Manual Deploy** button (top right)
2. Select **Deploy latest commit**
3. Wait for deployment to complete

## 🔍 Verify Fix

After deployment, check the logs. You should see:
```
Successfully installed streamlit-...
Successfully installed pandas-...
Successfully installed plotly-...
...
```

And then:
```
Running 'streamlit run app.py...'
```

## ❌ What NOT to Use

**DO NOT USE** this build command (it's malformed):
```bash
pip install streamlit>=1.28.0,<2.0.0
pip install pandas>=2.0.0,<3.0.0
...
```

**USE THIS** instead:
```bash
pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
```

## 🎯 Why This Works

- `requirements.txt` contains all packages with correct version constraints
- Single command installs everything properly
- No bash interpretation errors
- Streamlit gets installed correctly

## 📋 Quick Checklist

- [ ] Build Command: `pip install --upgrade pip setuptools wheel && pip install -r requirements.txt`
- [ ] Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
- [ ] Python Version: `3.11.9`
- [ ] Saved changes
- [ ] Manual deploy triggered
- [ ] Logs show "Successfully installed streamlit"

## 🆘 Still Not Working?

If still failing after these steps:

1. **Check requirements.txt exists** in your repo root
2. **Verify it contains streamlit**:
   ```
   streamlit>=1.28.0,<2.0.0
   ```
3. **Check service logs** for specific error messages
4. **Try deleting and recreating the service** using the Blueprint:
   - Delete existing service
   - New + → Blueprint
   - Connect repo: `Allorasays/financial-analyzer-pro`
   - Branch: `complete-app-restoration`
   - Blueprint: `render_final.yaml`

---

**This fix will resolve the "streamlit: command not found" error!** ✅








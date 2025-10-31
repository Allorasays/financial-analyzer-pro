# Render Deployment Fix - Step by Step

## Issue
Backend service is using old start command: `uvicorn proxy:app...` which fails because `uvicorn` isn't in PATH.

## Solution

### **Option 1: Update Start Command in Render Dashboard** (Recommended)

1. **Go to your backend service** (`moneta-backend-api`) in Render dashboard
2. **Click Settings** tab
3. **Find "Start Command"** field
4. **Replace with**:
   ```
   python -m uvicorn proxy:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips="*"
   ```
5. **Also verify**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Environment**: `PYTHON_VERSION = 3.11.9` (or remove it to use runtime.txt)
6. **Click "Save Changes"**
7. **Go to Deploy tab → "Manual Deploy" → "Clear build cache & deploy"**

---

### **Option 2: Use Startup Script** (Alternative)

If Option 1 doesn't work:

1. **Update Start Command** in Render Settings to:
   ```
   bash start_backend.sh
   ```
2. **Make sure** `start_backend.sh` is committed to your repo
3. **Deploy**

---

## Verification

After updating and deploying:

1. **Check logs** - Should see:
   ```
   INFO:     Uvicorn running on http://0.0.0.0:XXXX (Press CTRL+C to quit)
   ```

2. **Test endpoints**:
   ```bash
   curl https://your-backend-url.onrender.com/health
   # Should return: {"status":"ok"}
   
   curl https://your-backend-url.onrender.com/api/system/status
   # Should return JSON with services
   ```

3. **Web dashboard** should now load after backend is healthy

---

## Common Issues

### Still seeing "uvicorn: command not found"
- Make sure you updated **Start Command** in Settings (not just the blueprint file)
- Clear build cache before redeploying
- Verify the command uses `python -m uvicorn` not just `uvicorn`

### "No open ports detected"
- This means the server isn't starting
- Check that `proxy:app` is correct (your file is `proxy.py`)
- Check logs for Python errors

### Build succeeds but service won't start
- Check that all dependencies install correctly
- Verify `requirements.txt` includes `uvicorn[standard]`
- Check environment variables are set correctly

---

## Current Files

- `start_backend.sh` - Alternative startup script
- `render_final.yaml` - Blueprint (service settings override this)
- `runtime.txt` - Python version: `python-3.11.9`

---

**Next Step**: Update Start Command in Render dashboard for backend service, then redeploy.


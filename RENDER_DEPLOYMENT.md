# 🚀 Render Deployment Guide

## Quick Deploy

1. **Fork/Clone** this repository to your GitHub account
2. **Connect to Render:**
   - Go to [render.com](https://render.com)
   - Sign up/Login with GitHub
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

## Configuration

### Environment Variables (Optional)
- `API_BASE_URL`: Your FastAPI backend URL (if separate)
- `RENDER`: Automatically set by Render

### Build Settings
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Deployment Steps

1. **Repository Setup:**
   - Ensure all files are committed to GitHub
   - Verify `requirements.txt` is up to date

2. **Render Configuration:**
   - **Name:** `financial-analyzer-streamlit`
   - **Environment:** `Python 3`
   - **Region:** Choose closest to your users
   - **Branch:** `main` or `master`

3. **Deploy:**
   - Click "Create Web Service"
   - Wait for build to complete (5-10 minutes)
   - Your app will be available at `https://your-app-name.onrender.com`

## Troubleshooting

### Common Issues:
- **Build fails:** Check `requirements.txt` for incompatible packages
- **App won't start:** Verify start command in render.yaml
- **Port issues:** Render automatically sets `$PORT` environment variable

### Logs:
- Check Render dashboard for build and runtime logs
- Verify all dependencies installed successfully

## File Structure for Render

```
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── render.yaml           # Render configuration
├── Procfile             # Alternative deployment method
├── .streamlit/          # Streamlit configuration
│   └── config.toml
├── runtime.txt          # Python version
└── packages.txt         # System dependencies
```

## Next Steps

After deployment:
1. Test all app functionality
2. Configure custom domain (optional)
3. Set up monitoring and alerts
4. Configure environment variables if needed

## Support

- Render Documentation: [docs.render.com](https://docs.render.com)
- Streamlit Deployment: [docs.streamlit.io](https://docs.streamlit.io)
- Issues: Check GitHub repository


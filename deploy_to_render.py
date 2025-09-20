#!/usr/bin/env python3
"""
Deployment script for Financial Analyzer Pro Enhanced Performance to Render
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'app_render_enhanced.py',
        'requirements_render_enhanced.txt',
        'render_enhanced_performance.yaml'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def create_render_config():
    """Create optimized Render configuration"""
    config = {
        "services": [
            {
                "type": "web",
                "name": "financial-analyzer-pro-enhanced",
                "env": "python",
                "plan": "free",
                "buildCommand": "pip install --upgrade pip && pip install -r requirements_render_enhanced.txt --no-cache-dir",
                "startCommand": "streamlit run app_render_enhanced.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --server.fileWatcherType none",
                "envVars": [
                    {"key": "PYTHON_VERSION", "value": "3.11.0"},
                    {"key": "STREAMLIT_SERVER_HEADLESS", "value": "true"},
                    {"key": "STREAMLIT_SERVER_ADDRESS", "value": "0.0.0.0"},
                    {"key": "STREAMLIT_SERVER_ENABLE_CORS", "value": "false"},
                    {"key": "STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", "value": "false"},
                    {"key": "STREAMLIT_SERVER_FILE_WATCHER_TYPE", "value": "none"},
                    {"key": "STREAMLIT_BROWSER_GATHER_USAGE_STATS", "value": "false"},
                    {"key": "STREAMLIT_SERVER_RUN_ON_SAVE", "value": "false"},
                    {"key": "CACHE_TTL", "value": "300"},
                    {"key": "MAX_CACHE_SIZE", "value": "200"},
                    {"key": "DB_PATH", "value": "predictions.db"},
                    {"key": "LOG_LEVEL", "value": "INFO"}
                ],
                "healthCheckPath": "/",
                "autoDeploy": True,
                "region": "oregon"
            }
        ]
    }
    
    with open('render.yaml', 'w') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Render configuration created")

def create_procfile():
    """Create Procfile for Render"""
    with open('Procfile', 'w') as f:
        f.write('web: streamlit run app_render_enhanced.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --server.fileWatcherType none\n')
    
    print("✅ Procfile created")

def create_runtime_txt():
    """Create runtime.txt for Python version"""
    with open('runtime.txt', 'w') as f:
        f.write('python-3.11.0\n')
    
    print("✅ runtime.txt created")

def test_local_deployment():
    """Test the app locally before deployment"""
    print("🧪 Testing local deployment...")
    
    try:
        # Test imports
        import streamlit as st
        import pandas as pd
        import plotly.graph_objects as go
        import yfinance as yf
        import numpy as np
        from sklearn.linear_model import LinearRegression
        print("✅ All imports successful")
        
        # Test app file
        with open('app_render_enhanced.py', 'r') as f:
            content = f.read()
            if 'def main():' in content and 'streamlit' in content:
                print("✅ App file structure looks good")
            else:
                print("❌ App file structure issues")
                return False
        
        print("✅ Local deployment test passed")
        return True
        
    except Exception as e:
        print(f"❌ Local deployment test failed: {str(e)}")
        return False

def create_deployment_guide():
    """Create deployment guide"""
    guide = """# 🚀 Render Deployment Guide - Financial Analyzer Pro Enhanced

## 📋 Prerequisites
1. GitHub repository with your code
2. Render.com account
3. All required files in repository

## 🚀 Deployment Steps

### 1. Prepare Repository
```bash
# Ensure all files are committed
git add .
git commit -m "Enhanced performance version ready for Render"
git push origin main
```

### 2. Deploy on Render
1. Go to [Render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Use these settings:
   - **Name**: financial-analyzer-pro-enhanced
   - **Environment**: Python
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements_render_enhanced.txt --no-cache-dir`
   - **Start Command**: `streamlit run app_render_enhanced.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --server.fileWatcherType none`

### 3. Environment Variables
Add these environment variables in Render dashboard:
- `PYTHON_VERSION`: 3.11.0
- `STREAMLIT_SERVER_HEADLESS`: true
- `STREAMLIT_SERVER_ADDRESS`: 0.0.0.0
- `STREAMLIT_SERVER_ENABLE_CORS`: false
- `STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION`: false
- `STREAMLIT_SERVER_FILE_WATCHER_TYPE`: none
- `CACHE_TTL`: 300
- `MAX_CACHE_SIZE`: 200
- `DB_PATH`: predictions.db
- `LOG_LEVEL`: INFO

### 4. Deploy
Click "Create Web Service" and wait for deployment to complete.

## 🔧 Troubleshooting

### Common Issues
1. **Build fails**: Check requirements.txt and Python version
2. **App won't start**: Verify start command and port configuration
3. **Database errors**: Ensure DB_PATH is set correctly
4. **Import errors**: Check all dependencies are in requirements file

### Logs
Check Render logs for detailed error information.

## ✅ Success Indicators
- Build completes successfully
- App starts without errors
- Health check passes
- App is accessible via URL

## 🎉 Post-Deployment
1. Test all features
2. Monitor performance
3. Check prediction accuracy tracking
4. Verify caching is working

Your enhanced Financial Analyzer Pro is now live! 🚀
"""
    
    with open('RENDER_DEPLOYMENT_GUIDE.md', 'w') as f:
        f.write(guide)
    
    print("✅ Deployment guide created")

def main():
    """Main deployment preparation"""
    print("🚀 Financial Analyzer Pro - Render Deployment Preparation")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("❌ Deployment preparation failed - missing files")
        sys.exit(1)
    
    # Create configuration files
    create_render_config()
    create_procfile()
    create_runtime_txt()
    
    # Test locally
    if not test_local_deployment():
        print("❌ Local test failed - fix issues before deploying")
        sys.exit(1)
    
    # Create deployment guide
    create_deployment_guide()
    
    print("\n✅ Deployment preparation completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Commit all files to GitHub")
    print("2. Go to Render.com and create new web service")
    print("3. Follow the deployment guide in RENDER_DEPLOYMENT_GUIDE.md")
    print("4. Monitor deployment logs for any issues")
    
    print("\n🎉 Ready for Render deployment! 🚀")

if __name__ == "__main__":
    main()




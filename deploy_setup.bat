@echo off
echo ========================================
echo 🚀 Financial Analyzer Pro - Deploy Setup
echo ========================================
echo.

echo ✅ Checking current status...
echo.

echo 📁 Current directory: %CD%
echo 📊 Files ready for deployment:
dir /b *.py *.txt *.yaml *.toml *.md 2>nul | findstr /v "start_app.bat start_app.py proxy.py"

echo.
echo 🔧 Next steps to deploy:
echo.
echo 1. 📚 GitHub Setup:
echo    - Create new repository: financial-analyzer-pro
echo    - Initialize git and push code
echo.
echo 2. 🌐 Render Setup:
echo    - Go to render.com and sign up
echo    - Connect GitHub account
echo    - Create new Web Service
echo.
echo 3. 🚀 Deploy:
echo    - Select repository: financial-analyzer-pro
echo    - Build command: pip install -r requirements.txt
echo    - Start command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
echo.

echo 💡 Quick Commands (when ready):
echo    git init
echo    git add .
echo    git commit -m "Phase 1 Complete: Financial Analyzer Pro"
echo    git remote add origin https://github.com/YOUR_USERNAME/financial-analyzer-pro.git
echo    git push -u origin main
echo.

echo 🎯 Your app will be live at:
echo    https://financial-analyzer-pro.onrender.com
echo.

echo 📋 See DEPLOYMENT_PREPARATION_GUIDE.md for detailed instructions
echo.

pause

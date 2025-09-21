@echo off
echo 🚀 Deploying Render Fix for Financial Analyzer...
echo.

echo 📝 Adding all files to git...
git add .

echo.
echo 💾 Committing Render deployment fix...
git commit -m "Fix Render deployment - optimized version with authentication"

echo.
echo 📤 Pushing to GitHub...
git push origin main

echo.
echo ✅ Deployment fix complete! 
echo 🌐 Check your Render dashboard for deployment status.
echo 🔧 The optimized app should now work on Render!
pause

@echo off
echo 🚀 Deploying Financial Analyzer with Day 10 Authentication...
echo.

echo 📝 Adding all files to git...
git add .

echo.
echo 💾 Committing changes...
git commit -m "Add Day 10: User Authentication with optional sign-in, portfolio persistence, and business metrics"

echo.
echo 📤 Pushing to GitHub...
git push origin main

echo.
echo ✅ Deployment complete! 
echo 🌐 Check your Render dashboard for deployment status.
echo 🔐 Your app now includes user authentication features!
pause

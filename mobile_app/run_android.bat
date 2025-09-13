@echo off
echo 🚀 Starting Financial Analyzer Pro on Android...
echo.

echo 📱 Starting Metro bundler...
start "Metro Bundler" cmd /k "npm start"

echo.
echo ⏳ Waiting for Metro to start...
timeout /t 5 /nobreak >nul

echo.
echo 🤖 Running on Android...
call npm run android

if %errorlevel% neq 0 (
    echo.
    echo ❌ Failed to run on Android
    echo.
    echo 🔧 Troubleshooting:
    echo 1. Make sure Android emulator is running
    echo 2. Check if USB debugging is enabled
    echo 3. Try: npm run clean && npm run android
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ App should be running on your Android device/emulator!
echo.
pause







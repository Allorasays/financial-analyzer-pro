@echo off
echo 🚀 Running Financial Analyzer Pro with compatible Java...
echo.

echo 📱 Setting up environment...
set ANDROID_HOME=C:\Users\mmiddlebass\AppData\Local\Android\Sdk
set PATH=%ANDROID_HOME%\platform-tools;%PATH%

echo.
echo 🔧 Using system Java instead of JDK 24...
echo.

echo 📦 Starting Metro bundler...
start "Metro Bundler" cmd /k "npm start"

echo.
echo ⏳ Waiting for Metro to start...
timeout /t 3 /nobreak >nul

echo.
echo 🤖 Running on Android with system Java...
set JAVA_HOME=
call npm run android

echo.
echo ✅ App should be running on your emulator!
echo.
pause





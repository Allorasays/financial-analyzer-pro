@echo off
echo 🚀 Installing Prerequisites for Financial Analyzer Pro...
echo.

echo ========================================
echo 📦 AUTOMATIC INSTALLATION
echo ========================================
echo.

echo 1. Checking if Node.js is installed...
where node >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Node.js already installed
    node --version
) else (
    echo ❌ Node.js not found
    echo.
    echo 📥 Please install Node.js manually:
    echo    1. Go to: https://nodejs.org/
    echo    2. Download LTS version
    echo    3. Run installer
    echo    4. Restart command prompt
    echo.
    echo Press any key to open Node.js download page...
    pause >nul
    start https://nodejs.org/
    echo.
    echo After installing Node.js, run this script again.
    pause
    exit /b 1
)
echo.

echo 2. Checking if npm is available...
where npm >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ npm available
    npm --version
) else (
    echo ❌ npm not found
    echo Please reinstall Node.js
    pause
    exit /b 1
)
echo.

echo 3. Installing project dependencies...
call npm install
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)
echo ✅ Dependencies installed successfully
echo.

echo 4. Installing React Native CLI globally...
call npm install -g @react-native-community/cli
if %errorlevel% neq 0 (
    echo ⚠️  Failed to install React Native CLI globally
    echo This might require administrator privileges
    echo You can try running as administrator or install manually
) else (
    echo ✅ React Native CLI installed
)
echo.

echo ========================================
echo 📋 NEXT STEPS
echo ========================================
echo.

echo 1. Install Android Studio:
echo    - Go to: https://developer.android.com/studio
echo    - Download and install
echo    - Make sure to install Android SDK (API 33+)
echo.

echo 2. Install Java Development Kit (JDK):
echo    - Go to: https://adoptium.net/
echo    - Download JDK 11 or higher
echo    - Install and set JAVA_HOME environment variable
echo.

echo 3. Set Environment Variables:
echo    - ANDROID_HOME=C:\Users\YourUsername\AppData\Local\Android\Sdk
echo    - JAVA_HOME=C:\Program Files\Java\jdk-11.0.x
echo.

echo 4. Create Android Emulator:
echo    - Open Android Studio
echo    - Go to Tools > AVD Manager
echo    - Create new virtual device
echo.

echo 5. Run the app:
echo    - run_android.bat
echo.

echo 📚 See QUICK_START_GUIDE.md for detailed instructions
echo.
pause





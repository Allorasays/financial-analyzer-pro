@echo off
echo 🚀 Automated Prerequisites Installation for Financial Analyzer Pro
echo.

echo ========================================
echo 📦 AUTOMATED INSTALLATION
echo ========================================
echo.

echo This script will help you install all required prerequisites.
echo.

echo 1. Installing Node.js...
echo.
echo 📥 Downloading Node.js LTS...
echo Please wait while we download the installer...

:: Download Node.js LTS (Windows x64)
powershell -Command "& {Invoke-WebRequest -Uri 'https://nodejs.org/dist/v18.19.0/node-v18.19.0-x64.msi' -OutFile 'nodejs-installer.msi'}"

if exist nodejs-installer.msi (
    echo ✅ Node.js installer downloaded
    echo.
    echo 🔧 Installing Node.js...
    echo Please follow the installation wizard...
    start /wait nodejs-installer.msi
    
    echo.
    echo ✅ Node.js installation completed
    del nodejs-installer.msi
) else (
    echo ❌ Failed to download Node.js installer
    echo Please download manually from: https://nodejs.org/
    echo.
    echo Press any key to open Node.js download page...
    pause >nul
    start https://nodejs.org/
)

echo.
echo 2. Verifying Node.js installation...
call node --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Node.js installed successfully
    node --version
    npm --version
) else (
    echo ❌ Node.js installation failed
    echo Please restart your command prompt and try again
    echo.
    pause
    exit /b 1
)

echo.
echo 3. Installing project dependencies...
call npm install
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    echo Please check your internet connection
    pause
    exit /b 1
)
echo ✅ Dependencies installed successfully

echo.
echo 4. Installing React Native CLI...
call npm install -g @react-native-community/cli
if %errorlevel% neq 0 (
    echo ⚠️  Failed to install React Native CLI globally
    echo This might require administrator privileges
    echo You can try running as administrator
) else (
    echo ✅ React Native CLI installed
)

echo.
echo ========================================
echo 📋 MANUAL INSTALLATION REQUIRED
echo ========================================
echo.

echo The following components need to be installed manually:
echo.

echo 1. Java Development Kit (JDK):
echo    - Go to: https://adoptium.net/
echo    - Download JDK 11 or higher
echo    - Install and set JAVA_HOME environment variable
echo.

echo 2. Android Studio:
echo    - Go to: https://developer.android.com/studio
echo    - Download and install
echo    - Make sure to install Android SDK (API 33+)
echo    - Set ANDROID_HOME environment variable
echo.

echo 3. Environment Variables:
echo    - ANDROID_HOME=C:\Users\%USERNAME%\AppData\Local\Android\Sdk
echo    - JAVA_HOME=C:\Program Files\Java\jdk-11.0.x
echo.

echo ========================================
echo 🚀 NEXT STEPS
echo ========================================
echo.

echo After installing the manual components:
echo 1. Run: check_prerequisites.bat
echo 2. Run: setup_android.bat
echo 3. Run: run_android.bat
echo.

echo 📚 See QUICK_START_GUIDE.md for detailed instructions
echo.

pause





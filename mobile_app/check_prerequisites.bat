@echo off
echo 🔍 Checking Prerequisites for Financial Analyzer Pro...
echo.

echo ========================================
echo 📋 PREREQUISITES CHECK
echo ========================================
echo.

echo 1. Checking Node.js...
where node >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Node.js found
    node --version
) else (
    echo ❌ Node.js NOT FOUND
    echo    Please install from: https://nodejs.org/
)
echo.

echo 2. Checking npm...
where npm >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ npm found
    npm --version
) else (
    echo ❌ npm NOT FOUND
    echo    Please install Node.js from: https://nodejs.org/
)
echo.

echo 3. Checking Java...
where java >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Java found
    java -version
) else (
    echo ❌ Java NOT FOUND
    echo    Please install JDK from: https://adoptium.net/
)
echo.

echo 4. Checking Android SDK...
where adb >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Android SDK found
    adb version
) else (
    echo ❌ Android SDK NOT FOUND
    echo    Please install Android Studio from: https://developer.android.com/studio
)
echo.

echo 5. Checking Environment Variables...
if defined ANDROID_HOME (
    echo ✅ ANDROID_HOME is set: %ANDROID_HOME%
) else (
    echo ❌ ANDROID_HOME NOT SET
    echo    Please set ANDROID_HOME to your Android SDK path
)
echo.

if defined JAVA_HOME (
    echo ✅ JAVA_HOME is set: %JAVA_HOME%
) else (
    echo ❌ JAVA_HOME NOT SET
    echo    Please set JAVA_HOME to your JDK path
)
echo.

echo ========================================
echo 📊 SUMMARY
echo ========================================
echo.

where node >nul 2>&1 && where npm >nul 2>&1 && where java >nul 2>&1 && where adb >nul 2>&1
if %errorlevel% equ 0 (
    echo 🎉 ALL PREREQUISITES FOUND!
    echo.
    echo ✅ You can now run:
    echo    - setup_android.bat
    echo    - run_android.bat
    echo.
) else (
    echo ⚠️  SOME PREREQUISITES MISSING
    echo.
    echo 📋 Please install missing components:
    echo    1. Node.js: https://nodejs.org/
    echo    2. Android Studio: https://developer.android.com/studio
    echo    3. JDK: https://adoptium.net/
    echo.
    echo 📚 See QUICK_START_GUIDE.md for detailed instructions
    echo.
)

echo.
pause





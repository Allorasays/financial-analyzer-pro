@echo off
echo 🚀 Setting up Financial Analyzer Pro for Android Studio...
echo.

echo 📦 Installing dependencies...
call npm install
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo 🔧 Installing React Native CLI...
call npm install -g @react-native-community/cli
if %errorlevel% neq 0 (
    echo ❌ Failed to install React Native CLI
    pause
    exit /b 1
)

echo.
echo 🧹 Cleaning previous builds...
call npm run clean
if exist android\app\build rmdir /s /q android\app\build

echo.
echo ✅ Setup complete!
echo.
echo 📋 Next steps:
echo 1. Open Android Studio
echo 2. Open project: mobile_app\android
echo 3. Wait for Gradle sync
echo 4. Create Android emulator
echo 5. Run the app
echo.
echo 📚 See ANDROID_STUDIO_SETUP.md for detailed instructions
echo.
pause









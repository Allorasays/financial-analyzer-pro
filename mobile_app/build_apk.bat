@echo off
echo 🚀 Building Financial Analyzer Pro APK...
echo.

echo 📱 Setting up environment...
set JAVA_HOME=C:\Program Files\Java\jdk-24
set ANDROID_HOME=C:\Users\mmiddlebass\AppData\Local\Android\Sdk
set PATH=%JAVA_HOME%\bin;%ANDROID_HOME%\platform-tools;%PATH%

echo.
echo 🔧 Building APK with Android Studio...
echo.

cd android

echo 📦 Running Gradle build...
call gradlew assembleDebug --no-daemon --max-workers=1

if %errorlevel% equ 0 (
    echo.
    echo ✅ APK built successfully!
    echo.
    echo 📱 Installing APK on emulator...
    adb install app\build\outputs\apk\debug\app-debug.apk
    
    if %errorlevel% equ 0 (
        echo.
        echo 🎉 App installed successfully!
        echo.
        echo 📱 Launching app...
        adb shell am start -n com.financialanalyzer.mobile/.MainActivity
    ) else (
        echo ❌ Failed to install APK
    )
) else (
    echo ❌ Build failed
    echo.
    echo 💡 Try opening the project in Android Studio instead:
    echo    1. Open Android Studio
    echo    2. File > Open
    echo    3. Navigate to mobile_app\android
    echo    4. Click Run button
)

echo.
pause





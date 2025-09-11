# 🚀 Quick Start Guide - Financial Analyzer Pro Mobile

## 📋 **Prerequisites Checklist**

Before running the Android app, you need to install these tools:

### **1. Node.js & npm** ⚠️ **REQUIRED**
- Download from: https://nodejs.org/
- Install **Node.js LTS** (v18 or higher)
- This will also install npm automatically
- **Verify installation:**
  ```bash
  node --version
  npm --version
  ```

### **2. Android Studio** ⚠️ **REQUIRED**
- Download from: https://developer.android.com/studio
- Install with Android SDK (API 33+)
- Install Android Emulator
- **Verify installation:**
  - Open Android Studio
  - Go to Tools > SDK Manager
  - Ensure Android SDK Platform 33 is installed

### **3. Java Development Kit (JDK)** ⚠️ **REQUIRED**
- Download JDK 11 or higher from: https://adoptium.net/
- Set JAVA_HOME environment variable
- **Verify installation:**
  ```bash
  java -version
  echo $JAVA_HOME
  ```

---

## 🔧 **Step-by-Step Setup**

### **Step 1: Install Node.js**
1. Go to https://nodejs.org/
2. Download **LTS version** (recommended)
3. Run the installer
4. Restart your command prompt/PowerShell
5. Verify: `node --version` and `npm --version`

### **Step 2: Install Android Studio**
1. Go to https://developer.android.com/studio
2. Download and install Android Studio
3. During installation, make sure to install:
   - Android SDK
   - Android SDK Platform
   - Android Virtual Device
   - Android SDK Build-Tools

### **Step 3: Set Environment Variables**
Add these to your system environment variables:

**Windows:**
```
ANDROID_HOME=C:\Users\YourUsername\AppData\Local\Android\Sdk
JAVA_HOME=C:\Program Files\Java\jdk-11.0.x
```

**How to set:**
1. Press `Win + R`, type `sysdm.cpl`
2. Click "Environment Variables"
3. Add new system variables for ANDROID_HOME and JAVA_HOME
4. Add to PATH: `%ANDROID_HOME%\platform-tools`

### **Step 4: Create Android Emulator**
1. Open Android Studio
2. Go to **Tools > AVD Manager**
3. Click **Create Virtual Device**
4. Choose **Phone > Pixel 4** (or similar)
5. Download **API 33** system image
6. Name it **FinancialAnalyzer_Emulator**
7. Click **Finish**

---

## 🚀 **Running the App**

### **Method 1: Using Setup Scripts**
```bash
# 1. Install dependencies
setup_android.bat

# 2. Run the app
run_android.bat
```

### **Method 2: Manual Commands**
```bash
# 1. Install dependencies
npm install

# 2. Install React Native CLI globally
npm install -g @react-native-community/cli

# 3. Start Metro bundler (in one terminal)
npm start

# 4. Run on Android (in another terminal)
npm run android
```

### **Method 3: Android Studio**
1. Open Android Studio
2. **File > Open** → Navigate to `mobile_app/android` folder
3. Wait for Gradle sync to complete
4. Click **Run** (green play button)

---

## 🐛 **Troubleshooting**

### **"npm is not recognized"**
- **Solution:** Install Node.js from https://nodejs.org/
- **Verify:** Restart command prompt and run `npm --version`

### **"java is not recognized"**
- **Solution:** Install JDK and set JAVA_HOME
- **Verify:** Run `java -version`

### **"adb is not recognized"**
- **Solution:** Add Android SDK platform-tools to PATH
- **Path:** `%ANDROID_HOME%\platform-tools`

### **Gradle sync failed**
```bash
cd android
./gradlew clean
cd ..
npm run android
```

### **Metro bundler issues**
```bash
npm start --reset-cache
```

### **Android emulator not starting**
- Check if virtualization is enabled in BIOS
- Try different emulator configuration
- Restart Android Studio

---

## 📱 **Testing the App**

### **1. Verify Installation**
```bash
# Check Node.js
node --version

# Check npm
npm --version

# Check Java
java -version

# Check Android SDK
adb version
```

### **2. Test Build**
```bash
# Clean and build
npm run clean
npm run build:android-debug

# Check if APK was created
dir android\app\build\outputs\apk\debug\
```

### **3. Test on Emulator**
1. Start Android emulator
2. Run `npm run android`
3. App should install and launch automatically

---

## 🎯 **Success Indicators**

You'll know everything is working when:
- ✅ Node.js and npm are installed
- ✅ Android Studio opens the project without errors
- ✅ Gradle sync completes successfully
- ✅ Android emulator starts and runs
- ✅ App builds without errors
- ✅ App launches on emulator
- ✅ All screens navigate correctly

---

## 📚 **Next Steps After Setup**

1. **Test all features:**
   - Login/Register
   - Dashboard
   - Portfolio management
   - Market data
   - Technical analysis

2. **Connect to backend:**
   - Update API URL in `src/services/api.ts`
   - Test API connectivity

3. **Customize the app:**
   - Update app icons
   - Modify color scheme
   - Add new features

---

## 🆘 **Need Help?**

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Try the clean build process
4. Check Android Studio logs for specific errors

**The app is ready to run once you have the prerequisites installed!** 🎉





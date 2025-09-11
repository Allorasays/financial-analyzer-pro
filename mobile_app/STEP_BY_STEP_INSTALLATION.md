# 🚀 Step-by-Step Installation Guide

## 📋 **Complete Installation Process**

This guide will walk you through installing all prerequisites for the Financial Analyzer Pro mobile app.

---

## **Step 1: Install Node.js** ⭐ **START HERE**

### **Why:** Node.js is required for React Native development and package management.

### **How:**
1. **Download Node.js:**
   - Go to https://nodejs.org/
   - Click **"Download Node.js (LTS)"** - this is the recommended version
   - Choose **Windows Installer (.msi)** for 64-bit

2. **Install Node.js:**
   - Run the downloaded `.msi` file
   - Follow the installation wizard
   - **Important:** Check "Add to PATH" during installation
   - Complete the installation

3. **Verify Installation:**
   - Open a **new** command prompt (important!)
   - Type: `node --version`
   - Type: `npm --version`
   - Both should show version numbers

### **✅ Success Check:**
```bash
node --version  # Should show v18.x.x or higher
npm --version   # Should show 9.x.x or higher
```

---

## **Step 2: Install Java Development Kit (JDK)**

### **Why:** Required for Android development and building the app.

### **How:**
1. **Download JDK:**
   - Go to https://adoptium.net/
   - Click **"Download Temurin"**
   - Choose **JDK 11** or higher
   - Select **Windows x64** installer

2. **Install JDK:**
   - Run the downloaded installer
   - Follow the installation wizard
   - **Note the installation path** (usually `C:\Program Files\Eclipse Adoptium\jdk-11.x.x`)

3. **Set JAVA_HOME:**
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Click **"Environment Variables"**
   - Under **"System Variables"**, click **"New"**
   - Variable name: `JAVA_HOME`
   - Variable value: `C:\Program Files\Eclipse Adoptium\jdk-11.0.x` (your actual path)
   - Click **"OK"**

### **✅ Success Check:**
```bash
java -version  # Should show Java version
echo %JAVA_HOME%  # Should show your JDK path
```

---

## **Step 3: Install Android Studio**

### **Why:** Provides Android SDK, emulator, and development tools.

### **How:**
1. **Download Android Studio:**
   - Go to https://developer.android.com/studio
   - Click **"Download Android Studio"**
   - Download the Windows installer

2. **Install Android Studio:**
   - Run the installer
   - Choose **"Standard"** installation
   - **Important:** Make sure "Android SDK" is checked
   - Complete the installation

3. **Set up Android SDK:**
   - Open Android Studio
   - Go to **Tools > SDK Manager**
   - In **"SDK Platforms"** tab:
     - Check **Android 13.0 (API 33)** or higher
   - In **"SDK Tools"** tab:
     - Check **Android SDK Build-Tools**
     - Check **Android Emulator**
     - Check **Android SDK Platform-Tools**
   - Click **"Apply"** and install

4. **Set ANDROID_HOME:**
   - Note the SDK path (usually `C:\Users\YourUsername\AppData\Local\Android\Sdk`)
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Click **"Environment Variables"**
   - Under **"System Variables"**, click **"New"**
   - Variable name: `ANDROID_HOME`
   - Variable value: `C:\Users\YourUsername\AppData\Local\Android\Sdk`
   - Click **"OK"**

### **✅ Success Check:**
```bash
adb version  # Should show Android Debug Bridge version
echo %ANDROID_HOME%  # Should show your Android SDK path
```

---

## **Step 4: Create Android Emulator**

### **Why:** Needed to run and test the mobile app.

### **How:**
1. **Open Android Studio**
2. **Go to Tools > AVD Manager**
3. **Click "Create Virtual Device"**
4. **Choose Device:**
   - Select **Phone > Pixel 4** (or similar)
   - Click **"Next"**
5. **Choose System Image:**
   - Select **API 33** (Android 13.0) or higher
   - Click **"Download"** if not already downloaded
   - Click **"Next"**
6. **Configure AVD:**
   - Name: `FinancialAnalyzer_Emulator`
   - Click **"Finish"**
7. **Start Emulator:**
   - Click the **play button** next to your emulator
   - Wait for it to boot up

### **✅ Success Check:**
- Emulator should start and show Android home screen
- You should see a virtual Android device running

---

## **Step 5: Install Project Dependencies**

### **How:**
1. **Open Command Prompt** in the `mobile_app` folder
2. **Run the setup script:**
   ```bash
   setup_android.bat
   ```
   Or manually:
   ```bash
   npm install
   npm install -g @react-native-community/cli
   ```

### **✅ Success Check:**
- No error messages during installation
- `node_modules` folder created in mobile_app directory

---

## **Step 6: Test the Installation**

### **How:**
1. **Run prerequisite check:**
   ```bash
   check_prerequisites.bat
   ```
   All items should show ✅

2. **Test the app:**
   ```bash
   run_android.bat
   ```

### **✅ Success Check:**
- App builds without errors
- App launches on Android emulator
- All screens are accessible

---

## **🔧 Troubleshooting**

### **"node is not recognized"**
- **Solution:** Restart command prompt after installing Node.js
- **Verify:** Run `node --version` in a new command prompt

### **"java is not recognized"**
- **Solution:** Set JAVA_HOME environment variable
- **Verify:** Run `java -version`

### **"adb is not recognized"**
- **Solution:** Add Android SDK to PATH
- **Path:** `%ANDROID_HOME%\platform-tools`

### **Gradle sync failed**
- **Solution:** Clean and rebuild
  ```bash
  cd android
  ./gradlew clean
  cd ..
  npm run android
  ```

### **Emulator won't start**
- **Solution:** Enable virtualization in BIOS
- **Alternative:** Try different emulator configuration

---

## **📱 What You'll Get**

Once everything is installed, you'll have:
- ✅ **Professional mobile app** with financial analysis features
- ✅ **Real-time market data** and portfolio tracking
- ✅ **Machine learning predictions** for stock prices
- ✅ **Technical analysis** with advanced indicators
- ✅ **Authentication system** with secure login
- ✅ **Modern UI** with Material Design components

---

## **🎯 Quick Commands Reference**

```bash
# Check if everything is installed
check_prerequisites.bat

# Install project dependencies
setup_android.bat

# Run the app
run_android.bat

# Build debug APK
npm run build:android-debug

# Clean everything
npm run clean
```

---

## **📚 Need Help?**

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed correctly
3. Try the clean build process
4. Check Android Studio logs for specific errors

**The app is ready to run once you complete these steps!** 🎉





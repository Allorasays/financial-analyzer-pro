# 📱 Android Studio Setup - Step-by-Step Guide
## Copy Files & Configure API Connection

This guide will walk you through **exactly** where to copy files and how to configure your API connection.

---

## 🎯 Part 1: Create Your Android Project

### Step 1.1: Open Android Studio
1. Launch Android Studio
2. Click **"New Project"** or **File → New → New Project**

### Step 1.2: Choose Project Template
1. Select **"Empty Activity"**
2. Click **"Next"**

### Step 1.3: Configure Your Project
```
Name: Financial Analyzer Mobile
Package name: com.financialanalyzer.mobile
Save location: [Choose your location]
Language: Kotlin
Minimum SDK: API 24 ("Nougat"; Android 7.0)
Build configuration language: Kotlin DSL (build.gradle.kts)
```

3. Click **"Finish"**
4. Wait for Gradle sync to complete

---

## 📁 Part 2: Understanding the File Structure

Your Android Studio project will look like this:
```
FinancialAnalyzerMobile/
├── app/
│   ├── src/
│   │   └── main/
│   │       ├── java/
│   │       │   └── com/
│   │       │       └── financialanalyzer/
│   │       │           └── mobile/
│   │       │               ├── data/                    ← CREATE THIS
│   │       │               │   ├── model/               ← CREATE THIS
│   │       │               │   │   └── Models.kt        ← COPY HERE
│   │       │               │   ├── api/                 ← CREATE THIS
│   │       │               │   │   └── ApiService.kt    ← COPY PART 1 HERE
│   │       │               │   ├── network/             ← CREATE THIS
│   │       │               │   │   └── RetrofitClient.kt ← COPY PART 2 HERE
│   │       │               │   └── repository/          ← CREATE THIS
│   │       │               │       └── Repository.kt    ← COPY PART 3 HERE
│   │       │               ├── ui/                      ← CREATE THIS
│   │       │               │   └── viewmodel/           ← CREATE THIS
│   │       │               │       └── ViewModels.kt    ← COPY HERE
│   │       │               └── MainActivity.kt          ← REPLACE EXISTING
│   │       ├── res/
│   │       └── AndroidManifest.xml                      ← EDIT THIS
│   └── build.gradle.kts                                 ← EDIT THIS
└── build.gradle.kts                                     ← EDIT THIS (project level)
```

---

## 🔨 Part 3: Create Required Directories

In Android Studio:

### Step 3.1: Create Package Structure
1. Right-click on `app/src/main/java/com/financialanalyzer/mobile/`
2. Select **New → Package**
3. Create these packages (one at a time):
   - `data`
   - `data.model`
   - `data.api`
   - `data.network`
   - `data.repository`
   - `ui`
   - `ui.viewmodel`

Your package structure should now show:
```
com.financialanalyzer.mobile
├── data
│   ├── api
│   ├── model
│   ├── network
│   └── repository
└── ui
    └── viewmodel
```

---

## 📄 Part 4: Copy Files to Correct Locations

### Step 4.1: Copy Data Models

1. In Android Studio, right-click on `data.model` package
2. Select **New → Kotlin Class/File**
3. Name it: `Models`
4. Copy the **ENTIRE CONTENTS** from your `android/data_models.kt` file
5. Paste into the new `Models.kt` file

**Location:** `app/src/main/java/com/financialanalyzer/mobile/data/model/Models.kt`

---

### Step 4.2: Copy API Service (Part 1 of 3)

1. Right-click on `data.api` package
2. Select **New → Kotlin Class/File**
3. Name it: `ApiService`
4. Open your `android/api_service.kt` file
5. Copy **LINES 1-95** (the API interface section)
6. Paste into `ApiService.kt`

**Location:** `app/src/main/java/com/financialanalyzer/mobile/data/api/ApiService.kt`

---

### Step 4.3: Copy Retrofit Client (Part 2 of 3)

1. Right-click on `data.network` package
2. Select **New → Kotlin Class/File**
3. Name it: `RetrofitClient`
4. Open your `android/api_service.kt` file
5. Copy **LINES 97-149** (the RetrofitClient section)
6. Paste into `RetrofitClient.kt`

**Location:** `app/src/main/java/com/financialanalyzer/mobile/data/network/RetrofitClient.kt`

---

### Step 4.4: Copy Repository (Part 3 of 3)

1. Right-click on `data.repository` package
2. Select **New → Kotlin Class/File**
3. Name it: `Repository`
4. Open your `android/api_service.kt` file
5. Copy **LINES 151-321** (the Repository section)
6. Paste into `Repository.kt`

**Location:** `app/src/main/java/com/financialanalyzer/mobile/data/repository/Repository.kt`

---

### Step 4.5: Copy ViewModels

1. Right-click on `ui.viewmodel` package
2. Select **New → Kotlin Class/File**
3. Name it: `ViewModels`
4. Copy the **ENTIRE CONTENTS** from your `android/viewmodels.kt` file
5. Paste into the new `ViewModels.kt` file

**Location:** `app/src/main/java/com/financialanalyzer/mobile/ui/viewmodel/ViewModels.kt`

---

### Step 4.6: Copy MainActivity

1. Your project already has a `MainActivity.kt` in the root `mobile` package
2. Open it in Android Studio
3. **REPLACE ALL CONTENTS** with the content from your `android/main_activity.kt` file

**Location:** `app/src/main/java/com/financialanalyzer/mobile/MainActivity.kt`

---

## ⚙️ Part 5: Configure Build Files

### Step 5.1: Update Project-Level build.gradle.kts

1. Open `build.gradle.kts` (Project: Financial Analyzer Mobile)
2. Find or add the `allprojects` block:

```kotlin
// Top-level build file
plugins {
    id("com.android.application") version "8.2.0" apply false
    id("org.jetbrains.kotlin.android") version "1.9.20" apply false
}
```

### Step 5.2: Update settings.gradle.kts

1. Open `settings.gradle.kts`
2. Make sure it looks like this:

```kotlin
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }  // For MPAndroidChart
    }
}

rootProject.name = "Financial Analyzer Mobile"
include(":app")
```

---

### Step 5.3: Update App-Level build.gradle.kts

1. Open `app/build.gradle.kts`
2. **REPLACE IT COMPLETELY** with this:

```kotlin
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.financialanalyzer.mobile"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.financialanalyzer.mobile"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        
        // ⚡ API CONFIGURATION - CHANGE THIS BASED ON YOUR SETUP
        // For Android Emulator: http://10.0.2.2:8000
        // For Physical Device: http://YOUR_LOCAL_IP:8000
        buildConfigField("String", "API_BASE_URL", "\"http://10.0.2.2:8000\"")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    
    buildFeatures {
        viewBinding = true
        buildConfig = true
    }
    
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    
    kotlinOptions {
        jvmTarget = "11"
    }
}

dependencies {
    // Core Android
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    
    // Lifecycle & ViewModel
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-livedata-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    
    // Retrofit for API calls
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")
    
    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    
    // Charts
    implementation("com.github.PhilJay:MPAndroidChart:v3.1.0")
    
    // SwipeRefreshLayout
    implementation("androidx.swiperefreshlayout:swiperefreshlayout:1.1.0")
    
    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}
```

3. Click **"Sync Now"** in the notification bar at the top

---

## 🔐 Part 6: Update AndroidManifest.xml

1. Open `app/src/main/AndroidManifest.xml`
2. Add internet permissions:

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    
    <!-- ⚡ ADD THESE PERMISSIONS -->
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
    
    <application
        android:allowBackup="true"
        android:usesCleartextTraffic="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:roundIcon="@mipmap/ic_launcher_round"
        android:supportsRtl="true"
        android:theme="@style/Theme.FinancialAnalyzer">
        
        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
        
    </application>
</manifest>
```

**Important:** Add `android:usesCleartextTraffic="true"` to allow HTTP connections (for development)

---

## 🌐 Part 7: Configure API Connection

### Step 7.1: Determine Your Setup

**Option A: Using Android Emulator**
- API URL: `http://10.0.2.2:8000`
- ✅ This is ALREADY configured in build.gradle.kts
- ✅ No changes needed!

**Option B: Using Physical Android Device**
- Need to find your computer's local IP address
- Then update the API URL in build.gradle.kts

---

### Step 7.2: Find Your Local IP Address (Physical Device Only)

**On Windows:**
1. Open Command Prompt (cmd)
2. Type: `ipconfig`
3. Look for **"IPv4 Address"** under your active network adapter
4. It will look like: `192.168.x.x` or `10.0.x.x`

**Example:**
```
Wireless LAN adapter Wi-Fi:
   IPv4 Address. . . . . . . . . . . : 192.168.1.105
```

**On Mac/Linux:**
1. Open Terminal
2. Type: `ifconfig` or `ip addr`
3. Look for **inet** address under your active network
4. Example: `192.168.1.105`

---

### Step 7.3: Update API URL (Physical Device Only)

If using a **physical device**, update your `app/build.gradle.kts`:

**FIND THIS LINE:**
```kotlin
buildConfigField("String", "API_BASE_URL", "\"http://10.0.2.2:8000\"")
```

**CHANGE TO:**
```kotlin
buildConfigField("String", "API_BASE_URL", "\"http://192.168.1.105:8000\"")
```
*(Replace `192.168.1.105` with YOUR actual IP address)*

**Then:**
1. Click **"Sync Now"**
2. Rebuild your project: **Build → Rebuild Project**

---

### Step 7.4: Ensure Same WiFi Network (Physical Device Only)

⚠️ **IMPORTANT:** Your phone and computer MUST be on the same WiFi network!

✅ Check:
- Computer connected to WiFi: "HomeNetwork"
- Phone connected to WiFi: "HomeNetwork" ← Must match!

❌ Won't work:
- Computer on WiFi, phone on mobile data
- Computer on "Network1", phone on "Network2"

---

## 🚀 Part 8: Start Your API Server

### Step 8.1: Open Terminal/Command Prompt

Navigate to your Financial Analyzer project directory:
```bash
cd C:\Users\mmiddlebass\Downloads\financial_analyzer_web_latest
```

### Step 8.2: Start the API

```bash
python proxy.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

✅ **Keep this terminal open!** The API must stay running while you use the app.

---

### Step 8.3: Test API in Browser (Optional)

Open your browser and test:

**From your computer:**
- http://localhost:8000/api/ai/health
- Should show: `{"status":"healthy"}`

**From your phone (physical device only):**
- http://192.168.1.105:8000/api/ai/health
- *(Replace with your IP)*
- Should show: `{"status":"healthy"}`

---

## ▶️ Part 9: Run Your Android App

### Step 9.1: Connect Your Device

**Option A: Emulator**
1. Click device selector dropdown at top
2. Select existing emulator OR click "Device Manager" to create one
3. Start the emulator

**Option B: Physical Device**
1. Enable Developer Options:
   - Go to Settings → About Phone
   - Tap "Build Number" 7 times
   - Developer Options now enabled!

2. Enable USB Debugging:
   - Settings → Developer Options
   - Turn ON "USB Debugging"

3. Connect via USB cable
4. Allow debugging prompt on phone

---

### Step 9.2: Build and Run

1. Click the green **▶️ Run** button (or press Shift+F10)
2. Select your device/emulator
3. Click **OK**
4. Wait for build and installation
5. App should launch automatically!

---

## ✅ Part 10: Test the Connection

### Step 10.1: Watch for Connection Test

When the app starts, it should:
1. Show a loading indicator
2. Try to connect to your API
3. Show a Toast message: **"API Connected!"** ✅
   OR
4. Show error: **"API Connection Failed"** ❌

---

### Step 10.2: Check Logcat for Errors

In Android Studio:
1. Click **Logcat** tab at bottom
2. Filter by: `API_TEST`
3. Look for:
   - ✅ `Connected successfully!`
   - ❌ `Connection failed: [error message]`

---

## 🐛 Troubleshooting

### ❌ Problem: "Cannot connect to API"

**Solution 1 - Check API is Running**
```bash
# In your terminal, you should see:
INFO:     Uvicorn running on http://0.0.0.0:8000
```
If not, run: `python proxy.py`

---

**Solution 2 - Check API URL**

**Emulator:**
- Must use: `http://10.0.2.2:8000`
- DON'T use: `http://localhost:8000` ❌

**Physical Device:**
- Must use: `http://YOUR_LOCAL_IP:8000`
- Example: `http://192.168.1.105:8000`

---

**Solution 3 - Check Firewall**

Windows Firewall might be blocking port 8000:
1. Windows Security → Firewall & network protection
2. Allow an app through firewall
3. Find Python or allow port 8000

---

**Solution 4 - Test API Manually**

Open browser on your phone and go to:
- `http://YOUR_IP:8000/api/ai/health`

If this doesn't work, the problem is network/firewall, not your app!

---

### ❌ Problem: "Gradle sync failed"

**Solution:**
1. File → Invalidate Caches → Invalidate and Restart
2. Clean project: Build → Clean Project
3. Rebuild: Build → Rebuild Project
4. Check internet connection

---

### ❌ Problem: "Unresolved reference" errors

**Solution:**
1. Make sure all files are in correct packages
2. Check package names match at top of each file
3. Sync Gradle: File → Sync Project with Gradle Files
4. Rebuild project

---

### ❌ Problem: "cleartext traffic not permitted"

**Solution:**
Already configured! But check `AndroidManifest.xml` has:
```xml
<application
    android:usesCleartextTraffic="true"
    ...>
```

---

## 🎉 Success!

If you see:
- ✅ App builds without errors
- ✅ App launches on device
- ✅ "API Connected!" toast message
- ✅ No red errors in Logcat

**Congratulations! Your Android app is connected to your Financial Analyzer API!** 🎊

---

## 📋 Quick Reference: File Locations

| Your File | Copy To | Package |
|-----------|---------|---------|
| `android/data_models.kt` | `Models.kt` | `data.model` |
| `android/api_service.kt` (lines 1-95) | `ApiService.kt` | `data.api` |
| `android/api_service.kt` (lines 97-149) | `RetrofitClient.kt` | `data.network` |
| `android/api_service.kt` (lines 151-321) | `Repository.kt` | `data.repository` |
| `android/viewmodels.kt` | `ViewModels.kt` | `ui.viewmodel` |
| `android/main_activity.kt` | `MainActivity.kt` | (root mobile) |

---

## 🔧 Configuration Checklist

- [ ] Created all required packages
- [ ] Copied all Kotlin files to correct locations
- [ ] Updated project-level build.gradle.kts
- [ ] Updated settings.gradle.kts (added jitpack.io)
- [ ] Updated app-level build.gradle.kts with dependencies
- [ ] Set correct API_BASE_URL in build.gradle.kts
- [ ] Updated AndroidManifest.xml with permissions
- [ ] Synced Gradle successfully
- [ ] Started API server (python proxy.py)
- [ ] Connected device/emulator
- [ ] Built and ran app successfully
- [ ] Saw "API Connected!" message

---

## 📞 Need Help?

Check these files for more info:
- `ANDROID_QUICK_START_GUIDE.md` - Quick setup overview
- `ANDROID_STUDIO_INTEGRATION_GUIDE.md` - Detailed technical guide

**You're all set! Start building your financial app!** 🚀











# 📱 Financial Analyzer Pro - Android Quick Start Guide

## 🚀 Get Started in 15 Minutes!

This guide will help you quickly set up your Android app to connect with your Financial Analyzer Pro backend.

## ✅ Prerequisites Checklist

- [ ] Android Studio installed (Electric Eel or later)
- [ ] Financial Analyzer Pro API running (`python proxy.py`)
- [ ] Basic understanding of Android development

## 📋 Step-by-Step Setup

### **Step 1: Create Android Project** (3 minutes)

1. Open Android Studio
2. Click **"New Project"**
3. Select **"Empty Activity"**
4. Configure:
   ```
   Name: Financial Analyzer Mobile
   Package: com.financialanalyzer.mobile
   Language: Kotlin
   Minimum SDK: API 24 (Android 7.0)
   ```
5. Click **"Finish"**

### **Step 2: Copy Project Files** (2 minutes)

Copy these files to your Android project:

```
android/
├── data_models.kt          → app/src/main/java/com/financialanalyzer/mobile/data/model/Models.kt
├── api_service.kt          → app/src/main/java/com/financialanalyzer/mobile/data/api/ApiService.kt
├── main_activity.kt        → app/src/main/java/com/financialanalyzer/mobile/MainActivity.kt
└── viewmodels.kt           → app/src/main/java/com/financialanalyzer/mobile/ui/viewmodel/ViewModels.kt
```

### **Step 3: Update build.gradle** (3 minutes)

**1. Project level `build.gradle.kts`:**

```kotlin
plugins {
    id("com.android.application") version "8.2.0" apply false
    id("org.jetbrains.kotlin.android") version "1.9.20" apply false
}

// Add this for MPAndroidChart
allprojects {
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}
```

**2. App level `app/build.gradle.kts`:**

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
        
        // IMPORTANT: Change this to your API URL
        // Emulator: http://10.0.2.2:8000
        // Physical device: http://YOUR_LOCAL_IP:8000
        buildConfigField("String", "API_BASE_URL", "\"http://10.0.2.2:8000\"")
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
    // Core
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    
    // Lifecycle & ViewModel
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-livedata-ktx:2.7.0")
    
    // Retrofit
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")
    
    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    
    // Charts
    implementation("com.github.PhilJay:MPAndroidChart:v3.1.0")
    
    // SwipeRefreshLayout
    implementation("androidx.swiperefreshlayout:swiperefreshlayout:1.1.0")
}
```

**3. Sync Gradle:** Click "Sync Now" in Android Studio

### **Step 4: Update AndroidManifest.xml** (1 minute)

Add internet permission in `app/src/main/AndroidManifest.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    
    <!-- Add these permissions -->
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

### **Step 5: Start Your API Server** (1 minute)

```bash
# In your Financial Analyzer Pro directory
python proxy.py

# You should see:
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

### **Step 6: Configure API URL** (2 minutes)

**For Android Emulator:**
- API URL: `http://10.0.2.2:8000`
- This maps to `localhost:8000` on your computer

**For Physical Device:**
1. Find your computer's local IP:
   ```bash
   # Windows
   ipconfig
   # Look for IPv4 Address: 192.168.x.x
   
   # Mac/Linux
   ifconfig
   # Look for inet 192.168.x.x
   ```
2. Update `build.gradle.kts`:
   ```kotlin
   buildConfigField("String", "API_BASE_URL", "\"http://192.168.x.x:8000\"")
   ```
3. Ensure phone and computer are on same WiFi network
4. Rebuild project

### **Step 7: Test Connection** (3 minutes)

Create a simple test in MainActivity:

```kotlin
override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    
    // Quick connection test
    lifecycleScope.launch {
        try {
            val response = RetrofitClient.apiService.getHealthCheck()
            if (response.isSuccessful) {
                Log.d("API_TEST", "✅ Connected successfully!")
                Toast.makeText(this@MainActivity, "API Connected!", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e("API_TEST", "❌ Connection failed: ${e.message}")
            Toast.makeText(this@MainActivity, "API Connection Failed", Toast.LENGTH_SHORT).show()
        }
    }
}
```

### **Step 8: Run the App!** (1 minute)

1. Click ▶️ "Run" in Android Studio
2. Select emulator or connected device
3. Wait for app to install and launch
4. Check for "API Connected!" toast message

## 🎯 What You Can Do Now

### **Get Market Overview**
```kotlin
viewModelScope.launch {
    val response = repository.getMarketOverview()
    if (response.isSuccess) {
        // Use response.data
    }
}
```

### **Get Stock Data**
```kotlin
viewModelScope.launch {
    val response = repository.getMarketData("AAPL", period = "1y")
    if (response.isSuccess) {
        // Display stock data
    }
}
```

### **Get Portfolio**
```kotlin
viewModelScope.launch {
    val response = repository.getPortfolioData()
    if (response.isSuccess) {
        // Show portfolio
    }
}
```

## 🐛 Troubleshooting

### **Problem: Cannot connect to API**

**Solution 1 - Emulator:**
```
✅ Use: http://10.0.2.2:8000
❌ Don't use: http://localhost:8000
```

**Solution 2 - Physical Device:**
```
1. Check WiFi - Same network?
2. Check IP address - Correct?
3. Check firewall - Allowed?
4. Test in browser: http://YOUR_IP:8000/api/ai/health
```

**Solution 3 - API not running:**
```bash
# Start the API
python proxy.py

# Should show:
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

### **Problem: SSL/TLS errors**

**Solution:**
```xml
<!-- In AndroidManifest.xml -->
<application
    android:usesCleartextTraffic="true"
    ...>
```

### **Problem: Gradle sync failed**

**Solution:**
```
1. File → Invalidate Caches → Invalidate and Restart
2. Update Android Studio to latest version
3. Check internet connection
4. Clean and rebuild project
```

### **Problem: Module dependency errors**

**Solution:**
```kotlin
// Make sure settings.gradle.kts has:
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}
```

## 📱 Test on Physical Device

1. **Enable Developer Mode:**
   - Settings → About Phone
   - Tap "Build Number" 7 times

2. **Enable USB Debugging:**
   - Settings → Developer Options
   - Turn on "USB Debugging"

3. **Connect Device:**
   - Connect via USB
   - Allow debugging when prompted

4. **Update API URL:**
   ```kotlin
   buildConfigField("String", "API_BASE_URL", "\"http://YOUR_IP:8000\"")
   ```

5. **Rebuild and Run:**
   - Build → Clean Project
   - Build → Rebuild Project
   - Run → Run 'app'

## 🎨 Customize Your App

### **Change Colors**
Edit `res/values/colors.xml`:
```xml
<resources>
    <color name="primary">#667EEA</color>
    <color name="green">#28A745</color>
    <color name="red">#DC3545</color>
</resources>
```

### **Change App Name**
Edit `res/values/strings.xml`:
```xml
<resources>
    <string name="app_name">Financial Analyzer</string>
</resources>
```

### **Change Icon**
Replace files in `res/mipmap-*/`:
- `ic_launcher.png`
- `ic_launcher_round.png`

## 📊 Available API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/api/ai/market-data/{ticker}` | Get stock price & indicators |
| `/api/ai/market-overview` | Get market indices |
| `/api/ai/global-markets` | Get global markets data |
| `/api/ai/portfolio-data` | Get portfolio information |
| `/api/ai/technical-analysis/{ticker}` | Get technical indicators |
| `/api/ai/risk-analysis/{ticker}` | Get risk metrics |
| `/api/ai/predictions/{ticker}` | Get ML predictions |
| `/api/ai/batch-market-data` | Get multiple stocks |
| `/api/ai/health` | Check API health |

## 🚀 Next Steps

1. ✅ **Complete basic setup** (you're here!)
2. 📊 **Add charts** for price visualization
3. 🎨 **Customize UI** to match your brand
4. 📱 **Add more features** (alerts, watchlist, etc.)
5. 🚀 **Deploy to Google Play** Store

## 📚 Additional Resources

- [Android Developer Guide](https://developer.android.com/)
- [Kotlin Documentation](https://kotlinlang.org/docs/home.html)
- [Retrofit Guide](https://square.github.io/retrofit/)
- [MPAndroidChart](https://github.com/PhilJay/MPAndroidChart)

## 💡 Pro Tips

1. **Use ViewBinding** - Already configured!
2. **Test on real device** - More accurate than emulator
3. **Check logs** - Android Studio → Logcat
4. **Use Postman** - Test API endpoints independently
5. **Keep API running** - Don't close `python proxy.py`

## ✅ Success Checklist

- [ ] Android Studio project created
- [ ] Dependencies added to build.gradle
- [ ] Internet permission added
- [ ] API server running
- [ ] API URL configured correctly
- [ ] App builds successfully
- [ ] App connects to API
- [ ] Market data loads
- [ ] No error messages

## 🎉 You're Ready!

Your Android app is now connected to your Financial Analyzer Pro backend. You can:
- ✅ Get real-time market data
- ✅ View technical indicators
- ✅ Analyze risk metrics
- ✅ Get ML predictions
- ✅ Track portfolios

**Need help?** Check `ANDROID_STUDIO_INTEGRATION_GUIDE.md` for detailed documentation!












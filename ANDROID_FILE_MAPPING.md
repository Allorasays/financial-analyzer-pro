# 📁 Android File Mapping - Visual Reference

## Quick Copy Guide

This is a visual reference showing **exactly** where each file goes.

---

## 📊 File Structure Overview

```
financial_analyzer_web_latest/android/     →    Android Studio Project
├── data_models.kt                         →    app/src/main/java/com/financialanalyzer/mobile/data/model/Models.kt
├── api_service.kt (Split into 3 files)    →    (See below)
├── viewmodels.kt                          →    app/src/main/java/com/financialanalyzer/mobile/ui/viewmodel/ViewModels.kt
└── main_activity.kt                       →    app/src/main/java/com/financialanalyzer/mobile/MainActivity.kt
```

---

## 📦 Package Structure to Create

Create these packages in Android Studio:

```
com.financialanalyzer.mobile/
├── data/
│   ├── api/                    ← New package
│   ├── model/                  ← New package
│   ├── network/                ← New package
│   └── repository/             ← New package
└── ui/
    └── viewmodel/              ← New package
```

---

## 📄 File 1: Data Models

**Source:** `android/data_models.kt`  
**Destination:** `data.model` package  
**New Name:** `Models.kt`

```
android/data_models.kt
    ↓ Copy entire file
app/src/main/java/com/financialanalyzer/mobile/data/model/Models.kt
```

**What it contains:**
- MarketDataResponse
- PriceData
- TechnicalIndicators
- RiskMetrics
- PortfolioResponse
- All other data models

---

## 📄 File 2: API Service (Part 1 of 3)

**Source:** `android/api_service.kt` (Lines 1-95)  
**Destination:** `data.api` package  
**New Name:** `ApiService.kt`

```
android/api_service.kt (Lines 1-95)
    ↓ Copy lines 1-95 only
app/src/main/java/com/financialanalyzer/mobile/data/api/ApiService.kt
```

**What it contains:**
- FinancialAnalyzerApiService interface
- All API endpoint definitions
- GET/POST method declarations

**Starts with:**
```kotlin
package com.financialanalyzer.mobile.data.api
```

**Ends with:**
```kotlin
    @GET("api/ai/health")
    suspend fun getHealthCheck(): Response<HealthResponse>
}
```

---

## 📄 File 3: Retrofit Client (Part 2 of 3)

**Source:** `android/api_service.kt` (Lines 97-149)  
**Destination:** `data.network` package  
**New Name:** `RetrofitClient.kt`

```
android/api_service.kt (Lines 97-149)
    ↓ Copy lines 97-149 only
app/src/main/java/com/financialanalyzer/mobile/data/network/RetrofitClient.kt
```

**What it contains:**
- RetrofitClient object
- OkHttp configuration
- Gson configuration
- Logging interceptor
- API service instance

**Starts with:**
```kotlin
package com.financialanalyzer.mobile.data.network
```

**Ends with:**
```kotlin
    val apiService: FinancialAnalyzerApiService = 
        retrofit.create(FinancialAnalyzerApiService::class.java)
}
```

---

## 📄 File 4: Repository (Part 3 of 3)

**Source:** `android/api_service.kt` (Lines 151-321)  
**Destination:** `data.repository` package  
**New Name:** `Repository.kt`

```
android/api_service.kt (Lines 151-321)
    ↓ Copy lines 151-321 only
app/src/main/java/com/financialanalyzer/mobile/data/repository/Repository.kt
```

**What it contains:**
- FinancialAnalyzerRepository class
- All repository methods
- Error handling
- Response processing

**Starts with:**
```kotlin
package com.financialanalyzer.mobile.data.repository
```

**Ends with:**
```kotlin
        }
    }
}
```

---

## 📄 File 5: ViewModels

**Source:** `android/viewmodels.kt`  
**Destination:** `ui.viewmodel` package  
**New Name:** `ViewModels.kt`

```
android/viewmodels.kt
    ↓ Copy entire file
app/src/main/java/com/financialanalyzer/mobile/ui/viewmodel/ViewModels.kt
```

**What it contains:**
- DashboardViewModel
- StockDetailViewModel
- PortfolioViewModel
- LiveData definitions

---

## 📄 File 6: MainActivity

**Source:** `android/main_activity.kt`  
**Destination:** Root `mobile` package (already exists)  
**Action:** REPLACE existing MainActivity.kt

```
android/main_activity.kt
    ↓ Replace entire file
app/src/main/java/com/financialanalyzer/mobile/MainActivity.kt
```

**What it contains:**
- MainActivity class
- UI setup with ViewBinding
- Chart configuration
- API connection test

---

## 🔧 Configuration Files

### build.gradle.kts (App Level)

**Location:** `app/build.gradle.kts`  
**Action:** REPLACE or UPDATE

**Key configuration:**
```kotlin
buildConfigField("String", "API_BASE_URL", "\"http://10.0.2.2:8000\"")
```

**Change to for physical device:**
```kotlin
buildConfigField("String", "API_BASE_URL", "\"http://YOUR_IP:8000\"")
```

---

### settings.gradle.kts (Project Level)

**Location:** `settings.gradle.kts`  
**Action:** UPDATE

**Add this repository:**
```kotlin
repositories {
    google()
    mavenCentral()
    maven { url = uri("https://jitpack.io") }  // ← Add this line
}
```

---

### AndroidManifest.xml

**Location:** `app/src/main/AndroidManifest.xml`  
**Action:** ADD permissions

**Add before `<application>` tag:**
```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
```

**Add in `<application>` tag:**
```xml
android:usesCleartextTraffic="true"
```

---

## 🌐 API Configuration Reference

### For Android Emulator

```kotlin
// In app/build.gradle.kts
buildConfigField("String", "API_BASE_URL", "\"http://10.0.2.2:8000\"")
```

**Why `10.0.2.2`?**
- Android emulator uses `10.0.2.2` to access `localhost` on your computer
- This is a special IP address that routes to your host machine

---

### For Physical Android Device

**Step 1: Find your IP**
```bash
# Windows
ipconfig
# Look for: IPv4 Address . . . : 192.168.x.x

# Mac/Linux
ifconfig
# Look for: inet 192.168.x.x
```

**Step 2: Update build.gradle.kts**
```kotlin
// Replace 192.168.1.105 with YOUR IP
buildConfigField("String", "API_BASE_URL", "\"http://192.168.1.105:8000\"")
```

**Step 3: Sync and Rebuild**
- Click "Sync Now"
- Build → Rebuild Project

---

## ✅ Verification Checklist

After copying all files:

### Files Created
- [ ] `data/model/Models.kt` exists
- [ ] `data/api/ApiService.kt` exists
- [ ] `data/network/RetrofitClient.kt` exists
- [ ] `data/repository/Repository.kt` exists
- [ ] `ui/viewmodel/ViewModels.kt` exists
- [ ] `MainActivity.kt` replaced

### Package Declarations
- [ ] Each file has correct package name at top
- [ ] No "Unresolved reference" errors
- [ ] All imports are green (not red)

### Configuration
- [ ] `app/build.gradle.kts` has all dependencies
- [ ] `settings.gradle.kts` has jitpack.io repository
- [ ] `AndroidManifest.xml` has internet permissions
- [ ] `API_BASE_URL` is configured correctly
- [ ] Gradle sync completed successfully

### API Server
- [ ] Python API server is running (`python proxy.py`)
- [ ] Can access http://localhost:8000/api/ai/health in browser
- [ ] Firewall allows port 8000

### Build
- [ ] Project builds without errors
- [ ] No lint errors (red underlines)
- [ ] App installs on device/emulator
- [ ] "API Connected!" toast appears

---

## 🎯 Quick Command Reference

### Start API Server
```bash
cd C:\Users\mmiddlebass\Downloads\financial_analyzer_web_latest
python proxy.py
```

### Find Your IP (Windows)
```bash
ipconfig | findstr IPv4
```

### Find Your IP (Mac/Linux)
```bash
ifconfig | grep "inet "
```

### Test API from Terminal
```bash
# Should return {"status":"healthy"}
curl http://localhost:8000/api/ai/health
```

---

## 📞 Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| Cannot connect to API | Check API is running: `python proxy.py` |
| Emulator can't connect | Use `http://10.0.2.2:8000` not `localhost` |
| Device can't connect | Use `http://YOUR_LOCAL_IP:8000` |
| Gradle sync failed | Invalidate caches and restart |
| Unresolved reference | Check package names match |
| Build error | Clean and rebuild project |
| Cleartext traffic error | Add `usesCleartextTraffic="true"` to manifest |

---

## 📚 Visual Package Structure

```
com.financialanalyzer.mobile
│
├── 📱 MainActivity.kt                           [Root level - UI entry point]
│
├── 📊 data/                                     [Data layer]
│   │
│   ├── 🔌 api/                                  [API definitions]
│   │   └── ApiService.kt                        [Retrofit interface]
│   │
│   ├── 📦 model/                                [Data models]
│   │   └── Models.kt                            [All response models]
│   │
│   ├── 🌐 network/                              [Network setup]
│   │   └── RetrofitClient.kt                    [HTTP client]
│   │
│   └── 💾 repository/                           [Data repository]
│       └── Repository.kt                        [API calls + error handling]
│
└── 🎨 ui/                                       [UI layer]
    └── viewmodel/                               [ViewModels]
        └── ViewModels.kt                        [All ViewModels]
```

---

## 🎉 Summary

**6 files** need to be created/copied:

1. ✅ `Models.kt` → All data structures
2. ✅ `ApiService.kt` → API endpoint definitions  
3. ✅ `RetrofitClient.kt` → HTTP client setup
4. ✅ `Repository.kt` → API call wrapper
5. ✅ `ViewModels.kt` → UI data management
6. ✅ `MainActivity.kt` → Main UI screen

**3 config files** need to be edited:

1. ✅ `app/build.gradle.kts` → Dependencies + API URL
2. ✅ `settings.gradle.kts` → Add jitpack.io
3. ✅ `AndroidManifest.xml` → Internet permissions

**Total setup time:** ~15-20 minutes

---

**Ready to code?** Follow the detailed instructions in `ANDROID_SETUP_STEP_BY_STEP.md`!











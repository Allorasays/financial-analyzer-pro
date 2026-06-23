# Android Studio Rebuild Guide for ML Predictions & Technical Analysis

## Overview
This guide will help you rebuild your Android app to fix ML prediction errors and enable technical analysis functionality.

## Prerequisites
- Android Studio installed and configured
- Your project open in Android Studio
- API server running on http://localhost:8000

## Step-by-Step Rebuild Process

### Phase 1: Backup Working Functions (SAFETY FIRST)

#### Step 1.1: Create Backup Folder
```bash
# In Android Studio Terminal
mkdir backup_working_functions
```

#### Step 1.2: Backup Key Files
```bash
# Backup your current working files
cp app/src/main/java/com/financialanalyzer/mobile/MainActivity.kt backup_working_functions/
cp app/src/main/java/com/financialanalyzer/mobile/data_models.kt backup_working_functions/
cp app/src/main/java/com/financialanalyzer/mobile/api_service.kt backup_working_functions/
cp app/src/main/java/com/financialanalyzer/mobile/viewmodels.kt backup_working_functions/
cp app/src/main/res/layout/activity_stock_detail.xml backup_working_functions/
```

#### Step 1.3: Git Backup (if using Git)
```bash
git add .
git commit -m "Backup: Working functions before ML/Technical Analysis updates"
git push origin main
```

### Phase 2: Update Android Files

#### Step 2.1: Replace MainActivity.kt
The file `android/main_activity.kt` has been updated with:
- ✅ Fixed imports
- ✅ Added missing View import
- ✅ Added ML predictions observer
- ✅ Added updatePredictions function
- ✅ Commented out non-existent UI references

#### Step 2.2: Verify Data Models
Ensure `android/data_models.kt` contains:
- ✅ Complete PredictionsResponse class
- ✅ ModelMetrics and FuturePrediction classes
- ✅ TechnicalAnalysisResponse class

#### Step 2.3: Verify API Service
Ensure `android/api_service.kt` contains:
- ✅ Correct prediction_days parameter
- ✅ Technical analysis endpoint

#### Step 2.4: Verify ViewModels
Ensure `android/viewmodels.kt` contains:
- ✅ loadPredictions function with correct parameters
- ✅ loadTechnicalAnalysis function

#### Step 2.5: Verify Layout
Ensure `android/activity_stock_detail.xml` contains:
- ✅ Technical Analysis card with proper IDs
- ✅ All required TextView elements

### Phase 3: Clean and Rebuild

#### Step 3.1: Clean Project
```bash
# In Android Studio Terminal
./gradlew clean
```

#### Step 3.2: Sync Project
- Click "Sync Now" if prompted
- Wait for sync to complete

#### Step 3.3: Rebuild Project
```bash
# In Android Studio Terminal
./gradlew build
```

#### Step 3.4: Uninstall Old App
- Run → Uninstall App

#### Step 3.5: Install Fresh Build
- Run → Run 'app'

### Phase 4: Verification

#### Step 4.1: Test ML Predictions
1. Open stock detail view (e.g., AAPL)
2. Check that predictions show real data instead of errors
3. Verify next day, week, month, quarter predictions work

#### Step 4.2: Test Technical Analysis
1. Navigate to stock detail view
2. Check Technical Analysis card is visible
3. Verify SMA 20, SMA 50, MACD, Trend, Signal show real data

#### Step 4.3: Test Other Features
1. Stock analysis
2. Portfolio management
3. Market overview
4. Risk assessment

### Phase 5: Rollback (if needed)

If anything breaks:
```bash
# Restore from backup
cp backup_working_functions/*.kt app/src/main/java/com/financialanalyzer/mobile/
cp backup_working_functions/activity_stock_detail.xml app/src/main/res/layout/
./gradlew clean
./gradlew build
```

## Expected Results

After successful rebuild:
- ✅ ML predictions work with real data
- ✅ Technical analysis shows indicators
- ✅ All existing features continue to work
- ✅ No more "coming soon" messages
- ✅ No more "error" messages in predictions

## Troubleshooting

### If ML Predictions Still Show Errors:
1. Check API server is running: http://localhost:8000
2. Check API endpoint: http://localhost:8000/api/ai/predictions/AAPL
3. Verify network permissions in AndroidManifest.xml
4. Check API base URL in BuildConfig

### If Technical Analysis Still Shows "Coming Soon":
1. Check API endpoint: http://localhost:8000/api/ai/technical-analysis/AAPL
2. Verify updateTechnicalIndicators function is called
3. Check that UI elements exist in layout

### If Unresolved References Appear:
1. Clean and rebuild project
2. Invalidate caches and restart Android Studio
3. Check all imports are correct

## API Endpoints to Test

Test these endpoints in your browser or Postman:
- ML Predictions: http://localhost:8000/api/ai/predictions/AAPL?prediction_days=5
- Technical Analysis: http://localhost:8000/api/ai/technical-analysis/AAPL
- Market Overview: http://localhost:8000/api/market/overview

## Success Indicators

You'll know the rebuild was successful when:
1. Android Studio compiles without errors
2. App installs and runs on emulator/device
3. ML predictions show real numbers
4. Technical analysis displays indicators
5. No "error" or "coming soon" messages

## Support

If you encounter issues:
1. Check the backup files first
2. Verify API server is running
3. Check Android Studio logs
4. Test API endpoints directly

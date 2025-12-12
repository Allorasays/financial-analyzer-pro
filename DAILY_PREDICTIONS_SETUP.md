# Daily Predictions & Accuracy Tracking Setup

## ✅ **System Configured**

Moneta now automatically makes **10 predictions per day** and tracks their accuracy against actual outcomes.

---

## 🤖 **How It Works**

### **Daily Schedule**
- **Time**: 11 PM UTC (6 PM ET) - After market close
- **Action**: 
  1. Makes 10 new predictions for different stocks
  2. Validates any pending predictions (from previous days)

### **Prediction Selection**
- Rotates through a pool of 20+ popular stocks
- Selection varies daily (based on date hash) for diversity
- Includes: AAPL, MSFT, GOOGL, NVDA, TSLA, SPY, QQQ, and more

### **Automatic Tracking**
- Each prediction is stored with target date
- System automatically validates when target date arrives
- Accuracy metrics calculated and available via API

---

## 📊 **What Gets Tracked**

For each of the 10 daily predictions:
- **Next Day** prediction (1 day ahead)
- **Next Week** prediction (7 days ahead)  
- **Next Month** prediction (30 days ahead)
- **Next Quarter** prediction (90 days ahead)

**Total**: ~40 predictions stored per day (10 stocks × 4 timeframes)

---

## 🔧 **Files Created**

1. **`daily_predictions_job.py`**
   - Makes 10 predictions per day
   - Rotates stock selection
   - Logs results

2. **`daily_validation_job.py`**
   - Validates pending predictions
   - Calculates accuracy metrics
   - Shows direction accuracy

3. **`combined_daily_job.py`**
   - Runs both prediction and validation
   - Single job for daily execution
   - Comprehensive logging

4. **`render_final.yaml`** (updated)
   - Added cron job scheduled for 11 PM UTC daily

---

## 📈 **Viewing Results**

### **Check Daily Predictions Made**
```bash
# View pending validations (predictions waiting to be checked)
curl https://moneta-backend-api.onrender.com/api/prediction-pending
```

### **Check Accuracy Metrics**
```bash
# Overall accuracy
curl https://moneta-backend-api.onrender.com/api/prediction-accuracy

# Recent accuracy (last 30 days)
curl https://moneta-backend-api.onrender.com/api/prediction-accuracy/recent?days=30
```

### **Manually Trigger Jobs** (if needed)
```bash
# Make predictions now
python daily_predictions_job.py

# Validate pending predictions now
python daily_validation_job.py

# Run both
python combined_daily_job.py
```

---

## ⏰ **Schedule Details**

**Render Cron Job**:
- **Schedule**: `0 23 * * *` (11 PM UTC daily)
- **Runs**: `combined_daily_job.py`
- **Does**: Makes 10 predictions + validates pending ones

**Why 11 PM UTC?**
- Market closes at 4 PM ET (9 PM UTC)
- Gives time for final prices to settle
- Validates previous day's predictions
- Makes new predictions for next day

---

## 📊 **Expected Results**

After running for a week:
- **~70 predictions** stored (10/day × 7 days)
- **~30-40 validations** (next_day predictions validated)
- **Direction accuracy** visible
- **Price error metrics** available

After running for a month:
- **~300 predictions** stored
- **~280 validations** (most next_day and next_week)
- **Comprehensive accuracy stats**
- **Trends visible** (improving/degrading accuracy)

---

## 🎯 **Accuracy Metrics You'll See**

1. **Direction Accuracy**: % of correct up/down predictions
2. **Mean Absolute Error**: Average $ difference
3. **RMSE**: Root Mean Squared Error
4. **Mean % Error**: Average percentage difference
5. **Accuracy by Ticker**: Which stocks are easier to predict
6. **Accuracy by Horizon**: 1-day vs 7-day vs 30-day accuracy

---

## 🔍 **Monitoring**

### **Check Job Status**
Render dashboard will show cron job execution logs.

### **Check Database**
Predictions stored in: `data/prediction_tracker.db`

### **API Endpoints**
All metrics available via REST API (see above)

---

## ⚙️ **Configuration**

### **Change Number of Predictions**
Edit `daily_predictions_job.py`:
```python
make_daily_predictions(base_url=base_url, count=10)  # Change 10 to desired number
```

### **Change Schedule**
Edit `render_final.yaml`:
```yaml
schedule: "0 23 * * *"  # Change to desired time (cron format)
```

### **Change Stock Pool**
Edit `daily_predictions_job.py`:
```python
DEFAULT_TICKERS = [
    "AAPL", "MSFT", ...  # Add/remove tickers
]
```

---

## ✅ **Status**

- ✅ Daily predictions job created
- ✅ Validation job created  
- ✅ Combined job created
- ✅ Render cron job configured
- ✅ Automatic tracking enabled
- ✅ API endpoints available

**The system will start making 10 predictions per day automatically once deployed!**

---

## 🚀 **Next Steps**

1. **Deploy to Render** - Cron job will start automatically
2. **Wait 1-2 days** - Let predictions accumulate
3. **Check accuracy** - Use API endpoints to view metrics
4. **Monitor trends** - Track accuracy over time
5. **Optimize model** - Use real accuracy data to improve predictions



# Daily Predictions - Quick Start Guide

## ✅ **System Ready**

Moneta is now configured to automatically make **10 predictions per day** and track their accuracy.

---

## 🚀 **What Happens Automatically**

### **Every Day at 11 PM UTC (6 PM ET):**

1. **Makes 10 New Predictions**
   - Selects 10 different stocks (rotates daily)
   - Makes predictions for: next day, next week, next month, next quarter
   - Stores all predictions in database

2. **Validates Pending Predictions**
   - Checks predictions from previous days
   - Compares predicted vs actual prices
   - Calculates accuracy metrics

---

## 📊 **View Results**

### **Via API:**

```bash
# Check accuracy metrics
curl https://moneta-backend-api.onrender.com/api/prediction-accuracy

# Check recent accuracy (last 30 days)
curl https://moneta-backend-api.onrender.com/api/prediction-accuracy/recent?days=30

# See pending validations
curl https://moneta-backend-api.onrender.com/api/prediction-pending
```

### **Response Example:**
```json
{
  "status": "success",
  "metrics": {
    "total_validations": 150,
    "direction_accuracy_pct": 82.0,
    "mean_absolute_error": 2.45,
    "rmse": 3.12,
    "correct_predictions": 123,
    "total_predictions": 150
  }
}
```

---

## 🧪 **Test Locally**

```bash
# Test making predictions
python test_daily_predictions.py

# Or run the full daily job
python combined_daily_job.py
```

---

## ⏰ **Timeline**

- **Day 1**: 10 predictions made, stored
- **Day 2**: 10 more predictions + validate Day 1's "next_day" predictions
- **Day 8**: Validate Day 1's "next_week" predictions
- **Day 31**: Validate Day 1's "next_month" predictions
- **Day 91**: Validate Day 1's "next_quarter" predictions

**After 1 week**: ~70 predictions stored, ~70 validations (next_day)
**After 1 month**: ~300 predictions stored, ~280 validations
**After 3 months**: ~900 predictions stored, comprehensive accuracy data

---

## 📈 **What You'll Learn**

1. **Real-world accuracy** (not just model fit)
2. **Which stocks are easier to predict**
3. **Which timeframes are most accurate** (1-day vs 30-day)
4. **Direction accuracy** (% of correct up/down calls)
5. **Price error trends** (improving or degrading over time)

---

## ✅ **Status**

- ✅ Daily job configured
- ✅ Cron schedule set (11 PM UTC daily)
- ✅ Automatic tracking enabled
- ✅ API endpoints ready
- ✅ Ready to deploy!

**Once deployed to Render, the system will start automatically!**








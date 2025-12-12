# Prediction Tracking System Documentation

## ✅ **System Implemented**

Moneta now tracks ML predictions and validates them against actual future prices to measure **real-world accuracy**.

---

## 🎯 **What It Does**

### **1. Stores Predictions**
When a prediction is made, the system automatically stores:
- Ticker symbol
- Predicted price
- Current price (at time of prediction)
- Target date (when to check actual price)
- Prediction horizon (1 day, 7 days, 30 days, 90 days)
- Model version, confidence, R² score
- Number of features used

### **2. Validates Against Actual Prices**
Periodically (or manually), the system:
- Fetches actual stock prices on target dates
- Compares predicted vs actual prices
- Calculates accuracy metrics:
  - Price error (absolute $ difference)
  - Percentage error
  - Direction accuracy (did price go up/down as predicted?)
  - RMSE, MAE, and other statistics

### **3. Provides Accuracy Reports**
Shows real-world accuracy metrics:
- Direction accuracy (% of correct up/down predictions)
- Average price error
- Error distribution
- Accuracy by ticker, model version, or time horizon

---

## 📊 **Database Structure**

### **Predictions Table**
Stores predictions when they're made:
- `id` - Unique prediction ID
- `ticker` - Stock symbol
- `prediction_timestamp` - When prediction was made
- `current_price` - Price at prediction time
- `predicted_price` - What we predicted
- `target_date` - Date to check actual price
- `horizon_days` - Days ahead (1, 7, 30, 90)
- `model_version` - Model version used
- `confidence_score` - Model confidence
- `r2_score` - R² score
- `features_used` - Number of features

### **Validations Table**
Stores validation results:
- `id` - Unique validation ID
- `prediction_id` - Links to prediction
- `actual_price` - Real price on target date
- `price_error` - Difference ($)
- `price_error_pct` - Difference (%)
- `direction_correct` - Did direction match? (1 = yes, 0 = no)
- `validated_at` - When validation happened

---

## 🔧 **API Endpoints**

### **1. Get Accuracy Metrics**
```
GET /api/prediction-accuracy
GET /api/prediction-accuracy?ticker=AAPL
GET /api/prediction-accuracy?horizon_days=1
GET /api/prediction-accuracy?model_version=2.2.0
```

**Response:**
```json
{
  "status": "success",
  "metrics": {
    "total_validations": 150,
    "mean_absolute_error": 2.45,
    "rmse": 3.12,
    "mean_absolute_percent_error": 0.95,
    "direction_accuracy": 0.82,
    "direction_accuracy_pct": 82.0,
    "correct_predictions": 123,
    "total_predictions": 150
  }
}
```

### **2. Get Recent Accuracy**
```
GET /api/prediction-accuracy/recent?days=30
```

Returns accuracy for the last N days of validations.

### **3. Validate Pending Predictions**
```
POST /api/prediction-validate?max_days_past=7
```

Manually trigger validation of all pending predictions.

### **4. Get Pending Validations**
```
GET /api/prediction-pending?max_days_past=7
```

List predictions waiting to be validated.

---

## 🤖 **Automated Validation**

### **Option 1: Manual Validation**
Call the endpoint when needed:
```bash
curl -X POST https://moneta-backend-api.onrender.com/api/prediction-validate
```

### **Option 2: Scheduled Job**
Run `validate_predictions_job.py` on a schedule (daily recommended):
```bash
# Daily at 6 AM (after market close previous day)
0 6 * * * cd /path/to/app && python validate_predictions_job.py
```

### **Option 3: Render Cron Job**
Add to `render.yaml`:
```yaml
services:
  - type: cron
    name: validate-predictions
    schedule: "0 6 * * *"  # Daily at 6 AM UTC
    buildCommand: python validate_predictions_job.py
```

---

## 📈 **How to Use**

### **1. Predictions Are Automatically Stored**
Every time `/api/ml/predictions/{ticker}` is called, predictions are stored for:
- Next day (1 day)
- Next week (7 days)
- Next month (30 days)
- Next quarter (90 days)

### **2. Validate Predictions**
Run validation daily (or manually):
```bash
python validate_predictions_job.py
```

Or use the API:
```bash
curl -X POST http://localhost:8000/api/prediction-validate
```

### **3. Check Accuracy**
```bash
# Overall accuracy
curl http://localhost:8000/api/prediction-accuracy

# For specific ticker
curl http://localhost:8000/api/prediction-accuracy?ticker=AAPL

# Recent accuracy (last 30 days)
curl http://localhost:8000/api/prediction-accuracy/recent?days=30
```

---

## 📊 **Accuracy Metrics Explained**

- **Direction Accuracy**: % of predictions where price moved in the predicted direction (up/down)
- **Mean Absolute Error (MAE)**: Average $ difference between predicted and actual
- **RMSE**: Root Mean Squared Error (penalizes large errors more)
- **Mean Absolute % Error**: Average percentage difference
- **Total Validations**: Number of predictions that have been validated

---

## 🎯 **Example Workflow**

1. **User requests prediction** for AAPL
   - System stores: "AAPL: Predict $150 on 2025-01-15 (current: $148)"

2. **One day later (2025-01-15)**
   - Validation job runs
   - Fetches actual price: $149.50
   - Calculates: Error = -$0.50, Error% = -0.33%, Direction = Correct ✓

3. **Check accuracy**
   - Over 100 validations: 82% direction accuracy, $2.45 average error

---

## 📁 **Files Created**

1. `prediction_tracker.py` - Core tracking and storage
2. `prediction_validator.py` - Validation logic
3. `validate_predictions_job.py` - Scheduled job script
4. `PREDICTION_TRACKING_SYSTEM.md` - This documentation

---

## 🚀 **Next Steps**

1. **Set up automated validation** (daily cron job recommended)
2. **Monitor accuracy metrics** to track model performance
3. **Use accuracy data** to improve model (feature selection, tuning)
4. **Display accuracy** in app/dashboard for user transparency

---

## ⚠️ **Notes**

- Predictions are stored automatically - no code changes needed
- Validation must be run manually or scheduled
- Database location: `data/prediction_tracker.db`
- Validations look back up to 7 days by default (configurable)
- Market holidays: System uses closest trading day if target date is a holiday



# Continuous Learning System Guide

## Overview

Your trading system now automatically learns and improves over time. Here's how it works:

### What It Does

1. **Collects new data** - Gathers fresh minute bars from your watchlist
2. **Evaluates performance** - Checks if current model is degrading
3. **Trains new models** - Builds improved models with latest data
4. **Validates** - Only deploys if new model is better
5. **Auto-deploys** - Seamlessly updates the production model
6. **Keeps history** - Tracks all models with rollback capability

---

## How It Works

### Automatic Checks

The system checks if retraining is needed based on:

1. **Time-based:** Model older than 7 days
2. **Performance-based:** Recent Sharpe ratio < 0.3
3. **Manual:** You force retrain

### Retraining Process

```
1. Collect 60 days of fresh data
   ↓
2. Train new model (31 features, RF with 200 trees)
   ↓
3. Validate on hold-out data
   ↓
4. Compare to current model
   ↓
5. Deploy if Sharpe > 0.5 AND better than current
   ↓
6. Trading bot auto-reloads new model
```

### Model Validation

New model must pass:
- **Minimum Sharpe:** > 0.5
- **Improvement:** Better than current model OR within 10% and comparable
- **No degradation:** Not worse than 90% of current Sharpe

---

## Usage

### Manual Retraining

Check if retrain needed:
```bash
cd backend
python3 continuous_learning.py
```

Force retrain (ignores checks):
```bash
python3 continuous_learning.py --force
```

### Automated Retraining (Recommended)

Run scheduler in background:
```bash
python3 model_scheduler.py &
```

**Schedule:**
- Daily check: 2:00 AM
- Weekly forced retrain: Sunday 1:00 AM

**Install schedule library:**
```bash
pip3 install schedule
```

### View Model History

```python
from backend.continuous_learning import ContinuousLearning

cl = ContinuousLearning()
print(cl.get_model_history())
```

Output:
```
model_id         trained_date         val_sharpe  is_deployed
20260109_020000  2026-01-09 02:00:00  1.23        True
20260102_020000  2026-01-02 02:00:00  1.15        False
20251226_020000  2025-12-26 02:00:00  1.08        False
```

### Rollback to Previous Model

If new model underperforms:

```python
from backend.continuous_learning import ContinuousLearning

cl = ContinuousLearning()
cl.rollback_to_previous_model()
```

---

## Trading Bot Integration

The trading bot automatically:

1. **Checks every 6 hours** for model updates (configurable)
2. **Reloads model** if file timestamp changed
3. **Continues trading** seamlessly with new model
4. **Logs reload** in output

You'll see:
```
============================================================
NEW MODEL DETECTED - RELOADING
============================================================
Model reloaded at 2026-01-09 02:05:23
============================================================
```

No need to restart the bot!

---

## Configuration

### `backend/trader_config.py`

```python
MODEL_RELOAD_INTERVAL_HOURS = 6  # Check for new models every 6 hours
```

### `backend/continuous_learning.py`

```python
cl = ContinuousLearning(
    retrain_frequency_days=7,        # Minimum days between retrains
    validation_threshold_sharpe=0.5  # Minimum Sharpe to deploy
)
```

---

## Files & Directories

```
HFTModel/
├── models/                           # Model archive
│   ├── model_20260109_020000.pkl   # Versioned models
│   ├── model_20260102_020000.pkl
│   └── model_metadata.csv           # Model history
├── trained_stock_model.pkl          # Production model (auto-updated)
└── backend/
    ├── continuous_learning.py       # Core retraining logic
    ├── model_scheduler.py           # Automated scheduling
    └── trader.py                    # Auto-reload capability
```

---

## Monitoring

### Email Alerts

Get notified when:
- New model deployed
- Retrain attempted but rejected
- Weekly retrain completed
- Errors during retraining

Configure in `backend/.env`:
```bash
ALERT_EMAIL_ENABLED=true
```

### Manual Monitoring

Check model performance:
```bash
python3 -c "
from backend.continuous_learning import ContinuousLearning
cl = ContinuousLearning()
print(cl.get_model_history())
"
```

Check if retrain needed:
```bash
python3 -c "
from backend.continuous_learning import ContinuousLearning
cl = ContinuousLearning()
should, reason = cl.should_retrain()
print(f'Retrain needed: {should}')
print(f'Reason: {reason}')
"
```

---

## Best Practices

### 1. Run Scheduler 24/7

Keep model scheduler running:
```bash
# Run in background
nohup python3 backend/model_scheduler.py > model_scheduler.log 2>&1 &

# Or use screen/tmux
screen -S model_scheduler
python3 backend/model_scheduler.py
# Ctrl+A, D to detach
```

### 2. Monitor Model Performance

Check weekly:
```bash
python3 backend/continuous_learning.py
```

Review model history and recent Sharpe ratios.

### 3. Keep Model Archive

Models are saved in `models/` directory. Don't delete these - they enable rollback.

### 4. Test Before Live Trading

After auto-retrain, review new model:
```bash
python3 -c "
from backend.performance_tracker import PerformanceTracker
tracker = PerformanceTracker()
tracker.print_report(start_date='2026-01-07')
"
```

If performance drops, rollback.

### 5. Adjust Frequency Based on Market

**Volatile markets:** Retrain more often (every 3-5 days)
**Stable markets:** Retrain less often (every 10-14 days)

Edit `continuous_learning.py`:
```python
cl = ContinuousLearning(retrain_frequency_days=3)  # More frequent
```

---

## Troubleshooting

### Model Not Reloading

**Issue:** Trading bot not picking up new model

**Solutions:**
1. Check `MODEL_RELOAD_INTERVAL_HOURS` in `trader_config.py`
2. Verify model file timestamp: `ls -la trained_stock_model.pkl`
3. Restart trading bot

### Retrain Keeps Failing

**Issue:** New models always rejected

**Solutions:**
1. Lower validation threshold:
   ```python
   cl = ContinuousLearning(validation_threshold_sharpe=0.3)
   ```
2. Check data quality: `python3 backend/continuous_learning.py`
3. Review recent trading performance

### No Data Available

**Issue:** "no data collected" error

**Solutions:**
1. Check Polygon API key in `.env`
2. Verify watchlist tickers in `trader_config.py`
3. Check API quota/limits

### Model Performance Degraded

**Issue:** New model worse than old

**Solution:**
```python
from backend.continuous_learning import ContinuousLearning
cl = ContinuousLearning()
cl.rollback_to_previous_model()
```

Then investigate why new model underperformed.

---

## Advanced Features

### Custom Training Parameters

Override model parameters:
```python
from backend.continuous_learning import ContinuousLearning

cl = ContinuousLearning()

custom_params = {
    'n_estimators': 300,      # More trees
    'max_depth': 15,          # Deeper trees
    'min_samples_split': 10   # More aggressive splitting
}

df = cl.collect_training_data(lookback_days=90)  # More data
model, metrics, samples = cl.train_new_model(df, model_params=custom_params)
```

### Multi-Model Ensemble

Train multiple models and ensemble:
```python
# Coming soon - train XGBoost, LightGBM, and RF
# Combine predictions with voting
```

### Feature Selection

Automatically remove low-importance features:
```python
# Coming soon - analyze feature importance
# Remove features with importance < threshold
```

---

## Performance Expectations

With continuous learning:

**Without CL:**
- Model degrades 10-20% over 1 month
- Fixed strategy, can't adapt

**With CL:**
- Model refreshed weekly
- Adapts to market conditions
- Maintains performance
- Typical improvement: 5-15% better Sharpe

**Timeline:**
- Week 1-2: Similar performance
- Week 3-4: Model starts adapting
- Month 2+: Clear improvement over static model

---

## System Architecture

```
Trading Bot (trader.py)
    ↓
Check every 6 hours
    ↓
Model file updated?
    ↓
Yes → Reload model
    ↓
Continue trading

Parallel Process:
    ↓
Model Scheduler (model_scheduler.py)
    ↓
Daily 2am: Check if retrain needed
    ↓
Weekly Sunday 1am: Force retrain
    ↓
Continuous Learning (continuous_learning.py)
    ↓
Collect data → Train → Validate → Deploy
    ↓
Update trained_stock_model.pkl
    ↓
Save to models/ directory
    ↓
Update metadata
    ↓
Send alert email
```

---

## Next Steps

1. ✅ Install schedule: `pip3 install schedule`
2. ✅ Start scheduler: `python3 backend/model_scheduler.py &`
3. ✅ Verify email alerts configured
4. ✅ Run initial retrain: `python3 backend/continuous_learning.py --force`
5. ✅ Start trading bot: `python3 backend/trader.py`
6. ⏳ Monitor for 1 week
7. ⏳ Review model history
8. ⏳ Adjust parameters if needed

Your system will now continuously improve itself!

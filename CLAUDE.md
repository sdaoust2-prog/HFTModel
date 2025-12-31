# dev notes

when working on this project keep code style natural, no excessive comments or ai patterns. write like a normal programmer would.

## project structure

quantitative stock prediction system with modular feature engine. competition-ready.

- core modules: feature_engine.py, utils.py, train.py, backtest.py
- notebooks: feature_analysis, linear_models, predictor, model_comparison
- backend: fastapi server for live predictions

## running stuff

quick train + backtest:
```bash
python train.py      # train model, saves to .pkl
python backtest.py   # backtest it
```

api server:
```bash
cd backend && python api.py
```

analysis (jupyter):
```bash
jupyter notebook feature_analysis.ipynb  # IC analysis
jupyter notebook model_comparison.ipynb  # compare models
```

dependencies: pandas, sklearn, requests, joblib, matplotlib, seaborn, scipy

## how it works

**feature engine** (`feature_engine.py`):
- pattern: FeatureEngine.feature_name(df, lookback=N)
- add features by defining static methods
- all features auto-computed via compute_all_features()

**default features:**
- momentum, volatility, price_direction, vwap_dev, hour, minute

**models available:**
- single feature baseline
- linear regression
- ridge/lasso (L1/L2 regularization)
- random forest (current best)

**evaluation:**
- chronological train/test split (no lookahead)
- IC analysis, confusion matrix, ROC/AUC
- backtest with sharpe, max dd, profit factor

## key files

**core modules:**
- `feature_engine.py` - add features here
- `utils.py` - data loading, backtesting, metrics
- `train.py` - quick model training
- `backtest.py` - test any model

**notebooks:**
- `feature_analysis.ipynb` - IC curves, correlations
- `linear_models.ipynb` - compare linear/ridge/lasso
- `predictor.ipynb` - random forest + full eval
- `model_comparison.ipynb` - compare all models

**models:**
- `trained_stock_model.pkl` - saved random forest
- `trained_linear_model.pkl` - saved linear (optional)

**backend:**
- `backend/api.py` - fastapi server
- `backend/.env` - polygon api key

## extending the system

**add new features:**
edit `feature_engine.py`, add static method following pattern
```python
@staticmethod
def my_feature(df, lookback=5):
    return df['close'].rolling(lookback).std()
```

**try new models:**
edit `train.py`, swap in different sklearn model

**for competitions:**
see `WALKTHROUGH.md` for complete guide

## notes

- accuracy ~52-53% is normal for minute data (barely beats random)
- feature quality matters more than model complexity
- always check train-test gap for overfitting
- transaction costs can kill profitability

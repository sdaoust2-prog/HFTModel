# quantitative stock prediction system

comprehensive ML system for intraday price movement prediction with full feature analysis, multiple model types, and rigorous evaluation

## overview

end-to-end quantitative trading system that:
- analyzes feature predictiveness using Information Coefficient (IC)
- compares multiple model architectures (linear, regularized, ensemble)
- evaluates out-of-sample performance with proper metrics
- backtests strategies with transaction costs and risk metrics

**data source**: polygon.io minute-level ohlcv bars

**features** (6 total):
- momentum_1min: 1-minute price return
- volatility_1min: squared momentum (captures magnitude)
- price_direction: binary close > open (market sentiment)
- vwap_dev: deviation from volume-weighted average price (mean reversion signal)
- hour: time of day (session effects)
- minute: minute within hour (microstructure)

**models implemented**:
- single best feature baseline (z-scored)
- multi-feature linear regression
- ridge regression (L2 regularization)
- lasso regression (L1 regularization, feature selection)
- random forest classifier (100 trees)

**evaluation approach**:
- chronological train/test split (prevents lookahead bias)
- information coefficient analysis
- confusion matrix, precision/recall, ROC/AUC
- feature importance rankings
- overfitting gap analysis
- backtest with sharpe ratio, max drawdown, win rate, profit factor

## performance

backtest on AAPL (oct-nov 2025, out-of-sample):

```
total return: -4.47%
buy & hold: -0.95%
sharpe ratio: -33.96
max drawdown: -4.65%
win rate: 34.2%
profit factor: 0.70
num trades: 524
```

model underperforms buy and hold. this is expected for a simple 6-feature classifier on minute data - mostly demonstrates the framework rather than alpha generation.

## setup

```bash
pip install -r requirements.txt
cp backend/.env.example backend/.env
# add polygon api key to backend/.env
```

## usage

**1. feature analysis:**
```bash
jupyter notebook feature_analysis.ipynb
# IC analysis, correlation heatmaps, feature vs return plots
# winsorization impact, feature distributions
```

**2. linear models:**
```bash
jupyter notebook linear_models.ipynb
# compare single feature, linear, ridge, lasso
# train multiple regression models
# saves best linear model
```

**3. random forest (main model):**
```bash
jupyter notebook predictor.ipynb
# full evaluation: confusion matrix, ROC/AUC, feature importance
# saves trained_stock_model.pkl
```

**4. model comparison:**
```bash
jupyter notebook model_comparison.ipynb
# side-by-side comparison of all models
# backtest performance metrics for each
# identifies best model by multiple criteria
```

**5. backtest:**
```bash
python3 backtest.py
# detailed metrics: sharpe, max dd, win rate, avg win/loss, turnover
```

**6. live api:**
```bash
cd backend
python3 api.py
# serves model on localhost:8000
curl http://localhost:8000/api/predict/AAPL
```

## file structure

```
Project/
├── feature_analysis.ipynb      # IC analysis, feature-return correlations
├── linear_models.ipynb         # linear/ridge/lasso model comparison
├── predictor.ipynb            # random forest with full evaluation
├── model_comparison.ipynb     # compare all model types
├── backtest.py                # backtesting with enhanced metrics
├── trained_stock_model.pkl    # saved random forest model
├── trained_linear_model.pkl   # saved best linear model (optional)
├── backend/
│   └── api.py                # fastapi server for live predictions
└── requirements.txt
```

## for trading competitions

- api endpoint at `/api/predict/{ticker}` returns json with decision + probabilities
- adjustable threshold via `?prob_threshold=0.6` for more conservative signals
- chronological split prevents lookahead bias (critical for live trading)
- transaction costs included in backtest (10bps per trade)

## what makes this quant-ready

unlike toy ML projects, this system demonstrates production quantitative workflows:

1. **feature analysis first**: IC curves, correlation analysis, winsorization impact before modeling
2. **multiple model comparison**: baseline → linear → regularized → ensemble with rigorous evaluation
3. **proper time series handling**: chronological splits, no lookahead bias
4. **comprehensive evaluation**: not just accuracy, but precision/recall/AUC/sharpe/drawdown
5. **overfitting detection**: explicit train-test gaps reported for all metrics
6. **realistic backtesting**: transaction costs, position tracking, proper sharpe calculation
7. **model comparison framework**: systematic approach to choosing best model

these are the practices used in actual quant firms, not academic ML.

## current limitations & future improvements

**current state:**
- model barely beats random (52-53% accuracy typical for minute data)
- features are basic (no order book, no microstructure signals)
- no dynamic position sizing or risk management
- single ticker (no cross-sectional analysis)
- short training period (1 month)

**next steps for real alpha:**
- level 2 data: bid-ask spreads, order book imbalance, aggressor flow
- longer history: train on 6-12 months, validate on holdout period
- cross-sectional signals: relative momentum, sector effects
- regime detection: separate models for high/low volatility
- position sizing: scale with signal strength and volatility
- portfolio construction: diversification across multiple tickers
- online learning: retrain models periodically with new data

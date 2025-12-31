# complete walkthrough: from zero to trading

this guide walks you through every piece of code, every concept, and how to actually use this system for trading competitions.

## project structure

```
HFTModel/
├── feature_engine.py          # core: compute features from OHLCV data
├── utils.py                   # data loading, backtesting, metrics
├── train.py                   # quick model training script
├── backtest.py               # backtest any trained model
├── backend/api.py            # serve predictions via API
│
├── feature_analysis.ipynb    # analyze which features are predictive
├── linear_models.ipynb       # compare linear/ridge/lasso
├── predictor.ipynb          # train random forest with full eval
├── model_comparison.ipynb   # compare all model types
│
└── trained_stock_model.pkl  # saved model
```

## part 1: feature engine (the foundation)

**what it is:** functions that convert raw OHLCV bars into predictive signals

**why it matters:** features are everything in quant. no model can fix bad features.

### core pattern

```python
from feature_engine import FeatureEngine

# all features follow this pattern:
value = FeatureEngine.feature_name(df, lookback=N)
```

### built-in features

1. **momentum** - price return over N periods
   formula: `(close[t] - close[t-N]) / close[t-N]`
   captures: trend direction and strength

2. **volatility** - squared returns
   formula: `momentum^2`
   captures: how much price is moving (regardless of direction)

3. **price_direction** - candle color
   formula: `1 if close > open else 0`
   captures: immediate buying/selling pressure

4. **vwap_deviation** - distance from volume-weighted price
   formula: `(close - vwap) / vwap`
   captures: mean reversion opportunity

5. **hour/minute** - time of day
   captures: session effects (opening volatility, lunch lull, etc)

### how to add your own features

```python
@staticmethod
def your_feature(df, lookback=5):
    """describe what it captures"""
    return df['close'].rolling(lookback).std() / df['close']
```

then add to compute_all_features list.

## part 2: data pipeline

### loading data

```python
from utils import pull_polygon_data

df = pull_polygon_data("AAPL", "2025-10-01", "2025-11-01", api_key)
# returns: [timestamp, open, high, low, close, volume]
```

### creating features + target

```python
from feature_engine import load_features_for_training

X, y_binary, y_continuous, feature_names = load_features_for_training(df)

# X: feature matrix (momentum, volatility, etc)
# y_binary: 1=UP, 0=DOWN (for classification)
# y_continuous: actual return (for regression)
```

### train/test split

```python
from utils import train_test_split_chronological

X_train, X_test, y_train, y_test = train_test_split_chronological(X, y_binary)
# CRITICAL: splits by time, not randomly
# this prevents lookahead bias
```

## part 3: feature analysis (before modeling)

**never skip this step.** you need to know which features actually predict returns.

### information coefficient (IC)

IC = correlation between feature and forward return (NOT 2*#correct/#total)

Since features and returns are both continuous, we use correlation:

```python
from utils import calculate_ic

# Pearson: linear relationships
ic_pearson = calculate_ic(X_train, y_train_continuous, method='pearson')

# Spearman: monotonic relationships, robust to outliers (recommended)
ic_spearman = calculate_ic(X_train, y_train_continuous, method='spearman')
print(ic_spearman)
```

**What's a good IC?**
- >0.02: decent (sounds small but huge in finance)
- >0.05: great
- >0.10: amazing (rare)

**Use Spearman for feature selection** - handles outliers better

### what to look for

- which features have highest |IC|?
- do ICs stay consistent across different time periods?
- are features correlated with each other (multicollinearity)?

run `feature_analysis.ipynb` to see:
- IC rankings
- feature vs return scatter plots
- correlation heatmaps
- IC curves over different horizons

## part 4: modeling approaches

### approach 1: single best feature (baseline)

use ONLY the feature with highest IC, z-scored

```python
best_feature = ic_scores.iloc[0]['feature']
X_single = scaler.fit_transform(X[[best_feature]])
model = LinearRegression().fit(X_single, y)
```

why do this?
establishes baseline. if complex models don't beat this, they're useless.

### approach 2: linear regression

use all features, predict continuous returns

```python
model = LinearRegression().fit(X_train, y_train_continuous)
```

pros: simple, interpretable
cons: assumes linear relationships (often not true)

### approach 3: ridge/lasso (regularized linear)

same as linear but penalizes large coefficients

```python
model = Ridge(alpha=0.1).fit(X_train, y_train)  # L2
model = Lasso(alpha=0.01).fit(X_train, y_train)  # L1
```

ridge: shrinks all coefficients
lasso: can zero out features (automatic feature selection)

### approach 4: random forest (current best)

ensemble of decision trees

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_binary)
```

pros: handles non-linear relationships, robust
cons: can overfit, slower

### which to use?

run `model_comparison.ipynb` to compare all on your data.
usually RF wins but sometimes simpler is better.

## part 5: evaluation (critical)

### metrics that matter

**accuracy** - % correct predictions
baseline: 50% (random coin flip)
good: 52-53% (yes, that's actually good for minute data)

**AUC** - area under ROC curve
baseline: 0.5
good: >0.55

**sharpe ratio** - risk-adjusted return
formula: `mean(returns) / std(returns) * sqrt(252*390)`
good: >1.0
great: >2.0

**max drawdown** - worst peak-to-trough loss
keep: <10%

**profit factor** - gross profit / gross loss
baseline: 1.0
good: >1.5

### overfitting check

```
train accuracy: 0.85
test accuracy: 0.52
```
this is severe overfitting. model memorized noise.

```
train accuracy: 0.53
test accuracy: 0.52
```
this is good generalization.

rule: train-test gap should be <0.05

## part 6: backtesting

simulation of how strategy would have performed

```python
from utils import backtest_strategy

signals = np.where(predictions > threshold, 1, -1)  # 1=buy, -1=sell
metrics = backtest_strategy(signals, actual_returns, transaction_cost=0.0001)
```

**transaction costs matter!**
ignoring them turns profitable strategies into losers.

typical costs:
- 1-2 bps (0.0001-0.0002) per trade for stocks
- higher for options/futures

### what to watch

1. **sharpe ratio** - is risk-adjusted return positive?
2. **max drawdown** - can you stomach the losses?
3. **win rate** - how often do you win? (doesn't need to be high if wins are big)
4. **profit factor** - gross wins / gross losses
5. **turnover** - how often you trade (higher = more costs)

## part 7: for trading competitions

### typical competition format

- given: historical data (OHLCV)
- predict: next period return/direction
- judged on: sharpe ratio, total return, or custom metric

### workflow for competitions

1. **load competition data**
```python
df = pd.read_csv('competition_data.csv')
# ensure columns: timestamp, open, high, low, close, volume
```

2. **run feature analysis**
```bash
jupyter notebook feature_analysis.ipynb
# modify to use competition data
```

3. **train multiple models**
```bash
jupyter notebook model_comparison.ipynb
# see which performs best on competition data
```

4. **submit predictions**
```python
model = joblib.load('trained_stock_model.pkl')

test_df = pd.read_csv('test_data.csv')
X_test, _, _, _ = load_features_for_training(test_df)

predictions = model.predict_proba(X_test)[:, 1]
pd.DataFrame({'id': test_df.index, 'prediction': predictions}).to_csv('submission.csv')
```

### tips for winning

1. **feature engineering is 80% of the work**
   try: lagged features, rolling stats, cross-sectional signals

2. **avoid overfitting**
   use chronological validation, check train-test gaps

3. **ensemble models**
   average predictions from RF + linear models often wins

4. **calibrate thresholds**
   default 0.5 might not be optimal. try 0.52, 0.55, 0.6

5. **understand the metric**
   optimizing for sharpe is different than total return

## part 8: extending the system

### adding new features

edit `feature_engine.py`:
```python
@staticmethod
def awesome_feature(df, lookback=10):
    """your secret sauce"""
    return (df['high'] - df['low']) / df['close'].rolling(lookback).std()
```

then add to default list in `compute_all_features`

### trying new models

edit `train.py`:
```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=100)
```

### live trading (via API)

1. start API server:
```bash
cd backend && python api.py
```

2. get predictions:
```bash
curl http://localhost:8000/api/predict/AAPL
```

3. parse response and execute trades via broker API

## part 9: common pitfalls

### lookahead bias

**wrong:**
```python
X_train, X_test = train_test_split(X, y, shuffle=True)  # NEVER
```

**right:**
```python
X_train, X_test = train_test_split_chronological(X, y)
```

### ignoring transaction costs

your backtest shows +50% return without costs.
with 1bp costs it's actually -5%.

always include costs.

### overfitting to test set

**wrong:**
```python
for threshold in np.arange(0.5, 0.7, 0.01):
    if backtest(threshold) > best:
        best = threshold
# you just overfit to test set
```

**right:**
use validation set or cross-validation for hyperparameter tuning.
test set should only be touched ONCE at the very end.

### data quality issues

- check for gaps (missing minutes)
- check for outliers (flash crashes)
- check for survivorship bias (only looking at surviving stocks)

## next steps

1. **run everything once to understand the flow**
```bash
python train.py          # train model
python backtest.py       # test it
```

2. **explore notebooks one by one**
```bash
jupyter notebook feature_analysis.ipynb  # see what features work
jupyter notebook model_comparison.ipynb  # see what models work
```

3. **start experimenting**
- add your own features
- try different lookback periods
- test on other tickers (SPY, QQQ, TSLA)
- try longer time periods

4. **prepare for competitions**
- practice with kaggle finance competitions
- understand different evaluation metrics
- build reusable templates for quick iteration

## questions to ask yourself

- which features have highest IC? why?
- is my model overfitting? (check train-test gap)
- does my backtest include transaction costs?
- what's my sharpe ratio? is it positive?
- how does my model compare to buy-and-hold?
- can i explain why this model works?

## when something breaks

1. **"module not found"**
   make sure you're in the HFTModel directory

2. **"API error"**
   check your polygon api key in .env

3. **"not enough data"**
   polygon free tier is limited. try different date range.

4. **model performs worse than random**
   totally normal for minute data. try longer horizons (5min, 15min).

5. **sharpe ratio is negative**
   either the model doesn't work or transaction costs are too high.

good luck. now go build something.

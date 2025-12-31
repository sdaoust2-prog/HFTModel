# advanced features guide

key quantitative techniques for production trading systems

## 1. winsorization (outlier handling)

**location:** `feature_engine.py` line 129

**what it does:** caps extreme values to prevent outliers from dominating

**CRITICAL: avoid lookahead bias**

```python
# WRONG - uses test data to compute thresholds
X_test_winsorized = FeatureEngine.winsorize(X_test)  # LOOKAHEAD BIAS!

# RIGHT - compute thresholds on train, apply to test
from feature_engine import FeatureEngine

# compute thresholds from TRAINING data only
lower_thresh = X_train['momentum_1min'].quantile(0.01)
upper_thresh = X_train['momentum_1min'].quantile(0.99)

# apply same thresholds to test
X_test['momentum_1min_clipped'] = X_test['momentum_1min'].clip(lower_thresh, upper_thresh)
```

**when to use:**
- check if outliers skew your IC scores
- run feature_analysis.ipynb, look at winsorization section
- if IC changes significantly after winsorization, outliers matter

**when NOT to use:**
- if your feature is "extra predictive at extremes" (your cousin's point)
- example: huge momentum spikes might be very predictive
- removing them would throw away alpha

**how to check:**
```python
# compare IC with and without outliers
ic_normal = calculate_ic(X_train, y_train)
ic_winsorized = calculate_ic(X_train_winsorized, y_train)

# if ic_winsorized >> ic_normal: outliers were noise
# if ic_winsorized << ic_normal: outliers were signal!
```

---

## 2. realized out-of-sample correlation

**location:** `utils.py` line 168

**what it does:** checks if your predictions actually correlate with real returns

**why it matters:** you can have high accuracy but LOW correlation = overfit

```python
from utils import calculate_realized_correlation

# get model predictions on test set
y_prob = model.predict_proba(X_test)[:, 1]  # P(UP)

# convert to expected returns (distance from 0.5)
predicted_returns = y_prob - 0.5

# actual returns
actual_returns = y_test_continuous

# calculate correlation
result = calculate_realized_correlation(predicted_returns, actual_returns)

print(result)
# {
#   'correlation': 0.035,
#   'pvalue': 0.02,
#   'overfitting_check': 'PASS'
# }
```

**interpreting results:**
- correlation > 0.02: good, model generalizes
- correlation near 0: BAD, you overfit
- correlation < 0: VERY BAD, model is backwards

**example of overfitting:**
```
train accuracy: 0.65
test accuracy: 0.53  ← looks ok!
realized correlation: 0.005  ← OVERFIT! predictions don't actually predict returns
```

---

## 3. position sizing scaled by signal magnitude

**location:** `utils.py` line 124

**what it does:** trade bigger when confident, smaller when uncertain

**old way (binary):**
```python
P(UP) = 0.55  → go 100% long
P(UP) = 0.99  → go 100% long  (same as above!)
```

**new way (scaled):**
```python
P(UP) = 0.55  → go 10% long   (barely above threshold)
P(UP) = 0.70  → go 40% long   (moderate confidence)
P(UP) = 0.99  → go 98% long   (very confident)
```

**how to use:**

```python
from utils import generate_scaled_signals, backtest_strategy

# get probabilities from model
probabilities = model.predict_proba(X_test)[:, 1]

# binary signals (old way)
signals_binary = generate_scaled_signals(probabilities, threshold=0.55, scale_by_magnitude=False)

# scaled signals (new way)
signals_scaled = generate_scaled_signals(probabilities, threshold=0.55, scale_by_magnitude=True)

# backtest both
metrics_binary = backtest_strategy(signals_binary, actual_returns)
metrics_scaled = backtest_strategy(signals_scaled, actual_returns)

print(f"Binary Sharpe: {metrics_binary['sharpe_ratio']:.2f}")
print(f"Scaled Sharpe: {metrics_scaled['sharpe_ratio']:.2f}")
```

**the math:**

```python
P(UP) = 0.70
threshold = 0.55

# signal strength = distance from 0.5
signal_strength = abs(0.70 - 0.5) = 0.20

# threshold distance
threshold_distance = 0.55 - 0.5 = 0.05

# scale: (actual - threshold) / (max - threshold)
scaled = (0.20 - 0.05) / (0.5 - 0.05) = 0.15 / 0.45 = 0.33

# direction: UP → +1
signal = +0.33  → go 33% long
```

**benefits:**
- reduces risk when uncertain
- increases exposure when confident
- usually improves Sharpe ratio
- can reduce max drawdown

---

## 4. dynamic threshold based on sharpe

**concept:** adjust trading threshold based on model performance

**not yet implemented in code, but here's how:**

```python
def calculate_optimal_threshold(probabilities, returns, thresholds=[0.5, 0.55, 0.6, 0.65, 0.7]):
    """
    Find threshold that maximizes Sharpe ratio on validation set
    """
    best_sharpe = -np.inf
    best_threshold = 0.55

    for thresh in thresholds:
        signals = generate_scaled_signals(probabilities, threshold=thresh)
        metrics = backtest_strategy(signals, returns)

        if metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            best_threshold = thresh

    return best_threshold, best_sharpe

# use on validation set (NOT test set!)
optimal_thresh, sharpe = calculate_optimal_threshold(val_probs, val_returns)
print(f"Optimal threshold: {optimal_thresh} (Sharpe: {sharpe:.2f})")

# apply to test set
test_signals = generate_scaled_signals(test_probs, threshold=optimal_thresh)
```

**important:** optimize on validation set, NOT test set, or you'll overfit

---

## complete workflow with all advanced features

```python
from feature_engine import FeatureEngine, load_features_for_training
from utils import (pull_polygon_data, train_test_split_chronological,
                   calculate_ic, calculate_realized_correlation,
                   generate_scaled_signals, backtest_strategy)
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 1. load data
df = pull_polygon_data("AAPL", "2025-10-01", "2025-11-01", api_key)
X, y_binary, y_continuous, features = load_features_for_training(df)

# 2. chronological split
X_train, X_test, y_train, y_test = train_test_split_chronological(X, y_binary)
_, _, y_train_cont, y_test_cont = train_test_split_chronological(X, y_continuous)

# 3. WINSORIZATION - check if outliers matter
ic_normal = calculate_ic(X_train, y_train_cont, method='spearman')
print("Normal IC:", ic_normal.head())

# winsorize TRAIN only
X_train_wins = X_train.copy()
for col in features:
    lower = X_train[col].quantile(0.01)
    upper = X_train[col].quantile(0.99)
    X_train_wins[col] = X_train[col].clip(lower, upper)

ic_wins = calculate_ic(X_train_wins, y_train_cont, method='spearman')
print("Winsorized IC:", ic_wins.head())

# if IC improves significantly, use winsorized features
if ic_wins.iloc[0]['abs_ic'] > ic_normal.iloc[0]['abs_ic']:
    print("Using winsorized features")
    X_train = X_train_wins
    # apply same thresholds to test
    X_test_wins = X_test.copy()
    for col in features:
        lower = X_train[col].quantile(0.01)  # from TRAIN
        upper = X_train[col].quantile(0.99)
        X_test_wins[col] = X_test[col].clip(lower, upper)
    X_test = X_test_wins

# 4. train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. get probabilities
train_probs = model.predict_proba(X_train)[:, 1]
test_probs = model.predict_proba(X_test)[:, 1]

# 6. REALIZED CORRELATION - check overfitting
train_pred_returns = train_probs - 0.5
test_pred_returns = test_probs - 0.5

train_corr = calculate_realized_correlation(train_pred_returns, y_train_cont)
test_corr = calculate_realized_correlation(test_pred_returns, y_test_cont)

print("\nRealized Correlation:")
print(f"Train: {train_corr['correlation']:.4f}")
print(f"Test:  {test_corr['correlation']:.4f}")
print(f"Gap:   {train_corr['correlation'] - test_corr['correlation']:.4f}")

if test_corr['correlation'] < 0.01:
    print("WARNING: Low test correlation - likely overfit!")

# 7. POSITION SIZING - compare binary vs scaled
signals_binary = generate_scaled_signals(test_probs, threshold=0.55, scale_by_magnitude=False)
signals_scaled = generate_scaled_signals(test_probs, threshold=0.55, scale_by_magnitude=True)

metrics_binary = backtest_strategy(signals_binary, y_test_cont)
metrics_scaled = backtest_strategy(signals_scaled, y_test_cont)

print("\nBacktest Comparison:")
print(f"Binary - Sharpe: {metrics_binary['sharpe_ratio']:.2f}, Max DD: {metrics_binary['max_drawdown']*100:.2f}%")
print(f"Scaled - Sharpe: {metrics_scaled['sharpe_ratio']:.2f}, Max DD: {metrics_scaled['max_drawdown']*100:.2f}%")

# 8. use whichever is better
if metrics_scaled['sharpe_ratio'] > metrics_binary['sharpe_ratio']:
    print("\nScaled position sizing wins!")
    final_metrics = metrics_scaled
else:
    print("\nBinary signals win!")
    final_metrics = metrics_binary
```

---

## when to use each feature

**winsorization:**
- your feature distributions have extreme outliers
- IC improves after winsorization
- model accuracy improves on clean data

**realized correlation:**
- ALWAYS check this
- key overfitting detector
- shows if model actually predicts returns or just memorizes patterns

**scaled position sizing:**
- when your model has varying confidence
- helps in noisy markets (reduces exposure when uncertain)
- usually improves Sharpe ratio
- reduces max drawdown

**dynamic threshold:**
- when optimizing for specific metric (Sharpe, max DD, etc.)
- use validation set to find optimal threshold
- reoptimize periodically as market conditions change

---

## file locations reference

- `feature_engine.py` line 129: `winsorize()`
- `utils.py` line 168: `calculate_realized_correlation()`
- `utils.py` line 124: `generate_scaled_signals()`
- `utils.py` line 71: `backtest_strategy()` (main backtest function)
- `feature_analysis.ipynb`: winsorization analysis
- `model_comparison.ipynb`: compare different approaches

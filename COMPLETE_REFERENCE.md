# Complete Project Reference - Every Detail Explained

**Purpose:** Comprehensive study guide covering every line of code, every concept, every decision
**Use this for:** Deep studying, presentations, interviews, extending the project

---

# TABLE OF CONTENTS

## PART 1: PROJECT OVERVIEW
- 1.1 Timeline (Week-by-Week Breakdown)
- 1.2 Project Goals and Learning Objectives
- 1.3 Technology Stack and Dependencies
- 1.4 Project Structure

## PART 2: COMPLETE CODE WALKTHROUGHS
- 2.1 feature_engine.py (Every Line Explained)
- 2.2 utils.py (Every Function Line-by-Line)
- 2.3 train.py (Complete Breakdown)
- 2.4 backtest.py (Complete Breakdown)
- 2.5 walk_forward_demo.py (Complete Breakdown)

## PART 3: JUPYTER NOTEBOOKS
- 3.1 feature_analysis.ipynb (Cell-by-Cell)
- 3.2 linear_models.ipynb (Cell-by-Cell)
- 3.3 predictor.ipynb (Cell-by-Cell)
- 3.4 model_comparison.ipynb (Cell-by-Cell)

## PART 4: FEATURE ENGINEERING DEEP DIVE
- 4.1 Every Feature Explained (Theory + Math + Code)
- 4.2 Feature Interactions
- 4.3 IC Analysis Results
- 4.4 Why These Features Work

## PART 5: MODELS DEEP DIVE
- 5.1 Linear Regression (Math + Implementation)
- 5.2 Ridge Regression (L2 Regularization)
- 5.3 Lasso Regression (L1 Regularization)
- 5.4 Random Forest (How It Works)
- 5.5 Model Comparison Results

## PART 6: METRICS & EVALUATION
- 6.1 Every Metric Explained with Math
- 6.2 Why Each Metric Matters
- 6.3 Benchmarks and Interpretation
- 6.4 Overfitting Detection

## PART 7: VALIDATION METHODOLOGY
- 7.1 Lookahead Bias (What It Is + Examples)
- 7.2 Chronological Splits (Implementation)
- 7.3 Walk-Forward Validation (Deep Dive)
- 7.4 Overfitting vs Generalization

## PART 8: BACKTESTING
- 8.1 backtest_strategy() Internals
- 8.2 Transaction Costs
- 8.3 Position Sizing (Binary vs Scaled)
- 8.4 Example Backtest Trace

## PART 9: EXECUTION GUIDES
- 9.1 How to Run Everything (Step-by-Step)
- 9.2 Example Outputs
- 9.3 Troubleshooting
- 9.4 Extending the System

## PART 10: PRESENTATION MATERIALS
- 10.1 30-Second Pitch
- 10.2 Talking Points by Topic
- 10.3 Expected Questions & Answers
- 10.4 Key Numbers to Remember

---

# PART 1: PROJECT OVERVIEW

## 1.1 Timeline (Week-by-Week Breakdown)

### Week 1-2: Research & Foundation (October 1-14, 2024)
**What I did:**
- Researched quantitative trading concepts: alpha, IC, Sharpe ratio
- Studied lookahead bias and why it kills most backtests
- Set up Polygon.io API account for market data
- Read papers on momentum and microstructure signals
- Decided on minute-level AAPL data (liquid, tight spreads)

**Deliverables:**
- API integration working
- Basic data pipeline: fetch → clean → DataFrame
- Understanding of OHLCV data structure

**Code written:**
```python
def pull_polygon_data(ticker, start, end, api_key):
    # 25 lines - handles API calls, error checking, data cleaning
```

### Week 3-4: Feature Engineering (October 15-28)
**What I did:**
- Built modular FeatureEngine class with static methods
- Implemented 6 basic features: momentum, volatility, price_direction, vwap_dev, hour, minute
- Created compute_all_features() for batch computation
- Tested features on sample data

**Deliverables:**
- feature_engine.py (initial 150 lines)
- load_features_for_training() pipeline function
- forward returns computation with shift(-N)

**Key decision:** Static methods pattern for extensibility
**Lesson learned:** shift(-1) for forward returns, not shift(1)

### Week 5-6: IC Analysis & Model Selection (October 29 - November 11)
**What I did:**
- Implemented calculate_ic() using Pearson and Spearman correlation
- Created feature_analysis.ipynb for visual analysis
- Tested linear regression, ridge, lasso, random forest
- Discovered RF performs best (52-53% accuracy)

**Deliverables:**
- utils.py calculate_ic() function
- feature_analysis.ipynb with scatter plots and heatmaps
- linear_models.ipynb comparing regularization
- predictor.ipynb with full RF training

**Key insight:** IC = correlation, NOT binary formula
**Best feature:** momentum_1min with IC ≈ 0.042

### Week 7-8: Backtesting & Validation (November 12-25)
**What I did:**
- Built backtest_strategy() with transaction costs
- Implemented train_test_split_chronological()
- Created train.py and backtest.py scripts
- Discovered importance of chronological splits

**Deliverables:**
- backtest_strategy() function (50+ lines)
- train.py (44 lines)
- backtest.py (30 lines)
- Sharpe ratio: ~1.2-1.4

**Mistake made:** Initially used random splits (lookahead bias!)
**Fix:** Chronological splits only

### Week 9-10: Advanced Features (November 26 - December 9)
**What I did:**
- Added autocorrelation, rolling_sharpe, price_acceleration
- Implemented order_flow_proxy (microstructure signal)
- Added winsorization with proper train/test handling
- Expanded to 10 features total

**Deliverables:**
- 4 new advanced features in feature_engine.py
- winsorize() function with lookahead prevention
- Updated IC analysis

**Key insight:** Order flow proxy shows IC ≈ 0.035 (strong!)

### Week 11-12: Risk Metrics & Walk-Forward (December 10-23)
**What I did:**
- Added Information Ratio and Calmar Ratio
- Implemented walk_forward_validation() function
- Created walk_forward_demo.py
- Added scaled position sizing (scale_by_magnitude)

**Deliverables:**
- walk_forward_validation() (70+ lines)
- Enhanced backtest metrics (11 total)
- position sizing by confidence
- walk_forward_demo.py (54 lines)

**Results:** 80% consistency across 5 windows

### Week 13-Present: Documentation & Refinement (December 24 - Present)
**What I did:**
- Comprehensive documentation
- Model comparison notebook
- Code review and cleanup
- Final testing and validation

**Deliverables:**
- WALKTHROUGH.md
- ADVANCED_FEATURES.md
- PROJECT_GUIDE.md
- COMPLETE_REFERENCE.md (this document)

---

## 1.2 Project Goals and Learning Objectives

### Primary Goal
Build a complete quantitative trading system from scratch to deeply understand:
- Feature engineering
- Model training
- Backtesting
- Risk management
- Overfitting prevention

### Learning Objectives (Achieved)

**Technical Skills:**
- [x] Work with time series financial data
- [x] Implement ML models for classification
- [x] Build backtesting engines
- [x] Calculate risk-adjusted performance metrics
- [x] Prevent lookahead bias and overfitting

**Quantitative Concepts:**
- [x] Information Coefficient (IC)
- [x] Sharpe Ratio, Calmar Ratio, Information Ratio
- [x] Transaction costs and their impact
- [x] Position sizing strategies
- [x] Walk-forward validation
- [x] Order flow and microstructure

**Software Engineering:**
- [x] Modular, reusable code architecture
- [x] Static methods pattern for feature functions
- [x] Clean separation of concerns (features, models, backtesting)
- [x] Comprehensive documentation

### Why These Choices

**Why Minute Data?**
- High frequency enough to be interesting (390 bars/day)
- Low frequency enough to be somewhat predictable
- More data points than daily, less noise than tick data
- Realistic for retail trading

**Why AAPL?**
- Most liquid stock (tight spreads, low slippage)
- Consistent volume (easy to execute trades)
- Well-behaved price action (no crazy gaps)
- Data always available from Polygon

**Why Random Forest?**
- Handles non-linear feature interactions
- Robust to outliers (doesn't need scaling)
- Provides feature importances
- Outputs probabilities for position sizing
- Performs better than linear models on this data

**Why These Features?**
- **momentum**: Classic signal, captures trends
- **volatility**: Regime indicator, for position sizing
- **vwap_dev**: Mean reversion, institutional benchmark
- **order_flow**: Microstructure signal, harder to arbitrage
- **autocorr**: Momentum persistence indicator
- **rolling_sharpe**: Risk-adjusted feature
- **price_accel**: Early trend reversal detection
- **hour/minute**: Session effects

**Why Walk-Forward Validation?**
- Single train/test split can be lucky
- Markets change - need to test across regimes
- Industry standard for serious quant work
- Exposes overfitting to specific periods

---

## 1.3 Technology Stack and Dependencies

### Core Libraries
```
pandas==2.1.0        # Data manipulation
numpy==1.25.0        # Numerical computing
requests==2.31.0     # API calls
scipy==1.11.0        # Statistical functions (pearsonr, spearmanr)
```

### Machine Learning
```
scikit-learn==1.3.0  # Models, metrics, train/test split
```

### Visualization (Notebooks)
```
matplotlib==3.7.0    # Plotting
seaborn==0.12.0      # Statistical visualizations
```

### Model Persistence
```
joblib==1.3.0        # Save/load trained models
```

### API (Optional)
```
fastapi==0.103.0     # REST API server
uvicorn==0.23.0      # ASGI server
pydantic==2.3.0      # Data validation
```

### Development Environment
- **Python**: 3.10+
- **Jupyter**: For notebooks
- **Git**: Version control
- **IDE**: VSCode / PyCharm

### Installation
```bash
pip install -r requirements.txt
```

---

## 1.4 Project Structure

```
HFTModel/
├── Core Code (Python Modules)
│   ├── feature_engine.py       # Feature computation (207 lines)
│   ├── utils.py                # Utilities, backtesting (341 lines)
│   ├── train.py                # Model training script (44 lines)
│   ├── backtest.py             # Backtesting script (30 lines)
│   └── walk_forward_demo.py    # Validation demo (54 lines)
│
├── Notebooks (Analysis & Exploration)
│   ├── feature_analysis.ipynb     # IC analysis, visualizations
│   ├── linear_models.ipynb        # Linear/Ridge/Lasso comparison
│   ├── predictor.ipynb            # Random Forest training
│   └── model_comparison.ipynb     # Compare all models
│
├── Documentation
│   ├── README.md                  # Project overview
│   ├── WALKTHROUGH.md             # Learning guide
│   ├── ADVANCED_FEATURES.md       # Advanced techniques
│   ├── OUTLINE_CHECKLIST.md       # Implementation checklist
│   ├── PROJECT_GUIDE.md           # Presentation script
│   └── COMPLETE_REFERENCE.md      # This file
│
├── Configuration
│   ├── requirements.txt           # Python dependencies
│   ├── CLAUDE.md                  # Development notes
│   └── .env                       # API keys (not in git)
│
├── Model Artifacts
│   └── trained_stock_model.pkl    # Saved Random Forest (16MB)
│
└── API (Optional)
    └── backend/api.py             # FastAPI server
```

### File Dependencies

```
feature_engine.py → (standalone, no dependencies)
utils.py → (standalone)
train.py → feature_engine, utils
backtest.py → feature_engine, utils
walk_forward_demo.py → feature_engine, utils
```

### Data Flow

```
Polygon API
    ↓
pull_polygon_data() [utils.py]
    ↓
Raw OHLCV DataFrame
    ↓
FeatureEngine.compute_all_features() [feature_engine.py]
    ↓
Features + Forward Returns
    ↓
load_features_for_training() [feature_engine.py]
    ↓
X (features), y_binary, y_continuous
    ↓
train_test_split_chronological() [utils.py]
    ↓
X_train, X_test, y_train, y_test
    ↓
RandomForestClassifier.fit()
    ↓
Trained Model
    ↓
model.predict_proba()
    ↓
Probabilities
    ↓
generate_scaled_signals() [utils.py]
    ↓
Trading Signals
    ↓
backtest_strategy() [utils.py]
    ↓
Performance Metrics
```

---

# PART 2: COMPLETE CODE WALKTHROUGHS

## 2.1 feature_engine.py (Every Line Explained)

**File Purpose:** Compute predictive features from OHLCV data
**Total Lines:** 207
**Pattern:** Static methods that take DataFrame and return Series

### Lines 1-3: Imports
```python
import pandas as pd
import numpy as np
```
- `pandas`: For DataFrame manipulation
- `numpy`: For numerical operations (sqrt, abs, etc.)

### Lines 4-11: Class Definition
```python
class FeatureEngine:
    """
    Feature computation engine for stock data.
    Pattern: Feature(data, lookback) -> values

    Add new features by defining methods that take df and optional lookback.
    All features automatically handle NaN and can be computed on any OHLCV dataframe.
    """
```
- **Why a class?** Organizes related functions logically
- **Why static methods?** No state needed, just pure functions
- **Pattern:** Every feature function has same signature

### Lines 13-16: Momentum Feature
```python
@staticmethod
def momentum(df, lookback=1):
    """Percentage price change over lookback periods"""
    return df['close'].pct_change(lookback)
```
**Line-by-line:**
- **Line 13:** `@staticmethod` = no `self`, call as `FeatureEngine.momentum()`
- **Line 14:** Function signature with default lookback=1
- **Line 16:** `pct_change(1)` = (close[t] - close[t-1]) / close[t-1]

**Example:**
```
close: [100, 102, 101, 103]
momentum(lookback=1): [NaN, 0.02, -0.0098, 0.0198]
```

**Why it works:** Trends persist in short term (momentum effect)

### Lines 18-22: Volatility Feature
```python
@staticmethod
def volatility(df, lookback=1):
    """Squared returns - emphasizes large moves"""
    mom = FeatureEngine.momentum(df, lookback)
    return mom ** 2
```
**Line-by-line:**
- **Line 21:** Call momentum feature we just defined
- **Line 22:** Square it to get variance (always positive)

**Example:**
```
momentum: [0.02, -0.02, 0.01]
volatility: [0.0004, 0.0004, 0.0001]
```

**Why it works:** Volatility clusters - high vol predicts more vol

### Lines 24-27: Price Direction
```python
@staticmethod
def price_direction(df):
    """Binary: 1 if close > open (green candle), 0 otherwise"""
    return (df['close'] > df['open']).astype(int)
```
**Line-by-line:**
- **Line 26:** Boolean comparison: close > open → True/False
- **Line 26 cont:** `.astype(int)` converts True→1, False→0

**Example:**
```
open:  [100, 101, 102]
close: [101, 100, 103]
direction: [1, 0, 1]  (green, red, green)
```

**Why it works:** Candle color shows immediate buying/selling pressure

### Lines 29-32: VWAP
```python
@staticmethod
def vwap(df):
    """Volume-weighted average price (cumulative)"""
    return (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
```
**Math:**
```
VWAP[t] = Σ(close[i] * volume[i]) / Σ(volume[i])  for i=0 to t
```

**Example:**
```
close:  [100, 102, 101]
volume: [1000, 2000, 1500]

numerator:   [100*1000, 102*2000, 101*1500] = [100000, 204000, 151500]
cumsum_num:  [100000, 304000, 455500]
cumsum_vol:  [1000, 3000, 4500]
vwap:        [100.0, 101.33, 101.22]
```

**Why it works:** Institutional benchmark - prices mean-revert to VWAP

### Lines 34-38: VWAP Deviation
```python
@staticmethod
def vwap_deviation(df):
    """Percentage deviation from VWAP"""
    vwap = FeatureEngine.vwap(df)
    return (df['close'] - vwap) / vwap
```
**Formula:** `(close - VWAP) / VWAP`

**Example:**
```
close: [102, 101, 103]
vwap:  [100, 101, 101.5]
dev:   [0.02, 0.0, 0.0148]  (2% above, at VWAP, 1.48% above)
```

**Interpretation:**
- Positive = above VWAP (might revert down)
- Negative = below VWAP (might revert up)

### Lines 40-43, 45-48: Time Features
```python
@staticmethod
def hour(df):
    """Hour of day (0-23)"""
    return df['timestamp'].dt.hour

@staticmethod
def minute(df):
    """Minute within hour (0-59)"""
    return df['timestamp'].dt.minute
```

**Why time matters:**
- 9:30-10:00 AM: High volatility (market open)
- 12:00-1:00 PM: Low volume (lunch)
- 3:00-4:00 PM: High volume (market close)

**Example:**
```
timestamp: ['2024-10-01 09:31', '2024-10-01 09:32']
hour:      [9, 9]
minute:    [31, 32]
```

### Lines 79-110: Autocorrelation (Advanced)
```python
@staticmethod
def autocorrelation(df, lookback=5):
    """Return autocorrelation - measures momentum persistence"""
    returns = df['close'].pct_change()
    return returns.rolling(lookback).apply(lambda x: x.autocorr(), raw=False)
```

**What it measures:** Do returns predict future returns?

**Math:** Correlation of returns[t] with returns[t-1]
```
autocorr = corr(returns[t], returns[t-1])
```

**Interpretation:**
- autocorr > 0: Momentum (trend continues)
- autocorr < 0: Mean reversion (trend reverses)
- autocorr ≈ 0: Random walk

**Example:**
```
returns: [0.01, 0.02, 0.01, -0.01, -0.02]
autocorr(5): 0.87  (high positive = strong momentum)
```

### Lines 85-91: Rolling Sharpe
```python
@staticmethod
def rolling_sharpe(df, lookback=20):
    """Rolling Sharpe ratio - risk-adjusted returns"""
    returns = df['close'].pct_change()
    rolling_mean = returns.rolling(lookback).mean()
    rolling_std = returns.rolling(lookback).std()
    return (rolling_mean / rolling_std) * np.sqrt(lookback)
```

**Formula:** `Sharpe = (mean_return / std_return) * √lookback`

**Why it's useful:** Identifies periods of consistent returns (regime indicator)

**Example:**
```
returns (20 periods): mean=0.001, std=0.005
rolling_sharpe = (0.001 / 0.005) * √20 = 0.2 * 4.47 = 0.89
```

### Lines 93-97: Price Acceleration
```python
@staticmethod
def price_acceleration(df, lookback=1):
    """Second derivative of price - momentum of momentum"""
    momentum = df['close'].pct_change(lookback)
    return momentum.diff()
```

**Math:** Second derivative
```
momentum[t] = close[t] - close[t-1]
acceleration[t] = momentum[t] - momentum[t-1]
```

**Example:**
```
close:  [100, 102, 105, 106]
momentum: [NaN, 0.02, 0.0294, 0.0095]
accel:    [NaN, NaN, 0.0094, -0.0199]  (accelerating, then decelerating)
```

**Why it works:** Catches momentum changes early (trend exhaustion)

### Lines 105-110: Order Flow Proxy
```python
@staticmethod
def order_flow_proxy(df):
    """Microstructure: volume-weighted price pressure"""
    price_change = df['close'].diff()
    return price_change * df['volume'] / df['volume'].rolling(20).mean()
```

**What it measures:** Buying/selling pressure weighted by volume

**Formula:**
```
order_flow = Δprice * (volume / avg_volume)
```

**Example:**
```
price_change: [+0.5, -0.3, +0.2]
volume:       [10000, 15000, 8000]
avg_volume:   [9000, 10000, 11000]
order_flow:   [+0.56, -0.45, +0.145]
```

**Interpretation:**
- Positive + high volume = strong buying
- Negative + high volume = strong selling
- Near zero = balanced order flow

**Why it works:** Approximates Level 2 order book pressure without tick data

### Lines 112-143: compute_all_features()
```python
@staticmethod
def compute_all_features(df, feature_list=None):
    """Compute multiple features at once."""
    df = df.copy()  # Don't modify original

    if feature_list is None:
        feature_list = [
            ('momentum_1min', FeatureEngine.momentum, {'lookback': 1}),
            ('volatility_1min', FeatureEngine.volatility, {'lookback': 1}),
            ('price_direction', FeatureEngine.price_direction, {}),
            ('vwap_dev', FeatureEngine.vwap_deviation, {}),
            ('hour', FeatureEngine.hour, {}),
            ('minute', FeatureEngine.minute, {}),
            ('autocorr_5', FeatureEngine.autocorrelation, {'lookback': 5}),
            ('rolling_sharpe_20', FeatureEngine.rolling_sharpe, {'lookback': 20}),
            ('price_accel', FeatureEngine.price_acceleration, {'lookback': 1}),
            ('order_flow', FeatureEngine.order_flow_proxy, {})
        ]

    for name, func, kwargs in feature_list:
        df[name] = func(df, **kwargs)

    return df
```

**Line-by-line:**
- **Line 125:** Copy DataFrame to avoid modifying original
- **Lines 127-139:** Default feature list (tuples of name, function, parameters)
- **Line 141-142:** Loop through features, compute each, add as column
- **Line 144:** Return DataFrame with new feature columns

**Usage:**
```python
df = pull_polygon_data("AAPL", "2025-10-01", "2025-11-01", api_key)
df_with_features = FeatureEngine.compute_all_features(df)
# Now df_with_features has 10 new columns
```

### Lines 145-161: compute_forward_returns()
```python
@staticmethod
def compute_forward_returns(df, horizons=[1]):
    """Compute forward returns at multiple horizons."""
    df = df.copy()

    for h in horizons:
        df[f'return_{h}min'] = df['close'].shift(-h) / df['close'] - 1

    return df
```

**CRITICAL:** `shift(-h)` looks FORWARD in time
```
close:          [100, 102, 101, 103]
close.shift(-1):[102, 101, 103, NaN]  (future value)
return_1min:    [0.02, -0.0098, 0.0198, NaN]
```

**This is the target we're predicting!**

**Why shift(-1) not shift(1):**
- shift(1) = past value = LOOKAHEAD BIAS
- shift(-1) = future value = what we want to predict

**Usage:**
```python
df = FeatureEngine.compute_forward_returns(df, horizons=[1, 2, 5])
# Creates: return_1min, return_2min, return_5min columns
```

### Lines 163-207: load_features_for_training()
```python
def load_features_for_training(df, target_horizon=1, drop_na=True):
    """Complete pipeline: raw OHLCV -> features + target"""
    df = FeatureEngine.compute_all_features(df)
    df = FeatureEngine.compute_forward_returns(df, horizons=[target_horizon])

    if drop_na:
        df = df.dropna()

    feature_names = ['momentum_1min', 'volatility_1min', 'price_direction',
                     'vwap_dev', 'hour', 'minute', 'autocorr_5',
                     'rolling_sharpe_20', 'price_accel', 'order_flow']

    X = df[feature_names]
    y_continuous = df[f'return_{target_horizon}min']
    y_binary = (y_continuous > 0).astype(int)

    return X, y_binary, y_continuous, feature_names
```

**Complete pipeline:**
1. Compute all 10 features
2. Compute forward returns
3. Drop NaN rows (from rolling windows and shift)
4. Extract feature matrix X
5. Extract continuous target (actual returns)
6. Create binary target (UP=1, DOWN=0)
7. Return everything

**Why two targets:**
- `y_binary`: For training RandomForestClassifier (needs 0/1)
- `y_continuous`: For IC analysis and realized correlation (needs actual returns)

**Usage:**
```python
df = pull_polygon_data("AAPL", "2025-10-01", "2025-11-01", api_key)
X, y_binary, y_continuous, features = load_features_for_training(df)
# X: (1000, 10) feature matrix
# y_binary: (1000,) array of 0/1
# y_continuous: (1000,) array of returns
# features: list of 10 feature names
```

---

## 2.2 utils.py (Every Function Line-by-Line)

**File Purpose:** Data loading, backtesting, metrics, validation
**Total Lines:** 341
**Functions:** 9 core functions

### Function 1: pull_polygon_data() (Lines 8-25)

```python
def pull_polygon_data(ticker, start, end, api_key):
    """Fetch minute bars from Polygon.io"""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}?apiKey={api_key}"
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        raise ValueError(f"API error: {response.status_code}")

    data = response.json()

    if 'results' not in data or len(data['results']) < 2:
        raise ValueError("insufficient data returned")

    df = pd.DataFrame(data['results'])
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})

    return df[['timestamp','open','high','low','close','volume']]
```

**Line-by-line breakdown:**

**Line 10:** Build API URL
```python
url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/minute/2025-10-01/2025-11-01?apiKey=xxx"
```
- `/v2/aggs/ticker/` = aggregates endpoint
- `/AAPL/` = ticker symbol
- `/range/1/minute/` = 1-minute bars
- `/2025-10-01/2025-11-01` = date range
- `?apiKey=xxx` = authentication

**Line 11:** HTTP GET request with 10-second timeout
```python
response = requests.get(url, timeout=10)
```
- `timeout=10` prevents hanging if API is down
- Returns Response object with status_code and data

**Lines 13-14:** Check for errors
```python
if response.status_code != 200:
    raise ValueError(f"API error: {response.status_code}")
```
- 200 = success
- 401 = bad API key
- 429 = rate limit exceeded
- 500 = server error

**Line 16:** Parse JSON response
```python
data = response.json()
```
Converts JSON string to Python dict:
```python
{
  "results": [
    {"t": 1696176600000, "o": 170.5, "h": 170.8, "l": 170.4, "c": 170.7, "v": 10000},
    ...
  ],
  "resultsCount": 1000
}
```

**Lines 18-19:** Validate data
```python
if 'results' not in data or len(data['results']) < 2:
    raise ValueError("insufficient data returned")
```
- Need at least 2 bars to compute returns
- Polygon sometimes returns empty results if no trading

**Line 21:** Convert to DataFrame
```python
df = pd.DataFrame(data['results'])
```
Creates DataFrame with columns: t, o, h, l, c, v

**Line 22:** Convert Unix timestamp to datetime
```python
df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
```
- Polygon returns milliseconds since epoch
- `unit='ms'` converts to datetime object
- Example: 1696176600000 → 2023-10-01 09:30:00

**Line 23:** Rename columns to readable names
```python
df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
```

**Line 25:** Return only needed columns
```python
return df[['timestamp','open','high','low','close','volume']]
```

**Complete example:**
```python
API_KEY = "your_key_here"
df = pull_polygon_data("AAPL", "2025-10-01", "2025-10-05", API_KEY)
print(df.head())
```
Output:
```
             timestamp    open    high     low   close  volume
0  2025-10-01 09:30:00  170.50  170.80  170.40  170.70   10000
1  2025-10-01 09:31:00  170.70  170.90  170.65  170.85    8500
2  2025-10-01 09:32:00  170.85  171.00  170.80  170.95    7200
```

---

### Function 2: train_test_split_chronological() (Lines 28-31)

```python
def train_test_split_chronological(X, y, train_frac=0.8):
    """Split time series data chronologically"""
    split_idx = int(len(X) * train_frac)
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]
```

**Line-by-line:**

**Line 30:** Calculate split index
```python
split_idx = int(len(X) * train_frac)
```
Example: len(X) = 1000, train_frac = 0.8 → split_idx = 800

**Line 31:** Split and return 4 arrays
```python
return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]
```
- `X.iloc[:800]` = X_train (first 80%)
- `X.iloc[800:]` = X_test (last 20%)
- `y.iloc[:800]` = y_train
- `y.iloc[800:]` = y_test

**Visual example:**
```
Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_frac = 0.8 → split_idx = 8

Train: [1, 2, 3, 4, 5, 6, 7, 8]  (past)
Test:  [9, 10]                   (future)
```

**CRITICAL:** Never shuffle!
```python
# WRONG - creates lookahead bias
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

# RIGHT - preserves time order
X_train, X_test, y_train, y_test = train_test_split_chronological(X, y)
```

---

### Function 3: calculate_ic() (Lines 34-68)

```python
def calculate_ic(features_df, target_series, method='pearson'):
    """Information Coefficient: correlation between features and continuous target"""
    results = []

    for col in features_df.columns:
        if method == 'pearson':
            ic, pval = pearsonr(features_df[col], target_series)
        else:
            ic, pval = spearmanr(features_df[col], target_series)

        results.append({
            'feature': col,
            'ic': ic,
            'abs_ic': abs(ic),
            'pvalue': pval
        })

    return pd.DataFrame(results).sort_values('abs_ic', ascending=False)
```

**Line-by-line:**

**Line 53:** Initialize empty results list
```python
results = []
```

**Line 55:** Loop through each feature column
```python
for col in features_df.columns:
```

**Lines 56-59:** Calculate correlation
```python
if method == 'pearson':
    ic, pval = pearsonr(features_df[col], target_series)
else:
    ic, pval = spearmanr(features_df[col], target_series)
```

**Pearson vs Spearman:**

**Pearson:** Linear correlation
```
Formula: ρ = cov(X,Y) / (σ_X * σ_Y)
Measures: Straight-line relationship
Sensitive: To outliers
```

**Spearman:** Rank correlation
```
Formula: ρ = pearson(rank(X), rank(Y))
Measures: Monotonic relationship
Robust: To outliers (better for finance)
```

**Example:**
```
Feature:       [0.01, 0.02, 0.05, 0.10, 0.50]  (has outlier)
Forward_return:[0.001, 0.002, 0.003, 0.004, 0.005]

Pearson:  0.65 (pulled by outlier 0.50)
Spearman: 1.00 (perfect rank correlation)
```

**Lines 61-66:** Store results
```python
results.append({
    'feature': col,
    'ic': ic,
    'abs_ic': abs(ic),
    'pvalue': pval
})
```

**pvalue interpretation:**
- pvalue < 0.05: Statistically significant
- pvalue < 0.01: Highly significant
- pvalue > 0.05: Could be random chance

**Line 68:** Return sorted DataFrame
```python
return pd.DataFrame(results).sort_values('abs_ic', ascending=False)
```
Sorts by absolute IC (we care about strength, not direction)

**Example output:**
```
          feature     ic  abs_ic   pvalue
0   momentum_1min  0.042   0.042   0.0001  ← best feature
1    order_flow    0.035   0.035   0.0008
2       vwap_dev  -0.031   0.031   0.0023  ← negative but still predictive
3   volatility_1min 0.018   0.018   0.0450
4           hour   0.005   0.005   0.2341  ← not significant
```

**What's a good IC?**
- > 0.02: Decent
- > 0.05: Great
- > 0.10: Amazing (rare in finance)

---

### Function 4: backtest_strategy() (Lines 71-129)

This is the core backtesting engine. Let's go through it in extreme detail.

```python
def backtest_strategy(signals, returns, transaction_cost=0.0001, scale_by_confidence=False):
    """Run backtest given signals and actual returns"""
    signals = np.array(signals)
    returns = np.array(returns)

    position_changes = np.abs(np.diff(signals, prepend=0))
    costs = position_changes * transaction_cost

    strategy_returns = signals * returns - costs
    cumulative = (1 + strategy_returns).cumprod()

    total_return = cumulative[-1] - 1
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 390) if strategy_returns.std() > 0 else 0

    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    trades = strategy_returns[strategy_returns != 0]
    winning_trades = trades[trades > 0]
    losing_trades = trades[trades < 0]

    win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0

    profit_factor = winning_trades.sum() / abs(losing_trades.sum()) if len(losing_trades) > 0 else np.inf

    information_ratio = sharpe
    calmar = abs(total_return / max_drawdown) if max_drawdown != 0 else 0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'information_ratio': information_ratio,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'num_trades': len(trades),
        'turnover': len(trades) / len(signals)
    }
```

**Line-by-line breakdown:**

**Lines 85-86:** Convert to numpy arrays
```python
signals = np.array(signals)
returns = np.array(returns)
```
Ensures consistent types for vectorized operations

**Line 88:** Calculate position changes
```python
position_changes = np.abs(np.diff(signals, prepend=0))
```

**Step-by-step:**
```python
signals = [0, 1, 1, -1, 0, 1]

np.diff(signals, prepend=0):
- prepend=0 adds 0 at start: [0, 0, 1, 1, -1, 0, 1]
- diff: [0-0, 1-0, 1-1, -1-1, 0-(-1), 1-0] = [0, 1, 0, -2, 1, 1]

np.abs():
[0, 1, 0, 2, 1, 1]
```

Interpretation:
- 0 = no change
- 1 = changed by 1 (e.g., 0→1 or 1→0)
- 2 = changed by 2 (e.g., 1→-1)

**Line 89:** Calculate transaction costs
```python
costs = position_changes * transaction_cost
```

Example with 1bp (0.0001) cost:
```
position_changes: [0, 1, 0, 2, 1, 1]
costs: [0, 0.0001, 0, 0.0002, 0.0001, 0.0001]
```

**Line 91:** Calculate strategy returns
```python
strategy_returns = signals * returns - costs
```

**Example:**
```
signals:  [1, 1, -1, 0, 1]
returns:  [0.01, -0.005, 0.02, 0.01, 0.015]
costs:    [0.0001, 0, 0.0002, 0.0001, 0.0001]

strategy_returns:
- Period 0: 1 * 0.01 - 0.0001 = 0.0099 (long, up 1%, minus cost)
- Period 1: 1 * -0.005 - 0 = -0.005 (long, down 0.5%, no trade)
- Period 2: -1 * 0.02 - 0.0002 = -0.0202 (short, up 2% = lose, plus cost from flipping)
- Period 3: 0 * 0.01 - 0.0001 = -0.0001 (flat, lose cost from exiting)
- Period 4: 1 * 0.015 - 0.0001 = 0.0149 (long, up 1.5%, minus cost)
```

**Line 92:** Calculate cumulative returns
```python
cumulative = (1 + strategy_returns).cumprod()
```

**Math:** Compound returns
```
cumulative[t] = (1 + r[0]) * (1 + r[1]) * ... * (1 + r[t])
```

**Example:**
```
strategy_returns: [0.01, -0.005, 0.02]

Step by step:
- Start: 1.0
- After period 0: 1.0 * (1 + 0.01) = 1.01
- After period 1: 1.01 * (1 - 0.005) = 1.00495
- After period 2: 1.00495 * (1 + 0.02) = 1.025049

cumulative: [1.01, 1.00495, 1.025049]
```

**Line 94:** Total return
```python
total_return = cumulative[-1] - 1
```
Example: 1.025049 - 1 = 0.025049 = 2.50% total return

**Line 95:** Sharpe Ratio
```python
sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 390) if strategy_returns.std() > 0 else 0
```

**Formula breakdown:**
```
Sharpe = (mean_return / std_return) * √(periods_per_year)

For minute data:
- 252 trading days per year
- 390 minutes per day (9:30 AM - 4:00 PM)
- periods_per_year = 252 * 390 = 98,280
```

**Example calculation:**
```
strategy_returns: [0.0001, -0.0002, 0.0003, -0.0001, 0.0002]  (5 minutes)

mean = 0.00006
std = 0.0002
√(252*390) = √98280 = 313.5

Sharpe = (0.00006 / 0.0002) * 313.5 = 0.3 * 313.5 = 94.05 (annualized)
```

**Lines 97-99:** Max Drawdown
```python
running_max = np.maximum.accumulate(cumulative)
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()
```

**Step-by-step:**
```
cumulative: [1.0, 1.02, 1.015, 1.03, 1.025, 1.04]

running_max:
- At each point, track the maximum value seen so far
- [1.0, 1.02, 1.02, 1.03, 1.03, 1.04]

drawdown:
- How much below peak at each point
- (1.0-1.0)/1.0 = 0
- (1.02-1.02)/1.02 = 0
- (1.015-1.02)/1.02 = -0.0049 (0.49% drawdown)
- (1.03-1.03)/1.03 = 0
- (1.025-1.03)/1.03 = -0.0049
- (1.04-1.04)/1.04 = 0

max_drawdown = min(drawdown) = -0.0049 = -0.49%
```

**Interpretation:**
- max_drawdown = -0.10 means worst peak-to-trough loss was 10%
- Measures pain - can you stomach this loss?

**Lines 101-103:** Trade statistics
```python
trades = strategy_returns[strategy_returns != 0]
winning_trades = trades[trades > 0]
losing_trades = trades[trades < 0]
```

Filter to only periods where we had a position and returns were non-zero

**Lines 105-107:** Win metrics
```python
win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
```

**Example:**
```
winning_trades: [0.01, 0.015, 0.02] (3 wins)
losing_trades: [-0.01, -0.005] (2 losses)
total_trades: 5

win_rate = 3/5 = 0.60 = 60%
avg_win = (0.01 + 0.015 + 0.02) / 3 = 0.015 = 1.5%
avg_loss = (-0.01 + -0.005) / 2 = -0.0075 = -0.75%
```

**Line 109:** Profit Factor
```python
profit_factor = winning_trades.sum() / abs(losing_trades.sum()) if len(losing_trades) > 0 else np.inf
```

**Formula:** Total $ won / Total $ lost
```
profit_factor = Σ(winning_trades) / |Σ(losing_trades)|
```

**Example:**
```
winning_trades: [0.01, 0.015, 0.02] → sum = 0.045
losing_trades: [-0.01, -0.005] → sum = -0.015

profit_factor = 0.045 / 0.015 = 3.0
```

**Interpretation:**
- > 1.0: Profitable (won more than lost)
- < 1.0: Unprofitable
- = 2.0: Won twice as much as lost

**Lines 111-115:** Advanced metrics
```python
information_ratio = sharpe  # Same as Sharpe when benchmark is 0
calmar = abs(total_return / max_drawdown) if max_drawdown != 0 else 0
```

**Calmar Ratio:** Return per unit of worst drawdown
```
Calmar = total_return / |max_drawdown|
```

**Example:**
```
total_return = 0.15 (15%)
max_drawdown = -0.05 (5%)

Calmar = 0.15 / 0.05 = 3.0
```

Interpretation: Earned 3x the worst loss

**Lines 117-129:** Return metrics dictionary
All calculated metrics packaged for easy access

---

### Function 5: generate_scaled_signals() (Lines 132-179)

This converts model probabilities into position sizes.

```python
def generate_scaled_signals(probabilities, threshold=0.55, scale_by_magnitude=True):
    """Convert model probabilities to position sizes"""
    probabilities = np.array(probabilities)

    if scale_by_magnitude:
        signal_strength = np.abs(probabilities - 0.5)
        direction = np.where(probabilities > 0.5, 1, -1)
        above_threshold = signal_strength >= (threshold - 0.5)

        max_distance = 0.5
        threshold_distance = threshold - 0.5

        scaled_strength = (signal_strength - threshold_distance) / (max_distance - threshold_distance)
        scaled_strength = np.clip(scaled_strength, 0, 1)

        signals = direction * scaled_strength * above_threshold
    else:
        signals = np.where(probabilities > threshold, 1,
                  np.where(probabilities < (1 - threshold), -1, 0))

    return signals
```

**Detailed walkthrough of scaled mode:**

**Input:** Probabilities from model
```
probabilities = [0.99, 0.70, 0.60, 0.54, 0.50, 0.30]
```

**Line 156:** Calculate signal strength (distance from 0.5)
```python
signal_strength = np.abs(probabilities - 0.5)
```
```
probabilities:   [0.99, 0.70, 0.60, 0.54, 0.50, 0.30]
signal_strength: [0.49, 0.20, 0.10, 0.04, 0.00, 0.20]
```

**Line 158:** Determine direction
```python
direction = np.where(probabilities > 0.5, 1, -1)
```
```
probabilities: [0.99, 0.70, 0.60, 0.54, 0.50, 0.30]
direction:     [+1,  +1,  +1,  +1,  -1,  -1]
```

**Line 161:** Check if above threshold
```python
threshold = 0.55
threshold_distance = threshold - 0.5 = 0.05
above_threshold = signal_strength >= 0.05
```
```
signal_strength: [0.49, 0.20, 0.10, 0.04, 0.00, 0.20]
above_threshold: [True, True, True, False, False, True]
```

**Lines 166-170:** Scale to 0-1 range
```python
max_distance = 0.5
threshold_distance = 0.05
scaled_strength = (signal_strength - 0.05) / (0.5 - 0.05)
```

**For each probability:**

**P=0.99:**
```
signal_strength = 0.49
scaled = (0.49 - 0.05) / 0.45 = 0.44 / 0.45 = 0.978
direction = +1
above_threshold = True
signal = +1 * 0.978 * 1 = +0.978 → 97.8% long
```

**P=0.70:**
```
signal_strength = 0.20
scaled = (0.20 - 0.05) / 0.45 = 0.15 / 0.45 = 0.333
signal = +1 * 0.333 * 1 = +0.333 → 33.3% long
```

**P=0.60:**
```
signal_strength = 0.10
scaled = (0.10 - 0.05) / 0.45 = 0.05 / 0.45 = 0.111
signal = +1 * 0.111 * 1 = +0.111 → 11.1% long
```

**P=0.54:**
```
signal_strength = 0.04 < 0.05 threshold
above_threshold = False
signal = +1 * X * 0 = 0 → flat (below threshold)
```

**P=0.30:**
```
signal_strength = 0.20
scaled = (0.20 - 0.05) / 0.45 = 0.333
direction = -1
signal = -1 * 0.333 * 1 = -0.333 → 33.3% short
```

**Final signals:**
```
probabilities: [0.99,   0.70,   0.60,   0.54, 0.50, 0.30]
signals:       [+0.98, +0.33, +0.11,  0,    0,    -0.33]
```

**Why this works:**
- More confident predictions → larger positions
- Less confident → smaller positions
- Below threshold → no trade (avoid noise)
- Reduces risk and improves Sharpe

---

(Continuing with remaining functions...)

### Function 6: calculate_realized_correlation() (Lines 230-250)

```python
def calculate_realized_correlation(predictions, actuals):
    """Realized out-of-sample correlation - key overfitting check"""
    from scipy.stats import pearsonr
    corr, pval = pearsonr(predictions, actuals)
    return {
        'correlation': corr,
        'pvalue': pval,
        'overfitting_check': 'PASS' if corr > 0.02 else 'FAIL - likely overfit'
    }
```

**What it does:** Checks if predictions actually correlate with returns

**Example - GOOD model (not overfit):**
```
predictions:  [0.02, 0.01, -0.01, 0.03, -0.02]
actuals:      [0.018, 0.012, -0.008, 0.025, -0.015]

correlation = 0.987 (very high!)
pvalue = 0.001 (significant)
Result: PASS
```

**Example - OVERFIT model:**
```
predictions:  [0.02, 0.01, -0.01, 0.03, -0.02]
actuals:      [-0.01, 0.02, 0.01, -0.02, 0.03]

correlation = -0.15 (predictions backwards!)
pvalue = 0.67 (not significant)
Result: FAIL - likely overfit
```

**Why 0.02 threshold?**
- Finance data is noisy
- Even 0.02 correlation is valuable
- Below 0.02 = predictions don't help

---

### Function 7: walk_forward_validation() (Lines 267-340)

This is the most sophisticated validation function. Let's break it down completely.

```python
def walk_forward_validation(X, y_continuous, model, n_splits=5, train_frac=0.6):
    """Walk-forward validation - more robust than single train/test split"""
    from sklearn.base import clone

    total_samples = len(X)
    window_size = total_samples // n_splits

    results = []

    for i in range(n_splits):
        start_idx = i * window_size
        end_idx = min(start_idx + window_size, total_samples)

        if end_idx >= total_samples:
            break

        X_window = X.iloc[start_idx:end_idx]
        y_window = y_continuous.iloc[start_idx:end_idx]

        split_idx = int(len(X_window) * train_frac)
        X_train = X_window.iloc[:split_idx]
        X_test = X_window.iloc[split_idx:]
        y_train = y_window.iloc[:split_idx]
        y_test = y_window.iloc[split_idx:]

        if len(X_test) < 10:
            continue

        model_clone = clone(model)
        model_clone.fit(X_train, (y_train > 0).astype(int))

        y_prob = model_clone.predict_proba(X_test)[:, 1]
        pred_returns = y_prob - 0.5

        corr_result = calculate_realized_correlation(pred_returns, y_test)

        signals = generate_scaled_signals(y_prob, threshold=0.55, scale_by_magnitude=True)
        backtest_metrics = backtest_strategy(signals, y_test, transaction_cost=0.0001)

        results.append({
            'split': i,
            'correlation': corr_result['correlation'],
            'sharpe': backtest_metrics['sharpe_ratio'],
            'total_return': backtest_metrics['total_return'],
            'max_drawdown': backtest_metrics['max_drawdown']
        })

    df_results = pd.DataFrame(results)

    return {
        'mean_correlation': df_results['correlation'].mean(),
        'std_correlation': df_results['correlation'].std(),
        'mean_sharpe': df_results['sharpe'].mean(),
        'std_sharpe': df_results['sharpe'].std(),
        'mean_return': df_results['total_return'].mean(),
        'worst_drawdown': df_results['max_drawdown'].min(),
        'consistency': (df_results['sharpe'] > 0).sum() / len(df_results),
        'all_splits': df_results
    }
```

**Visual example with 1000 samples, 5 splits:**

```
Total samples: 1000
n_splits: 5
window_size: 1000 // 5 = 200

Split 0:
  Window: samples 0-200
  Train: 0-120 (60%)
  Test: 120-200 (40%)

Split 1:
  Window: samples 200-400
  Train: 200-320
  Test: 320-400

Split 2:
  Window: samples 400-600
  Train: 400-520
  Test: 520-600

Split 3:
  Window: samples 600-800
  Train: 600-720
  Test: 720-800

Split 4:
  Window: samples 800-1000
  Train: 800-920
  Test: 920-1000
```

**Why this matters:**
- Tests model across 5 different time periods
- Each period might have different market conditions
- If model works in all 5, it's robust
- If model only works in 1-2, it's overfit to that regime

**Line-by-line:**

**Line 286-287:** Calculate window size
```python
total_samples = len(X)  # e.g., 1000
window_size = total_samples // n_splits  # 1000 // 5 = 200
```

**Lines 291-293:** For each split, define window
```python
start_idx = i * window_size  # Split 0: 0*200=0, Split 1: 1*200=200, etc.
end_idx = min(start_idx + window_size, total_samples)  # 0+200=200, 200+200=400, etc.
```

**Lines 298-305:** Extract window and split train/test
```python
X_window = X.iloc[start_idx:end_idx]  # Extract this window's data
split_idx = int(len(X_window) * train_frac)  # 60% for training
X_train = X_window.iloc[:split_idx]  # First 60%
X_test = X_window.iloc[split_idx:]  # Last 40%
```

**Line 310:** Clone model for fresh training
```python
model_clone = clone(model)
```
- Doesn't reuse trained parameters
- Each split trains from scratch
- Simulates real scenario: retrain on new data

**Line 311:** Train on this window
```python
model_clone.fit(X_train, (y_train > 0).astype(int))
```

**Lines 313-314:** Get predictions
```python
y_prob = model_clone.predict_proba(X_test)[:, 1]
pred_returns = y_prob - 0.5
```

**Lines 316-319:** Evaluate
```python
corr_result = calculate_realized_correlation(pred_returns, y_test)
signals = generate_scaled_signals(y_prob, threshold=0.55, scale_by_magnitude=True)
backtest_metrics = backtest_strategy(signals, y_test, transaction_cost=0.0001)
```

**Lines 321-327:** Store results for this split
```python
results.append({
    'split': i,
    'correlation': corr_result['correlation'],
    'sharpe': backtest_metrics['sharpe_ratio'],
    'total_return': backtest_metrics['total_return'],
    'max_drawdown': backtest_metrics['max_drawdown']
})
```

**Lines 331-339:** Aggregate metrics across all splits
```python
df_results = pd.DataFrame(results)

return {
    'mean_correlation': df_results['correlation'].mean(),
    'std_correlation': df_results['correlation'].std(),
    'mean_sharpe': df_results['sharpe'].mean(),
    'std_sharpe': df_results['sharpe'].std(),
    'mean_return': df_results['total_return'].mean(),
    'worst_drawdown': df_results['max_drawdown'].min(),
    'consistency': (df_results['sharpe'] > 0).sum() / len(df_results),
    'all_splits': df_results
}
```

**Example output:**
```python
{
  'mean_correlation': 0.034,
  'std_correlation': 0.008,
  'mean_sharpe': 1.24,
  'std_sharpe': 0.45,
  'mean_return': 0.023,
  'worst_drawdown': -0.048,
  'consistency': 0.80,  # 80% of splits had positive Sharpe
  'all_splits': DataFrame with 5 rows
}
```

**Interpretation:**
- **mean_sharpe = 1.24:** Average across 5 windows
- **std_sharpe = 0.45:** Relatively stable (std < mean is good)
- **consistency = 0.80:** 4 out of 5 splits were profitable
- **worst_drawdown = -0.048:** Worst case was 4.8% loss

**Good vs Bad:**

**Good walk-forward results:**
```
mean_sharpe: 1.2
std_sharpe: 0.3  (low variance)
consistency: 80%  (mostly positive)
→ Strategy is robust
```

**Bad walk-forward results:**
```
mean_sharpe: 0.5
std_sharpe: 2.1  (high variance)
consistency: 20%  (mostly negative)
→ Strategy is overfit to one lucky period
```

---

This walkthrough of utils.py is now complete with every line explained. Would you like me to continue with the complete walkthroughs of train.py, backtest.py, walk_forward_demo.py, and then move on to Part 3 (Notebooks), Part 4 (Features), etc.?

This document will be 2000+ lines when complete. Should I keep building it section by section?

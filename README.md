# ml stock predictor

random forest classifier for intraday price movement prediction using minute-level data

## model overview

**objective**: predict next-minute price direction (up/down) and convert to trading signals (buy/sell/hold)

**data source**: polygon.io minute-level ohlcv bars

**features** (6 total):
- momentum_1min: 1-minute price return
- volatility_1min: squared momentum
- price_direction: binary close > open
- vwap_dev: deviation from volume-weighted average price
- hour: time of day
- minute: minute within hour

**model**: random forest (100 trees), chronological train/test split to avoid lookahead bias

**decision logic**:
- buy if P(up) > 0.55
- sell if P(down) > 0.55
- hold otherwise

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

**train model:**
```bash
jupyter notebook predictor.ipynb
# runs training, saves to trained_stock_model.pkl
```

**backtest:**
```bash
python3 backtest.py
# outputs sharpe, max dd, win rate, profit factor, etc
```

**live api:**
```bash
cd backend
python3 api.py
# serves model on localhost:8000
curl http://localhost:8000/api/predict/AAPL
```

## file structure

```
Project/
├── predictor.ipynb          # model training notebook
├── backtest.py              # backtesting with proper metrics
├── trained_stock_model.pkl  # saved random forest model
├── backend/
│   └── api.py              # fastapi server for live predictions
└── requirements.txt
```

## for trading competitions

- api endpoint at `/api/predict/{ticker}` returns json with decision + probabilities
- adjustable threshold via `?prob_threshold=0.6` for more conservative signals
- chronological split prevents lookahead bias (critical for live trading)
- transaction costs included in backtest (10bps per trade)

## limitations

- model barely beats random (52-53% accuracy)
- negative sharpe in backtest period
- high trade frequency (524 trades in 1 month) = high transaction costs
- no risk management (fixed position sizing)
- features are too simple for real alpha

this is a starting framework. real improvements would need:
- more sophisticated features (order flow, market microstructure)
- longer training period
- ensemble methods or deep learning
- proper position sizing and risk controls
- regime detection

# Complete Advanced Trading System Guide

Everything you need to know about the upgraded trading system.

## What Was Added

### 1. Extended Features (17 new technical indicators)
- **RSI** (7 & 14 period)
- **MACD** Histogram
- **Bollinger Bands** Position
- **ATR** (Average True Range)
- **Stochastic** Oscillator
- **OBV** (On-Balance Volume)
- **MFI** (Money Flow Index)
- **Williams %R**
- **CCI** (Commodity Channel Index)
- **ROC** (Rate of Change)
- **PVT** (Price Volume Trend)
- **VWMA** (Volume-Weighted MA)
- **Keltner Channels**
- **ADX** (Trend Strength)
- **Volume Spike Detection**
- **Gap Detection**
- Additional momentum/volatility variants

**Total: 31 features** (was 10)

### 2. Performance Tracking System
- SQLite database logging all trades
- Position tracking with entry/exit details
- Daily performance summaries
- Prediction accuracy tracking
- Equity curve generation
- Sharpe/Sortino/Calmar ratios
- Max drawdown calculation
- Win/loss distribution analysis
- Performance by ticker
- Performance by time of day

### 3. Advanced Backtesting Suite
- Parameter sweep (test multiple thresholds/costs)
- Walk-forward validation (5+ time windows)
- Monte Carlo simulation (1000+ runs)
- Model comparison framework
- Heatmap visualization
- Statistical significance testing

### 4. Enhanced Risk Management
- Portfolio heat monitoring (total capital at risk)
- Sector exposure limits
- Correlation-based position limits
- Volatility-adjusted position sizing
- Kelly criterion calculator
- Value at Risk (VaR) & Conditional VaR
- Dynamic position sizing based on performance
- Consecutive loss detection
- ATR-based stop losses

### 5. Advanced Model Training
- **XGBoost** support
- **LightGBM** support
- **Ensemble** models (voting classifier)
- Gradient Boosting
- Automated model comparison
- Feature importance analysis
- Best model auto-selection

### 6. Web Dashboard
- Real-time performance metrics
- Live equity curve
- Open positions display
- Recent trades table
- Auto-refresh every 10 seconds
- Clean, modern UI
- Multiple API endpoints

### 7. Alert System
- Email notifications
- Trade execution alerts
- Position closed alerts
- Daily summary emails
- Risk warning alerts
- System error notifications
- Large move detection
- Daily loss limit alerts

---

## Quick Start

### Install Additional Dependencies

```bash
pip3 install xgboost lightgbm matplotlib seaborn
```

### Train Advanced Model

```bash
python train_advanced.py
```

This compares RF, GB, XGBoost, LightGBM, and Ensemble models.

### Run Comprehensive Backtest

```bash
python advanced_backtest.py
```

Tests multiple parameters and validates across time windows.

### Start the Dashboard

```bash
cd backend
python3 dashboard_api.py
```

Open http://localhost:8001

### Start Trading (Enhanced)

```bash
cd backend
python3 trader.py --test
```

Bot now uses:
- All 31 features
- Trade logging
- Risk management
- Alert system

---

## Usage Examples

### Training with Extended Features

```python
from feature_engine import load_features_for_training

# Old way (10 features)
X, y_binary, y_continuous, names = load_features_for_training(df)

# New way (31 features)
X, y_binary, y_continuous, names = load_features_for_training(
    df,
    use_extended=True
)

print(f"Using {len(names)} features")
```

### Logging Trades

```python
from backend.trade_logger import TradeLogger

logger = TradeLogger()

# Log a trade
logger.log_trade(
    ticker='AAPL',
    side='BUY',
    qty=10,
    price=180.50,
    confidence=0.67
)

# Log a completed position
logger.log_position(
    ticker='AAPL',
    side='BUY',
    qty=10,
    entry_price=180.50,
    exit_price=185.20,
    entry_time=datetime.now(),
    exit_time=datetime.now(),
    exit_reason='take_profit'
)

# Get performance summary
summary = logger.get_performance_summary()
print(summary)
```

### Performance Analysis

```python
from backend.performance_tracker import PerformanceTracker

tracker = PerformanceTracker()

# Print full report
tracker.print_report()

# Plot equity curve
tracker.plot_equity_curve()

# Plot drawdown
tracker.plot_drawdown()

# Analyze by ticker
by_ticker = tracker.analyze_by_ticker()
print(by_ticker)

# Analyze by time of day
by_time = tracker.analyze_by_time_of_day()
print(by_time)
```

### Advanced Backtesting

```python
from advanced_backtest import AdvancedBacktester

backtester = AdvancedBacktester(df)

# Parameter sweep
param_grid = {
    'threshold': [0.50, 0.55, 0.60],
    'transaction_cost': [0.0001, 0.0005]
}
results = backtester.parameter_sweep(param_grid, use_extended_features=True)

# Walk-forward validation
wf_results = backtester.walk_forward_test(n_splits=5)
print(f"Mean Sharpe: {wf_results['mean_sharpe']:.2f}")

# Monte Carlo simulation
mc_results = backtester.monte_carlo_simulation(n_simulations=1000)
print(f"Percentile: {mc_results['percentile']*100:.1f}%")
```

### Risk Management

```python
from backend.risk_manager import RiskManager

rm = RiskManager(
    max_portfolio_heat=0.02,
    max_correlated_positions=3,
    max_sector_exposure=0.30
)

# Check if trade allowed
can_trade, reason = rm.should_allow_trade(
    ticker='AAPL',
    positions=current_positions,
    account_equity=100000,
    daily_pnl=-200,
    max_daily_loss=500
)

# Calculate position size
qty = rm.calculate_position_size_volatility_adjusted(
    ticker='AAPL',
    current_price=180,
    account_equity=100000,
    historical_volatility=0.25
)

# Risk report
rm.print_risk_report(positions, account_equity, daily_pnl)
```

### Alerts Setup

1. Edit `backend/.env`:

```bash
ALERT_EMAIL_ENABLED=true
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
RECIPIENT_EMAIL=your_email@gmail.com
```

2. Use in code:

```python
from backend.alert_system import AlertSystem

alerts = AlertSystem()

alerts.alert_trade_executed('AAPL', 'BUY', 10, 180.50, 0.67)
alerts.alert_position_closed('AAPL', 'BUY', 10, 180.50, 185.20, 47.00, 0.026, 'take_profit')
alerts.alert_daily_summary(15, 9, 6, 234.50, 102345.67)
```

---

## Dashboard Features

**Metrics Display:**
- Total P&L
- Win Rate
- Total Trades
- Sharpe Ratio
- Max Drawdown
- Current Equity

**Recent Positions Table:**
- Ticker, Side, Qty
- Entry/Exit Prices
- P&L
- Timestamps

**API Endpoints:**
- `/api/performance` - Performance metrics
- `/api/positions/recent` - Recent positions
- `/api/trades/recent` - Recent trades
- `/api/equity_curve` - Equity curve data
- `/api/performance/by_ticker` - Per-ticker stats
- `/api/performance/by_time` - Hourly stats

---

## File Structure

```
HFTModel/
├── feature_engine.py              # 31 features
├── train_advanced.py              # XGBoost/LightGBM/Ensemble
├── advanced_backtest.py           # Comprehensive backtesting
├── backend/
│   ├── trader.py                 # Enhanced trading bot
│   ├── trader_config.py          # Bot configuration
│   ├── trade_logger.py           # Database logging
│   ├── performance_tracker.py    # Performance analysis
│   ├── risk_manager.py           # Risk controls
│   ├── alert_system.py           # Email alerts
│   ├── dashboard_api.py          # Web dashboard
│   ├── trading.db                # SQLite database
│   └── .env                      # Configuration
└── COMPLETE_SYSTEM_GUIDE.md      # This file
```

---

## Tips & Best Practices

1. **Always backtest first** - Never deploy untested strategies
2. **Use extended features** - More features = better predictions
3. **Monitor the dashboard** - Keep it open during trading hours
4. **Set up email alerts** - Get notified immediately
5. **Review trade logs daily** - Learn from wins and losses
6. **Start with conservative risk** - Increase gradually
7. **Compare models** - XGBoost often beats Random Forest
8. **Walk-forward validate** - Most realistic test
9. **Check sector exposure** - Don't overtrade correlated stocks
10. **Track performance metrics** - Sharpe > 1.0 is good, > 1.5 is excellent

---

## Performance Expectations

With extended features and advanced models:

**Model Accuracy:**
- 52-55% (minute data)
- 55-58% (with good features)
- 60%+ rare but possible

**Sharpe Ratio:**
- < 0.5: Poor
- 0.5-1.0: Acceptable
- 1.0-1.5: Good
- 1.5-2.0: Very Good
- 2.0+: Excellent

**Win Rate:**
- 50-55%: Normal
- 55-60%: Good
- 60%+: Excellent

**Profit Factor:**
- < 1.0: Losing
- 1.0-1.5: Breakeven/Slight profit
- 1.5-2.0: Good
- 2.0+: Excellent

---

## Common Issues & Solutions

**Issue:** Model accuracy not improving with more features
- **Solution:** Feature selection, try different models, check for collinearity

**Issue:** Good backtest, poor live performance
- **Solution:** Check for lookahead bias, overfitting, market regime change

**Issue:** Dashboard shows no data
- **Solution:** Ensure trading.db exists, check if bot has logged trades

**Issue:** Email alerts not sending
- **Solution:** Verify .env config, use Gmail app password, check SMTP settings

**Issue:** Risk manager blocking all trades
- **Solution:** Adjust limits in risk_manager.py initialization

---

## Next Steps

1. Run `python train_advanced.py` to compare models
2. Run `python advanced_backtest.py` for validation
3. Start dashboard: `cd backend && python3 dashboard_api.py`
4. Configure alerts in `backend/.env`
5. Test with paper trading: `cd backend && python3 trader.py --test`
6. Monitor dashboard at http://localhost:8001
7. Review performance daily using `performance_tracker.py`
8. Adjust parameters based on results
9. When satisfied, run during market hours without --test flag

Remember: Paper trading first, real money later!

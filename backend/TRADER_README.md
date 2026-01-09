# Trading Bot Setup

## Quick Start

**During market hours (9:30am-4pm ET):**
```bash
cd backend
python3 trader.py
```

**Test mode (anytime, skips market hours check):**
```bash
cd backend
python3 trader.py --test
```

## Configuration

Edit `trader_config.py`:

- `WATCHLIST` - stocks to trade (smaller caps = more volatility)
- `PREDICTION_THRESHOLD` - minimum confidence to trade (0.55 = 55%)
- `POSITION_SIZE_USD` - dollar amount per trade
- `MAX_POSITIONS` - max concurrent positions
- `MAX_DAILY_LOSS` - stop trading if hit
- `ALLOW_SHORTING` - True = go long AND short, False = long only
- `STOP_LOSS_PCT` - auto-exit at 2% loss
- `TAKE_PROFIT_PCT` - auto-exit at 3% profit

## How It Works

1. Every 60 seconds, scans watchlist
2. Gets latest minute bars from Alpaca
3. Calculates features
4. Gets model prediction
5. If bullish (confidence > threshold): BUY (go long)
6. If bearish (confidence > threshold): SELL (go short)
7. Monitors positions for stop loss / take profit
8. Auto-exits on limits

## Risk Controls

- Max positions limit
- Daily loss limit
- Stop loss per position (2%)
- Take profit per position (3%)
- Two-way trading: long AND short

## Monitoring

Bot prints status every cycle:
- Current equity & PnL
- Open positions
- Entry prices & current P&L

Press Ctrl+C to stop gracefully.

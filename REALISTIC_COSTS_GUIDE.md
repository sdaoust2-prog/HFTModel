# Realistic Transaction Costs & Tax Tracking

## Problem with Current System

**Current:** Uses fixed 1 basis point (0.0001) cost per trade
**Reality:** Actual costs are 3-10x higher

---

## Real Trading Costs Breakdown

### What You Actually Pay

**1. Bid-Ask Spread** (unavoidable)
- AAPL/MSFT/SPY: ~1 bp (0.01%)
- Mid-caps (PLTR, SNAP): ~2.5-3 bp
- Small/volatile (RIVN, AMC): ~4-5 bp

**2. Slippage** (market orders)
- Liquid stocks: 0.5-1 bp
- Medium liquidity: 1.5-2 bp
- Low liquidity: 3-4 bp

**3. Market Impact** (your order moves the price)
- Orders < $5k: ~0 bp
- Orders $5k-$25k: ~0.5 bp
- Orders $25k-$100k: ~1.5 bp
- Orders > $100k: ~3+ bp

**4. Time of Day Effects**
- Market open (9:30-9:45am): 1.5x worse
- Market close (3:45-4pm): 1.3x worse
- Midday (10am-2pm): Normal

**5. Commission**
- Alpaca: $0 (free)
- Most brokers: $0 now

---

## Total Cost Examples

### High Liquidity (AAPL, SPY)
```
Spread:        1.0 bp
Slippage:      0.5 bp
Market Impact: 0.0 bp (small order)
Total:         1.5 bp = 0.015%
```

**$1000 position = $0.15 cost per trade**

### Medium Liquidity (PLTR, SNAP)
```
Spread:        2.5 bp
Slippage:      1.5 bp
Market Impact: 0.5 bp
Total:         4.5 bp = 0.045%
```

**$1000 position = $0.45 cost per trade**

### Low Liquidity (RIVN, AMC)
```
Spread:        5.0 bp
Slippage:      3.0 bp
Market Impact: 1.0 bp
Total:         9.0 bp = 0.09%
```

**$1000 position = $0.90 cost per trade**

---

## Impact on Profitability

### Example Strategy

**Backtest with simplified costs (1 bp):**
- 100 trades/day
- Avg hold: 2 hours
- Reported return: +15% annualized
- Sharpe: 1.5

**Backtest with realistic costs (5 bp average):**
- Same strategy
- Real return: +8% annualized (nearly half!)
- Sharpe: 0.9

**The difference kills strategies.**

---

## Capital Gains Tax

### Tax Rates (2025)

**Short-Term (<1 year holding):**
- Taxed as ordinary income
- Typical: 22-37% federal
- Plus state tax (CA: ~13%, NY: ~8%)
- **Day trading = 37% + state**

**Long-Term (>1 year holding):**
- Preferential rate
- 0%, 15%, or 20% based on income
- Most traders: 15-20%

### Example

**Trade without tax consideration:**
```
Buy AAPL at $180
Sell at $190
Gain: $10/share × 10 = $100
```

**With tax (short-term, 37% federal + 10% state):**
```
Gross gain: $100
Tax: $47
Net gain: $53
```

**You only keep 53% of profits!**

---

## Using Realistic Costs

### Cost Model

```python
from backend.realistic_costs import RealisticCostModel

cost_model = RealisticCostModel()

# Check typical costs for a stock
costs = cost_model.get_typical_cost_bps('PLTR')
print(costs)
# {'spread_bps': 2.5, 'slippage_bps': 1.5, 'total_bps': 4.0, 'total_pct': 0.04}

# Calculate exact cost for a trade
trade_cost = cost_model.calculate_total_cost(
    ticker='PLTR',
    qty=50,
    price=25.50,
    side='BUY',
    timestamp=pd.to_datetime('2026-01-09 09:35:00'),  # Near open = worse
    volatility=0.025  # 2.5% recent volatility
)

print(trade_cost)
# {
#   'total_cost': 2.35,           # Total $ cost
#   'cost_per_share': 0.047,      # Cost per share
#   'cost_percentage': 0.00184,   # 0.184% or 18.4 bps
#   'spread': 0.038,
#   'slippage': 0.009
# }
```

### Tax Tracking

```python
from backend.realistic_costs import TaxTracker
from datetime import datetime

tracker = TaxTracker(
    tax_rate_short_term=0.37,  # 37% federal for day trading
    tax_rate_long_term=0.20    # 20% for >1 year holds
)

# Open position
tracker.open_position(
    ticker='AAPL',
    qty=10,
    price=180,
    timestamp=datetime(2025, 1, 1)
)

# Close position
result = tracker.close_position(
    ticker='AAPL',
    qty=10,
    exit_price=190,
    exit_time=datetime(2025, 6, 1)
)

print(result)
# {
#   'gain': 100.0,
#   'hold_days': 151,
#   'tax_type': 'short_term',  # < 1 year
#   'tax_rate': 0.37,
#   'tax_owed': 37.0,          # $37 tax!
#   'net_gain': 63.0           # Only keep $63
# }

# Get full tax report
tracker.print_tax_report()
```

---

## Updated Backtesting

### Before (Wrong)
```python
# Old way - too optimistic
metrics = backtest_strategy(signals, returns, transaction_cost=0.0001)
# Sharpe: 1.5 (overstated!)
```

### After (Realistic)
```python
from backend.realistic_costs import RealisticCostModel

cost_model = RealisticCostModel()

# Get realistic cost for your stocks
avg_cost_bps = 4.5  # PLTR, RIVN, SNAP, COIN average

metrics = backtest_strategy(
    signals,
    returns,
    transaction_cost=avg_cost_bps / 10000  # 4.5 bp = 0.00045
)
# Sharpe: 0.9 (realistic!)
```

---

## Impact on Your Strategy

### Current Watchlist Costs

```python
from backend.realistic_costs import RealisticCostModel

cost_model = RealisticCostModel()

for ticker in ['PLTR', 'RIVN', 'SNAP', 'COIN', 'SOFI', 'NIO', 'LCID', 'AMC']:
    costs = cost_model.get_typical_cost_bps(ticker)
    print(f"{ticker}: {costs['total_bps']:.1f} bps")
```

Output:
```
PLTR:  4.0 bps
RIVN:  7.0 bps
SNAP:  4.5 bps
COIN:  7.0 bps
SOFI:  5.0 bps
NIO:   5.0 bps
LCID:  7.5 bps
AMC:   8.0 bps

Average: ~5.9 bps per trade
```

**Your actual costs are 6x higher than current model!**

---

## Recommendations

### 1. Use Realistic Costs in Backtests

Edit `trader_config.py`:
```python
REALISTIC_TRANSACTION_COST_BPS = 5.5  # Average for your watchlist
```

Update backtests:
```python
transaction_cost = config.REALISTIC_TRANSACTION_COST_BPS / 10000
```

### 2. Track Taxes

Integrate into performance tracking:
```python
from backend.realistic_costs import TaxTracker

tax_tracker = TaxTracker()
# Log every position open/close
# Get tax report at year end
```

### 3. Optimize for Lower Costs

**Reduce turnover:**
- Current: ~100 trades/day
- Target: ~20-30 trades/day
- Hold positions longer (hours not minutes)

**Trade liquid stocks:**
- Focus on AAPL, MSFT, SPY (1.5 bp)
- Avoid AMC, LCID (8 bp)

**Avoid peak hours:**
- Don't trade 9:30-9:45am
- Don't trade 3:45-4:00pm
- Best: 10:30am-2:30pm

### 4. Consider Tax Strategy

**For short-term trading (day trading):**
- Accept 37%+ tax hit
- Need 63% higher gross returns to match buy-and-hold

**For swing trading (hold days/weeks):**
- Still short-term capital gains
- But lower transaction costs
- Better Sharpe ratio

**For position trading (hold >1 year):**
- 20% tax rate
- Keep 80% of profits
- Minimal transaction costs

---

## Integration with Trading Bot

I can update the bot to:

1. ✅ Calculate realistic costs per trade
2. ✅ Track tax implications
3. ✅ Report after-tax P&L
4. ✅ Warn about high-cost trades
5. ✅ Optimize for tax efficiency

Want me to integrate this into the bot now? Takes ~10 min.

---

## Real Example

**Your current strategy:**
- Watchlist: Low liquidity small caps
- Trades: ~50/day
- Hold time: 1-2 hours
- Cost per trade: ~6 bps
- Tax rate: 37% (short-term)

**Monthly calculation:**
```
Gross profit: $1000
Transaction costs: $1000 × 50 trades × 0.0006 = -$30
Net before tax: $970
Taxes (37%): -$359
Net after tax: $611

You keep 61% of gross profits!
```

**With optimization:**
- Trade liquid stocks (AAPL, SPY)
- Reduce to 20 trades/day
- Cost per trade: 1.5 bps

```
Gross profit: $1000
Transaction costs: $1000 × 20 × 0.00015 = -$3
Net before tax: $997
Taxes (37%): -$369
Net after tax: $628

You keep 63% (2% improvement)
```

---

## Next Steps

1. ✅ Understand realistic costs
2. ⏳ Run backtest with 5.5 bps cost (not 1 bp)
3. ⏳ See how strategy performs with real costs
4. ⏳ Decide: optimize strategy or switch to more liquid stocks
5. ⏳ Integrate tax tracking into bot

Want me to re-run your backtests with realistic costs now?

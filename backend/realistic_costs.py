import numpy as np
import pandas as pd
from datetime import datetime

class RealisticCostModel:
    def __init__(self):
        self.stock_params = {
            'AAPL': {'spread_bps': 1.0, 'liquidity': 'high'},
            'MSFT': {'spread_bps': 1.0, 'liquidity': 'high'},
            'GOOGL': {'spread_bps': 1.5, 'liquidity': 'high'},
            'TSLA': {'spread_bps': 2.0, 'liquidity': 'medium'},
            'SPY': {'spread_bps': 0.5, 'liquidity': 'high'},
            'PLTR': {'spread_bps': 2.5, 'liquidity': 'medium'},
            'RIVN': {'spread_bps': 4.0, 'liquidity': 'low'},
            'SNAP': {'spread_bps': 3.0, 'liquidity': 'medium'},
            'COIN': {'spread_bps': 4.0, 'liquidity': 'low'},
            'SOFI': {'spread_bps': 3.5, 'liquidity': 'medium'},
            'NIO': {'spread_bps': 3.5, 'liquidity': 'medium'},
            'LCID': {'spread_bps': 4.5, 'liquidity': 'low'},
            'AMC': {'spread_bps': 5.0, 'liquidity': 'low'},
        }

        self.base_commission = 0

    def get_spread_cost(self, ticker, price):
        params = self.stock_params.get(ticker, {'spread_bps': 3.0})
        spread_bps = params['spread_bps']
        return price * (spread_bps / 10000)

    def get_slippage(self, ticker, qty, price, side, volatility=None):
        params = self.stock_params.get(ticker, {'spread_bps': 3.0, 'liquidity': 'medium'})

        base_slippage_bps = {
            'high': 0.5,
            'medium': 1.5,
            'low': 3.0
        }[params['liquidity']]

        market_impact = self.calculate_market_impact(ticker, qty, price)

        if volatility and volatility > 0.02:
            volatility_multiplier = 1.0 + (volatility - 0.02) * 20
            base_slippage_bps *= volatility_multiplier

        total_slippage_bps = base_slippage_bps + market_impact

        slippage_dollars = price * (total_slippage_bps / 10000)

        return slippage_dollars if side == 'BUY' else -slippage_dollars

    def calculate_market_impact(self, ticker, qty, price):
        position_value = qty * price

        if position_value < 5000:
            impact_bps = 0.0
        elif position_value < 25000:
            impact_bps = 0.5
        elif position_value < 100000:
            impact_bps = 1.5
        else:
            impact_bps = 3.0

        return impact_bps

    def get_time_of_day_multiplier(self, timestamp):
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)

        hour = timestamp.hour
        minute = timestamp.minute

        if hour == 9 and minute < 45:
            return 1.5
        elif hour == 15 and minute >= 45:
            return 1.3
        elif 10 <= hour <= 14:
            return 1.0
        else:
            return 1.1

    def calculate_total_cost(self, ticker, qty, price, side, timestamp=None, volatility=None):
        spread = self.get_spread_cost(ticker, price)

        slippage = self.get_slippage(ticker, qty, price, side, volatility)

        if timestamp:
            time_multiplier = self.get_time_of_day_multiplier(timestamp)
            spread *= time_multiplier
            slippage *= time_multiplier

        total_cost_per_share = spread + abs(slippage)

        total_cost = total_cost_per_share * qty + self.base_commission

        cost_percentage = total_cost / (price * qty) if price * qty > 0 else 0

        return {
            'total_cost': total_cost,
            'cost_per_share': total_cost_per_share,
            'cost_percentage': cost_percentage,
            'spread': spread,
            'slippage': slippage,
            'commission': self.base_commission
        }

    def adjust_backtest_for_costs(self, trades_df):
        if 'ticker' not in trades_df.columns:
            trades_df['ticker'] = 'UNKNOWN'

        if 'timestamp' not in trades_df.columns:
            trades_df['timestamp'] = None

        adjusted_trades = []

        for idx, trade in trades_df.iterrows():
            costs = self.calculate_total_cost(
                ticker=trade.get('ticker', 'UNKNOWN'),
                qty=abs(trade.get('qty', 0)),
                price=trade.get('price', 0),
                side=trade.get('side', 'BUY'),
                timestamp=trade.get('timestamp'),
                volatility=trade.get('volatility')
            )

            adjusted_trade = trade.copy()
            adjusted_trade['transaction_cost'] = costs['total_cost']
            adjusted_trade['cost_percentage'] = costs['cost_percentage']
            adjusted_trade['spread_cost'] = costs['spread'] * abs(trade.get('qty', 0))
            adjusted_trade['slippage_cost'] = costs['slippage'] * abs(trade.get('qty', 0))

            adjusted_trades.append(adjusted_trade)

        return pd.DataFrame(adjusted_trades)

    def get_typical_cost_bps(self, ticker):
        params = self.stock_params.get(ticker, {'spread_bps': 3.0, 'liquidity': 'medium'})

        spread_bps = params['spread_bps']
        slippage_bps = {
            'high': 0.5,
            'medium': 1.5,
            'low': 3.0
        }[params['liquidity']]

        typical_total = spread_bps + slippage_bps

        return {
            'spread_bps': spread_bps,
            'slippage_bps': slippage_bps,
            'total_bps': typical_total,
            'total_pct': typical_total / 100
        }

class TaxTracker:
    def __init__(self, tax_rate_short_term=0.37, tax_rate_long_term=0.20):
        self.tax_rate_short_term = tax_rate_short_term
        self.tax_rate_long_term = tax_rate_long_term

        self.positions = {}
        self.realized_gains = []
        self.unrealized_gains = []

    def open_position(self, ticker, qty, price, timestamp, side='LONG'):
        if ticker not in self.positions:
            self.positions[ticker] = []

        self.positions[ticker].append({
            'qty': qty,
            'entry_price': price,
            'entry_time': timestamp,
            'side': side
        })

    def close_position(self, ticker, qty, exit_price, exit_time):
        if ticker not in self.positions or not self.positions[ticker]:
            return None

        position = self.positions[ticker].pop(0)

        if position['side'] == 'LONG':
            gain = (exit_price - position['entry_price']) * qty
        else:
            gain = (position['entry_price'] - exit_price) * qty

        hold_days = (exit_time - position['entry_time']).days

        is_long_term = hold_days >= 365

        if is_long_term:
            tax_rate = self.tax_rate_long_term
            tax_type = 'long_term'
        else:
            tax_rate = self.tax_rate_short_term
            tax_type = 'short_term'

        tax_owed = gain * tax_rate if gain > 0 else 0

        realized = {
            'ticker': ticker,
            'qty': qty,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'gain': gain,
            'gain_pct': gain / (position['entry_price'] * qty),
            'hold_days': hold_days,
            'tax_type': tax_type,
            'tax_rate': tax_rate,
            'tax_owed': tax_owed,
            'net_gain': gain - tax_owed,
            'entry_time': position['entry_time'],
            'exit_time': exit_time
        }

        self.realized_gains.append(realized)

        return realized

    def get_unrealized_pnl(self, ticker, current_price, current_time):
        if ticker not in self.positions or not self.positions[ticker]:
            return []

        unrealized = []
        for pos in self.positions[ticker]:
            if pos['side'] == 'LONG':
                gain = (current_price - pos['entry_price']) * pos['qty']
            else:
                gain = (pos['entry_price'] - current_price) * pos['qty']

            hold_days = (current_time - pos['entry_time']).days
            is_long_term = hold_days >= 365
            tax_rate = self.tax_rate_long_term if is_long_term else self.tax_rate_short_term

            tax_owed = gain * tax_rate if gain > 0 else 0

            unrealized.append({
                'ticker': ticker,
                'qty': pos['qty'],
                'entry_price': pos['entry_price'],
                'current_price': current_price,
                'gain': gain,
                'hold_days': hold_days,
                'tax_type': 'long_term' if is_long_term else 'short_term',
                'tax_rate': tax_rate,
                'estimated_tax': tax_owed,
                'estimated_net_gain': gain - tax_owed
            })

        return unrealized

    def get_tax_summary(self):
        if not self.realized_gains:
            return {
                'total_realized_gain': 0,
                'total_tax_owed': 0,
                'net_after_tax': 0,
                'short_term_gains': 0,
                'long_term_gains': 0,
                'num_trades': 0
            }

        df = pd.DataFrame(self.realized_gains)

        short_term = df[df['tax_type'] == 'short_term']
        long_term = df[df['tax_type'] == 'long_term']

        return {
            'total_realized_gain': df['gain'].sum(),
            'total_tax_owed': df['tax_owed'].sum(),
            'net_after_tax': df['net_gain'].sum(),
            'short_term_gains': short_term['gain'].sum(),
            'long_term_gains': long_term['gain'].sum(),
            'short_term_tax': short_term['tax_owed'].sum(),
            'long_term_tax': long_term['tax_owed'].sum(),
            'num_trades': len(df),
            'avg_hold_days': df['hold_days'].mean()
        }

    def print_tax_report(self):
        summary = self.get_tax_summary()

        print("\n" + "="*60)
        print("TAX REPORT")
        print("="*60)
        print(f"Total Realized Gain:    ${summary['total_realized_gain']:,.2f}")
        print(f"Total Tax Owed:         ${summary['total_tax_owed']:,.2f}")
        print(f"Net After Tax:          ${summary['net_after_tax']:,.2f}")
        print(f"\nShort-Term (<1 year):")
        print(f"  Gains:                ${summary['short_term_gains']:,.2f}")
        print(f"  Tax (37%):            ${summary['short_term_tax']:,.2f}")
        print(f"\nLong-Term (>1 year):")
        print(f"  Gains:                ${summary['long_term_gains']:,.2f}")
        print(f"  Tax (20%):            ${summary['long_term_tax']:,.2f}")
        print(f"\nAvg Hold Time:          {summary['avg_hold_days']:.0f} days")
        print(f"Number of Trades:       {summary['num_trades']}")
        print("="*60 + "\n")

if __name__ == "__main__":
    cost_model = RealisticCostModel()

    print("Transaction Cost Examples:\n")

    tickers = ['AAPL', 'PLTR', 'RIVN', 'AMC']
    for ticker in tickers:
        costs = cost_model.get_typical_cost_bps(ticker)
        print(f"{ticker:6s}: {costs['total_bps']:.1f} bps total ({costs['spread_bps']:.1f} spread + {costs['slippage_bps']:.1f} slippage)")

    print("\n" + "="*60)
    print("Tax Tracking Example:\n")

    tax_tracker = TaxTracker()

    tax_tracker.open_position('AAPL', 10, 180, pd.to_datetime('2025-01-01'))
    tax_tracker.close_position('AAPL', 10, 190, pd.to_datetime('2025-06-01'))

    tax_tracker.open_position('TSLA', 5, 250, pd.to_datetime('2024-01-01'))
    tax_tracker.close_position('TSLA', 5, 280, pd.to_datetime('2025-06-01'))

    tax_tracker.print_tax_report()

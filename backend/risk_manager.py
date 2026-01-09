import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

class RiskManager:
    def __init__(self, max_portfolio_heat=0.02, max_correlated_positions=3,
                 max_sector_exposure=0.30, volatility_target=0.15):
        self.max_portfolio_heat = max_portfolio_heat
        self.max_correlated_positions = max_correlated_positions
        self.max_sector_exposure = max_sector_exposure
        self.volatility_target = volatility_target

        self.sector_map = {
            'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech',
            'TSLA': 'auto', 'RIVN': 'auto', 'LCID': 'auto', 'NIO': 'auto',
            'COIN': 'fintech', 'SOFI': 'fintech',
            'PLTR': 'tech', 'SNAP': 'tech',
            'AMC': 'entertainment',
            'SPY': 'index'
        }

        self.position_history = []
        self.daily_pnl_history = []

    def calculate_portfolio_heat(self, positions, account_equity):
        total_risk = 0
        for ticker, pos in positions.items():
            position_value = abs(pos['qty']) * pos['current_price']
            stop_loss_distance = 0.02
            risk_per_position = position_value * stop_loss_distance
            total_risk += risk_per_position

        portfolio_heat = total_risk / account_equity
        return portfolio_heat

    def calculate_position_size_volatility_adjusted(self, ticker, current_price, account_equity,
                                                    historical_volatility, base_size_usd=1000):
        if historical_volatility == 0:
            return int(base_size_usd / current_price)

        volatility_scalar = self.volatility_target / historical_volatility
        volatility_scalar = np.clip(volatility_scalar, 0.5, 2.0)

        adjusted_size_usd = base_size_usd * volatility_scalar

        max_position_value = account_equity * 0.05
        adjusted_size_usd = min(adjusted_size_usd, max_position_value)

        qty = int(adjusted_size_usd / current_price)
        return max(qty, 1)

    def calculate_kelly_criterion(self, win_rate, avg_win, avg_loss):
        if avg_loss == 0:
            return 0

        win_loss_ratio = abs(avg_win / avg_loss)
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        kelly_fraction = max(0, min(kelly_fraction, 0.25))

        return kelly_fraction

    def check_correlation_limit(self, ticker, positions, correlation_threshold=0.7):
        if len(positions) == 0:
            return True

        sector = self.sector_map.get(ticker, 'other')

        correlated_count = 0
        for existing_ticker in positions.keys():
            existing_sector = self.sector_map.get(existing_ticker, 'other')
            if sector == existing_sector and sector != 'other':
                correlated_count += 1

        return correlated_count < self.max_correlated_positions

    def check_sector_exposure(self, ticker, positions, account_equity):
        sector = self.sector_map.get(ticker, 'other')

        sector_exposure = defaultdict(float)
        for existing_ticker, pos in positions.items():
            existing_sector = self.sector_map.get(existing_ticker, 'other')
            position_value = abs(pos['qty']) * pos['current_price']
            sector_exposure[existing_sector] += position_value

        current_sector_exposure = sector_exposure[sector] / account_equity

        return current_sector_exposure < self.max_sector_exposure

    def calculate_var(self, returns, confidence=0.95):
        if len(returns) == 0:
            return 0
        return np.percentile(returns, (1 - confidence) * 100)

    def calculate_cvar(self, returns, confidence=0.95):
        if len(returns) == 0:
            return 0
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    def check_daily_loss_limit(self, daily_pnl, max_daily_loss):
        return abs(daily_pnl) < max_daily_loss

    def check_consecutive_losses(self, recent_trades, max_consecutive_losses=5):
        if len(recent_trades) < max_consecutive_losses:
            return True

        last_n_trades = recent_trades[-max_consecutive_losses:]
        return not all(trade['pnl'] < 0 for trade in last_n_trades)

    def dynamic_position_sizing(self, base_size, current_equity, starting_equity, performance_window):
        equity_ratio = current_equity / starting_equity

        if equity_ratio > 1.1:
            size_multiplier = 1.2
        elif equity_ratio > 1.05:
            size_multiplier = 1.1
        elif equity_ratio < 0.95:
            size_multiplier = 0.8
        elif equity_ratio < 0.90:
            size_multiplier = 0.6
        else:
            size_multiplier = 1.0

        if len(performance_window) > 0:
            recent_sharpe = self.calculate_rolling_sharpe(performance_window)

            if recent_sharpe > 1.5:
                size_multiplier *= 1.1
            elif recent_sharpe < 0.5:
                size_multiplier *= 0.8

        return int(base_size * size_multiplier)

    def calculate_rolling_sharpe(self, returns, periods_per_year=252*390):
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        return (np.mean(returns) / np.std(returns)) * np.sqrt(periods_per_year)

    def should_allow_trade(self, ticker, positions, account_equity, daily_pnl,
                          max_daily_loss, recent_trades=None):
        if not self.check_daily_loss_limit(daily_pnl, max_daily_loss):
            return False, "daily loss limit exceeded"

        if ticker in positions:
            return False, "already have position in this ticker"

        portfolio_heat = self.calculate_portfolio_heat(positions, account_equity)
        if portfolio_heat > self.max_portfolio_heat:
            return False, f"portfolio heat too high: {portfolio_heat:.2%}"

        if not self.check_correlation_limit(ticker, positions):
            return False, "too many correlated positions"

        if not self.check_sector_exposure(ticker, positions, account_equity):
            return False, "sector exposure limit exceeded"

        if recent_trades and not self.check_consecutive_losses(recent_trades):
            return False, "too many consecutive losses"

        return True, "approved"

    def calculate_stop_loss_price(self, entry_price, side, atr=None, atr_multiplier=2.0, fixed_pct=0.02):
        if atr is not None:
            stop_distance = atr * atr_multiplier
        else:
            stop_distance = entry_price * fixed_pct

        if side == 'BUY':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_take_profit_price(self, entry_price, side, risk_reward_ratio=1.5, fixed_pct=0.03):
        profit_distance = entry_price * fixed_pct

        if side == 'BUY':
            return entry_price + profit_distance
        else:
            return entry_price - profit_distance

    def get_risk_report(self, positions, account_equity, daily_pnl):
        portfolio_heat = self.calculate_portfolio_heat(positions, account_equity)

        sector_exposure = defaultdict(float)
        for ticker, pos in positions.items():
            sector = self.sector_map.get(ticker, 'other')
            position_value = abs(pos['qty']) * pos['current_price']
            sector_exposure[sector] += position_value

        report = {
            'portfolio_heat': portfolio_heat,
            'portfolio_heat_pct': portfolio_heat * 100,
            'num_positions': len(positions),
            'daily_pnl': daily_pnl,
            'sector_exposure': dict(sector_exposure),
            'largest_sector': max(sector_exposure, key=sector_exposure.get) if sector_exposure else None,
            'largest_sector_pct': max(sector_exposure.values()) / account_equity * 100 if sector_exposure else 0
        }

        return report

    def print_risk_report(self, positions, account_equity, daily_pnl):
        report = self.get_risk_report(positions, account_equity, daily_pnl)

        print("\n" + "="*60)
        print("RISK REPORT")
        print("="*60)
        print(f"Portfolio Heat:     {report['portfolio_heat_pct']:.2f}% (max: {self.max_portfolio_heat*100:.2f}%)")
        print(f"Num Positions:      {report['num_positions']}")
        print(f"Daily P&L:          ${report['daily_pnl']:+,.2f}")

        if report['sector_exposure']:
            print(f"\nSector Exposure:")
            for sector, value in report['sector_exposure'].items():
                pct = value / account_equity * 100
                print(f"  {sector:15s} ${value:10,.2f} ({pct:.1f}%)")

        print("="*60 + "\n")

if __name__ == "__main__":
    rm = RiskManager()

    positions = {
        'AAPL': {'qty': 10, 'current_price': 180, 'entry_price': 175},
        'MSFT': {'qty': 5, 'current_price': 380, 'entry_price': 375}
    }

    account_equity = 100000
    daily_pnl = 250

    rm.print_risk_report(positions, account_equity, daily_pnl)

    can_trade, reason = rm.should_allow_trade('GOOGL', positions, account_equity, daily_pnl, max_daily_loss=500)
    print(f"can trade GOOGL: {can_trade} - {reason}")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trade_logger import TradeLogger
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceTracker:
    def __init__(self, logger=None):
        self.logger = logger or TradeLogger()

    def calculate_sharpe_ratio(self, returns, periods_per_year=252*390):
        if len(returns) == 0 or returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)

    def calculate_sortino_ratio(self, returns, periods_per_year=252*390):
        if len(returns) == 0:
            return 0
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        return (returns.mean() / downside_returns.std()) * np.sqrt(periods_per_year)

    def calculate_max_drawdown(self, equity_curve):
        if len(equity_curve) == 0:
            return 0
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()

    def calculate_calmar_ratio(self, total_return, max_drawdown):
        if max_drawdown == 0:
            return 0
        return abs(total_return / max_drawdown)

    def get_equity_curve(self, start_date=None, end_date=None, starting_capital=100000):
        positions = self.logger.get_all_positions(start_date=start_date, end_date=end_date)

        if positions.empty:
            return pd.Series([starting_capital])

        positions = positions.sort_values('exit_timestamp')
        equity = starting_capital + positions['pnl'].cumsum()
        equity = pd.concat([pd.Series([starting_capital]), equity])
        return equity

    def get_returns(self, start_date=None, end_date=None):
        positions = self.logger.get_all_positions(start_date=start_date, end_date=end_date)

        if positions.empty:
            return pd.Series([])

        positions = positions.sort_values('exit_timestamp')
        return positions['pnl_pct']

    def generate_report(self, start_date=None, end_date=None, starting_capital=100000):
        summary = self.logger.get_performance_summary(start_date, end_date)
        equity_curve = self.get_equity_curve(start_date, end_date, starting_capital)
        returns = self.get_returns(start_date, end_date)

        if summary:
            total_return = (equity_curve.iloc[-1] - starting_capital) / starting_capital
            max_dd = self.calculate_max_drawdown(equity_curve.values)
            sharpe = self.calculate_sharpe_ratio(returns)
            sortino = self.calculate_sortino_ratio(returns)
            calmar = self.calculate_calmar_ratio(total_return, max_dd)

            report = {
                **summary,
                'starting_capital': starting_capital,
                'ending_capital': equity_curve.iloc[-1],
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_dd,
                'max_drawdown_pct': max_dd * 100,
                'calmar_ratio': calmar
            }
        else:
            report = {
                'starting_capital': starting_capital,
                'ending_capital': starting_capital,
                'total_return': 0,
                'total_trades': 0
            }

        return report

    def print_report(self, start_date=None, end_date=None, starting_capital=100000):
        report = self.generate_report(start_date, end_date, starting_capital)

        print("\n" + "="*70)
        print("TRADING PERFORMANCE REPORT")
        print("="*70)

        if start_date:
            print(f"Period: {start_date} to {end_date or 'now'}")

        print(f"\nCAPITAL:")
        print(f"  Starting: ${report['starting_capital']:,.2f}")
        print(f"  Ending:   ${report['ending_capital']:,.2f}")
        print(f"  P&L:      ${report['ending_capital'] - report['starting_capital']:+,.2f} ({report.get('total_return_pct', 0):+.2f}%)")

        if report.get('total_trades', 0) > 0:
            print(f"\nTRADE STATISTICS:")
            print(f"  Total Trades:     {report['total_trades']}")
            print(f"  Wins:             {report['num_wins']} ({report['win_rate']*100:.1f}%)")
            print(f"  Losses:           {report['num_losses']}")
            print(f"  Avg Win:          ${report['avg_win']:,.2f}")
            print(f"  Avg Loss:         ${report['avg_loss']:,.2f}")
            print(f"  Largest Win:      ${report['largest_win']:,.2f}")
            print(f"  Largest Loss:     ${report['largest_loss']:,.2f}")
            print(f"  Profit Factor:    {report['profit_factor']:.2f}")
            print(f"  Avg Hold Time:    {report['avg_hold_time_minutes']:.1f} minutes")

            print(f"\nLONG vs SHORT:")
            print(f"  Long Trades:      {report['total_long_trades']} (P&L: ${report['long_pnl']:+,.2f})")
            print(f"  Short Trades:     {report['total_short_trades']} (P&L: ${report['short_pnl']:+,.2f})")

            print(f"\nRISK METRICS:")
            print(f"  Sharpe Ratio:     {report['sharpe_ratio']:.2f}")
            print(f"  Sortino Ratio:    {report['sortino_ratio']:.2f}")
            print(f"  Max Drawdown:     {report['max_drawdown_pct']:.2f}%")
            print(f"  Calmar Ratio:     {report['calmar_ratio']:.2f}")

        print("="*70 + "\n")

    def plot_equity_curve(self, start_date=None, end_date=None, starting_capital=100000, save_path=None):
        equity = self.get_equity_curve(start_date, end_date, starting_capital)

        plt.figure(figsize=(14, 7))
        plt.plot(equity.values, linewidth=2)
        plt.title('Equity Curve', fontsize=16, fontweight='bold')
        plt.xlabel('Trade Number')
        plt.ylabel('Equity ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_drawdown(self, start_date=None, end_date=None, starting_capital=100000, save_path=None):
        equity = self.get_equity_curve(start_date, end_date, starting_capital)
        running_max = np.maximum.accumulate(equity.values)
        drawdown = (equity.values - running_max) / running_max * 100

        plt.figure(figsize=(14, 7))
        plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        plt.plot(drawdown, color='red', linewidth=2)
        plt.title('Drawdown (%)', fontsize=16, fontweight='bold')
        plt.xlabel('Trade Number')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_win_loss_distribution(self, start_date=None, end_date=None, save_path=None):
        positions = self.logger.get_all_positions(start_date, end_date)

        if positions.empty:
            print("no data to plot")
            return

        plt.figure(figsize=(14, 7))
        wins = positions[positions['pnl'] > 0]['pnl']
        losses = positions[positions['pnl'] < 0]['pnl']

        plt.hist([wins, losses], bins=30, label=['Wins', 'Losses'], color=['green', 'red'], alpha=0.7)
        plt.title('Win/Loss Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('P&L ($)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def analyze_by_ticker(self, start_date=None, end_date=None):
        positions = self.logger.get_all_positions(start_date, end_date)

        if positions.empty:
            return pd.DataFrame()

        ticker_stats = positions.groupby('ticker').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_pct': 'mean',
            'hold_time_minutes': 'mean'
        }).round(2)

        ticker_stats.columns = ['num_trades', 'total_pnl', 'avg_pnl', 'avg_pnl_pct', 'avg_hold_time']

        for ticker in ticker_stats.index:
            ticker_positions = positions[positions['ticker'] == ticker]
            wins = ticker_positions[ticker_positions['pnl'] > 0]
            ticker_stats.loc[ticker, 'win_rate'] = len(wins) / len(ticker_positions) if len(ticker_positions) > 0 else 0

        return ticker_stats.sort_values('total_pnl', ascending=False)

    def analyze_by_time_of_day(self, start_date=None, end_date=None):
        positions = self.logger.get_all_positions(start_date, end_date)

        if positions.empty:
            return pd.DataFrame()

        positions['hour'] = pd.to_datetime(positions['entry_timestamp']).dt.hour

        time_stats = positions.groupby('hour').agg({
            'pnl': ['count', 'sum', 'mean']
        }).round(2)

        time_stats.columns = ['num_trades', 'total_pnl', 'avg_pnl']

        return time_stats

if __name__ == "__main__":
    tracker = PerformanceTracker()
    tracker.print_report()

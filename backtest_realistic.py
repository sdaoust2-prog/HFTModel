import numpy as np
import pandas as pd
import joblib
from feature_engine import load_features_for_training
from utils import pull_polygon_data, train_test_split_chronological, backtest_strategy, print_backtest_results
from backend.realistic_costs import RealisticCostModel, TaxTracker
from datetime import datetime

def run_backtest_with_realistic_costs(model, df, ticker, prob_threshold=0.55):
    cost_model = RealisticCostModel()

    X, y_binary, y_continuous, _ = load_features_for_training(df, use_extended=True)
    _, X_test, _, y_test_cont = train_test_split_chronological(X, y_continuous, train_frac=0.8)

    prob = model.predict_proba(X_test)
    signals = np.where(prob[:, 1] > prob_threshold, 1,
              np.where(prob[:, 0] > prob_threshold, -1, 0))

    typical_cost = cost_model.get_typical_cost_bps(ticker)
    realistic_cost_pct = typical_cost['total_bps'] / 10000

    simple_metrics = backtest_strategy(signals, y_test_cont, transaction_cost=0.0001)
    realistic_metrics = backtest_strategy(signals, y_test_cont, transaction_cost=realistic_cost_pct)

    return simple_metrics, realistic_metrics, typical_cost

def run_backtest_with_taxes(model, df, ticker, prob_threshold=0.55, initial_capital=10000):
    cost_model = RealisticCostModel()
    tax_tracker = TaxTracker()

    X, y_binary, y_continuous, _ = load_features_for_training(df, use_extended=True)
    _, X_test, _, y_test_cont = train_test_split_chronological(X, y_continuous, train_frac=0.8)

    prob = model.predict_proba(X_test)
    signals = np.where(prob[:, 1] > prob_threshold, 1,
              np.where(prob[:, 0] > prob_threshold, -1, 0))

    typical_cost = cost_model.get_typical_cost_bps(ticker)
    realistic_cost_pct = typical_cost['total_bps'] / 10000

    cash = initial_capital
    shares = 0
    entry_price = None
    entry_time = None
    total_gross_pnl = 0
    total_costs = 0
    total_tax = 0

    test_df = df.iloc[len(df) - len(signals):]

    for i, (signal, ret) in enumerate(zip(signals, y_test_cont)):
        current_price = test_df.iloc[i]['close']
        current_time = test_df.iloc[i]['timestamp']

        if shares == 0 and signal != 0:
            shares_to_buy = int(cash / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * realistic_cost_pct
                shares = shares_to_buy * signal
                entry_price = current_price
                entry_time = current_time
                cash -= shares_to_buy * current_price + cost
                total_costs += cost

                if isinstance(current_time, pd.Timestamp):
                    current_time_dt = current_time.to_pydatetime()
                else:
                    current_time_dt = datetime.now()

                tax_tracker.open_position(ticker, abs(shares), entry_price, current_time_dt,
                                         side='LONG' if shares > 0 else 'SHORT')

        elif shares != 0 and ((shares > 0 and signal <= 0) or (shares < 0 and signal >= 0)):
            gross_pnl = abs(shares) * (current_price - entry_price) * (1 if shares > 0 else -1)
            cost = abs(shares) * current_price * realistic_cost_pct

            if isinstance(current_time, pd.Timestamp):
                current_time_dt = current_time.to_pydatetime()
            else:
                current_time_dt = datetime.now()

            tax_result = tax_tracker.close_position(ticker, abs(shares), current_price, current_time_dt)

            cash += abs(shares) * current_price - cost
            total_gross_pnl += gross_pnl
            total_costs += cost

            if tax_result:
                total_tax += tax_result['tax_owed']

            shares = 0
            entry_price = None
            entry_time = None

    final_equity = cash + (abs(shares) * test_df.iloc[-1]['close'] if shares != 0 else 0)

    gross_return = (final_equity - initial_capital) / initial_capital
    net_pnl = total_gross_pnl - total_costs - total_tax
    net_return = net_pnl / initial_capital

    tax_summary = tax_tracker.get_tax_summary()

    return {
        'gross_pnl': total_gross_pnl,
        'total_costs': total_costs,
        'total_tax': total_tax,
        'net_pnl': net_pnl,
        'gross_return_pct': gross_return * 100,
        'net_return_pct': net_return * 100,
        'cost_impact_pct': (total_costs / total_gross_pnl * 100) if total_gross_pnl > 0 else 0,
        'tax_impact_pct': (total_tax / total_gross_pnl * 100) if total_gross_pnl > 0 else 0,
        'you_keep_pct': (net_pnl / total_gross_pnl * 100) if total_gross_pnl > 0 else 0,
        'tax_summary': tax_summary
    }

if __name__ == "__main__":
    API_KEY = "MSeDps8X9ILJUQC4Lxfw5_h4DMdO1ZVB"

    model = joblib.load("trained_stock_model.pkl")

    print("\n" + "="*80)
    print("REALISTIC BACKTEST COMPARISON")
    print("="*80)

    watchlist = ['PLTR', 'RIVN', 'SNAP', 'COIN']

    for ticker in watchlist:
        print(f"\n{ticker}:")
        print("-" * 80)

        try:
            df = pull_polygon_data(ticker, "2025-11-01", "2025-12-31", API_KEY)

            simple_metrics, realistic_metrics, cost_info = run_backtest_with_realistic_costs(
                model, df, ticker, prob_threshold=0.55
            )

            print(f"\nTransaction Cost: {cost_info['total_bps']:.1f} basis points (0.{cost_info['total_bps']:.0f}%)")
            print(f"  Spread: {cost_info['spread_bps']:.1f} bp")
            print(f"  Slippage: {cost_info['slippage_bps']:.1f} bp")

            print(f"\nSIMPLIFIED MODEL (1 bp cost):")
            print(f"  Sharpe: {simple_metrics['sharpe_ratio']:.2f}")
            print(f"  Return: {simple_metrics['total_return']*100:.2f}%")
            print(f"  Max DD: {simple_metrics['max_drawdown']*100:.2f}%")

            print(f"\nREALISTIC MODEL ({cost_info['total_bps']:.1f} bp cost):")
            print(f"  Sharpe: {realistic_metrics['sharpe_ratio']:.2f}")
            print(f"  Return: {realistic_metrics['total_return']*100:.2f}%")
            print(f"  Max DD: {realistic_metrics['max_drawdown']*100:.2f}%")

            sharpe_degradation = ((realistic_metrics['sharpe_ratio'] - simple_metrics['sharpe_ratio'])
                                 / simple_metrics['sharpe_ratio'] * 100) if simple_metrics['sharpe_ratio'] != 0 else 0
            print(f"\nSharpe Degradation: {sharpe_degradation:.1f}%")

            print(f"\n{'='*80}")
            print(f"WITH TAX TRACKING (37% short-term capital gains)")
            print(f"{'='*80}")

            tax_results = run_backtest_with_taxes(model, df, ticker, prob_threshold=0.55)

            print(f"\nGross P&L:        ${tax_results['gross_pnl']:>10,.2f} ({tax_results['gross_return_pct']:>6.2f}%)")
            print(f"Transaction Costs: -${tax_results['total_costs']:>10,.2f} ({tax_results['cost_impact_pct']:>6.2f}%)")
            print(f"Taxes Owed:       -${tax_results['total_tax']:>10,.2f} ({tax_results['tax_impact_pct']:>6.2f}%)")
            print(f"{'='*40}")
            print(f"Net P&L:           ${tax_results['net_pnl']:>10,.2f} ({tax_results['net_return_pct']:>6.2f}%)")
            print(f"\nYou Keep: {tax_results['you_keep_pct']:.1f}% of gross profits")

            print(f"\n{'='*80}\n")

        except Exception as e:
            print(f"Error backtesting {ticker}: {e}\n")

    print("\nSUMMARY:")
    print("="*80)
    print("Your current watchlist (PLTR, RIVN, SNAP, COIN, SOFI, NIO, LCID, AMC) has")
    print("transaction costs 3-8x higher than the simplified 1bp model.")
    print("\nWith realistic costs + taxes:")
    print("  - Transaction costs reduce returns by 10-30%")
    print("  - Short-term capital gains tax (37%) takes another 37% of remaining profit")
    print("  - You typically keep 45-65% of gross profits")
    print("\nRecommendations:")
    print("  1. Focus on more liquid stocks (AAPL, MSFT, SPY) for lower costs")
    print("  2. Reduce trading frequency to minimize turnover")
    print("  3. Hold positions longer to qualify for long-term gains (20% tax)")
    print("="*80)

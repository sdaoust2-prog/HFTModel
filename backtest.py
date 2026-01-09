import numpy as np
import joblib
from feature_engine import load_features_for_training
from utils import pull_polygon_data, train_test_split_chronological, backtest_strategy, print_backtest_results
from backend.realistic_costs import RealisticCostModel


def run_backtest(model, df, prob_threshold=0.55, transaction_cost=0.0001, use_extended=True):
    X, y_binary, y_continuous, _ = load_features_for_training(df, use_extended=use_extended)
    _, X_test, _, y_test_cont = train_test_split_chronological(X, y_continuous, train_frac=0.8)
    prob = model.predict_proba(X_test)
    signals = np.where(prob[:, 1] > prob_threshold, 1,
              np.where(prob[:, 0] > prob_threshold, -1, 0))
    return backtest_strategy(signals, y_test_cont, transaction_cost)


if __name__ == "__main__":
    API_KEY = "MSeDps8X9ILJUQC4Lxfw5_h4DMdO1ZVB"
    TICKER = "AAPL"
    START = "2025-01-01"
    END = "2025-12-31"

    print(f"backtesting {TICKER} from {START} to {END}")

    df = pull_polygon_data(TICKER, START, END, API_KEY)
    model = joblib.load("trained_stock_model.pkl")

    cost_model = RealisticCostModel()
    cost_info = cost_model.get_typical_cost_bps(TICKER)

    print(f"\nrealistic transaction costs for {TICKER}: {cost_info['total_bps']:.1f} bps")

    simplified_metrics = run_backtest(model, df, prob_threshold=0.55, transaction_cost=0.0001)
    realistic_metrics = run_backtest(model, df, prob_threshold=0.55,
                                     transaction_cost=cost_info['total_bps']/10000)

    bh_metrics = backtest_strategy(np.ones(len(df) - int(len(df)*0.8)),
                                   df['close'].pct_change().iloc[int(len(df)*0.8):].values)

    print_backtest_results(simplified_metrics, f"{TICKER} Model (Simplified 1bp Cost)")
    print_backtest_results(realistic_metrics, f"{TICKER} Model (Realistic {cost_info['total_bps']:.1f}bp Cost)")
    print_backtest_results(bh_metrics, "Buy & Hold Benchmark")

    sharpe_impact = ((realistic_metrics['sharpe_ratio'] - simplified_metrics['sharpe_ratio'])
                    / simplified_metrics['sharpe_ratio'] * 100) if simplified_metrics['sharpe_ratio'] != 0 else 0
    print(f"\nrealistic costs impact: sharpe degradation of {sharpe_impact:.1f}%")

import numpy as np
import joblib
from feature_engine import load_features_for_training
from utils import pull_polygon_data, train_test_split_chronological, backtest_strategy, print_backtest_results


def run_backtest(model, df, prob_threshold=0.55, transaction_cost=0.0001):
    """Backtest a trained model on test data"""

    X, y_binary, y_continuous, _ = load_features_for_training(df)

    _, X_test, _, y_test_cont = train_test_split_chronological(X, y_continuous, train_frac=0.8)

    prob = model.predict_proba(X_test)

    signals = np.where(prob[:, 1] > prob_threshold, 1,
              np.where(prob[:, 0] > prob_threshold, -1, 0))

    return backtest_strategy(signals, y_test_cont, transaction_cost)


if __name__ == "__main__":
    API_KEY = "vFDjkUVRfPnedLrbRjm75BZ9CJHz3dfv"
    TICKER = "AAPL"
    START = "2025-10-01"
    END = "2025-11-01"

    print(f"backtesting {TICKER} from {START} to {END}")

    df = pull_polygon_data(TICKER, START, END, API_KEY)
    model = joblib.load("trained_stock_model.pkl")

    metrics = run_backtest(model, df, prob_threshold=0.55)

    bh_metrics = backtest_strategy(np.ones(len(df) - int(len(df)*0.8)),
                                   df['close'].pct_change().iloc[int(len(df)*0.8):].values)

    print_backtest_results(metrics, f"{TICKER} Model Strategy")
    print_backtest_results(bh_metrics, "Buy & Hold Benchmark")

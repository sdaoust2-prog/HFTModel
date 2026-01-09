from sklearn.ensemble import RandomForestClassifier
from feature_engine import load_features_for_training
from utils import pull_polygon_data, walk_forward_validation
import pandas as pd

if __name__ == "__main__":
    API_KEY = "MSeDps8X9ILJUQC4Lxfw5_h4DMdO1ZVB"
    TICKER = "AAPL"
    START = "2025-01-01"
    END = "2025-12-31"

    print(f"walk-forward validation on {TICKER}")
    print(f"splitting data into 5 windows, testing each\n")

    df = pull_polygon_data(TICKER, START, END, API_KEY)
    X, y_binary, y_continuous, _ = load_features_for_training(df)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    results = walk_forward_validation(X, y_continuous, model, n_splits=5)

    print("="*50)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("="*50)
    print(f"Mean Correlation:  {results['mean_correlation']:>8.4f} ± {results['std_correlation']:.4f}")
    print(f"Mean Sharpe:       {results['mean_sharpe']:>8.2f} ± {results['std_sharpe']:.2f}")
    print(f"Mean Return:       {results['mean_return']*100:>8.2f}%")
    print(f"Worst Drawdown:    {results['worst_drawdown']*100:>8.2f}%")
    print(f"Consistency:       {results['consistency']*100:>8.1f}% (% positive Sharpe)")
    print("="*50)

    print("\nPER-SPLIT BREAKDOWN:")
    print(results['all_splits'].to_string(index=False))

    print("\nInterpretation:")
    if results['consistency'] > 0.6:
        print("consistent across time periods")
    else:
        print("inconsistent - likely overfit")
    if results['std_sharpe'] < results['mean_sharpe']:
        print("stable performance")
    else:
        print("high variance")

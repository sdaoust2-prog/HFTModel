import pandas as pd
import numpy as np
import requests
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def pull_polygon_data(ticker, start, end, api_key):
    """Fetch minute bars from Polygon.io"""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}?apiKey={api_key}"
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        raise ValueError(f"API error: {response.status_code}")

    data = response.json()

    if 'results' not in data or len(data['results']) < 2:
        raise ValueError("insufficient data returned")

    df = pd.DataFrame(data['results'])
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})

    return df[['timestamp','open','high','low','close','volume']]


def train_test_split_chronological(X, y, train_frac=0.8):
    """Split time series data chronologically"""
    split_idx = int(len(X) * train_frac)
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def calculate_ic(features_df, target_series, method='pearson'):
    """
    Information Coefficient: correlation between features and continuous target

    Note: IC in quant finance = correlation between feature and forward returns
    NOT the binary formula IC = 2*#correct/#total (that's for binary bets)

    Since both feature and target are continuous, we use:
    - Pearson correlation (linear relationships)
    - Spearman rank correlation (monotonic relationships, robust to outliers)

    Args:
        features_df: dataframe of feature values
        target_series: continuous target (forward returns)
        method: 'pearson' or 'spearman'

    Returns:
        DataFrame with IC scores ranked by absolute value
    """
    results = []

    for col in features_df.columns:
        if method == 'pearson':
            ic, pval = pearsonr(features_df[col], target_series)
        else:
            ic, pval = spearmanr(features_df[col], target_series)

        results.append({
            'feature': col,
            'ic': ic,
            'abs_ic': abs(ic),
            'pvalue': pval
        })

    return pd.DataFrame(results).sort_values('abs_ic', ascending=False)


def backtest_strategy(signals, returns, transaction_cost=0.0001, scale_by_confidence=False):
    """
    Run backtest given signals and actual returns

    Args:
        signals: array of positions (1=long, -1=short, 0=flat)
                 OR array of continuous signals (e.g., 0.3 = 30% long)
        returns: array of actual forward returns
        transaction_cost: cost per trade (e.g. 0.0001 = 1bp)
        scale_by_confidence: if True, treats signals as position sizes (0-1 scale)

    Returns:
        dict with performance metrics
    """
    signals = np.array(signals)
    returns = np.array(returns)

    position_changes = np.abs(np.diff(signals, prepend=0))
    costs = position_changes * transaction_cost

    strategy_returns = signals * returns - costs
    cumulative = (1 + strategy_returns).cumprod()

    total_return = cumulative[-1] - 1
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 390) if strategy_returns.std() > 0 else 0

    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    trades = strategy_returns[strategy_returns != 0]
    winning_trades = trades[trades > 0]
    losing_trades = trades[trades < 0]

    win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0

    profit_factor = winning_trades.sum() / abs(losing_trades.sum()) if len(losing_trades) > 0 else np.inf

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'num_trades': len(trades),
        'turnover': len(trades) / len(signals)
    }


def generate_scaled_signals(probabilities, threshold=0.55, scale_by_magnitude=True):
    """
    Convert model probabilities to position sizes

    Your cousin's advice: scale position size by signal strength

    Args:
        probabilities: array of P(UP) from model (0-1)
        threshold: minimum confidence to trade (default 0.55)
        scale_by_magnitude: if True, position size = distance from 0.5

    Returns:
        signals: array of position sizes (-1 to +1)

    Examples:
        P(UP)=0.7, threshold=0.55, scale=True  → signal = +0.4 (40% long)
        P(UP)=0.6, threshold=0.55, scale=True  → signal = +0.2 (20% long)
        P(UP)=0.52, threshold=0.55, scale=True → signal = 0 (below threshold)
        P(UP)=0.3, threshold=0.55, scale=True  → signal = -0.4 (40% short)
    """
    probabilities = np.array(probabilities)

    if scale_by_magnitude:
        # Distance from 0.5 = signal strength
        signal_strength = np.abs(probabilities - 0.5)

        # Direction: +1 if P(UP) > 0.5, -1 otherwise
        direction = np.where(probabilities > 0.5, 1, -1)

        # Only trade if above threshold
        above_threshold = signal_strength >= (threshold - 0.5)

        # Scale position: 0 at threshold, 1.0 at extreme (0 or 1)
        # Normalize so threshold→0 and 0.5 distance→1.0
        max_distance = 0.5  # maximum possible distance from 0.5
        threshold_distance = threshold - 0.5

        scaled_strength = (signal_strength - threshold_distance) / (max_distance - threshold_distance)
        scaled_strength = np.clip(scaled_strength, 0, 1)  # ensure 0-1 range

        signals = direction * scaled_strength * above_threshold

    else:
        # Binary signals (old behavior)
        signals = np.where(probabilities > threshold, 1,
                  np.where(probabilities < (1 - threshold), -1, 0))

    return signals


def print_backtest_results(metrics, title="Backtest Results"):
    """Pretty print backtest metrics"""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(f"Total Return:     {metrics['total_return']*100:>8.2f}%")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:>8.2f}")
    print(f"Max Drawdown:     {metrics['max_drawdown']*100:>8.2f}%")
    print(f"Win Rate:         {metrics['win_rate']*100:>8.1f}%")
    print(f"Avg Win:          {metrics['avg_win']*100:>8.4f}%")
    print(f"Avg Loss:         {metrics['avg_loss']*100:>8.4f}%")
    print(f"Profit Factor:    {metrics['profit_factor']:>8.2f}")
    print(f"Num Trades:       {metrics['num_trades']:>8}")
    print(f"Turnover:         {metrics['turnover']*100:>8.2f}%")
    print(f"{'='*50}\n")


def evaluate_classifier(y_true, y_pred, y_prob=None):
    """Complete evaluation for binary classifier"""
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    results = {
        'accuracy': acc,
        'confusion_matrix': cm
    }

    if y_prob is not None:
        from sklearn.metrics import roc_auc_score
        results['auc'] = roc_auc_score(y_true, y_prob)

    print(f"\nAccuracy: {acc:.4f}")
    if y_prob is not None:
        print(f"AUC: {results['auc']:.4f}")

    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              DOWN    UP")
    print(f"Actual DOWN   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       UP     {cm[1,0]:4d}  {cm[1,1]:4d}")

    print("\n" + classification_report(y_true, y_pred, target_names=['DOWN', 'UP']))

    return results


def calculate_realized_correlation(predictions, actuals):
    """
    Realized out-of-sample correlation - key overfitting check

    Measures correlation between predicted returns and actual returns on test set.
    Low correlation despite good accuracy = you overfit.

    Args:
        predictions: model predictions (probabilities or returns)
        actuals: actual forward returns

    Returns:
        correlation coefficient
    """
    from scipy.stats import pearsonr
    corr, pval = pearsonr(predictions, actuals)
    return {
        'correlation': corr,
        'pvalue': pval,
        'overfitting_check': 'PASS' if corr > 0.02 else 'FAIL - likely overfit'
    }


def compare_models(results_dict):
    """
    Compare multiple models

    Args:
        results_dict: {'model_name': {'test_acc': ..., 'test_auc': ..., ...}}

    Returns:
        DataFrame with comparison
    """
    df = pd.DataFrame(results_dict).T
    return df.sort_values('test_acc', ascending=False)

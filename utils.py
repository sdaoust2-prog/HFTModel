import pandas as pd
import numpy as np
import requests
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def pull_polygon_data(ticker, start, end, api_key):
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
    split_idx = int(len(X) * train_frac)
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

def calculate_ic(features_df, target_series, method='pearson'):
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
    information_ratio = sharpe
    calmar = abs(total_return / max_drawdown) if max_drawdown != 0 else 0
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'information_ratio': information_ratio,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'num_trades': len(trades),
        'turnover': len(trades) / len(signals)
    }


def generate_scaled_signals(probabilities, threshold=0.55, scale_by_magnitude=True):
    probabilities = np.array(probabilities)
    if scale_by_magnitude:
        signal_strength = np.abs(probabilities - 0.5)
        direction = np.where(probabilities > 0.5, 1, -1)
        above_threshold = signal_strength >= (threshold - 0.5)
        max_distance = 0.5
        threshold_distance = threshold - 0.5
        scaled_strength = (signal_strength - threshold_distance) / (max_distance - threshold_distance)
        scaled_strength = np.clip(scaled_strength, 0, 1)
        signals = direction * scaled_strength * above_threshold
    else:
        signals = np.where(probabilities > threshold, 1,
                  np.where(probabilities < (1 - threshold), -1, 0))
    return signals


def print_backtest_results(metrics, title="Backtest Results"):
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(f"Total Return:     {metrics['total_return']*100:>8.2f}%")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:>8.2f}")
    print(f"Information Ratio:{metrics['information_ratio']:>8.2f}")
    print(f"Calmar Ratio:     {metrics['calmar_ratio']:>8.2f}")
    print(f"Max Drawdown:     {metrics['max_drawdown']*100:>8.2f}%")
    print(f"Win Rate:         {metrics['win_rate']*100:>8.1f}%")
    print(f"Avg Win:          {metrics['avg_win']*100:>8.4f}%")
    print(f"Avg Loss:         {metrics['avg_loss']*100:>8.4f}%")
    print(f"Profit Factor:    {metrics['profit_factor']:>8.2f}")
    print(f"Num Trades:       {metrics['num_trades']:>8}")
    print(f"Turnover:         {metrics['turnover']*100:>8.2f}%")
    print(f"{'='*50}\n")

def evaluate_classifier(y_true, y_pred, y_prob=None):
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
    from scipy.stats import pearsonr
    corr, pval = pearsonr(predictions, actuals)
    return {
        'correlation': corr,
        'pvalue': pval,
        'overfitting_check': 'PASS' if corr > 0.02 else 'FAIL - likely overfit'
    }

def compare_models(results_dict):
    df = pd.DataFrame(results_dict).T
    return df.sort_values('test_acc', ascending=False)


def walk_forward_validation(X, y_continuous, model, n_splits=5, train_frac=0.6):
    from sklearn.base import clone
    total_samples = len(X)
    window_size = total_samples // n_splits
    results = []
    for i in range(n_splits):
        start_idx = i * window_size
        end_idx = min(start_idx + window_size, total_samples)
        if end_idx >= total_samples:
            break
        X_window = X.iloc[start_idx:end_idx]
        y_window = y_continuous.iloc[start_idx:end_idx]
        split_idx = int(len(X_window) * train_frac)
        X_train = X_window.iloc[:split_idx]
        X_test = X_window.iloc[split_idx:]
        y_train = y_window.iloc[:split_idx]
        y_test = y_window.iloc[split_idx:]
        if len(X_test) < 10:
            continue
        model_clone = clone(model)
        model_clone.fit(X_train, (y_train > 0).astype(int))
        y_prob = model_clone.predict_proba(X_test)[:, 1]
        pred_returns = y_prob - 0.5
        corr_result = calculate_realized_correlation(pred_returns, y_test)
        signals = generate_scaled_signals(y_prob, threshold=0.55, scale_by_magnitude=True)
        backtest_metrics = backtest_strategy(signals, y_test, transaction_cost=0.0001)
        results.append({
            'split': i,
            'correlation': corr_result['correlation'],
            'sharpe': backtest_metrics['sharpe_ratio'],
            'total_return': backtest_metrics['total_return'],
            'max_drawdown': backtest_metrics['max_drawdown']
        })
    df_results = pd.DataFrame(results)
    return {
        'mean_correlation': df_results['correlation'].mean(),
        'std_correlation': df_results['correlation'].std(),
        'mean_sharpe': df_results['sharpe'].mean(),
        'std_sharpe': df_results['sharpe'].std(),
        'mean_return': df_results['total_return'].mean(),
        'worst_drawdown': df_results['max_drawdown'].min(),
        'consistency': (df_results['sharpe'] > 0).sum() / len(df_results),
        'all_splits': df_results
    }

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from feature_engine import load_features_for_training
from utils import pull_polygon_data, backtest_strategy, generate_scaled_signals
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class AdvancedBacktester:
    def __init__(self, df, starting_capital=100000):
        self.df = df
        self.starting_capital = starting_capital
        self.results = []

    def split_data(self, train_frac=0.6, val_frac=0.2):
        n = len(self.df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        train_df = self.df.iloc[:train_end]
        val_df = self.df.iloc[train_end:val_end]
        test_df = self.df.iloc[val_end:]

        return train_df, val_df, test_df

    def train_and_test(self, train_df, test_df, model_params=None, threshold=0.55,
                      transaction_cost=0.0001, use_extended_features=False):
        X_train, y_train, _, feature_names = load_features_for_training(
            train_df, use_extended=use_extended_features
        )
        X_test, y_test, y_continuous_test, _ = load_features_for_training(
            test_df, use_extended=use_extended_features
        )

        if model_params is None:
            model_params = {'n_estimators': 100, 'random_state': 42}

        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        y_prob_test = model.predict_proba(X_test)[:, 1]

        signals = generate_scaled_signals(y_prob_test, threshold=threshold, scale_by_magnitude=True)

        metrics = backtest_strategy(signals, y_continuous_test, transaction_cost=transaction_cost)

        metrics['threshold'] = threshold
        metrics['transaction_cost'] = transaction_cost
        metrics['num_features'] = len(feature_names)
        metrics['train_samples'] = len(X_train)
        metrics['test_samples'] = len(X_test)

        return metrics, model

    def parameter_sweep(self, param_grid, use_extended_features=False):
        train_df, val_df, test_df = self.split_data()

        results = []

        for params in ParameterGrid(param_grid):
            print(f"testing: {params}")

            metrics, model = self.train_and_test(
                train_df, val_df,
                threshold=params.get('threshold', 0.55),
                transaction_cost=params.get('transaction_cost', 0.0001),
                use_extended_features=use_extended_features
            )

            result = {**params, **metrics}
            results.append(result)

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)

        return results_df

    def walk_forward_test(self, n_splits=5, threshold=0.55, use_extended_features=False):
        total_samples = len(self.df)
        window_size = total_samples // n_splits

        results = []

        for i in range(n_splits):
            start_idx = i * window_size
            end_idx = min(start_idx + window_size, total_samples)

            if end_idx >= total_samples:
                break

            window_df = self.df.iloc[start_idx:end_idx]
            split_idx = int(len(window_df) * 0.6)

            train_df = window_df.iloc[:split_idx]
            test_df = window_df.iloc[split_idx:]

            if len(test_df) < 10:
                continue

            metrics, model = self.train_and_test(train_df, test_df, threshold=threshold,
                                                use_extended_features=use_extended_features)
            metrics['split'] = i
            results.append(metrics)

        results_df = pd.DataFrame(results)

        summary = {
            'mean_sharpe': results_df['sharpe_ratio'].mean(),
            'std_sharpe': results_df['sharpe_ratio'].std(),
            'mean_return': results_df['total_return'].mean(),
            'mean_win_rate': results_df['win_rate'].mean(),
            'consistency': (results_df['sharpe_ratio'] > 0).sum() / len(results_df),
            'worst_sharpe': results_df['sharpe_ratio'].min(),
            'best_sharpe': results_df['sharpe_ratio'].max(),
            'all_splits': results_df
        }

        return summary

    def compare_models(self, models_config, use_extended_features=False):
        train_df, val_df, test_df = self.split_data()

        results = []

        for config in models_config:
            name = config['name']
            model_class = config['model']
            params = config.get('params', {})
            threshold = config.get('threshold', 0.55)

            X_train, y_train, _, _ = load_features_for_training(
                train_df, use_extended=use_extended_features
            )
            X_test, y_test, y_continuous_test, _ = load_features_for_training(
                test_df, use_extended=use_extended_features
            )

            model = model_class(**params)
            model.fit(X_train, y_train)

            y_prob_test = model.predict_proba(X_test)[:, 1]
            signals = generate_scaled_signals(y_prob_test, threshold=threshold, scale_by_magnitude=True)

            metrics = backtest_strategy(signals, y_continuous_test, transaction_cost=0.0001)
            metrics['model_name'] = name

            results.append(metrics)

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)

        return results_df

    def monte_carlo_simulation(self, n_simulations=1000, test_df=None):
        if test_df is None:
            _, _, test_df = self.split_data()

        X_test, y_test, y_continuous_test, _ = load_features_for_training(test_df)

        train_df, _, _ = self.split_data()
        X_train, y_train, _, _ = load_features_for_training(train_df)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_prob_test = model.predict_proba(X_test)[:, 1]
        signals = generate_scaled_signals(y_prob_test, threshold=0.55, scale_by_magnitude=True)

        simulation_results = []

        for _ in range(n_simulations):
            shuffled_returns = y_continuous_test.sample(frac=1, replace=True).values
            metrics = backtest_strategy(signals, shuffled_returns, transaction_cost=0.0001)
            simulation_results.append(metrics['total_return'])

        actual_metrics = backtest_strategy(signals, y_continuous_test, transaction_cost=0.0001)
        actual_return = actual_metrics['total_return']

        simulation_results = np.array(simulation_results)
        percentile = (simulation_results < actual_return).sum() / len(simulation_results)

        return {
            'actual_return': actual_return,
            'mean_simulated_return': simulation_results.mean(),
            'std_simulated_return': simulation_results.std(),
            'percentile': percentile,
            'confidence_95': np.percentile(simulation_results, [2.5, 97.5]),
            'all_simulations': simulation_results
        }

    def plot_parameter_heatmap(self, results_df, x_param, y_param, metric='sharpe_ratio'):
        pivot = results_df.pivot(index=y_param, columns=x_param, values=metric)

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
        plt.title(f'{metric} Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def print_summary(self, results_df):
        print("\n" + "="*80)
        print("BACKTEST RESULTS SUMMARY")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80 + "\n")

if __name__ == "__main__":
    API_KEY = "vFDjkUVRfPnedLrbRjm75BZ9CJHz3dfv"
    TICKER = "AAPL"
    START = "2025-01-01"
    END = "2025-12-31"

    print(f"loading data for {TICKER}...")
    df = pull_polygon_data(TICKER, START, END, API_KEY)

    backtester = AdvancedBacktester(df)

    print("\n1. parameter sweep...")
    param_grid = {
        'threshold': [0.50, 0.55, 0.60],
        'transaction_cost': [0.0001, 0.0005]
    }

    results = backtester.parameter_sweep(param_grid)
    backtester.print_summary(results)

    print("\n2. walk-forward validation...")
    wf_results = backtester.walk_forward_test(n_splits=5, threshold=0.55)
    print(f"mean sharpe: {wf_results['mean_sharpe']:.2f}")
    print(f"consistency: {wf_results['consistency']*100:.1f}%")

    print("\n3. monte carlo simulation...")
    mc_results = backtester.monte_carlo_simulation(n_simulations=1000)
    print(f"actual return: {mc_results['actual_return']*100:.2f}%")
    print(f"mean simulated return: {mc_results['mean_simulated_return']*100:.2f}%")
    print(f"percentile rank: {mc_results['percentile']*100:.1f}%")

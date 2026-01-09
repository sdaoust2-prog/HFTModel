import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import sqlite3

class OnlineLearner:
    def __init__(self, model_path='../trained_stock_model.pkl', db_path='trading.db'):
        self.model_path = os.path.join(os.path.dirname(__file__), '..', model_path.replace('../', ''))
        self.db_path = os.path.join(os.path.dirname(__file__), db_path)
        self.model = joblib.load(self.model_path)

    def collect_trade_outcomes(self, lookback_hours=24):
        if not os.path.exists(self.db_path):
            print("no trading database found")
            return None

        conn = sqlite3.connect(self.db_path)

        cutoff = datetime.now() - timedelta(hours=lookback_hours)

        query = """
        SELECT p.*,
               t1.features as entry_features,
               t1.confidence as entry_confidence,
               t2.timestamp as exit_timestamp
        FROM positions p
        LEFT JOIN trades t1 ON p.ticker = t1.ticker
            AND abs(strftime('%s', p.entry_time) - strftime('%s', t1.timestamp)) < 60
            AND t1.side = p.side
        LEFT JOIN trades t2 ON p.ticker = t2.ticker
            AND abs(strftime('%s', p.exit_time) - strftime('%s', t2.timestamp)) < 60
        WHERE p.exit_time IS NOT NULL
            AND p.exit_time >= ?
        ORDER BY p.exit_time DESC
        """

        df = pd.read_sql_query(query, conn, params=(cutoff,))
        conn.close()

        if df.empty:
            return None

        print(f"collected {len(df)} completed trades from last {lookback_hours}h")
        return df

    def extract_learning_samples(self, trades_df):
        samples = []

        for idx, trade in trades_df.iterrows():
            try:
                if pd.isna(trade['entry_features']):
                    continue

                features = eval(trade['entry_features'])

                feature_vector = [
                    features.get('momentum_1min', 0),
                    features.get('volatility_1min', 0),
                    features.get('price_direction', 0),
                    features.get('vwap_dev', 0),
                    features.get('hour', 0),
                    features.get('minute', 0),
                    features.get('autocorr_5', 0),
                    features.get('rolling_sharpe_20', 0),
                    features.get('price_accel', 0),
                    features.get('order_flow', 0)
                ]

                actual_outcome = 1 if trade['pnl'] > 0 else 0

                predicted_direction = 1 if trade['side'] == 'BUY' else 0

                reward = trade['pnl'] / abs(trade['qty'] * trade['entry_price'])

                samples.append({
                    'features': feature_vector,
                    'predicted': predicted_direction,
                    'actual': actual_outcome,
                    'reward': reward,
                    'pnl': trade['pnl'],
                    'ticker': trade['ticker'],
                    'confidence': trade.get('entry_confidence', 0.5)
                })

            except Exception as e:
                print(f"error processing trade {idx}: {e}")
                continue

        return pd.DataFrame(samples)

    def analyze_model_performance(self, samples_df):
        if samples_df.empty:
            return None

        correct = (samples_df['predicted'] == samples_df['actual']).sum()
        total = len(samples_df)
        accuracy = correct / total

        avg_reward = samples_df['reward'].mean()
        total_pnl = samples_df['pnl'].sum()

        winners = samples_df[samples_df['actual'] == 1]
        losers = samples_df[samples_df['actual'] == 0]

        win_rate = len(winners) / total if total > 0 else 0

        high_conf_correct = samples_df[samples_df['confidence'] > 0.6]
        high_conf_accuracy = ((high_conf_correct['predicted'] == high_conf_correct['actual']).sum()
                             / len(high_conf_correct)) if len(high_conf_correct) > 0 else 0

        performance = {
            'total_trades': total,
            'accuracy': accuracy,
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'total_pnl': total_pnl,
            'high_conf_accuracy': high_conf_accuracy,
            'best_tickers': samples_df.groupby('ticker')['pnl'].sum().sort_values(ascending=False).head(3).to_dict(),
            'worst_tickers': samples_df.groupby('ticker')['pnl'].sum().sort_values().head(3).to_dict()
        }

        return performance

    def incremental_update(self, lookback_hours=24, min_samples=20):
        print(f"\n{'='*60}")
        print("ONLINE LEARNING - INCREMENTAL UPDATE")
        print(f"{'='*60}\n")

        trades_df = self.collect_trade_outcomes(lookback_hours)

        if trades_df is None or len(trades_df) == 0:
            print("no trades to learn from")
            return False

        samples_df = self.extract_learning_samples(trades_df)

        if len(samples_df) < min_samples:
            print(f"insufficient samples: {len(samples_df)} < {min_samples}")
            return False

        print(f"\nlearning from {len(samples_df)} trades...")

        performance = self.analyze_model_performance(samples_df)

        print(f"\nCurrent Performance:")
        print(f"  Accuracy: {performance['accuracy']*100:.1f}%")
        print(f"  Win Rate: {performance['win_rate']*100:.1f}%")
        print(f"  Avg Reward: {performance['avg_reward']*100:.3f}%")
        print(f"  Total P&L: ${performance['total_pnl']:.2f}")
        print(f"  High Confidence Accuracy: {performance['high_conf_accuracy']*100:.1f}%")

        if performance['accuracy'] < 0.48:
            print("\nWARNING: Model accuracy below 48%, significant retraining recommended")

        X = np.array(samples_df['features'].tolist())
        y = samples_df['actual'].values

        sample_weights = np.abs(samples_df['reward'].values)
        sample_weights = sample_weights / sample_weights.sum()

        try:
            new_model = clone(self.model)
            new_model.fit(X, y, sample_weight=sample_weights)

            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(new_model, X, y, cv=min(5, len(X)//2))

            print(f"\nUpdated Model CV Score: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

            if scores.mean() > 0.48:
                backup_path = self.model_path.replace('.pkl', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
                joblib.dump(self.model, backup_path)

                joblib.dump(new_model, self.model_path)
                self.model = new_model

                print(f"\n✓ Model updated successfully")
                print(f"  Backup saved: {os.path.basename(backup_path)}")

                return True
            else:
                print(f"\n✗ Updated model performance too low, keeping current model")
                return False

        except Exception as e:
            print(f"\nerror during update: {e}")
            return False

    def reinforcement_learning_update(self, lookback_hours=24):
        print(f"\n{'='*60}")
        print("REINFORCEMENT LEARNING UPDATE")
        print(f"{'='*60}\n")

        trades_df = self.collect_trade_outcomes(lookback_hours)

        if trades_df is None or len(trades_df) < 10:
            print("insufficient trades for RL update")
            return False

        samples_df = self.extract_learning_samples(trades_df)

        performance = self.analyze_model_performance(samples_df)

        print(f"Learning Signal:")
        print(f"  Win Rate: {performance['win_rate']*100:.1f}%")
        print(f"  Avg Reward: {performance['avg_reward']*100:.3f}%")
        print(f"  Total P&L: ${performance['total_pnl']:.2f}")

        positive_rewards = samples_df[samples_df['reward'] > 0]
        negative_rewards = samples_df[samples_df['reward'] < 0]

        print(f"\nGood Trades: {len(positive_rewards)}")
        print(f"Bad Trades: {len(negative_rewards)}")

        if len(positive_rewards) > 0:
            print(f"\nBest Performers:")
            for ticker, pnl in performance['best_tickers'].items():
                print(f"  {ticker}: ${pnl:.2f}")

        if len(negative_rewards) > 0:
            print(f"\nWorst Performers:")
            for ticker, pnl in performance['worst_tickers'].items():
                print(f"  {ticker}: ${pnl:.2f}")

        X_good = np.array(positive_rewards['features'].tolist())
        y_good = positive_rewards['actual'].values
        reward_weights = np.abs(positive_rewards['reward'].values)

        if len(X_good) > 5:
            print(f"\nBoosting model on {len(X_good)} profitable trades...")

            for i in range(3):
                self.model.fit(X_good, y_good, sample_weight=reward_weights)

            joblib.dump(self.model, self.model_path)
            print("✓ RL update complete")

            return True

        return False

    def print_learning_report(self, lookback_hours=24):
        trades_df = self.collect_trade_outcomes(lookback_hours)

        if trades_df is None or len(trades_df) == 0:
            print("no recent trades")
            return

        samples_df = self.extract_learning_samples(trades_df)
        performance = self.analyze_model_performance(samples_df)

        print(f"\n{'='*60}")
        print(f"LEARNING REPORT - Last {lookback_hours}h")
        print(f"{'='*60}\n")

        print(f"Total Trades: {performance['total_trades']}")
        print(f"Prediction Accuracy: {performance['accuracy']*100:.1f}%")
        print(f"Win Rate: {performance['win_rate']*100:.1f}%")
        print(f"Avg Reward per Trade: {performance['avg_reward']*100:.3f}%")
        print(f"Total P&L: ${performance['total_pnl']:.2f}")
        print(f"High Confidence Accuracy: {performance['high_conf_accuracy']*100:.1f}%")

        print(f"\nBest Tickers:")
        for ticker, pnl in performance['best_tickers'].items():
            ticker_trades = len(samples_df[samples_df['ticker'] == ticker])
            print(f"  {ticker}: ${pnl:.2f} ({ticker_trades} trades)")

        print(f"\nWorst Tickers:")
        for ticker, pnl in performance['worst_tickers'].items():
            ticker_trades = len(samples_df[samples_df['ticker'] == ticker])
            print(f"  {ticker}: ${pnl:.2f} ({ticker_trades} trades)")

        if performance['accuracy'] > 0.55:
            print(f"\n✓ Model performing well (>55% accuracy)")
        elif performance['accuracy'] > 0.50:
            print(f"\n⚠ Model marginal (50-55% accuracy)")
        else:
            print(f"\n✗ Model underperforming (<50% accuracy) - retrain recommended")

if __name__ == "__main__":
    import sys

    learner = OnlineLearner()

    if len(sys.argv) == 1 or sys.argv[1] == 'report':
        learner.print_learning_report(lookback_hours=24)

    elif sys.argv[1] == 'update':
        hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
        learner.incremental_update(lookback_hours=hours)

    elif sys.argv[1] == 'rl':
        hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
        learner.reinforcement_learning_update(lookback_hours=hours)

    else:
        print("usage:")
        print("  python online_learning.py report        # show learning report")
        print("  python online_learning.py update [24]   # incremental update from last N hours")
        print("  python online_learning.py rl [24]       # reinforcement learning update")

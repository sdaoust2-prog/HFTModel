import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from feature_engine import load_features_for_training
from utils import pull_polygon_data, train_test_split_chronological, backtest_strategy, generate_scaled_signals
from trade_logger import TradeLogger
from performance_tracker import PerformanceTracker

class ContinuousLearning:
    def __init__(self, retrain_frequency_days=7, validation_threshold_sharpe=0.5):
        self.retrain_frequency = retrain_frequency_days
        self.validation_threshold = validation_threshold_sharpe
        self.logger = TradeLogger()
        self.tracker = PerformanceTracker(self.logger)

        self.model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(self.model_dir, exist_ok=True)

        self.metadata_path = os.path.join(self.model_dir, 'model_metadata.csv')
        self.load_metadata()

    def load_metadata(self):
        if os.path.exists(self.metadata_path):
            self.metadata = pd.read_csv(self.metadata_path)
        else:
            self.metadata = pd.DataFrame(columns=[
                'model_id', 'trained_date', 'deployed_date', 'data_start', 'data_end',
                'train_samples', 'val_sharpe', 'is_deployed', 'notes'
            ])

    def save_metadata(self):
        self.metadata.to_csv(self.metadata_path, index=False)

    def should_retrain(self):
        if self.metadata.empty:
            return True, "no model exists"

        last_deployed = self.metadata[self.metadata['is_deployed'] == True]
        if last_deployed.empty:
            return True, "no deployed model"

        last_date = pd.to_datetime(last_deployed.iloc[-1]['trained_date'])
        days_since = (datetime.now() - last_date).days

        if days_since >= self.retrain_frequency:
            return True, f"last trained {days_since} days ago"

        recent_performance = self.tracker.generate_report(
            start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        )

        if recent_performance.get('sharpe_ratio', 0) < 0.3:
            return True, f"performance degraded (sharpe: {recent_performance.get('sharpe_ratio', 0):.2f})"

        return False, f"model is recent ({days_since} days old) and performing well"

    def collect_training_data(self, lookback_days=60):
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv('POLYGON_API_KEY')
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=lookback_days)

        all_data = []

        import trader_config as config
        for ticker in config.WATCHLIST[:3]:
            try:
                print(f"fetching {ticker} data...")
                df = pull_polygon_data(ticker, str(start_date), str(end_date), api_key)
                df['ticker'] = ticker
                all_data.append(df)
                print(f"  got {len(df)} bars")
            except Exception as e:
                print(f"  failed: {e}")

        if not all_data:
            raise ValueError("no data collected")

        combined = pd.concat(all_data, ignore_index=True)
        print(f"\ntotal data: {len(combined)} bars from {len(all_data)} tickers")
        return combined

    def train_new_model(self, df, model_params=None):
        print("\ntraining new model...")

        X, y_binary, y_continuous, feature_names = load_features_for_training(df, use_extended=True)
        X_train, X_val, y_train, y_val = train_test_split_chronological(X, y_binary, train_frac=0.8)
        _, _, y_train_cont, y_val_cont = train_test_split_chronological(X, y_continuous, train_frac=0.8)

        if model_params is None:
            model_params = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 20,
                'random_state': 42
            }

        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        y_prob_val = model.predict_proba(X_val)[:, 1]
        signals = generate_scaled_signals(y_prob_val, threshold=0.55, scale_by_magnitude=True)
        val_metrics = backtest_strategy(signals, y_val_cont, transaction_cost=0.0001)

        print(f"\nvalidation results:")
        print(f"  sharpe: {val_metrics['sharpe_ratio']:.2f}")
        print(f"  return: {val_metrics['total_return']*100:.2f}%")
        print(f"  win rate: {val_metrics['win_rate']*100:.1f}%")

        return model, val_metrics, len(X_train)

    def validate_model(self, new_model, new_metrics, current_model=None):
        if new_metrics['sharpe_ratio'] < self.validation_threshold:
            return False, f"new model sharpe {new_metrics['sharpe_ratio']:.2f} below threshold {self.validation_threshold}"

        if current_model is None:
            return True, "no existing model, deploying new one"

        improvement_threshold = 0.1
        current_sharpe = self.get_current_model_sharpe()

        if new_metrics['sharpe_ratio'] > current_sharpe + improvement_threshold:
            return True, f"new model better (sharpe {new_metrics['sharpe_ratio']:.2f} vs {current_sharpe:.2f})"

        if new_metrics['sharpe_ratio'] >= current_sharpe * 0.9:
            return True, "new model comparable or better"

        return False, f"new model worse (sharpe {new_metrics['sharpe_ratio']:.2f} vs {current_sharpe:.2f})"

    def get_current_model_sharpe(self):
        if self.metadata.empty:
            return 0

        deployed = self.metadata[self.metadata['is_deployed'] == True]
        if deployed.empty:
            return 0

        return deployed.iloc[-1]['val_sharpe']

    def deploy_model(self, model, metrics, train_samples, data_start, data_end, notes=""):
        model_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_filename = f'model_{model_id}.pkl'
        model_path = os.path.join(self.model_dir, model_filename)
        joblib.dump(model, model_path)

        production_path = os.path.join(os.path.dirname(__file__), '..', 'trained_stock_model.pkl')
        joblib.dump(model, production_path)

        self.metadata.loc[self.metadata['is_deployed'] == True, 'is_deployed'] = False

        new_row = {
            'model_id': model_id,
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'deployed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_start': data_start,
            'data_end': data_end,
            'train_samples': train_samples,
            'val_sharpe': metrics['sharpe_ratio'],
            'is_deployed': True,
            'notes': notes
        }

        self.metadata = pd.concat([self.metadata, pd.DataFrame([new_row])], ignore_index=True)
        self.save_metadata()

        print(f"\nmodel deployed:")
        print(f"  id: {model_id}")
        print(f"  path: {production_path}")
        print(f"  sharpe: {metrics['sharpe_ratio']:.2f}")

        return model_id

    def run_retraining_cycle(self, force=False):
        print("="*70)
        print("CONTINUOUS LEARNING - RETRAINING CYCLE")
        print("="*70)

        should_retrain, reason = self.should_retrain()
        print(f"\ncheck: {reason}")

        if not should_retrain and not force:
            print("skipping retrain")
            return None

        print("\nstarting retrain process...")

        try:
            current_model_path = os.path.join(os.path.dirname(__file__), '..', 'trained_stock_model.pkl')
            current_model = joblib.load(current_model_path) if os.path.exists(current_model_path) else None
        except:
            current_model = None

        df = self.collect_training_data(lookback_days=60)

        new_model, new_metrics, train_samples = self.train_new_model(df)

        is_valid, validation_msg = self.validate_model(new_model, new_metrics, current_model)
        print(f"\nvalidation: {validation_msg}")

        if not is_valid:
            print("model rejected, keeping current model")
            return None

        data_start = df['timestamp'].min().strftime('%Y-%m-%d')
        data_end = df['timestamp'].max().strftime('%Y-%m-%d')

        model_id = self.deploy_model(
            new_model, new_metrics, train_samples,
            data_start, data_end,
            notes=validation_msg
        )

        print(f"\nSUCCESS: new model deployed (id: {model_id})")
        print("="*70)

        return model_id

    def get_model_history(self):
        if self.metadata.empty:
            return pd.DataFrame()

        return self.metadata.sort_values('trained_date', ascending=False)

    def get_current_model_age(self):
        model_path = os.path.join(os.path.dirname(__file__), '..', 'trained_stock_model.pkl')
        if os.path.exists(model_path):
            return os.path.getmtime(model_path)
        return 0

    def rollback_to_previous_model(self):
        if self.metadata.empty or len(self.metadata) < 2:
            print("no previous model to rollback to")
            return False

        deployed = self.metadata[self.metadata['is_deployed'] == True]
        if deployed.empty:
            print("no deployed model")
            return False

        current_model_id = deployed.iloc[-1]['model_id']

        previous_models = self.metadata[
            (self.metadata['is_deployed'] == False) &
            (self.metadata['trained_date'] < deployed.iloc[-1]['trained_date'])
        ].sort_values('trained_date', ascending=False)

        if previous_models.empty:
            print("no previous model available")
            return False

        previous_model_id = previous_models.iloc[0]['model_id']
        previous_model_path = os.path.join(self.model_dir, f'model_{previous_model_id}.pkl')

        if not os.path.exists(previous_model_path):
            print(f"previous model file not found: {previous_model_path}")
            return False

        model = joblib.load(previous_model_path)
        production_path = os.path.join(os.path.dirname(__file__), '..', 'trained_stock_model.pkl')
        joblib.dump(model, production_path)

        self.metadata.loc[self.metadata['model_id'] == current_model_id, 'is_deployed'] = False
        self.metadata.loc[self.metadata['model_id'] == previous_model_id, 'is_deployed'] = True
        self.metadata.loc[self.metadata['model_id'] == previous_model_id, 'deployed_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.save_metadata()

        print(f"\nrollback successful:")
        print(f"  from: {current_model_id}")
        print(f"  to: {previous_model_id}")

        return True

if __name__ == "__main__":
    import sys

    cl = ContinuousLearning(retrain_frequency_days=7)

    if len(sys.argv) > 1 and sys.argv[1] == '--force':
        print("forcing retrain...")
        cl.run_retraining_cycle(force=True)
    else:
        cl.run_retraining_cycle(force=False)

    print("\nmodel history:")
    print(cl.get_model_history().to_string(index=False))

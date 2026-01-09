from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import numpy as np
from feature_engine import load_features_for_training
from utils import pull_polygon_data, train_test_split_chronological, evaluate_classifier, backtest_strategy, generate_scaled_signals

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("xgboost not available, install with: pip3 install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("lightgbm not available, install with: pip3 install lightgbm")

def train_xgboost(X_train, y_train, X_test, y_test):
    if not XGBOOST_AVAILABLE:
        return None

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             verbose=False)

    return model

def train_lightgbm(X_train, y_train, X_test, y_test):
    if not LIGHTGBM_AVAILABLE:
        return None

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train,
             eval_set=[(X_test, y_test)])

    return model

def train_ensemble(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)

    estimators = [('rf', rf), ('gb', gb), ('lr', lr)]

    if XGBOOST_AVAILABLE:
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        estimators.append(('xgb', xgb_model))

    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    ensemble.fit(X_train, y_train)

    return ensemble

def evaluate_model(model, X_test, y_test, y_continuous_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    signals = generate_scaled_signals(y_prob, threshold=0.55, scale_by_magnitude=True)
    backtest_metrics = backtest_strategy(signals, y_continuous_test, transaction_cost=0.0001)

    print(f"\n{model_name} Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Sharpe: {backtest_metrics['sharpe_ratio']:.2f}")
    print(f"  Total Return: {backtest_metrics['total_return']*100:.2f}%")
    print(f"  Win Rate: {backtest_metrics['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {backtest_metrics['profit_factor']:.2f}")

    return {
        'model_name': model_name,
        'accuracy': acc,
        'auc': auc,
        **backtest_metrics
    }

def compare_all_models(df, use_extended_features=True):
    X, y_binary, y_continuous, feature_names = load_features_for_training(df, use_extended=use_extended_features)
    X_train, X_test, y_train, y_test = train_test_split_chronological(X, y_binary)
    _, _, y_train_cont, y_test_cont = train_test_split_chronological(X, y_continuous)

    print(f"training models with {len(feature_names)} features")
    print(f"train samples: {len(X_train)}, test samples: {len(X_test)}")

    results = []

    print("\n1. Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_results = evaluate_model(rf, X_test, y_test, y_test_cont, "Random Forest")
    results.append(rf_results)

    print("\n2. Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train)
    gb_results = evaluate_model(gb, X_test, y_test, y_test_cont, "Gradient Boosting")
    results.append(gb_results)

    if XGBOOST_AVAILABLE:
        print("\n3. XGBoost...")
        xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
        xgb_results = evaluate_model(xgb_model, X_test, y_test, y_test_cont, "XGBoost")
        results.append(xgb_results)

    if LIGHTGBM_AVAILABLE:
        print("\n4. LightGBM...")
        lgb_model = train_lightgbm(X_train, y_train, X_test, y_test)
        lgb_results = evaluate_model(lgb_model, X_test, y_test, y_test_cont, "LightGBM")
        results.append(lgb_results)

    print("\n5. Ensemble...")
    ensemble = train_ensemble(X_train, y_train)
    ensemble_results = evaluate_model(ensemble, X_test, y_test, y_test_cont, "Ensemble")
    results.append(ensemble_results)

    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    for r in sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True):
        print(f"{r['model_name']:20s} | Sharpe: {r['sharpe_ratio']:6.2f} | Return: {r['total_return']*100:6.2f}% | "
              f"Win Rate: {r['win_rate']*100:5.1f}% | AUC: {r['auc']:.4f}")
    print("="*80)

    best_model_result = max(results, key=lambda x: x['sharpe_ratio'])
    best_model_name = best_model_result['model_name']

    print(f"\nBest model: {best_model_name}")

    if best_model_name == "Random Forest":
        best_model = rf
    elif best_model_name == "Gradient Boosting":
        best_model = gb
    elif best_model_name == "XGBoost":
        best_model = xgb_model
    elif best_model_name == "LightGBM":
        best_model = lgb_model
    else:
        best_model = ensemble

    return best_model, results

if __name__ == "__main__":
    API_KEY = "vFDjkUVRfPnedLrbRjm75BZ9CJHz3dfv"
    TICKER = "AAPL"
    START = "2025-01-01"
    END = "2025-12-31"

    print(f"training advanced models on {TICKER} from {START} to {END}")

    df = pull_polygon_data(TICKER, START, END, API_KEY)

    best_model, results = compare_all_models(df, use_extended_features=True)

    joblib.dump(best_model, 'trained_advanced_model.pkl')
    print(f"\nbest model saved to trained_advanced_model.pkl")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from feature_engine import load_features_for_training
from utils import pull_polygon_data, train_test_split_chronological, evaluate_classifier


def train_model(df, model_type='rf', **model_params):
    """Train a model on the data"""

    X, y_binary, _, feature_names = load_features_for_training(df)

    X_train, X_test, y_train, y_test = train_test_split_chronological(X, y_binary)

    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42, **model_params)
    else:
        raise ValueError(f"unknown model type: {model_type}")

    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    print(f"\ntrain: {len(X_train)} samples")
    print(f"test: {len(X_test)} samples")

    evaluate_classifier(y_test, y_pred_test, y_prob_test)

    if hasattr(model, 'feature_importances_'):
        print("\nfeature importance:")
        for feat, imp in sorted(zip(feature_names, model.feature_importances_),
                               key=lambda x: x[1], reverse=True):
            print(f"  {feat:20s} {imp:.4f}")

    return model, X_test, y_test, y_prob_test


if __name__ == "__main__":
    API_KEY = "vFDjkUVRfPnedLrbRjm75BZ9CJHz3dfv"
    TICKER = "AAPL"
    START = "2025-10-01"
    END = "2025-11-01"

    print(f"training on {TICKER} from {START} to {END}")

    df = pull_polygon_data(TICKER, START, END, API_KEY)
    df.to_parquet(f"/Users/shaundaoust/data/polygon/{TICKER}_{START}_{END}.pq")
    model, X_test, y_test, y_prob = train_model(df, model_type='rf')

    breakpoint()

    joblib.dump(model, 'trained_stock_model.pkl')
    print("\nmodel saved to trained_stock_model.pkl")

from feature_engine import load_features_for_training
from utils import pull_polygon_data, train_test_split_chronological
from advanced_backtest import AdvancedBacktester
from backend.performance_tracker import PerformanceTracker
from backend.trade_logger import TradeLogger
from backend.risk_manager import RiskManager
from backend.alert_system import AlertSystem
from train_advanced import compare_all_models
import joblib

def demo_complete_system():
    print("="*80)
    print("COMPLETE TRADING SYSTEM DEMONSTRATION")
    print("="*80)

    API_KEY = "vFDjkUVRfPnedLrbRjm75BZ9CJHz3dfv"
    TICKER = "AAPL"
    START = "2025-01-01"
    END = "2025-12-31"

    print(f"\n1. Loading data for {TICKER}...")
    df = pull_polygon_data(TICKER, START, END, API_KEY)
    print(f"   Loaded {len(df)} minute bars")

    print("\n2. Feature Engineering (Extended Features)...")
    X, y_binary, y_continuous, feature_names = load_features_for_training(
        df,
        use_extended=True
    )
    print(f"   Created {len(feature_names)} features")
    print(f"   Sample features: {', '.join(feature_names[:5])}...")

    print("\n3. Training Advanced Models...")
    best_model, results = compare_all_models(df, use_extended_features=True)
    print(f"   Best model selected")

    print("\n4. Advanced Backtesting...")
    backtester = AdvancedBacktester(df)

    print("\n   4a. Parameter Sweep...")
    param_grid = {
        'threshold': [0.50, 0.55, 0.60],
        'transaction_cost': [0.0001, 0.0005]
    }
    sweep_results = backtester.parameter_sweep(param_grid, use_extended_features=True)
    best_params = sweep_results.iloc[0]
    print(f"      Best threshold: {best_params['threshold']}")
    print(f"      Best Sharpe: {best_params['sharpe_ratio']:.2f}")

    print("\n   4b. Walk-Forward Validation...")
    wf_results = backtester.walk_forward_test(n_splits=5, threshold=0.55, use_extended_features=True)
    print(f"      Mean Sharpe: {wf_results['mean_sharpe']:.2f}")
    print(f"      Consistency: {wf_results['consistency']*100:.1f}%")

    print("\n   4c. Monte Carlo Simulation...")
    mc_results = backtester.monte_carlo_simulation(n_simulations=100)
    print(f"      Actual Return: {mc_results['actual_return']*100:.2f}%")
    print(f"      Percentile: {mc_results['percentile']*100:.1f}%")

    print("\n5. Performance Tracking Setup...")
    logger = TradeLogger()
    tracker = PerformanceTracker(logger)
    print("   Trade logger initialized")
    print("   Performance tracker ready")

    print("\n6. Risk Management Setup...")
    risk_manager = RiskManager(
        max_portfolio_heat=0.02,
        max_correlated_positions=3,
        max_sector_exposure=0.30
    )
    print("   Risk manager configured")
    print(f"   Max portfolio heat: 2%")
    print(f"   Max sector exposure: 30%")

    test_positions = {
        'AAPL': {'qty': 10, 'current_price': 180, 'entry_price': 175}
    }
    can_trade, reason = risk_manager.should_allow_trade(
        'MSFT', test_positions, 100000, -100, 500
    )
    print(f"   Sample trade check: {can_trade} - {reason}")

    print("\n7. Alert System Setup...")
    alert_system = AlertSystem()
    print(f"   Email alerts: {'enabled' if alert_system.email_enabled else 'disabled (configure in .env)'}")

    print("\n8. Saving Best Model...")
    joblib.dump(best_model, 'trained_advanced_model.pkl')
    print("   Model saved to: trained_advanced_model.pkl")

    print("\n" + "="*80)
    print("SYSTEM DEMONSTRATION COMPLETE")
    print("="*80)

    print("\nNext Steps:")
    print("1. Start dashboard: cd backend && python3 dashboard_api.py")
    print("2. View dashboard: http://localhost:8001")
    print("3. Run trading bot: cd backend && python3 trader.py --test")
    print("4. Monitor trades in real-time")
    print("5. Review performance: python3 -c 'from backend.performance_tracker import PerformanceTracker; PerformanceTracker().print_report()'")

    print("\nAll components tested and ready!")

if __name__ == "__main__":
    demo_complete_system()

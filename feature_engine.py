import pandas as pd
import numpy as np

class FeatureEngine:
    """
    Feature computation engine for stock data.
    Pattern: Feature(data, lookback) -> values

    Add new features by defining methods that take df and optional lookback.
    All features automatically handle NaN and can be computed on any OHLCV dataframe.
    """

    @staticmethod
    def momentum(df, lookback=1):
        """Percentage price change over lookback periods"""
        return df['close'].pct_change(lookback)

    @staticmethod
    def volatility(df, lookback=1):
        """Squared returns - emphasizes large moves"""
        mom = FeatureEngine.momentum(df, lookback)
        return mom ** 2

    @staticmethod
    def price_direction(df):
        """Binary: 1 if close > open (green candle), 0 otherwise"""
        return (df['close'] > df['open']).astype(int)

    @staticmethod
    def vwap(df):
        """Volume-weighted average price (cumulative)"""
        return (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    @staticmethod
    def vwap_deviation(df):
        """Percentage deviation from VWAP"""
        vwap = FeatureEngine.vwap(df)
        return (df['close'] - vwap) / vwap

    @staticmethod
    def hour(df):
        """Hour of day (0-23)"""
        return df['timestamp'].dt.hour

    @staticmethod
    def minute(df):
        """Minute within hour (0-59)"""
        return df['timestamp'].dt.minute

    @staticmethod
    def rolling_mean(df, lookback=5):
        """Rolling mean of close price"""
        return df['close'].rolling(lookback).mean()

    @staticmethod
    def rolling_std(df, lookback=5):
        """Rolling standard deviation of close price"""
        return df['close'].rolling(lookback).std()

    @staticmethod
    def returns_z_score(df, lookback=20):
        """Z-scored returns over lookback window"""
        returns = df['close'].pct_change()
        rolling_mean = returns.rolling(lookback).mean()
        rolling_std = returns.rolling(lookback).std()
        return (returns - rolling_mean) / rolling_std

    @staticmethod
    def volume_ratio(df, lookback=5):
        """Current volume / average volume"""
        avg_vol = df['volume'].rolling(lookback).mean()
        return df['volume'] / avg_vol

    @staticmethod
    def high_low_range(df):
        """(high - low) / close - measures intrabar volatility"""
        return (df['high'] - df['low']) / df['close']

    @staticmethod
    def autocorrelation(df, lookback=5):
        """Return autocorrelation - measures momentum persistence"""
        returns = df['close'].pct_change()
        return returns.rolling(lookback).apply(lambda x: x.autocorr(), raw=False)

    @staticmethod
    def rolling_sharpe(df, lookback=20):
        """Rolling Sharpe ratio - risk-adjusted returns"""
        returns = df['close'].pct_change()
        rolling_mean = returns.rolling(lookback).mean()
        rolling_std = returns.rolling(lookback).std()
        return (rolling_mean / rolling_std) * np.sqrt(lookback)

    @staticmethod
    def price_acceleration(df, lookback=1):
        """Second derivative of price - momentum of momentum"""
        momentum = df['close'].pct_change(lookback)
        return momentum.diff()

    @staticmethod
    def volume_price_correlation(df, lookback=10):
        """Rolling correlation between volume and price changes"""
        returns = df['close'].pct_change()
        return returns.rolling(lookback).corr(df['volume'])

    @staticmethod
    def order_flow_proxy(df):
        """Microstructure: volume-weighted price pressure"""
        # Positive when price moves up with volume, negative when down
        price_change = df['close'].diff()
        return price_change * df['volume'] / df['volume'].rolling(20).mean()

    @staticmethod
    def compute_all_features(df, feature_list=None):
        """
        Compute multiple features at once.

        Args:
            df: OHLCV dataframe with timestamp
            feature_list: list of (name, function, kwargs) tuples
                         if None, uses default feature set

        Returns:
            dataframe with original data + feature columns
        """
        df = df.copy()

        if feature_list is None:
            feature_list = [
                ('momentum_1min', FeatureEngine.momentum, {'lookback': 1}),
                ('volatility_1min', FeatureEngine.volatility, {'lookback': 1}),
                ('price_direction', FeatureEngine.price_direction, {}),
                ('vwap_dev', FeatureEngine.vwap_deviation, {}),
                ('hour', FeatureEngine.hour, {}),
                ('minute', FeatureEngine.minute, {}),
                ('autocorr_5', FeatureEngine.autocorrelation, {'lookback': 5}),
                ('rolling_sharpe_20', FeatureEngine.rolling_sharpe, {'lookback': 20}),
                ('price_accel', FeatureEngine.price_acceleration, {'lookback': 1}),
                ('order_flow', FeatureEngine.order_flow_proxy, {})
            ]

        for name, func, kwargs in feature_list:
            df[name] = func(df, **kwargs)

        return df

    @staticmethod
    def compute_forward_returns(df, horizons=[1]):
        """
        Compute forward returns at multiple horizons.

        Args:
            df: dataframe with 'close' column
            horizons: list of forward periods (e.g. [1, 2, 5])

        Returns:
            dataframe with added return_Nmin columns
        """
        df = df.copy()

        for h in horizons:
            df[f'return_{h}min'] = df['close'].shift(-h) / df['close'] - 1

        return df

    @staticmethod
    def winsorize(series, lower=0.01, upper=0.99):
        """
        Cap extreme values at percentiles

        CRITICAL: To avoid lookahead bias, compute percentiles on TRAIN set only,
        then apply those thresholds to test set.

        Example:
            train_momentum = FeatureEngine.momentum(train_df)
            lower_thresh = train_momentum.quantile(0.01)
            upper_thresh = train_momentum.quantile(0.99)

            # Apply same thresholds to test
            test_momentum_clipped = test_momentum.clip(lower_thresh, upper_thresh)
        """
        lower_val = series.quantile(lower)
        upper_val = series.quantile(upper)
        return series.clip(lower=lower_val, upper=upper_val)


def load_features_for_training(df, target_horizon=1, drop_na=True):
    """
    Complete pipeline: raw OHLCV -> features + target

    Returns:
        X: feature matrix
        y_binary: binary target (UP/DOWN)
        y_continuous: continuous target (return)
        feature_names: list of feature column names
    """
    df = FeatureEngine.compute_all_features(df)
    df = FeatureEngine.compute_forward_returns(df, horizons=[target_horizon])

    if drop_na:
        df = df.dropna()

    feature_names = ['momentum_1min', 'volatility_1min', 'price_direction',
                     'vwap_dev', 'hour', 'minute', 'autocorr_5',
                     'rolling_sharpe_20', 'price_accel', 'order_flow']

    X = df[feature_names]
    y_continuous = df[f'return_{target_horizon}min']
    y_binary = (y_continuous > 0).astype(int)

    return X, y_binary, y_continuous, feature_names

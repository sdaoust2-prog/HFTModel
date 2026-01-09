import pandas as pd
import numpy as np

class FeatureEngine:
    @staticmethod
    def momentum(df, lookback=1):
        return df['close'].pct_change(lookback)

    @staticmethod
    def volatility(df, lookback=1):
        mom = FeatureEngine.momentum(df, lookback)
        return mom ** 2

    @staticmethod
    def price_direction(df):
        return (df['close'] > df['open']).astype(int)

    @staticmethod
    def vwap(df):
        return (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    @staticmethod
    def vwap_deviation(df):
        vwap = FeatureEngine.vwap(df)
        return (df['close'] - vwap) / vwap

    @staticmethod
    def hour(df):
        return df['timestamp'].dt.hour

    @staticmethod
    def minute(df):
        return df['timestamp'].dt.minute

    @staticmethod
    def rolling_mean(df, lookback=5):
        return df['close'].rolling(lookback).mean()

    @staticmethod
    def rolling_std(df, lookback=5):
        return df['close'].rolling(lookback).std()

    @staticmethod
    def returns_z_score(df, lookback=20):
        returns = df['close'].pct_change()
        rolling_mean = returns.rolling(lookback).mean()
        rolling_std = returns.rolling(lookback).std()
        return (returns - rolling_mean) / rolling_std

    @staticmethod
    def volume_ratio(df, lookback=5):
        avg_vol = df['volume'].rolling(lookback).mean()
        return df['volume'] / avg_vol

    @staticmethod
    def high_low_range(df):
        return (df['high'] - df['low']) / df['close']

    @staticmethod
    def autocorrelation(df, lookback=5):
        returns = df['close'].pct_change()
        return returns.rolling(lookback).apply(lambda x: x.autocorr(), raw=False)

    @staticmethod
    def rolling_sharpe(df, lookback=20):
        returns = df['close'].pct_change()
        rolling_mean = returns.rolling(lookback).mean()
        rolling_std = returns.rolling(lookback).std()
        return (rolling_mean / rolling_std) * np.sqrt(lookback)

    @staticmethod
    def price_acceleration(df, lookback=1):
        momentum = df['close'].pct_change(lookback)
        return momentum.diff()

    @staticmethod
    def volume_price_correlation(df, lookback=10):
        returns = df['close'].pct_change()
        return returns.rolling(lookback).corr(df['volume'])

    @staticmethod
    def order_flow_proxy(df):
        price_change = df['close'].diff()
        return price_change * df['volume'] / df['volume'].rolling(20).mean()

    @staticmethod
    def compute_all_features(df, feature_list=None):
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
        df = df.copy()
        for h in horizons:
            df[f'return_{h}min'] = df['close'].shift(-h) / df['close'] - 1
        return df

    @staticmethod
    def winsorize(series, lower=0.01, upper=0.99):
        lower_val = series.quantile(lower)
        upper_val = series.quantile(upper)
        return series.clip(lower=lower_val, upper=upper_val)


def load_features_for_training(df, target_horizon=1, drop_na=True):
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

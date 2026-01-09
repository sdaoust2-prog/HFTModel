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
    def rsi(df, lookback=14):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(lookback).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(lookback).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(df, fast=12, slow=26, signal=9):
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line

    @staticmethod
    def bollinger_bands(df, lookback=20, num_std=2):
        sma = df['close'].rolling(lookback).mean()
        std = df['close'].rolling(lookback).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return (df['close'] - lower) / (upper - lower)

    @staticmethod
    def atr(df, lookback=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(lookback).mean()

    @staticmethod
    def stochastic(df, lookback=14):
        low_min = df['low'].rolling(lookback).min()
        high_max = df['high'].rolling(lookback).max()
        return 100 * (df['close'] - low_min) / (high_max - low_min)

    @staticmethod
    def obv(df):
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv / obv.rolling(20).mean()

    @staticmethod
    def mfi(df, lookback=14):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(lookback).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(lookback).sum()
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi

    @staticmethod
    def williams_r(df, lookback=14):
        high_max = df['high'].rolling(lookback).max()
        low_min = df['low'].rolling(lookback).min()
        return -100 * (high_max - df['close']) / (high_max - low_min)

    @staticmethod
    def cci(df, lookback=20):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(lookback).mean()
        mad = typical_price.rolling(lookback).apply(lambda x: np.abs(x - x.mean()).mean())
        return (typical_price - sma) / (0.015 * mad)

    @staticmethod
    def roc(df, lookback=12):
        return ((df['close'] - df['close'].shift(lookback)) / df['close'].shift(lookback)) * 100

    @staticmethod
    def price_volume_trend(df):
        pvt = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
        return pvt / pvt.rolling(20).mean()

    @staticmethod
    def vwma(df, lookback=20):
        return (df['close'] * df['volume']).rolling(lookback).sum() / df['volume'].rolling(lookback).sum()

    @staticmethod
    def keltner_channel(df, lookback=20, multiplier=2):
        ema = df['close'].ewm(span=lookback, adjust=False).mean()
        atr_val = FeatureEngine.atr(df, lookback)
        upper = ema + (multiplier * atr_val)
        lower = ema - (multiplier * atr_val)
        return (df['close'] - lower) / (upper - lower)

    @staticmethod
    def adx(df, lookback=14):
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        atr_val = FeatureEngine.atr(df, lookback)
        plus_di = 100 * (plus_dm.rolling(lookback).mean() / atr_val)
        minus_di = 100 * (minus_dm.rolling(lookback).mean() / atr_val)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(lookback).mean()

    @staticmethod
    def volume_spike(df, lookback=20):
        avg_volume = df['volume'].rolling(lookback).mean()
        std_volume = df['volume'].rolling(lookback).std()
        return (df['volume'] - avg_volume) / std_volume

    @staticmethod
    def gap(df):
        return (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

    @staticmethod
    def compute_all_features(df, feature_list=None, use_extended=False):
        df = df.copy()
        if feature_list is None:
            if use_extended:
                feature_list = [
                    ('momentum_1min', FeatureEngine.momentum, {'lookback': 1}),
                    ('momentum_5min', FeatureEngine.momentum, {'lookback': 5}),
                    ('volatility_1min', FeatureEngine.volatility, {'lookback': 1}),
                    ('volatility_5min', FeatureEngine.volatility, {'lookback': 5}),
                    ('price_direction', FeatureEngine.price_direction, {}),
                    ('vwap_dev', FeatureEngine.vwap_deviation, {}),
                    ('hour', FeatureEngine.hour, {}),
                    ('minute', FeatureEngine.minute, {}),
                    ('autocorr_5', FeatureEngine.autocorrelation, {'lookback': 5}),
                    ('rolling_sharpe_20', FeatureEngine.rolling_sharpe, {'lookback': 20}),
                    ('price_accel', FeatureEngine.price_acceleration, {'lookback': 1}),
                    ('order_flow', FeatureEngine.order_flow_proxy, {}),
                    ('rsi_14', FeatureEngine.rsi, {'lookback': 14}),
                    ('rsi_7', FeatureEngine.rsi, {'lookback': 7}),
                    ('macd', FeatureEngine.macd, {}),
                    ('bb_position', FeatureEngine.bollinger_bands, {'lookback': 20}),
                    ('atr_14', FeatureEngine.atr, {'lookback': 14}),
                    ('stochastic', FeatureEngine.stochastic, {'lookback': 14}),
                    ('obv_normalized', FeatureEngine.obv, {}),
                    ('mfi', FeatureEngine.mfi, {'lookback': 14}),
                    ('williams_r', FeatureEngine.williams_r, {'lookback': 14}),
                    ('cci', FeatureEngine.cci, {'lookback': 20}),
                    ('roc', FeatureEngine.roc, {'lookback': 12}),
                    ('pvt', FeatureEngine.price_volume_trend, {}),
                    ('vwma_dev', lambda df: (df['close'] / FeatureEngine.vwma(df, 20)) - 1, {}),
                    ('keltner', FeatureEngine.keltner_channel, {'lookback': 20}),
                    ('adx', FeatureEngine.adx, {'lookback': 14}),
                    ('volume_spike', FeatureEngine.volume_spike, {'lookback': 20}),
                    ('gap', FeatureEngine.gap, {}),
                    ('high_low_range', FeatureEngine.high_low_range, {}),
                    ('volume_ratio', FeatureEngine.volume_ratio, {'lookback': 5})
                ]
            else:
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


def load_features_for_training(df, target_horizon=1, drop_na=True, use_extended=False):
    df = FeatureEngine.compute_all_features(df, use_extended=use_extended)
    df = FeatureEngine.compute_forward_returns(df, horizons=[target_horizon])
    if drop_na:
        df = df.dropna()

    if use_extended:
        feature_names = [
            'momentum_1min', 'momentum_5min', 'volatility_1min', 'volatility_5min',
            'price_direction', 'vwap_dev', 'hour', 'minute', 'autocorr_5',
            'rolling_sharpe_20', 'price_accel', 'order_flow', 'rsi_14', 'rsi_7',
            'macd', 'bb_position', 'atr_14', 'stochastic', 'obv_normalized',
            'mfi', 'williams_r', 'cci', 'roc', 'pvt', 'vwma_dev', 'keltner',
            'adx', 'volume_spike', 'gap', 'high_low_range', 'volume_ratio'
        ]
    else:
        feature_names = ['momentum_1min', 'volatility_1min', 'price_direction',
                         'vwap_dev', 'hour', 'minute', 'autocorr_5',
                         'rolling_sharpe_20', 'price_accel', 'order_flow']

    X = df[feature_names]
    y_continuous = df[f'return_{target_horizon}min']
    y_binary = (y_continuous > 0).astype(int)
    return X, y_binary, y_continuous, feature_names

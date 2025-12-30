import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import requests

def pull_polygon_data(ticker, start, end, api_key):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}?apiKey={api_key}"
    response = requests.get(url)
    data = response.json()

    if 'results' not in data or len(data['results']) < 2:
        raise ValueError("not enough data")

    df = pd.DataFrame(data['results'])
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    df = df[['timestamp','open','high','low','close','volume']]
    return df

def calculate_features(df):
    df = df.copy()

    df['momentum_1min'] = df['close'].pct_change()
    df['volatility_1min'] = df['momentum_1min'] ** 2
    df['price_direction'] = (df['close'] > df['open']).astype(int)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_dev'] = (df['close'] - df['vwap']) / df['vwap']
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['next_return'] = df['close'].shift(-1) / df['close'] - 1

    df = df.dropna()
    return df

def backtest(model, df, prob_threshold=0.55, transaction_cost=0.0001):
    features = ['momentum_1min', 'volatility_1min', 'price_direction', 'vwap_dev', 'hour', 'minute']
    X = df[features]

    # get predictions
    pred_proba = model.predict_proba(X)

    # convert to signals: 1=buy, -1=sell, 0=hold
    signals = np.zeros(len(pred_proba))
    signals[pred_proba[:, 1] > prob_threshold] = 1  # buy
    signals[pred_proba[:, 0] > prob_threshold] = -1  # sell

    # calculate returns
    df['signal'] = signals
    df['position'] = df['signal']  # simplified: instant entry/exit
    df['strategy_return'] = df['position'] * df['next_return'] - np.abs(df['position'].diff()) * transaction_cost
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    df['buy_hold_return'] = (1 + df['next_return']).cumprod()

    return df

def calculate_metrics(backtest_df):
    returns = backtest_df['strategy_return'].dropna()

    # basic stats
    total_return = backtest_df['cumulative_return'].iloc[-1] - 1
    buy_hold = backtest_df['buy_hold_return'].iloc[-1] - 1

    # sharpe ratio (annualized, assuming 252*390 minute bars per year)
    sharpe = returns.mean() / returns.std() * np.sqrt(252 * 390) if returns.std() > 0 else 0

    # max drawdown
    cumulative = backtest_df['cumulative_return']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # win rate
    trades = returns[returns != 0]
    win_rate = (trades > 0).sum() / len(trades) if len(trades) > 0 else 0

    # profit factor
    gross_profit = trades[trades > 0].sum()
    gross_loss = abs(trades[trades < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # number of trades
    position_changes = backtest_df['position'].diff().fillna(0)
    num_trades = (position_changes != 0).sum()

    metrics = {
        'total_return': total_return,
        'buy_hold_return': buy_hold,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'num_trades': num_trades,
        'avg_trade_return': trades.mean() if len(trades) > 0 else 0
    }

    return metrics

if __name__ == "__main__":
    # load model
    model = joblib.load("trained_stock_model.pkl")

    # backtest params
    API_KEY = "vFDjkUVRfPnedLrbRjm75BZ9CJHz3dfv"
    TICKER = "AAPL"
    START_DATE = "2025-10-01"
    END_DATE = "2025-11-01"

    print(f"backtesting {TICKER} from {START_DATE} to {END_DATE}")

    # pull data and prepare features
    df = pull_polygon_data(TICKER, START_DATE, END_DATE, API_KEY)
    df = calculate_features(df)

    # chronological split (use last 20% for testing)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    # run backtest
    results = backtest(model, test_df, prob_threshold=0.55)
    metrics = calculate_metrics(results)

    # print results
    print("\nbacktest results:")
    print(f"total return: {metrics['total_return']*100:.2f}%")
    print(f"buy & hold: {metrics['buy_hold_return']*100:.2f}%")
    print(f"sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"max drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"win rate: {metrics['win_rate']*100:.1f}%")
    print(f"profit factor: {metrics['profit_factor']:.2f}")
    print(f"num trades: {metrics['num_trades']}")
    print(f"avg trade return: {metrics['avg_trade_return']*100:.4f}%")

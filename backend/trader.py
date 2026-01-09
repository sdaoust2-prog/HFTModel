import os
import time
import joblib
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import trader_config as config

load_dotenv()

if config.USE_REALISTIC_COSTS or config.ENABLE_TAX_TRACKING:
    from realistic_costs import RealisticCostModel, TaxTracker

class TradingBot:
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')

        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)

        self.model_path = os.path.join(os.path.dirname(__file__), "..", "trained_stock_model.pkl")
        self.model = joblib.load(self.model_path)
        self.model_last_modified = os.path.getmtime(self.model_path)
        self.last_model_check = datetime.now()

        self.positions = {}
        self.daily_pnl = 0.0
        self.start_cash = None

        if config.USE_REALISTIC_COSTS:
            self.cost_model = RealisticCostModel()
        else:
            self.cost_model = None

        if config.ENABLE_TAX_TRACKING:
            self.tax_tracker = TaxTracker()
        else:
            self.tax_tracker = None

        print(f"bot initialized at {datetime.now()}")
        print(f"watchlist: {config.WATCHLIST}")
        print(f"max positions: {config.MAX_POSITIONS}")
        print(f"position size: ${config.POSITION_SIZE_USD}")
        print(f"shorting: {'ENABLED' if config.ALLOW_SHORTING else 'DISABLED'}")
        print(f"model auto-reload: every {config.MODEL_RELOAD_INTERVAL_HOURS} hours")
        print(f"realistic costs: {'ENABLED' if config.USE_REALISTIC_COSTS else 'DISABLED'}")
        print(f"tax tracking: {'ENABLED' if config.ENABLE_TAX_TRACKING else 'DISABLED'}")

    def get_account_info(self):
        account = self.trading_client.get_account()
        if self.start_cash is None:
            self.start_cash = float(account.cash)
        return account

    def check_and_reload_model(self):
        hours_since_check = (datetime.now() - self.last_model_check).total_seconds() / 3600

        if hours_since_check < config.MODEL_RELOAD_INTERVAL_HOURS:
            return False

        self.last_model_check = datetime.now()

        try:
            current_mtime = os.path.getmtime(self.model_path)

            if current_mtime > self.model_last_modified:
                print(f"\n{'='*60}")
                print("NEW MODEL DETECTED - RELOADING")
                print(f"{'='*60}")

                self.model = joblib.load(self.model_path)
                self.model_last_modified = current_mtime

                print(f"Model reloaded at {datetime.now()}")
                print(f"{'='*60}\n")

                return True

        except Exception as e:
            print(f"error reloading model: {e}")

        return False

    def get_market_data(self, ticker, lookback_minutes=60):
        end = datetime.now()
        start = end - timedelta(minutes=lookback_minutes)

        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end
        )

        bars = self.data_client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            return None

        df = df.reset_index()
        df = df.rename(columns={'symbol': 'ticker'})
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    def calculate_features(self, df):
        if len(df) < 2:
            return None

        last_two = df.iloc[-2:]

        momentum_1min = (last_two['close'].iloc[1] - last_two['close'].iloc[0]) / last_two['close'].iloc[0]
        volatility_1min = momentum_1min ** 2
        price_direction = int(last_two['close'].iloc[1] > last_two['open'].iloc[1])

        vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        vwap_dev = (last_two['close'].iloc[1] - vwap.iloc[-1]) / vwap.iloc[-1]

        hour = last_two['timestamp'].iloc[1].hour
        minute = last_two['timestamp'].iloc[1].minute

        returns = df['close'].pct_change()
        autocorr = returns.rolling(5).apply(lambda x: x.autocorr() if len(x) >= 2 else 0, raw=False).iloc[-1]

        rolling_mean = returns.rolling(20).mean().iloc[-1]
        rolling_std = returns.rolling(20).std().iloc[-1]
        rolling_sharpe = (rolling_mean / rolling_std) * (20 ** 0.5) if rolling_std > 0 else 0

        price_accel = returns.diff().iloc[-1]

        price_change = df['close'].diff()
        order_flow = (price_change * df['volume'] / df['volume'].rolling(20).mean()).iloc[-1]

        features = {
            'momentum_1min': momentum_1min,
            'volatility_1min': volatility_1min,
            'price_direction': price_direction,
            'vwap_dev': vwap_dev,
            'hour': hour,
            'minute': minute,
            'autocorr_5': autocorr,
            'rolling_sharpe_20': rolling_sharpe,
            'price_accel': price_accel,
            'order_flow': order_flow
        }

        return features, last_two['close'].iloc[1]

    def get_prediction(self, ticker):
        try:
            df = self.get_market_data(ticker)
            if df is None or len(df) < 20:
                return None, None, None

            result = self.calculate_features(df)
            if result is None:
                return None, None, None

            features, current_price = result

            feature_row = pd.DataFrame([features])
            pred_proba = self.model.predict_proba(feature_row)[0]

            return pred_proba[1], pred_proba[0], current_price

        except Exception as e:
            print(f"prediction error for {ticker}: {e}")
            return None, None, None

    def get_current_positions(self):
        try:
            positions = self.trading_client.get_all_positions()
            pos_dict = {}
            for pos in positions:
                pos_dict[pos.symbol] = {
                    'qty': float(pos.qty),
                    'entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'pnl': float(pos.unrealized_pl),
                    'pnl_pct': float(pos.unrealized_plpc)
                }
            return pos_dict
        except Exception as e:
            print(f"error getting positions: {e}")
            return {}

    def place_order(self, ticker, side, qty, price=None):
        try:
            order_data = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            order = self.trading_client.submit_order(order_data)

            if self.cost_model and price:
                costs = self.cost_model.calculate_total_cost(
                    ticker=ticker,
                    qty=qty,
                    price=price,
                    side='BUY' if side == OrderSide.BUY else 'SELL',
                    timestamp=datetime.now()
                )
                print(f"order placed: {side} {qty} {ticker} @ ${price:.2f}")
                print(f"  estimated cost: ${costs['total_cost']:.2f} ({costs['cost_percentage']*100:.3f}%)")
            else:
                print(f"order placed: {side} {qty} {ticker} (order id: {order.id})")

            if self.tax_tracker and side == OrderSide.BUY:
                self.tax_tracker.open_position(ticker, qty, price or 0, datetime.now(), side='LONG')
            elif self.tax_tracker and side == OrderSide.SELL:
                positions = self.get_current_positions()
                if ticker not in positions:
                    self.tax_tracker.open_position(ticker, qty, price or 0, datetime.now(), side='SHORT')

            return order
        except Exception as e:
            print(f"order failed for {ticker}: {e}")
            return None

    def check_risk_limits(self):
        account = self.get_account_info()
        current_cash = float(account.cash)
        daily_change = self.start_cash - current_cash

        if daily_change > config.MAX_DAILY_LOSS:
            print(f"daily loss limit hit: ${daily_change:.2f}")
            return False

        positions = self.get_current_positions()
        if len(positions) >= config.MAX_POSITIONS:
            print(f"max positions reached: {len(positions)}")
            return False

        return True

    def check_exit_signals(self):
        positions = self.get_current_positions()

        for ticker, pos in positions.items():
            pnl_pct = pos['pnl_pct']
            is_long = pos['qty'] > 0

            if pnl_pct <= -config.STOP_LOSS_PCT:
                print(f"stop loss triggered for {ticker}: {pnl_pct*100:.2f}%")
                close_side = OrderSide.SELL if is_long else OrderSide.BUY
                self.place_order(ticker, close_side, int(abs(pos['qty'])), pos['current_price'])

                if self.tax_tracker:
                    tax_result = self.tax_tracker.close_position(ticker, abs(pos['qty']),
                                                                 pos['current_price'], datetime.now())
                    if tax_result:
                        print(f"  tax liability: ${tax_result['tax_owed']:.2f} " +
                              f"({tax_result['tax_type']}, net: ${tax_result['net_gain']:.2f})")

            elif pnl_pct >= config.TAKE_PROFIT_PCT:
                print(f"take profit triggered for {ticker}: {pnl_pct*100:.2f}%")
                close_side = OrderSide.SELL if is_long else OrderSide.BUY
                self.place_order(ticker, close_side, int(abs(pos['qty'])), pos['current_price'])

                if self.tax_tracker:
                    tax_result = self.tax_tracker.close_position(ticker, abs(pos['qty']),
                                                                 pos['current_price'], datetime.now())
                    if tax_result:
                        print(f"  tax liability: ${tax_result['tax_owed']:.2f} " +
                              f"({tax_result['tax_type']}, net: ${tax_result['net_gain']:.2f})")

    def execute_strategy(self, ticker):
        current_positions = self.get_current_positions()

        if ticker in current_positions:
            return

        if not self.check_risk_limits():
            return

        prob_up, prob_down, current_price = self.get_prediction(ticker)

        if prob_up is None:
            return

        if prob_up > config.PREDICTION_THRESHOLD:
            qty = int(config.POSITION_SIZE_USD / current_price)
            if qty > 0:
                print(f"{ticker}: BUY signal (confidence: {prob_up:.3f}, price: ${current_price:.2f})")
                self.place_order(ticker, OrderSide.BUY, qty, current_price)

        elif prob_down > config.PREDICTION_THRESHOLD:
            if config.ALLOW_SHORTING:
                qty = int(config.POSITION_SIZE_USD / current_price)
                if qty > 0:
                    print(f"{ticker}: SHORT signal (confidence: {prob_down:.3f}, price: ${current_price:.2f})")
                    self.place_order(ticker, OrderSide.SELL, qty, current_price)
            else:
                print(f"{ticker}: SELL signal ignored (shorting disabled), confidence: {prob_down:.3f}")

    def is_market_open(self):
        now = datetime.now()
        current_time = now.time()

        if now.weekday() >= 5:
            return False

        market_open = datetime.strptime(f"{config.MARKET_OPEN_HOUR}:{config.MARKET_OPEN_MINUTE}", "%H:%M").time()
        market_close = datetime.strptime(f"{config.MARKET_CLOSE_HOUR}:{config.MARKET_CLOSE_MINUTE}", "%H:%M").time()

        return market_open <= current_time <= market_close

    def print_status(self):
        account = self.get_account_info()
        positions = self.get_current_positions()

        total_equity = float(account.equity)
        cash = float(account.cash)
        pnl = total_equity - self.start_cash if self.start_cash else 0

        print(f"\n{'='*60}")
        print(f"status update: {datetime.now()}")
        print(f"{'='*60}")
        print(f"equity: ${total_equity:,.2f} | cash: ${cash:,.2f} | pnl: ${pnl:+,.2f}")
        print(f"positions: {len(positions)}/{config.MAX_POSITIONS}")

        if positions:
            print(f"\nopen positions:")
            for ticker, pos in positions.items():
                direction = "LONG" if pos['qty'] > 0 else "SHORT"
                print(f"  {ticker} {direction}: {abs(pos['qty'])} shares @ ${pos['entry_price']:.2f} | "
                      f"current: ${pos['current_price']:.2f} | pnl: ${pos['pnl']:+.2f} ({pos['pnl_pct']*100:+.2f}%)")

                if self.cost_model:
                    cost_info = self.cost_model.get_typical_cost_bps(ticker)
                    print(f"    typical costs: {cost_info['total_bps']:.1f} bps")

        if self.tax_tracker:
            tax_summary = self.tax_tracker.get_tax_summary()
            if tax_summary['num_trades'] > 0:
                print(f"\ntax summary:")
                print(f"  realized gain: ${tax_summary['total_realized_gain']:,.2f}")
                print(f"  tax owed: ${tax_summary['total_tax_owed']:,.2f}")
                print(f"  net after tax: ${tax_summary['net_after_tax']:,.2f}")

        print(f"{'='*60}\n")

    def run(self, skip_hours_check=False):
        print("starting trading bot...")

        while True:
            try:
                if not skip_hours_check and not self.is_market_open():
                    print("market closed, waiting...")
                    time.sleep(300)
                    continue

                self.check_and_reload_model()

                self.check_exit_signals()

                for ticker in config.WATCHLIST:
                    self.execute_strategy(ticker)
                    time.sleep(1)

                self.print_status()

                print(f"sleeping for {config.TRADE_INTERVAL_SECONDS}s...")
                time.sleep(config.TRADE_INTERVAL_SECONDS)

            except KeyboardInterrupt:
                print("\nbot stopped by user")
                self.print_status()
                break
            except Exception as e:
                print(f"error in main loop: {e}")
                time.sleep(10)

if __name__ == "__main__":
    import sys
    skip_hours = '--test' in sys.argv

    bot = TradingBot()
    if skip_hours:
        print("TEST MODE: skipping market hours check")
    bot.run(skip_hours_check=skip_hours)

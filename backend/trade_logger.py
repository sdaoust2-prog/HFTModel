import sqlite3
import pandas as pd
from datetime import datetime
import os

class TradeLogger:
    def __init__(self, db_path='trading.db'):
        self.db_path = os.path.join(os.path.dirname(__file__), db_path)
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                ticker TEXT,
                side TEXT,
                qty INTEGER,
                price REAL,
                order_id TEXT,
                strategy_signal TEXT,
                confidence REAL,
                features TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_timestamp DATETIME,
                exit_timestamp DATETIME,
                ticker TEXT,
                side TEXT,
                qty INTEGER,
                entry_price REAL,
                exit_price REAL,
                entry_order_id TEXT,
                exit_order_id TEXT,
                pnl REAL,
                pnl_pct REAL,
                hold_time_minutes REAL,
                exit_reason TEXT,
                max_favorable_excursion REAL,
                max_adverse_excursion REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_performance (
                date DATE PRIMARY KEY,
                starting_equity REAL,
                ending_equity REAL,
                num_trades INTEGER,
                num_wins INTEGER,
                num_losses INTEGER,
                total_pnl REAL,
                gross_profit REAL,
                gross_loss REAL,
                win_rate REAL,
                profit_factor REAL,
                sharpe_ratio REAL,
                max_drawdown REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                ticker TEXT,
                prob_up REAL,
                prob_down REAL,
                actual_return_1min REAL,
                actual_return_5min REAL,
                prediction_correct BOOLEAN
            )
        ''')

        conn.commit()
        conn.close()

    def log_trade(self, ticker, side, qty, price, order_id=None, strategy_signal=None, confidence=None, features=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO trades (timestamp, ticker, side, qty, price, order_id, strategy_signal, confidence, features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now(), ticker, side, qty, price, order_id, strategy_signal, confidence, str(features)))

        conn.commit()
        conn.close()

    def log_position(self, ticker, side, qty, entry_price, exit_price, entry_time, exit_time,
                    entry_order_id=None, exit_order_id=None, exit_reason=None, mfe=None, mae=None):
        pnl = (exit_price - entry_price) * qty if side == 'BUY' else (entry_price - exit_price) * qty
        pnl_pct = pnl / (entry_price * qty)
        hold_time = (exit_time - entry_time).total_seconds() / 60

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO positions (entry_timestamp, exit_timestamp, ticker, side, qty,
                                 entry_price, exit_price, entry_order_id, exit_order_id,
                                 pnl, pnl_pct, hold_time_minutes, exit_reason,
                                 max_favorable_excursion, max_adverse_excursion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (entry_time, exit_time, ticker, side, qty, entry_price, exit_price,
              entry_order_id, exit_order_id, pnl, pnl_pct, hold_time, exit_reason, mfe, mae))

        conn.commit()
        conn.close()

    def log_daily_performance(self, date, starting_equity, ending_equity, metrics):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO daily_performance
            (date, starting_equity, ending_equity, num_trades, num_wins, num_losses,
             total_pnl, gross_profit, gross_loss, win_rate, profit_factor, sharpe_ratio, max_drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date, starting_equity, ending_equity,
              metrics.get('num_trades', 0), metrics.get('num_wins', 0), metrics.get('num_losses', 0),
              metrics.get('total_pnl', 0), metrics.get('gross_profit', 0), metrics.get('gross_loss', 0),
              metrics.get('win_rate', 0), metrics.get('profit_factor', 0),
              metrics.get('sharpe_ratio', 0), metrics.get('max_drawdown', 0)))

        conn.commit()
        conn.close()

    def log_prediction(self, ticker, prob_up, prob_down, actual_return_1min=None, actual_return_5min=None):
        prediction_correct = None
        if actual_return_1min is not None:
            prediction_correct = (prob_up > 0.5 and actual_return_1min > 0) or (prob_down > 0.5 and actual_return_1min < 0)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO predictions (timestamp, ticker, prob_up, prob_down, actual_return_1min,
                                   actual_return_5min, prediction_correct)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now(), ticker, prob_up, prob_down, actual_return_1min, actual_return_5min, prediction_correct))

        conn.commit()
        conn.close()

    def get_all_trades(self, ticker=None, start_date=None, end_date=None):
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def get_all_positions(self, ticker=None, start_date=None, end_date=None):
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM positions WHERE 1=1"
        params = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if start_date:
            query += " AND entry_timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND exit_timestamp <= ?"
            params.append(end_date)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def get_performance_summary(self, start_date=None, end_date=None):
        positions = self.get_all_positions(start_date=start_date, end_date=end_date)

        if positions.empty:
            return {}

        total_trades = len(positions)
        wins = positions[positions['pnl'] > 0]
        losses = positions[positions['pnl'] < 0]

        summary = {
            'total_trades': total_trades,
            'num_wins': len(wins),
            'num_losses': len(losses),
            'win_rate': len(wins) / total_trades if total_trades > 0 else 0,
            'total_pnl': positions['pnl'].sum(),
            'avg_win': wins['pnl'].mean() if not wins.empty else 0,
            'avg_loss': losses['pnl'].mean() if not losses.empty else 0,
            'largest_win': wins['pnl'].max() if not wins.empty else 0,
            'largest_loss': losses['pnl'].min() if not losses.empty else 0,
            'profit_factor': wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty and losses['pnl'].sum() != 0 else 0,
            'avg_hold_time_minutes': positions['hold_time_minutes'].mean(),
            'total_long_trades': len(positions[positions['side'] == 'BUY']),
            'total_short_trades': len(positions[positions['side'] == 'SELL']),
            'long_pnl': positions[positions['side'] == 'BUY']['pnl'].sum(),
            'short_pnl': positions[positions['side'] == 'SELL']['pnl'].sum()
        }

        return summary

    def get_prediction_accuracy(self, start_date=None, end_date=None):
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM predictions WHERE prediction_correct IS NOT NULL"
        params = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            return {}

        return {
            'total_predictions': len(df),
            'correct_predictions': df['prediction_correct'].sum(),
            'accuracy': df['prediction_correct'].mean(),
            'avg_confidence': df[['prob_up', 'prob_down']].max(axis=1).mean()
        }

    def export_to_csv(self, output_dir='exports'):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        trades = self.get_all_trades()
        positions = self.get_all_positions()

        trades.to_csv(f'{output_dir}/trades_{timestamp}.csv', index=False)
        positions.to_csv(f'{output_dir}/positions_{timestamp}.csv', index=False)

        print(f"exported to {output_dir}/")

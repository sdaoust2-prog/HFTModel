import os
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide
from dotenv import load_dotenv
import trader_config as config

load_dotenv()

class PositionManager:
    def __init__(self):
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.client = TradingClient(api_key, secret_key, paper=True)

    def get_all_positions(self):
        positions = self.client.get_all_positions()
        pos_list = []
        for pos in positions:
            pos_list.append({
                'ticker': pos.symbol,
                'qty': float(pos.qty),
                'side': 'LONG' if float(pos.qty) > 0 else 'SHORT',
                'entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pnl': float(pos.unrealized_pl),
                'unrealized_pnl_pct': float(pos.unrealized_plpc),
                'cost_basis': float(pos.cost_basis)
            })
        return pos_list

    def print_position_summary(self):
        positions = self.get_all_positions()

        if not positions:
            print("no open positions")
            return

        print(f"\n{'='*80}")
        print(f"POSITION SUMMARY - {datetime.now()}")
        print(f"{'='*80}")

        total_value = sum(p['market_value'] for p in positions)
        total_pnl = sum(p['unrealized_pnl'] for p in positions)

        print(f"\nTotal Positions: {len(positions)}")
        print(f"Total Market Value: ${total_value:,.2f}")
        print(f"Total Unrealized P&L: ${total_pnl:+,.2f}\n")

        print(f"{'Ticker':<8} {'Side':<6} {'Qty':<6} {'Entry':<10} {'Current':<10} {'P&L':<12} {'P&L %':<8}")
        print("-" * 80)

        for pos in sorted(positions, key=lambda x: abs(x['unrealized_pnl']), reverse=True):
            print(f"{pos['ticker']:<8} {pos['side']:<6} {abs(pos['qty']):<6.0f} "
                  f"${pos['entry_price']:<9.2f} ${pos['current_price']:<9.2f} "
                  f"${pos['unrealized_pnl']:>+10.2f} {pos['unrealized_pnl_pct']*100:>+6.2f}%")

        print(f"{'='*80}\n")

        winners = [p for p in positions if p['unrealized_pnl'] > 0]
        losers = [p for p in positions if p['unrealized_pnl'] < 0]

        print(f"Winners: {len(winners)} (avg: +${sum(p['unrealized_pnl'] for p in winners)/len(winners):.2f})"
              if winners else "Winners: 0")
        print(f"Losers: {len(losers)} (avg: -${abs(sum(p['unrealized_pnl'] for p in losers)/len(losers)):.2f})"
              if losers else "Losers: 0")

    def close_position(self, ticker):
        try:
            position = self.client.get_open_position(ticker)
            qty = abs(float(position.qty))
            side = OrderSide.SELL if float(position.qty) > 0 else OrderSide.BUY

            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import TimeInForce

            order = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )

            result = self.client.submit_order(order)
            print(f"closed {ticker}: {side} {qty} shares (order: {result.id})")
            return True
        except Exception as e:
            print(f"error closing {ticker}: {e}")
            return False

    def close_all_positions(self):
        positions = self.get_all_positions()
        print(f"\nclosing {len(positions)} positions...")

        closed = 0
        for pos in positions:
            if self.close_position(pos['ticker']):
                closed += 1

        print(f"\nclosed {closed}/{len(positions)} positions")

    def close_losers(self, min_loss_pct=0.02):
        positions = self.get_all_positions()
        losers = [p for p in positions if p['unrealized_pnl_pct'] <= -min_loss_pct]

        print(f"\nclosing {len(losers)} positions with loss >= {min_loss_pct*100}%...")

        for pos in losers:
            print(f"  {pos['ticker']}: {pos['unrealized_pnl_pct']*100:.2f}%")
            self.close_position(pos['ticker'])

    def close_winners(self, min_profit_pct=0.03):
        positions = self.get_all_positions()
        winners = [p for p in positions if p['unrealized_pnl_pct'] >= min_profit_pct]

        print(f"\nclosing {len(winners)} positions with profit >= {min_profit_pct*100}%...")

        for pos in winners:
            print(f"  {pos['ticker']}: {pos['unrealized_pnl_pct']*100:.2f}%")
            self.close_position(pos['ticker'])

    def check_position_age(self):
        account = self.client.get_account()

        print(f"\nAccount Status:")
        print(f"  Buying Power: ${float(account.buying_power):,.2f}")
        print(f"  Cash: ${float(account.cash):,.2f}")
        print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"  Equity: ${float(account.equity):,.2f}")

if __name__ == "__main__":
    import sys

    pm = PositionManager()

    if len(sys.argv) == 1:
        pm.print_position_summary()
        pm.check_position_age()

    elif sys.argv[1] == 'close-all':
        pm.print_position_summary()
        confirm = input("\nclose ALL positions? (yes/no): ")
        if confirm.lower() == 'yes':
            pm.close_all_positions()

    elif sys.argv[1] == 'close-losers':
        loss_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
        pm.print_position_summary()
        pm.close_losers(min_loss_pct=loss_pct)

    elif sys.argv[1] == 'close-winners':
        profit_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
        pm.print_position_summary()
        pm.close_winners(min_profit_pct=profit_pct)

    elif sys.argv[1] == 'close':
        ticker = sys.argv[2]
        pm.close_position(ticker)

    else:
        print("usage:")
        print("  python position_manager.py                    # show positions")
        print("  python position_manager.py close-all          # close all positions")
        print("  python position_manager.py close-losers 0.02  # close losers >= 2%")
        print("  python position_manager.py close-winners 0.03 # close winners >= 3%")
        print("  python position_manager.py close TICKER       # close specific ticker")

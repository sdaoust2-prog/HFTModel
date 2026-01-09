import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class AlertSystem:
    def __init__(self):
        self.email_enabled = os.getenv('ALERT_EMAIL_ENABLED', 'false').lower() == 'true'
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.sender_email = os.getenv('SENDER_EMAIL', '')
        self.sender_password = os.getenv('SENDER_PASSWORD', '')
        self.recipient_email = os.getenv('RECIPIENT_EMAIL', '')

        self.alert_history = []

    def send_email(self, subject, body):
        if not self.email_enabled:
            print(f"email alerts disabled. would have sent: {subject}")
            return False

        if not self.sender_email or not self.recipient_email:
            print("email not configured")
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()

            print(f"email sent: {subject}")
            return True

        except Exception as e:
            print(f"failed to send email: {e}")
            return False

    def alert_trade_executed(self, ticker, side, qty, price, confidence):
        subject = f"Trade Executed: {side} {qty} {ticker} @ ${price:.2f}"
        body = f"""
Trading Alert
=============

Trade Executed:
  Ticker: {ticker}
  Side: {side}
  Quantity: {qty}
  Price: ${price:.2f}
  Confidence: {confidence:.1%}
  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.log_alert('trade_executed', subject)
        self.send_email(subject, body)

    def alert_position_closed(self, ticker, side, qty, entry_price, exit_price, pnl, pnl_pct, reason):
        subject = f"Position Closed: {ticker} - P&L: ${pnl:+.2f} ({pnl_pct*100:+.1f}%)"
        body = f"""
Trading Alert
=============

Position Closed:
  Ticker: {ticker}
  Side: {side}
  Quantity: {qty}
  Entry Price: ${entry_price:.2f}
  Exit Price: ${exit_price:.2f}
  P&L: ${pnl:+.2f} ({pnl_pct*100:+.2f}%)
  Exit Reason: {reason}
  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.log_alert('position_closed', subject)
        self.send_email(subject, body)

    def alert_daily_summary(self, num_trades, wins, losses, total_pnl, ending_equity):
        win_rate = wins / num_trades * 100 if num_trades > 0 else 0

        subject = f"Daily Summary - P&L: ${total_pnl:+.2f} ({num_trades} trades)"
        body = f"""
Daily Trading Summary
====================

Date: {datetime.now().strftime('%Y-%m-%d')}

Performance:
  Total Trades: {num_trades}
  Wins: {wins}
  Losses: {losses}
  Win Rate: {win_rate:.1f}%
  Total P&L: ${total_pnl:+.2f}
  Ending Equity: ${ending_equity:,.2f}

Time: {datetime.now().strftime('%H:%M:%S')}
"""
        self.log_alert('daily_summary', subject)
        self.send_email(subject, body)

    def alert_risk_warning(self, warning_type, message, details=None):
        subject = f"Risk Warning: {warning_type}"
        body = f"""
Risk Warning Alert
=================

Type: {warning_type}
Message: {message}

"""
        if details:
            body += "Details:\n"
            for key, value in details.items():
                body += f"  {key}: {value}\n"

        body += f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        self.log_alert('risk_warning', subject)
        self.send_email(subject, body)

    def alert_system_error(self, error_type, error_message, traceback=None):
        subject = f"System Error: {error_type}"
        body = f"""
System Error Alert
=================

Type: {error_type}
Error: {error_message}

"""
        if traceback:
            body += f"Traceback:\n{traceback}\n"

        body += f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        self.log_alert('system_error', subject)
        self.send_email(subject, body)

    def alert_large_move(self, ticker, direction, move_pct, current_price):
        subject = f"Large Move Alert: {ticker} {direction} {move_pct:.1f}%"
        body = f"""
Large Price Move Alert
=====================

Ticker: {ticker}
Direction: {direction}
Move: {move_pct:.2f}%
Current Price: ${current_price:.2f}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.log_alert('large_move', subject)
        self.send_email(subject, body)

    def alert_daily_loss_limit(self, daily_pnl, limit, current_equity):
        subject = f"Daily Loss Limit Approaching - P&L: ${daily_pnl:+.2f}"
        body = f"""
Daily Loss Limit Alert
=====================

Current Daily P&L: ${daily_pnl:+.2f}
Daily Loss Limit: ${limit:.2f}
Current Equity: ${current_equity:,.2f}

Trading has been paused for the day.

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.log_alert('daily_loss_limit', subject)
        self.send_email(subject, body)

    def log_alert(self, alert_type, message):
        self.alert_history.append({
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message
        })

        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

    def get_alert_history(self, limit=50):
        return self.alert_history[-limit:]

if __name__ == "__main__":
    alerts = AlertSystem()

    print("testing alert system...")

    alerts.alert_trade_executed('AAPL', 'BUY', 10, 180.50, 0.67)
    alerts.alert_position_closed('AAPL', 'BUY', 10, 180.50, 185.20, 47.00, 0.026, 'take_profit')
    alerts.alert_daily_summary(15, 9, 6, 234.50, 102345.67)
    alerts.alert_risk_warning('Portfolio Heat', 'Portfolio heat at 1.8%, approaching 2% limit',
                             {'current_heat': '1.8%', 'limit': '2.0%'})

    print(f"\nalert history ({len(alerts.get_alert_history())} alerts):")
    for alert in alerts.get_alert_history():
        print(f"  {alert['timestamp']}: {alert['type']} - {alert['message']}")

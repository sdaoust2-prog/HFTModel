from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os
from trade_logger import TradeLogger
from performance_tracker import PerformanceTracker

app = FastAPI(title="Trading Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = TradeLogger()
tracker = PerformanceTracker(logger)

class PerformanceMetrics(BaseModel):
    total_trades: int
    num_wins: int
    num_losses: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    starting_capital: float
    ending_capital: float

class Position(BaseModel):
    id: int
    ticker: str
    side: str
    qty: int
    entry_price: float
    exit_price: Optional[float]
    pnl: Optional[float]
    entry_timestamp: str
    exit_timestamp: Optional[str]

@app.get("/")
async def dashboard():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                background: #0a0e27;
                color: #e0e0e0;
                padding: 20px;
            }
            .container { max-width: 1400px; margin: 0 auto; }
            h1 {
                font-size: 2.5em;
                margin-bottom: 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }
            .metric-card {
                background: linear-gradient(135deg, #1e2139 0%, #2a2f4f 100%);
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                border: 1px solid rgba(255,255,255,0.05);
            }
            .metric-label {
                font-size: 0.9em;
                color: #888;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #fff;
            }
            .positive { color: #4ade80; }
            .negative { color: #f87171; }
            table {
                width: 100%;
                background: linear-gradient(135deg, #1e2139 0%, #2a2f4f 100%);
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            }
            th, td {
                padding: 15px;
                text-align: left;
                border-bottom: 1px solid rgba(255,255,255,0.05);
            }
            th {
                background: rgba(102, 126, 234, 0.1);
                color: #667eea;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.85em;
                letter-spacing: 1px;
            }
            tr:hover { background: rgba(255,255,255,0.02); }
            .refresh-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 1em;
                font-weight: 600;
                margin-bottom: 20px;
                transition: transform 0.2s;
            }
            .refresh-btn:hover { transform: translateY(-2px); }
            .section-title {
                font-size: 1.5em;
                margin: 40px 0 20px 0;
                color: #667eea;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Trading Dashboard</h1>
            <button class="refresh-btn" onclick="loadData()">Refresh Data</button>

            <div class="metrics-grid" id="metrics"></div>

            <h2 class="section-title">Recent Positions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Side</th>
                        <th>Qty</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>P&L</th>
                        <th>Entry Time</th>
                        <th>Exit Time</th>
                    </tr>
                </thead>
                <tbody id="positions"></tbody>
            </table>
        </div>

        <script>
            async function loadData() {
                try {
                    const metricsRes = await fetch('/api/performance');
                    const metrics = await metricsRes.json();

                    const positionsRes = await fetch('/api/positions/recent');
                    const positions = await positionsRes.json();

                    displayMetrics(metrics);
                    displayPositions(positions);
                } catch (error) {
                    console.error('Error loading data:', error);
                }
            }

            function displayMetrics(metrics) {
                const metricsDiv = document.getElementById('metrics');
                metricsDiv.innerHTML = `
                    <div class="metric-card">
                        <div class="metric-label">Total P&L</div>
                        <div class="metric-value ${metrics.total_pnl >= 0 ? 'positive' : 'negative'}">
                            $${metrics.total_pnl.toFixed(2)}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">${(metrics.win_rate * 100).toFixed(1)}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Trades</div>
                        <div class="metric-value">${metrics.total_trades}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">${metrics.sharpe_ratio.toFixed(2)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value negative">${(metrics.max_drawdown * 100).toFixed(2)}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Equity</div>
                        <div class="metric-value">$${metrics.ending_capital.toFixed(2)}</div>
                    </div>
                `;
            }

            function displayPositions(positions) {
                const tbody = document.getElementById('positions');
                tbody.innerHTML = positions.map(p => `
                    <tr>
                        <td>${p.ticker}</td>
                        <td>${p.side}</td>
                        <td>${p.qty}</td>
                        <td>$${p.entry_price.toFixed(2)}</td>
                        <td>${p.exit_price ? '$' + p.exit_price.toFixed(2) : '-'}</td>
                        <td class="${p.pnl >= 0 ? 'positive' : 'negative'}">
                            ${p.pnl ? '$' + p.pnl.toFixed(2) : '-'}
                        </td>
                        <td>${new Date(p.entry_timestamp).toLocaleString()}</td>
                        <td>${p.exit_timestamp ? new Date(p.exit_timestamp).toLocaleString() : 'Open'}</td>
                    </tr>
                `).join('');
            }

            loadData();
            setInterval(loadData, 10000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/performance")
async def get_performance():
    report = tracker.generate_report()
    return report

@app.get("/api/positions/recent")
async def get_recent_positions(limit: int = 50):
    positions = logger.get_all_positions()

    if positions.empty:
        return []

    positions = positions.sort_values('entry_timestamp', ascending=False).head(limit)
    return positions.to_dict('records')

@app.get("/api/trades/recent")
async def get_recent_trades(limit: int = 100):
    trades = logger.get_all_trades()

    if trades.empty:
        return []

    trades = trades.sort_values('timestamp', ascending=False).head(limit)
    return trades.to_dict('records')

@app.get("/api/equity_curve")
async def get_equity_curve():
    equity = tracker.get_equity_curve()
    return {
        'equity': equity.tolist(),
        'trades': len(equity)
    }

@app.get("/api/performance/by_ticker")
async def get_performance_by_ticker():
    analysis = tracker.analyze_by_ticker()

    if analysis.empty:
        return {}

    return analysis.to_dict('index')

@app.get("/api/performance/by_time")
async def get_performance_by_time():
    analysis = tracker.analyze_by_time_of_day()

    if analysis.empty:
        return {}

    return analysis.to_dict('index')

if __name__ == "__main__":
    import uvicorn
    print("starting dashboard on http://localhost:8001")
    uvicorn.run("dashboard_api:app", host="0.0.0.0", port=8001, reload=True)

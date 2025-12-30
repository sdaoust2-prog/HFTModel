from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import requests
import joblib
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Stock Predictor API",
    description="ML API for stock predictions",
    version="1.0.0"
)

# cors for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_stock_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
    print(f"model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"error loading model: {e}")
    model = None

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "vFDjkUVRfPnedLrbRjm75BZ9CJHz3dfv")

class PredictionResponse(BaseModel):
    ticker: str
    decision: str
    confidence: float
    timestamp: str
    current_price: float
    features: Dict[str, float]
    probabilities: Dict[str, float]

class StockDataResponse(BaseModel):
    ticker: str
    data: List[Dict]
    count: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    api_key_configured: bool

def pull_polygon_data(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}?apiKey={api_key}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'results' not in data or len(data['results']) < 2:
            raise ValueError(f"not enough data from polygon api")

        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')

        df = df.rename(columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        })

        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        return df

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"polygon api error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"data processing error: {str(e)}")


def calculate_features_for_prediction(df: pd.DataFrame) -> tuple:
    # need last 2 bars for momentum calc
    last_two = df.iloc[-2:]

    momentum_1min = (last_two['close'].iloc[1] - last_two['close'].iloc[0]) / last_two['close'].iloc[0]
    volatility_1min = momentum_1min ** 2
    price_direction = int(last_two['close'].iloc[1] > last_two['open'].iloc[1])

    # vwap deviation
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    vwap_dev = (last_two['close'].iloc[1] - vwap.iloc[-1]) / vwap.iloc[-1]

    hour = last_two['timestamp'].iloc[1].hour
    minute = last_two['timestamp'].iloc[1].minute

    features = {
        'momentum_1min': momentum_1min,
        'volatility_1min': volatility_1min,
        'price_direction': price_direction,
        'vwap_dev': vwap_dev,
        'hour': hour,
        'minute': minute
    }

    return features, last_two.iloc[1]

@app.get("/", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "api_key_configured": POLYGON_API_KEY != ""
    }


@app.get("/api/predict/{ticker}", response_model=PredictionResponse)
async def predict_stock(ticker: str, prob_threshold: float = 0.55):
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    ticker = ticker.upper().strip()
    if not ticker.isalpha() or len(ticker) > 5:
        raise HTTPException(status_code=400, detail="invalid ticker")

    try:
        today = datetime.now().date()
        start_date = today - timedelta(days=1)

        df = pull_polygon_data(ticker, str(start_date), str(today), POLYGON_API_KEY)
        features, last_bar = calculate_features_for_prediction(df)

        feature_row = pd.DataFrame([{
            'momentum_1min': features['momentum_1min'],
            'volatility_1min': features['volatility_1min'],
            'price_direction': features['price_direction'],
            'vwap_dev': features['vwap_dev'],
            'hour': features['hour'],
            'minute': features['minute']
        }])

        # returns [P(down), P(up)]
        pred_proba = model.predict_proba(feature_row)[0]

        if pred_proba[1] > prob_threshold:
            decision = "BUY"
            confidence = pred_proba[1]
        elif pred_proba[0] > prob_threshold:
            decision = "SELL"
            confidence = pred_proba[0]
        else:
            decision = "HOLD"
            confidence = max(pred_proba)

        return {
            "ticker": ticker,
            "decision": decision,
            "confidence": float(confidence),
            "timestamp": str(last_bar['timestamp']),
            "current_price": float(last_bar['close']),
            "features": {k: float(v) for k, v in features.items()},
            "probabilities": {
                "down": float(pred_proba[0]),
                "up": float(pred_proba[1])
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction error: {str(e)}")


@app.get("/api/data/{ticker}", response_model=StockDataResponse)
async def get_stock_data(ticker: str, days: int = 1):
    ticker = ticker.upper().strip()
    if not ticker.isalpha() or len(ticker) > 5:
        raise HTTPException(status_code=400, detail="invalid ticker")

    if days < 1 or days > 30:
        raise HTTPException(status_code=400, detail="days must be 1-30")

    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        df = pull_polygon_data(ticker, str(start_date), str(end_date), POLYGON_API_KEY)
        data = df.to_dict('records')

        for record in data:
            record['timestamp'] = str(record['timestamp'])

        return {
            "ticker": ticker,
            "data": data,
            "count": len(data)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"data error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

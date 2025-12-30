# dev notes

when working on this project keep code style natural, no excessive comments or ai patterns. write like a normal programmer would.

## project structure

stock predictor using ML + react frontend. pulls minute data from polygon.io api.

- backend: jupyter notebooks + fastapi server
- frontend: react/ts/tailwind

## running stuff

frontend:
```bash
cd ui_figma_stock_predictor
npm install
npm run dev
```

backend:
```bash
cd backend
python api.py
```

notebooks need pandas, sklearn, requests, joblib

## how it works

model uses 6 features:
- momentum_1min - pct change
- volatility_1min - squared momentum
- price_direction - close vs open
- vwap_dev - distance from vwap
- hour/minute - time features

random forest classifier, threshold at 0.55 for buy/sell signals

train/test split is chronological not random (prevents lookahead)

## key files

- trained_stock_model.pkl - saved model
- backend/api.py - fastapi server
- predictor.ipynb - main training notebook
- ui_figma_stock_predictor/src - react app

## frontend structure

```
src/
├── App.tsx
├── main.tsx
├── components/
└── pages/
```

vite for dev server, tailwind for styles

## api stuff

polygon api key in .env file
model gets loaded on startup
endpoints: /api/predict/{ticker}, /api/data/{ticker}

accuracy is around 52-53%, barely better than random but thats the point

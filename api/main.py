from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="ASX Portfolio Platform API")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_prices():
    df = pd.read_csv("data/asx_prices.csv", index_col=0, parse_dates=True)
    prices = df.pivot_table(index="Date", columns="Ticker", values="Close")
    prices.index = pd.to_datetime(prices.index, utc=True)
    return prices

def get_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_macd(series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd, signal

def engineer_features(series):
    df = pd.DataFrame()
    df["return_1d"]        = series.pct_change(1)
    df["return_5d"]        = series.pct_change(5)
    df["return_10d"]       = series.pct_change(10)
    df["return_20d"]       = series.pct_change(20)
    sma10                  = series.rolling(10).mean()
    sma50                  = series.rolling(50).mean()
    df["price_vs_sma10"]   = (series - sma10) / sma10
    df["price_vs_sma50"]   = (series - sma50) / sma50
    df["volatility_10d"]   = df["return_1d"].rolling(10).std()
    df["volatility_30d"]   = df["return_1d"].rolling(30).std()
    df["rsi"]              = get_rsi(series)
    macd, signal           = get_macd(series)
    df["macd"]             = macd
    df["macd_signal"]      = signal
    df["macd_hist"]        = macd - signal
    df["target"]           = (series.shift(-1) > series).astype(int)
    return df.dropna()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "ASX Portfolio Platform API", "status": "running"}

@app.get("/api/stocks")
def get_stocks():
    """Latest price and 1-day return for every stock in the watchlist."""
    prices = load_prices()
    returns = prices.pct_change()
    result = []
    for ticker in prices.columns:
        result.append({
            "ticker":       ticker,
            "price":        round(float(prices[ticker].iloc[-1]), 2),
            "change_1d":    round(float(returns[ticker].iloc[-1]) * 100, 2),
            "change_5d":    round(float(prices[ticker].pct_change(5).iloc[-1]) * 100, 2),
        })
    return result

@app.get("/api/portfolio")
def get_portfolio():
    """Optimal portfolio weights via max Sharpe ratio."""
    prices = load_prices()
    mu        = expected_returns.mean_historical_return(prices)
    cov       = risk_models.sample_cov(prices)
    ef        = EfficientFrontier(mu, cov)
    weights   = ef.max_sharpe()
    cleaned   = ef.clean_weights()
    perf      = ef.portfolio_performance(verbose=False)
    return {
        "weights": {k: round(v, 4) for k, v in cleaned.items() if v > 0.001},
        "expected_return":  round(perf[0] * 100, 2),
        "annual_volatility": round(perf[1] * 100, 2),
        "sharpe_ratio":     round(perf[2], 2),
    }

@app.get("/api/signals")
def get_signals():
    """Technical indicator signals for every stock."""
    prices = load_prices()
    result = []
    for ticker in prices.columns:
        series     = prices[ticker]
        rsi        = get_rsi(series).iloc[-1]
        macd, sig  = get_macd(series)
        sma50      = series.rolling(50).mean().iloc[-1]
        price      = series.iloc[-1]
        vs_sma50   = ((price - sma50) / sma50) * 100

        if rsi > 70:   rsi_label = "Overbought"
        elif rsi < 30: rsi_label = "Oversold"
        else:          rsi_label = "Neutral"

        result.append({
            "ticker":     ticker,
            "rsi":        round(float(rsi), 1),
            "rsi_label":  rsi_label,
            "macd_bull":  bool(macd.iloc[-1] > sig.iloc[-1]),
            "vs_sma50":   round(float(vs_sma50), 1),
        })
    return result

@app.get("/api/predictions")
def get_predictions():
    """XGBoost next-day direction predictions for every stock."""
    prices  = load_prices()
    FEATS   = [
        "return_1d","return_5d","return_10d","return_20d",
        "price_vs_sma10","price_vs_sma50",
        "volatility_10d","volatility_30d",
        "rsi","macd","macd_signal","macd_hist",
    ]
    result  = []
    for ticker in prices.columns:
        try:
            df      = engineer_features(prices[ticker])
            X, y    = df[FEATS], df["target"]
            scaler  = StandardScaler()
            Xs      = scaler.fit_transform(X)
            model   = XGBClassifier(
                n_estimators=100, max_depth=3,
                learning_rate=0.05, random_state=42,
                eval_metric="logloss", verbosity=0
            )
            model.fit(Xs, y)
            prob    = model.predict_proba(scaler.transform(X.iloc[-1:]))[0][1]
            result.append({
                "ticker":     ticker,
                "direction":  "UP" if prob > 0.5 else "DOWN",
                "confidence": round(float(max(prob, 1 - prob)) * 100, 1),
                "prob_up":    round(float(prob) * 100, 1),
            })
        except Exception as e:
            result.append({"ticker": ticker, "error": str(e)})
    return result
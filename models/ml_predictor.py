import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def load_prices():
    df = pd.read_csv("data/asx_prices.csv", index_col=0, parse_dates=True)
    prices = df.pivot_table(index="Date", columns="Ticker", values="Close")
    return prices

def engineer_features(price_series, ticker):
    """
    Build features the model learns from.
    These are all things a trader would look at manually.
    """
    df = pd.DataFrame(index=price_series.index)
    df["price"] = price_series

    # Returns over different windows
    df["return_1d"] = price_series.pct_change(1)
    df["return_5d"] = price_series.pct_change(5)
    df["return_10d"] = price_series.pct_change(10)
    df["return_20d"] = price_series.pct_change(20)

    # Moving averages
    df["sma_10"] = price_series.rolling(10).mean()
    df["sma_50"] = price_series.rolling(50).mean()
    df["price_vs_sma10"] = (price_series - df["sma_10"]) / df["sma_10"]
    df["price_vs_sma50"] = (price_series - df["sma_50"]) / df["sma_50"]

    # Volatility (how much it's been swinging lately)
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_30d"] = df["return_1d"].rolling(30).std()

    # RSI
    delta = price_series.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12 = price_series.ewm(span=12).mean()
    ema26 = price_series.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Volume momentum (are people trading more or less than usual?)
    df["ticker"] = ticker

    # Target: did the price go UP the next day? (1 = yes, 0 = no)
    df["target"] = (price_series.shift(-1) > price_series).astype(int)

    return df.dropna()

def train_and_evaluate(ticker, price_series):
    """Train an XGBoost model and evaluate it honestly."""
    
    df = engineer_features(price_series, ticker)
    
    feature_cols = [
        "return_1d", "return_5d", "return_10d", "return_20d",
        "price_vs_sma10", "price_vs_sma50",
        "volatility_10d", "volatility_30d",
        "rsi", "macd", "macd_signal", "macd_hist"
    ]
    
    X = df[feature_cols]
    y = df["target"]
    
    # Time series split — IMPORTANT: we never train on future data
    # This is how real quant teams validate models
    tscv = TimeSeriesSplit(n_splits=3)
    
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
    
    # Train final model on all data for feature importance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    final_model = XGBClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.05, random_state=42,
        eval_metric="logloss", verbosity=0
    )
    final_model.fit(X_scaled, y)
    
    avg_accuracy = np.mean(accuracies)
    
    # What does the model predict for TOMORROW?
    latest_features = X.iloc[-1:].values
    latest_scaled = scaler.transform(latest_features)
    tomorrow_prob = final_model.predict_proba(latest_scaled)[0][1]
    
    return avg_accuracy, tomorrow_prob, final_model, feature_cols

def run_ml_analysis():
    print("=== ASX ML Prediction Engine ===")
    print("Using XGBoost with Time Series Cross-Validation\n")
    print("DISCLAIMER: This model is for educational purposes.")
    print("Past patterns do not guarantee future returns.\n")
    print("-" * 65)
    
    prices = load_prices()
    
    feature_importance_all = {}
    
    print(f"\n{'Stock':<10} {'CV Accuracy':>12} {'Tomorrow':>12} {'Confidence':>12}")
    print("-" * 50)
    
    for ticker in prices.columns:
        try:
            acc, tomorrow_prob, model, feature_cols = train_and_evaluate(
                ticker, prices[ticker]
            )
            
            direction = "UP ▲" if tomorrow_prob > 0.5 else "DOWN ▼"
            confidence = max(tomorrow_prob, 1 - tomorrow_prob)
            
            print(f"{ticker:<10} {acc:>11.1%} {direction:>12} {confidence:>11.1%}")
            
            # Store feature importances
            for feat, imp in zip(feature_cols, model.feature_importances_):
                feature_importance_all[feat] = feature_importance_all.get(feat, 0) + imp
                
        except Exception as e:
            print(f"{ticker:<10} Error: {e}")
    
    # Show which features matter most across all stocks
    print("\n--- Most Predictive Features (across all stocks) ---")
    sorted_features = sorted(feature_importance_all.items(), 
                            key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_features[:6]:
        bar = "█" * int(imp / max(v for _, v in sorted_features) * 20)
        print(f"  {feat:<20} {bar}")
    
    print("\n=== ML Analysis Complete ===")

if __name__ == "__main__":
    run_ml_analysis()
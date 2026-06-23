import pandas as pd
import numpy as np

def load_prices():
    df = pd.read_csv("data/asx_prices.csv", index_col=0, parse_dates=True)
    prices = df.pivot_table(index="Date", columns="Ticker", values="Close")
    return prices

def calculate_sma(prices, window):
    """Simple Moving Average — smooths out price noise over N days."""
    return prices.rolling(window=window).mean()

def calculate_rsi(prices, window=14):
    """
    Relative Strength Index (0-100).
    Above 70 = overbought (might be due for a drop).
    Below 30 = oversold (might be due for a bounce).
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    MACD — Moving Average Convergence Divergence.
    When MACD line crosses above signal line = bullish signal.
    When MACD line crosses below signal line = bearish signal.
    """
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def generate_signals(prices):
    """
    Combine indicators to generate simple buy/sell signals for each stock.
    """
    sma_50 = calculate_sma(prices, 50)
    sma_200 = calculate_sma(prices, 200)
    rsi = calculate_rsi(prices)
    macd_line, signal_line, _ = calculate_macd(prices)
    
    latest_price = prices.iloc[-1]
    latest_sma50 = sma_50.iloc[-1]
    latest_sma200 = sma_200.iloc[-1]
    latest_rsi = rsi.iloc[-1]
    latest_macd = macd_line.iloc[-1]
    latest_signal = signal_line.iloc[-1]
    
    print("=== Technical Indicator Signals (Today) ===\n")
    print(f"{'Stock':<10} {'Price':>8} {'RSI':>7} {'vs SMA50':>10} {'MACD':>8} {'Signal':>10}")
    print("-" * 60)
    
    for ticker in prices.columns:
        price = latest_price[ticker]
        rsi_val = latest_rsi[ticker]
        vs_sma50 = ((price - latest_sma50[ticker]) / latest_sma50[ticker]) * 100
        macd_val = latest_macd[ticker]
        signal_val = latest_signal[ticker]
        
        # Simple signal logic
        signals = []
        if rsi_val < 30:
            signals.append("OVERSOLD")
        elif rsi_val > 70:
            signals.append("OVERBOUGHT")
        if macd_val > signal_val:
            signals.append("MACD+")
        else:
            signals.append("MACD-")
        if price > latest_sma50[ticker]:
            signals.append("ABOVE50MA")
        
        signal_str = " ".join(signals)
        print(f"{ticker:<10} ${price:>7.2f} {rsi_val:>7.1f} {vs_sma50:>+9.1f}% {signal_str}")
    
    print("\n--- RSI Guide: <30 Oversold | 30-70 Neutral | >70 Overbought ---")
    print("--- MACD+: Bullish momentum | MACD-: Bearish momentum ---")

if __name__ == "__main__":
    prices = load_prices()
    generate_signals(prices)
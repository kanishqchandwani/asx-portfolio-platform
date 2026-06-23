import yfinance as yf
import pandas as pd
from datetime import datetime

# ASX stocks we're tracking (the .AX suffix tells Yahoo Finance these are Australian)
ASX_WATCHLIST = [
    "CBA.AX",   # Commonwealth Bank
    "BHP.AX",   # BHP Group
    "CSL.AX",   # CSL Limited
    "NAB.AX",   # National Australia Bank
    "WBC.AX",   # Westpac
    "ANZ.AX",   # ANZ Bank
    "WES.AX",   # Wesfarmers
    "MQG.AX",   # Macquarie Group
    "SUN.AX",   # Suncorp
    "QAN.AX",   # Qantas
]

def fetch_stock_data(ticker, period="1y"):
    """Fetch historical price data for a single ASX stock."""
    print(f"Fetching {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df["Ticker"] = ticker
    return df

def fetch_all_stocks():
    """Loop through watchlist and collect all stock data."""
    all_data = []

    for ticker in ASX_WATCHLIST:
        try:
            df = fetch_stock_data(ticker)
            all_data.append(df)
            print(f"  {ticker}: {len(df)} days of data pulled")
        except Exception as e:
            print(f"  {ticker}: Failed — {e}")

    combined = pd.concat(all_data)
    return combined

if __name__ == "__main__":
    print("=== ASX Portfolio Platform — Data Fetch ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    data = fetch_all_stocks()

    # Save to CSV so we can inspect it
    output_path = "data/asx_prices.csv"
    data.to_csv(output_path)

    print()
    print(f"Done. {len(data)} rows saved to {output_path}")
    print()
    print("Preview of the data:")
    print(data[["Ticker", "Close", "Volume"]].tail(10))

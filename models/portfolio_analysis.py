import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import plotting
import warnings
warnings.filterwarnings("ignore")

def load_price_data():
    """Load the ASX price data we fetched in Phase 1."""
    df = pd.read_csv("data/asx_prices.csv", index_col=0, parse_dates=True)
    
    # Pivot so each column is a stock and rows are dates
    prices = df.pivot_table(index="Date", columns="Ticker", values="Close")
    prices.index = pd.to_datetime(prices.index, utc=True)
    return prices

def calculate_returns(prices):
    """Calculate daily percentage returns for each stock."""
    returns = prices.pct_change().dropna()
    return returns

def optimize_portfolio(prices):
    """
    Find the optimal portfolio weights using Modern Portfolio Theory.
    Maximises the Sharpe Ratio — best return for the least risk.
    """
    # Calculate expected annual returns and covariance matrix
    mu = expected_returns.mean_historical_return(prices)
    cov_matrix = risk_models.sample_cov(prices)
    
    # Build the efficient frontier and maximise Sharpe ratio
    ef = EfficientFrontier(mu, cov_matrix)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    
    # Get portfolio performance stats
    performance = ef.portfolio_performance(verbose=False)
    
    return cleaned_weights, performance

def calculate_var(returns, confidence=0.95):
    """
    Value at Risk — answers the question:
    'On a bad day, what's the most I could lose?'
    """
    var = np.percentile(returns, (1 - confidence) * 100)
    return var

def calculate_volatility(returns):
    """Annualised volatility for each stock (how risky each one is)."""
    volatility = returns.std() * np.sqrt(252)  # 252 trading days in a year
    return volatility

def run_analysis():
    print("=== ASX Portfolio Analysis Engine ===\n")
    
    # Load data
    prices = load_price_data()
    returns = calculate_returns(prices)
    
    print(f"Analysing {len(prices.columns)} stocks over {len(prices)} trading days\n")
    
    # --- OPTIMAL PORTFOLIO ---
    print("--- Optimal Portfolio Weights (Max Sharpe Ratio) ---")
    weights, performance = optimize_portfolio(prices)
    
    for ticker, weight in weights.items():
        if weight > 0.001:
            print(f"  {ticker}: {weight:.1%}")
    
    print(f"\n  Expected Annual Return:  {performance[0]:.1%}")
    print(f"  Annual Volatility:       {performance[1]:.1%}")
    print(f"  Sharpe Ratio:            {performance[2]:.2f}")
    
    # --- RISK METRICS ---
    print("\n--- Individual Stock Risk (Annual Volatility) ---")
    volatility = calculate_volatility(returns)
    for ticker, vol in volatility.sort_values(ascending=False).items():
        bar = "█" * int(vol * 100)
        print(f"  {ticker}: {vol:.1%}  {bar}")
    
    # --- VALUE AT RISK ---
    print("\n--- Value at Risk (95% Confidence, Daily) ---")
    portfolio_returns = returns.mean(axis=1)  # Equal-weighted for now
    var_95 = calculate_var(portfolio_returns)
    print(f"  On a bad day (worst 5%), portfolio could lose: {abs(var_95):.2%}")
    print(f"  On a $10,000 portfolio that's: ${abs(var_95) * 10000:.0f}")
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    run_analysis()
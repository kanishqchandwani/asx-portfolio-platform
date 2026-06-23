# ASX Portfolio Platform

A full-stack financial analytics platform for the Australian Securities Exchange (ASX), built with Python, FastAPI, and React.

**Live Demo:** https://asx-portfolio-platform.vercel.app

## What it does
- Pulls live price data for 10 ASX-listed stocks via Yahoo Finance
- Optimises portfolio weights using Modern Portfolio Theory (max Sharpe Ratio)
- Calculates technical indicators: RSI, MACD, Moving Averages
- Predicts next-day price direction using an XGBoost ML model
- Displays everything in a real-time interactive dashboard

## Tech Stack
- **Data**: yfinance, pandas, numpy
- **Quant Models**: PyPortfolioOpt (Markowitz efficient frontier)
- **Machine Learning**: XGBoost, scikit-learn (Time Series Cross-Validation)
- **Backend**: FastAPI, Python 3.12
- **Frontend**: React, Recharts, Vite
- **Deployment**: Railway (backend), Vercel (frontend)

## Run locally
```bash
# Backend
pip install -r requirements.txt
uvicorn api.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

## Author
Kanishq Chandwani — QUT Bachelor of IT & Business (Computer Science & Finance)
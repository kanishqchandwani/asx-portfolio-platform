import { useState, useEffect } from "react"
import { PieChart, Pie, Cell, Tooltip, BarChart, Bar, XAxis, YAxis, ResponsiveContainer } from "recharts"
import "./App.css"

const API = "http://127.0.0.1:8000"
const PIE_COLORS = ["#00C49F","#0088FE","#FFBB28","#FF8042","#a855f7","#ec4899"]

function App() {
  const [stocks,      setStocks]      = useState([])
  const [portfolio,   setPortfolio]   = useState(null)
  const [signals,     setSignals]     = useState([])
  const [predictions, setPredictions] = useState([])
  const [loading,     setLoading]     = useState(true)
  const [updated,     setUpdated]     = useState("")

  useEffect(() => { fetchAll() }, [])

  async function fetchAll() {
    setLoading(true)
    try {
      const [s, p, sig, pred] = await Promise.all([
        fetch(`${API}/api/stocks`).then(r => r.json()),
        fetch(`${API}/api/portfolio`).then(r => r.json()),
        fetch(`${API}/api/signals`).then(r => r.json()),
        fetch(`${API}/api/predictions`).then(r => r.json()),
      ])
      setStocks(s)
      setPortfolio(p)
      setSignals(sig)
      setPredictions(pred)
      setUpdated(new Date().toLocaleTimeString("en-AU"))
    } catch(e) { console.error(e) }
    setLoading(false)
  }

  const pieData = portfolio
    ? Object.entries(portfolio.weights).map(([k,v]) => ({ name: k, value: +(v*100).toFixed(1) }))
    : []

  const rsiData = signals.map(s => ({ ticker: s.ticker.replace(".AX",""), rsi: s.rsi }))

  return (
    <div className="app">
      {/* HEADER */}
      <header className="header">
        <div className="header-left">
          <span className="logo">📈 ASX Portfolio Platform</span>
          <span className="subtitle">Live Australian Market Analytics</span>
        </div>
        <div className="header-right">
          {updated && <span className="updated">Last updated: {updated}</span>}
          <button className="refresh-btn" onClick={fetchAll}>↻ Refresh</button>
        </div>
      </header>

      {loading ? (
        <div className="loading">Loading market data...</div>
      ) : (
        <main className="main">

          {/* STOCK PRICES */}
          <section className="card full-width">
            <h2>ASX Watchlist — Live Prices</h2>
            <table className="table">
              <thead>
                <tr>
                  <th>Stock</th>
                  <th>Price (AUD)</th>
                  <th>1D Change</th>
                  <th>5D Change</th>
                </tr>
              </thead>
              <tbody>
                {stocks.map(s => (
                  <tr key={s.ticker}>
                    <td className="ticker">{s.ticker}</td>
                    <td>${s.price.toFixed(2)}</td>
                    <td className={s.change_1d >= 0 ? "green" : "red"}>
                      {s.change_1d >= 0 ? "▲" : "▼"} {Math.abs(s.change_1d).toFixed(2)}%
                    </td>
                    <td className={s.change_5d >= 0 ? "green" : "red"}>
                      {s.change_5d >= 0 ? "▲" : "▼"} {Math.abs(s.change_5d).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          {/* PORTFOLIO + RSI */}
          <section className="card">
            <h2>Optimal Portfolio</h2>
            <p className="hint">Maximised Sharpe Ratio via Modern Portfolio Theory</p>
            {portfolio && (
              <>
                <div className="metrics-row">
                  <div className="metric">
                    <span className="metric-label">Expected Return</span>
                    <span className="metric-value green">{portfolio.expected_return}%</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Volatility</span>
                    <span className="metric-value yellow">{portfolio.annual_volatility}%</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Sharpe Ratio</span>
                    <span className="metric-value blue">{portfolio.sharpe_ratio}</span>
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={pieData} cx="50%" cy="50%" outerRadius={80}
                      dataKey="value" label={({name,value}) => `${name}: ${value}%`}>
                      {pieData.map((_,i) => (
                        <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(v) => `${v}%`} />
                  </PieChart>
                </ResponsiveContainer>
              </>
            )}
          </section>

          <section className="card">
            <h2>RSI by Stock</h2>
            <p className="hint">Above 70 = Overbought · Below 30 = Oversold</p>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={rsiData} layout="vertical"
                margin={{ left: 10, right: 30, top: 5, bottom: 5 }}>
                <XAxis type="number" domain={[0,100]} tick={{ fill:"#aaa", fontSize:11 }}
                  tickLine={false} axisLine={false}/>
                <YAxis type="category" dataKey="ticker" tick={{ fill:"#ccc", fontSize:12 }}
                  tickLine={false} axisLine={false} width={50}/>
                <Tooltip formatter={(v) => [`RSI: ${v}`, ""]} />
                <Bar dataKey="rsi" radius={[0,4,4,0]}>
                  {rsiData.map((d,i) => (
                    <Cell key={i}
                      fill={d.rsi > 70 ? "#ef4444" : d.rsi < 30 ? "#22c55e" : "#3b82f6"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </section>

          {/* SIGNALS */}
          <section className="card">
            <h2>Technical Signals</h2>
            <table className="table">
              <thead>
                <tr>
                  <th>Stock</th>
                  <th>RSI</th>
                  <th>Status</th>
                  <th>MACD</th>
                  <th>vs SMA50</th>
                </tr>
              </thead>
              <tbody>
                {signals.map(s => (
                  <tr key={s.ticker}>
                    <td className="ticker">{s.ticker}</td>
                    <td>{s.rsi}</td>
                    <td className={
                      s.rsi_label === "Overbought" ? "red" :
                      s.rsi_label === "Oversold"   ? "green" : "gray"
                    }>{s.rsi_label}</td>
                    <td className={s.macd_bull ? "green" : "red"}>
                      {s.macd_bull ? "Bullish ▲" : "Bearish ▼"}
                    </td>
                    <td className={s.vs_sma50 >= 0 ? "green" : "red"}>
                      {s.vs_sma50 >= 0 ? "+" : ""}{s.vs_sma50}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          {/* ML PREDICTIONS */}
          <section className="card">
            <h2>ML Predictions — Tomorrow's Direction</h2>
            <p className="hint">XGBoost model trained on 12 technical features</p>
            <div className="predictions-grid">
              {predictions.map(p => (
                <div key={p.ticker} className={`prediction-card ${p.direction === "UP" ? "pred-up" : "pred-down"}`}>
                  <span className="pred-ticker">{p.ticker}</span>
                  <span className="pred-direction">{p.direction === "UP" ? "▲ UP" : "▼ DOWN"}</span>
                  <span className="pred-conf">{p.confidence}% confidence</span>
                  <div className="conf-bar">
                    <div className="conf-fill" style={{ width: `${p.confidence}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </section>

        </main>
      )}

      <footer className="footer">
        Built by Kanishq Chandwani · QUT IT & Business · ASX Portfolio Platform
      </footer>
    </div>
  )
}

export default App
## Statistical Arbitrage in US Equities (Streamlit)

Streamlit app that replicates a mean-reversion statistical arbitrage workflow inspired by Avellaneda & Lee (2010). The app pulls US equity data from Yahoo Finance, fits factor models to extract idiosyncratic residuals, estimates Ornstein–Uhlenbeck parameters, and produces contrarian long/short signals you can backtest or view live.

## Collaborators
- Tanish Patel <tp2899@columbia.edu>
- Aditya Pendyala <ap4839@columbia.edu>
- Hardik Saurabh Gupta <hg2770@columbia.edu>

### Features
- Multiple factor models: robust PCA, single-factor SPY, sector ETF betas (auto-mapped from ticker sectors), full SPY+sector+PCA stack, plus a cointegration/pairs mode with ADF and half-life filters.
- Residual modeling: OU/AR(1) fit with optional volume-based “trading time” scaling, configurable estimation windows, entry/exit s-score bands, and transaction cost + leverage inputs.
- Backtesting: equity curve, exposure charts, per-ticker trade log, s-score traces, and pair-trade visualization (spread bands + OHLC overlays).
- Live signals: latest-day s-scores and suggested actions (Go LONG/SHORT, CLOSE, or NO TRADE).
- Analytics pages: factor diagnostics (correlation heatmaps, PCA loadings/eigenportfolios, residual t-stats) and trade analytics (PnL KPIs, distributions, streaks, heatmaps).

### Project Structure
- `stat_arb2.py` — main Streamlit entry point and final published app.
- `pages/01_Factor_Diagnostics.py` — factor model visuals once a run has been executed.
- `pages/02_Trade_Analytics.py` — backtest KPI and trade-distribution dashboards.
- `stat_arb.py` — earlier version kept for reference.
- `notebook1.ipynb`, `notebook2.ipynb` — exploratory notebooks.
- `requirements.txt` — Python dependencies.

### Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```
The app fetches market data via `yfinance`, so internet access is required.

### Running the App
```bash
streamlit run stat_arb2.py
```
Streamlit will open a browser tab. Use the sidebar to:
1) Enter a comma-separated ticker universe and choose **Backtest** or **Live Signals**.  
2) Set the date range (live mode still loads history for estimation).  
3) Pick a factor model; cointegration mode exposes additional correlation/ADF/half-life/notional sliders.  
4) Configure OU/residual window, PCA variance target, beta windows, entry/exit z-thresholds, transaction costs, leverage, and optional volume-based scaling.  
5) Click **Run Backtest** (or **Compute Live Signals**). Outputs and downstream pages rely on the session state from this run.

### Notes & Limitations
- Data quality and coverage depend on Yahoo Finance; thinly traded tickers or missing sectors may reduce usable signals.
- PCA and OU fits require enough history and at least 3 tickers after cleaning; otherwise the app will warn and skip those assets.
- Cointegration backtests only run when qualifying pairs are found within the chosen filters.

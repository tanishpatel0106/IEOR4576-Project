# Statistical Arbitrage Streamlit App

Mean-reversion stat-arb inspired by Avellaneda & Lee (2010) with multiple factor models and a cointegration-based pair trading mode. Includes multipage dashboards for factor diagnostics and trade analytics.

## Quick Start

1. Install deps:
   ```bash
   pip install streamlit pandas numpy yfinance plotly scikit-learn statsmodels
   ```
2. Run:
   ```bash
   streamlit run stat_arb2.py
   ```
3. Open the local URL (default `http://localhost:8501`) and use the sidebar to configure a backtest or live signals.

## Modes (Factor Model selector)

- **PCA**: Robust PCA on log returns; residuals = reconstruction errors.
- **Single Market Factor (SPY)**: Rolling beta to SPY; residual = stock return – β·SPY.
- **Sector ETF factors (auto mapping)**: Auto-map stocks to sector ETFs via yfinance sector field; rolling beta per stock to ETF.
- **SPY + Sector + PCA**: Avellaneda–Lee 3-stage: strip SPY, strip sector ETF, sector-level PCA residual.
- **Cointegration (pairs)**: Find cointegrated pairs (corr filter, ADF on spread, half-life band), then trade spread s-scores.

## Key Sidebar Controls

- Universe tickers (comma-separated)
- Mode: Backtest / Live Signals
- Dates: backtest start / data end
- PCA: correlation window, target explained variance (UI), OU window, entry/exit z
- Beta window for SPY/ETF models
- Costs (bps), leverage long/short, initial equity
- Trading time (volume-based scaling) toggle
- Cointegration-only (shown only in that mode): min corr, ADF p-value cutoff, per-pair notional fraction, min/max half-life

## What Happens

- Data via yfinance (Adj Close, Volume). Returns = log(price/lag).
- OU/AR(1) fit on residuals spreads → s-score → entry/exit bands.
- Factor modes: build residuals via chosen model, trade equal-notional long/short with leverage and costs.
- Cointegration: OLS hedge ratio on log prices, ADF stationarity on spread, half-life filter; trade spread long/short by s-score with fixed per-pair notional.

## Outputs

- Main page: KPIs (return, vol, Sharpe, trades), equity curve, exposure (factor modes), per-stock P&L (factor modes), trade logs and charts. Cointegration mode shows candidate pairs, cointegration stats (pairs scanned/kept, corr, half-life, entries/exits), spread chart, dual-LEG OHLC with long/short markers, and pair trade log.
- Pages (non-cointegration):
  - **Factor Diagnostics**: correlation matrix, PCA loadings/eigenportfolios, residual t-stats, factor returns.
  - **Trade Analytics**: win rate, profit factor, expectancy, streaks, PnL/return histograms, cumulative PnL, long vs short PnL, holding-period vs PnL, ticker contribution, time heatmaps.

## Theory (brief)

- Residual mean-reversion: factors remove common risk; residuals modeled as OU/AR(1); trade when residual deviates (s-score) from equilibrium.
- Cointegration: two nonstationary prices form a stationary spread after hedging by β; ADF validates stationarity; half-life ensures actionable mean reversion; trade spread bands.

## Example (cointegration run)

- Universe: AAPL, MSFT, AMZN, GOOGL, NVDA, JPM, XOM, JNJ
- Filters: corr ≥0.60, ADF p <0.05, half-life 2–90 days; 7/7 pairs kept; avg corr ~0.75; median half-life ~11d
- Performance (2022-12-16 to 2025-12-15): Final ~$1.55M from $1M (+54.7% total; ~16.3% ann; ~8.7% vol; Sharpe ~1.88)
- Top signals (by latest s-score): e.g., JPM/NVDA, XOM/NVDA, GOOG/XOM, AAPL/GOOGL

## Notes

- PCA needs ≥3 valid tickers with sufficient history.
- If no cointegrated pairs are found, relax corr/ADF/half-life filters.
- Adjust entry/exit z and OU window to balance churn vs responsiveness.

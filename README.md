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

## Detailed Guide (How to Use and Why It Works)

### What this app is for
- Implements a market-neutral, mean-reversion stat-arb workflow inspired by Avellaneda & Lee (2010).
- Two families of engines: (1) Factor-residual contrarian trading; (2) Cointegration-based pair trading.
- Multipage Streamlit layout: the main page runs backtests or live signals; the other pages explain factors and trades.

### Running the app
1) Install dependencies (Python 3.9+ recommended):
   ```bash
   pip install streamlit pandas numpy yfinance plotly scikit-learn statsmodels
   ```
2) Launch:
   ```bash
   streamlit run stat_arb2.py
   ```
3) Open the local URL (default http://localhost:8501). Use the sidebar to configure, then click **Run Backtest** (or **Compute Live Signals**).

### High-level workflow (backtest)
1) Choose **Mode** = Backtest (Live Signals skips trading simulation and only prints latest s-scores).
2) Set **Universe tickers** (comma separated). Include SPY and major sector ETFs if you plan to use those factor modes.
3) Pick **Factor model** (see next section).
4) Adjust **data window** (Backtest start date and Data end date).
5) Tune **PCA** window/variance and **OU** window, entry/exit z-bands, and optional volume-based trading time.
6) For cointegration mode, set correlation floor, ADF p-value cutoff, half-life bounds, and per-pair notional.
7) For backtests, choose costs (bps) and long/short leverage; set initial equity.
8) Click **Run Backtest**. Results and charts render on the main page; extra analytics are on the two subpages.

### Modes (Factor model selector)
- **PCA**: Robust PCA on log returns (Ledoit-Wolf shrinkage, eigenvalue flooring). Residuals = reconstruction errors; trades on stocks whose residual deviates from mean.
- **Single Market Factor (SPY)**: Rolling beta of each stock to SPY over `beta_window_days`; residual = stock return minus beta*SPY.
- **Sector ETF factors (auto mapping)**: Automatically maps each stock to a sector ETF via yfinance metadata, then runs rolling single-factor betas to those ETFs.
- **SPY + Sector + PCA**: Three-stage de-risking: (1) strip market (SPY), (2) strip sector ETF, (3) PCA on sector residuals to capture remaining common modes.
- **Cointegration (pairs)**: Screen all pairs by correlation, fit hedge ratio by OLS on log prices, test stationarity with ADF, filter by half-life, and trade spread s-scores with fixed per-pair notional.

### Sidebar parameters (what they do)
- **PCA: Target explained variance** and **PCA correlation window**: Control how many components are retained; larger windows smooth noise but react slower.
- **Rolling beta window (SPY/ETF)**: Sample length for market/sector betas; shorter windows adapt faster but can be noisy.
- **OU / residual estimation window**: How much history feeds the AR(1)/OU fit used for s-scores.
- **Entry / Exit |s-score| thresholds**: Z-band for opening and closing positions; widen to reduce churn, tighten to increase responsiveness.
- **Use volume-based trading time**: Scales residuals by relative volume so high-volume days are down-weighted (Avellaneda–Lee trading time idea).
- **Transaction cost (bps)** and **leverage** (backtest only): Cost per side and gross leverage for long and short books.
- **Cointegration controls** (only when that mode is chosen):
  - Min correlation: prefilter for sufficiently linked pairs.
  - ADF p-value cutoff: keep only stationary spreads.
  - Notional per pair (fraction of equity): allocates fixed slice per active pair.
  - Half-life bounds: drop spreads that mean revert too slowly or too fast.

### Outputs and navigation
- **Main page – Backtest (factor modes)**: KPI cards (return, vol, Sharpe, trades), equity curve, exposure charts, per-ticker PnL, trade log, and ticker-level price/s-score charts with signal markers.
- **Main page – Backtest (cointegration)**: Table of candidate pairs with corr/ADF/half-life stats, spread chart with entry/exit markers, dual-leg OHLC with long/short markers, and pair trade log.
- **Main page – Live Signals**: Latest s-scores and suggested actions (Go LONG, Go SHORT, CLOSE, or NO TRADE) plus residual and OU stats for context.
- **Page: Factor Diagnostics**: Correlation matrix (ETFs vs stocks), PCA loadings/eigenportfolios, residual t-stats, and factor return series (or SPY/sector returns for non-PCA modes).
- **Page: Trade Analytics**: Trade KPIs (win rate, profit factor, expectancy, streaks), PnL/return histograms, cumulative PnL, long vs short curves, holding-period vs PnL scatter, ticker contribution bars, and time-based heatmaps.

### Theory cheat sheet (why the signals work)
- **Residual mean reversion**: Remove common risk (market, sector, or latent PCA factors) to isolate idiosyncratic residuals. If those residuals follow a mean-reverting OU/AR(1), deviations measured by `s-score = (residual - mu_hat) / eq_std` are tradable: short when rich (high s-score), long when cheap (low s-score), close near zero.
- **PCA variant**: Uses shrinkage covariance and eigenvalue flooring to stabilize eigenvectors; reconstruction errors act as residuals. Keeping enough PCs to hit the target explained variance preserves common risk while discarding noise.
- **Rolling betas**: SPY or sector ETF regressions keep exposures current; residuals show stock-specific performance after controlling for its systematic leg.
- **Three-stage model**: Sequentially removes market and sector, then de-noises remaining common modes with PCA, approximating the Avellaneda–Lee factor stripping pipeline.
- **Trading time**: Volume-scaled residuals approximate business-time volatility, dampening signals on unusually high-volume days that might overstate price moves.
- **Cointegration**: Two nonstationary prices can form a stationary spread after hedging by the OLS beta. ADF tests for stationarity; half-life from AR(1) on the spread measures how quickly it mean-reverts. Trade s-scores of the spread with symmetric bands.
- **Risk/positioning**: Gross leverage is applied equally across long and short books in factor modes; cointegration mode allocates fixed notional per qualified pair for balanced diversification.

### Practical tips
- Include SPY and sector ETFs in the universe when using factor-based modes to avoid missing required benchmarks.
- If PCA yields no signals, widen the correlation window, lower the explained-variance target, or add more tickers (needs at least 3 with clean data).
- For cointegration, start with looser filters (e.g., corr 0.5–0.6, ADF 0.10, half-life 2–120) to confirm data sufficiency, then tighten.
- Entry/exit z-bands are the main churn control: larger bands mean fewer, higher-conviction trades; smaller bands increase turnover and cost drag.
- Live signals still need sufficient lookback to estimate OU params; if the table is empty, extend the start date or reduce filters.
- When interpreting results, compare long vs short PnL, streaks, and heatmaps to spot regime sensitivity and calendar effects.

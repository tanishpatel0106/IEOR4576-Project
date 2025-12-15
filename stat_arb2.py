import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

if "run_backtest" not in st.session_state:
    st.session_state["run_backtest"] = False

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="Statistical Arbitrage – Avellaneda & Lee",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Statistical Arbitrage in US Equities (Avellaneda & Lee Style)")

st.markdown(
    """
This app implements a **mean-reversion stat-arb strategy** inspired by
*Avellaneda & Lee (2010)*:

- Decompose stock returns into **systematic factors** (PCA / SPY / sector ETFs)  
- Model idiosyncratic residuals as **mean-reverting (OU / AR(1))**  
- Generate **contrarian long/short signals** using s-scores  
- Run a **backtest** or see **live signals** based on recent data  
- Optionally use **volume-based “trading time” scaling**
"""
)

# =========================================================
# SIDEBAR – PARAMETERS
# =========================================================

st.sidebar.header("Configuration")

default_universe = "AAPL, MSFT, AMZN, GOOGL, META, JPM, XOM, JNJ, NVDA, BRK-B"
universe_str = st.sidebar.text_input(
    "Universe tickers (comma-separated)",
    value=default_universe
)
user_tickers = [t.strip().upper() for t in universe_str.split(",") if t.strip()]

mode = st.sidebar.selectbox(
    "Mode",
    ["Backtest", "Live Signals"]
)

if mode == "Backtest":
    start_date = st.sidebar.date_input(
        "Backtest start date",
        value=datetime.today().date() - timedelta(days=365 * 3)
    )
else:
    # Live mode: still need history for estimation
    start_date = datetime.today().date() - timedelta(days=365 * 3)

end_date = st.sidebar.date_input(
    "Data end date",
    value=datetime.today().date()
)

factor_model_type = st.sidebar.selectbox(
    "Factor model",
    [
        "PCA",
        "Single Market Factor (SPY)",
        "Sector ETF factors (auto mapping)",
        "SPY + Sector + PCA",
        "Cointegration (pairs)"
    ]
)

pair_corr_floor = pair_adf_alpha = pair_notional_frac = pair_min_hl = pair_max_hl = None
if factor_model_type == "Cointegration (pairs)":
    pair_corr_floor = st.sidebar.number_input(
        "Cointegration: min correlation",
        min_value=0.0,
        max_value=0.99,
        value=0.6,
        step=0.05,
        help="Screen out pairs that are too weakly correlated."
    )
    pair_adf_alpha = st.sidebar.number_input(
        "Cointegration: ADF p-value cutoff",
        min_value=0.001,
        max_value=0.2,
        value=0.05,
        step=0.005,
        help="Keep pairs whose spread ADF p-value is below this."
    )
    pair_notional_frac = st.sidebar.slider(
        "Cointegration: notional per pair (fraction of equity)",
        min_value=0.02,
        max_value=0.5,
        value=0.1,
        step=0.02
    )
    pair_min_hl = st.sidebar.number_input(
        "Cointegration: min half-life (days)",
        min_value=1,
        max_value=200,
        value=2,
        step=1
    )
    pair_max_hl = st.sidebar.number_input(
        "Cointegration: max half-life (days)",
        min_value=5,
        max_value=400,
        value=90,
        step=5
    )


pca_explained_var = st.sidebar.slider(
    "PCA: Target explained variance",
    min_value=0.3,
    max_value=0.9,
    value=0.55,
    step=0.05
)

window_corr_days = st.sidebar.number_input(
    "PCA correlation window (days)",
    min_value=120,
    max_value=504,
    value=252,
    step=21
)

# Rolling beta window for SPY / ETF models
beta_window_days = st.sidebar.number_input(
    "Rolling beta window (days) for SPY / ETF models",
    min_value=40,
    max_value=252,
    value=60,
    step=5
)

ou_window_days = st.sidebar.number_input(
    "OU / residual estimation window (days)",
    min_value=40,
    max_value=120,
    value=60,
    step=5
)

entry_z = st.sidebar.number_input(
    "Entry |s-score| threshold",
    min_value=0.5,
    max_value=3.0,
    value=1.25,
    step=0.25
)
exit_z = st.sidebar.number_input(
    "Exit |s-score| threshold (for closing)",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.1
)

use_trading_time = st.sidebar.checkbox(
    "Use volume-based 'trading time' scaling",
    value=True
)

if mode == "Backtest":
    tc_bps = st.sidebar.number_input(
        "Transaction cost per side (bps)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=1.0,
        help="5 bps per trade side ≈ 10 bps round trip"
    )
    leverage_long = st.sidebar.number_input("Long leverage (× equity)", 0.0, 5.0, 2.0, 0.5)
    leverage_short = st.sidebar.number_input("Short leverage (× equity)", 0.0, 5.0, 2.0, 0.5)

    initial_equity = st.sidebar.number_input(
        "Initial equity",
        min_value=10_000.0,
        max_value=10_000_000.0,
        value=1_000_000.0,
        step=50_000.0
    )
    # --- Replace old run_button with persistent session state ---
    if st.sidebar.button("Run Backtest" if mode == "Backtest" else "Compute Live Signals"):
        st.session_state["run_backtest"] = True

else:
    tc_bps = 0.0
    leverage_long = 2.0
    leverage_short = 2.0
    initial_equity = 1_000_000.0

# run_button = st.sidebar.button("Run" if mode == "Backtest" else "Compute Live Signals")

# =========================================================
# SECTOR → ETF MAPPING (AUTOMATIC)
# =========================================================

SECTOR_TO_ETF = {
    # These sector labels are common in yfinance metadata
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Financial Services": "XLF",
    "Financial": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Communications": "XLC",
}

@st.cache_data(show_spinner=True)
def auto_sector_etf_mapping(tickers: List[str]) -> Dict[str, str]:
    """
    Automatically map each stock ticker to a sector ETF
    via yfinance .info['sector'] and SECTOR_TO_ETF mapping.
    Returns: {stock: etf}
    """
    mapping: Dict[str, str] = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
        except Exception:
            continue
        sector = info.get("sector")
        if sector is None:
            continue
        etf = SECTOR_TO_ETF.get(sector)
        if etf:
            mapping[t] = etf
    return mapping

# =========================================================
# CORE HELPERS
# =========================================================

@st.cache_data(show_spinner=True)
def load_price_and_volume(
    tickers: List[str],
    start: datetime,
    end: datetime
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Adjusted Close and Volume for tickers via yfinance.
    Returns:
        prices_df: columns = tickers, index = dates
        volume_df: columns = tickers, index = dates
    """
    data = yf.download(
        tickers,
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=False,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.levels[0]:
            prices = data["Adj Close"]
        elif "Close" in data.columns.levels[0]:
            prices = data["Close"]
        else:
            raise ValueError("No Close or Adj Close in yfinance data.")

        if "Volume" in data.columns.levels[0]:
            volume = data["Volume"]
        else:
            volume = pd.DataFrame(
                0, index=data.index, columns=prices.columns
            )
    else:
        prices = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
        volume = data["Volume"] if "Volume" in data.columns else pd.DataFrame(
            0, index=data.index, columns=[prices.name]
        )

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    if isinstance(volume, pd.Series):
        volume = volume.to_frame()

    prices = prices.dropna(how="all")
    volume = volume.reindex_like(prices)

    return prices, volume


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1))
    return returns.dropna(how="all")

# ---------------- PCA FACTOR MODEL ----------------

# ============================================================
#                ROBUST PCA MODULE (OPTION B)
# ============================================================
from sklearn.covariance import LedoitWolf

def robust_pca_factors(returns: pd.DataFrame, n_components: int = None):
    """
    Enhanced PCA for statistical arbitrage (Option B):
    - Fill missing data
    - Ledoit-Wolf shrinkage covariance (robust)
    - Symmetrize covariance
    - Eigenvalue flooring
    - Stable eigenvector orientation
    - Factor projection -> reconstruction -> residuals
    """

    # 1. Clean missing data
    X = returns.copy()
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # Drop columns with remaining NaNs
    X = X.dropna(axis=1, how="any")

    if X.shape[1] < 3:
        return pd.DataFrame(), None, None

    # 2. Ledoit-Wolf shrinkage covariance
    lw = LedoitWolf().fit(X)
    cov = lw.covariance_

    # 3. Symmetrize covariance
    cov = 0.5 * (cov + cov.T)

    # 4. Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # sort by variance descending
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 5. Floor eigenvalues
    eigvals = np.clip(eigvals, 1e-6, None)

    # 6. Choose number of components
    if n_components is None:
        n_components = int(np.sqrt(X.shape[1]))
        n_components = max(1, min(n_components, X.shape[1] - 1))

    # 7. Extract eigenvectors
    factors = eigvecs[:, :n_components]

    # 8. Stable orientation
    for i in range(n_components):
        if factors[np.argmax(np.abs(factors[:, i])), i] < 0:
            factors[:, i] *= -1

    # 9. Factor scores F = X * factors
    F = X.values @ factors

    # 10. Reconstruction X_hat = F * factors^T
    X_hat = F @ factors.T

    # 11. Residuals
    residuals = pd.DataFrame(X.values - X_hat, index=X.index, columns=X.columns)

    return residuals, eigvals, eigvecs


# ============================================================
#             PCA Residual Builder (REPLACEMENT)
# ============================================================
def build_residuals_pca(
    prices: pd.DataFrame,
    tickers: List[str],
    pca_lookback: int
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    CORRECTED PCA RESIDUAL BUILDER:
    - uses only tickers that exist in prices
    - uses log returns (stable)
    - enforces minimum asset count
    - avoids empty PCA residuals
    Returns (residual_df, eigvals, eigvecs)
    """
    # ensure tickers exist in price DataFrame
    tickers = [t for t in tickers if t in prices.columns]
    if len(tickers) < 3:
        return pd.DataFrame(), None, None  # PCA needs > 2 assets

    # Slice window
    price_window = prices[tickers].iloc[-pca_lookback:].copy()
    price_window = price_window.dropna(axis=1, how="any")

    if price_window.shape[1] < 3:
        return pd.DataFrame(), None, None

    # log returns (numerically stable)
    ret_window = np.log(price_window / price_window.shift(1)).dropna()
    if ret_window.empty or ret_window.shape[1] < 3:
        return pd.DataFrame(), None, None

    # robust PCA
    resid_df, eigvals, eigvecs = robust_pca_factors(ret_window)
    if resid_df is None or resid_df.empty:
        return pd.DataFrame(), None, None

    return resid_df, eigvals, eigvecs
# def compute_pca_factors(
#     returns: pd.DataFrame,
#     window: int,
#     explained_target: float
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Rolling PCA factor model:
#     - For each date t, use previous `window` days to estimate correlation matrix
#     - Get eigenvectors/values, keep enough components to hit explained_target
#     - Build eigenportfolio factor returns
#     Returns:
#         factors_df: factor returns (F1, F2, ...)
#         betas_panel: MultiIndex (date, ticker) with betas on those factors
#     """
#     dates = returns.index
#     factor_returns_list = []
#     betas_records = []

#     for t_idx in range(window, len(dates)):
#         end_idx = t_idx
#         start_idx = t_idx - window
#         window_data = returns.iloc[start_idx:end_idx].dropna(axis=1, how="any")
#         if window_data.shape[1] < 3:
#             continue

#         R = window_data.values
#         mu = R.mean(axis=0)
#         std = R.std(axis=0, ddof=1)
#         std[std == 0] = 1e-8
#         Y = (R - mu) / std

#         C = np.corrcoef(Y, rowvar=False)

#         eigvals, eigvecs = np.linalg.eigh(C)
#         idx = np.argsort(eigvals)[::-1]
#         eigvals = eigvals[idx]
#         eigvecs = eigvecs[:, idx]

#         total_var = eigvals.sum()
#         cum_var = np.cumsum(eigvals) / total_var
#         m = np.searchsorted(cum_var, explained_target) + 1
#         m = min(m, window_data.shape[1])

#         eigvecs_m = eigvecs[:, :m]

#         sigma_i = std
#         sigma_i[sigma_i == 0] = 1e-8
#         weights = eigvecs_m / sigma_i[:, None]  # n_stocks × m

#         Rt_last = window_data.iloc[-1].values
#         F_last = weights.T @ Rt_last
#         factor_date = dates[end_idx]
#         factor_returns_list.append(
#             pd.Series(F_last, index=[f"F{k+1}" for k in range(m)], name=factor_date)
#         )

#         F_window = window_data @ weights
#         F_window = pd.DataFrame(
#             F_window, index=window_data.index,
#             columns=[f"F{k+1}" for k in range(m)]
#         )

#         for ticker in window_data.columns:
#             y = window_data[ticker].values
#             X = F_window.values
#             XtX = X.T @ X
#             try:
#                 beta = np.linalg.solve(XtX, X.T @ y)
#             except np.linalg.LinAlgError:
#                 beta = np.linalg.lstsq(X, y, rcond=None)[0]
#             betas_records.append({
#                 "date": factor_date,
#                 "ticker": ticker,
#                 **{f"F{k+1}": beta[k] for k in range(m)}
#             })

#     if not factor_returns_list:
#         return pd.DataFrame(), pd.DataFrame()

#     factors_df = pd.DataFrame(factor_returns_list).sort_index()
#     betas_df = pd.DataFrame(betas_records)
#     betas_df.set_index(["date", "ticker"], inplace=True)
#     betas_df.sort_index(inplace=True)

#     return factors_df, betas_df


# def build_residuals_pca(
#     returns: pd.DataFrame,
#     window_corr_days: int,
#     explained_target: float
# ) -> Tuple[Dict[str, pd.Series], Dict]:
#     factors_df, betas_panel = compute_pca_factors(
#         returns,
#         window=window_corr_days,
#         explained_target=explained_target
#     )
#     if factors_df.empty:
#         return {}, {}

#     common_dates = returns.index.intersection(factors_df.index)
#     returns_aligned = returns.loc[common_dates]
#     factors_aligned = factors_df.loc[common_dates]

#     residuals_dict = {}
#     for date in common_dates:
#         day_ret = returns_aligned.loc[date]
#         if date not in betas_panel.index.get_level_values(0):
#             continue
#         betas_today = betas_panel.loc[date]
#         common_stocks = day_ret.index.intersection(betas_today.index)
#         if len(common_stocks) == 0:
#             continue

#         F = factors_aligned.loc[date].values
#         for ticker in common_stocks:
#             beta_vec = betas_today.loc[ticker].values
#             y = day_ret[ticker]
#             y_hat = float(beta_vec @ F)
#             resid = y - y_hat
#             residuals_dict.setdefault(ticker, {})[date] = resid

#     residuals = {
#         ticker: pd.Series(d, name=ticker).sort_index()
#         for ticker, d in residuals_dict.items()
#     }

#     meta = {"factors": factors_df, "betas": betas_panel}
#     return residuals, meta

# ---------------- SINGLE-FACTOR ROLLING BETA (SPY) ----------------

def build_residuals_single_factor_rolling(
    returns: pd.DataFrame,
    market_ticker: str,
    beta_window: int
) -> Tuple[Dict[str, pd.Series], Dict]:
    """
    Rolling beta vs SPY (or other market ETF) with window beta_window.
    """
    if market_ticker not in returns.columns:
        raise ValueError(f"{market_ticker} must be in returns for single-factor model.")

    factor_ret = returns[market_ticker].dropna()
    residuals: Dict[str, pd.Series] = {}
    beta_records = []

    for ticker in returns.columns:
        if ticker == market_ticker:
            continue
        aligned = pd.concat(
            [returns[ticker], factor_ret], axis=1, keys=[ticker, market_ticker]
        ).dropna()
        if len(aligned) <= beta_window:
            continue

        betas = []
        dates = aligned.index
        for i in range(beta_window, len(dates)):
            window_data = aligned.iloc[i - beta_window:i]
            y = window_data[ticker].values
            x = window_data[market_ticker].values
            xx = x @ x
            if xx < 1e-10:
                b = 0.0
            else:
                b = float((x @ y) / xx)
            betas.append({"date": dates[i], "beta": b})

        if not betas:
            continue

        beta_series = pd.Series(
            [b["beta"] for b in betas],
            index=[b["date"] for b in betas],
            name="beta"
        )

        r_stock = aligned[ticker].loc[beta_series.index]
        r_factor = aligned[market_ticker].loc[beta_series.index]
        resid = r_stock - beta_series * r_factor
        residuals[ticker] = resid

        for b in betas:
            beta_records.append({
                "ticker": ticker,
                "date": b["date"],
                "beta": b["beta"]
            })

    beta_df = pd.DataFrame(beta_records).set_index(["date", "ticker"]).sort_index()
    meta = {"factor": factor_ret, "betas": beta_df}
    return residuals, meta

# ---------------- SECTOR ETF ROLLING BETA MODEL ----------------

def build_residuals_sector_etf_rolling(
    returns: pd.DataFrame,
    sector_mapping: Dict[str, str],
    beta_window: int
) -> Tuple[Dict[str, pd.Series], Dict]:
    """
    For each stock, regress on its sector ETF using rolling beta_window-day window.
    residual_t = r_stock_t - beta_t * r_etf_t
    """
    residuals: Dict[str, pd.Series] = {}
    beta_records = []

    # Ensure ETF returns exist
    etfs_needed = sorted(set(sector_mapping.values()))
    missing_etfs = [e for e in etfs_needed if e not in returns.columns]
    if missing_etfs:
        raise ValueError(f"Missing ETF returns for: {missing_etfs}")

    for stock, etf in sector_mapping.items():
        if stock not in returns.columns or etf not in returns.columns:
            continue
        aligned = returns[[stock, etf]].dropna()
        if len(aligned) <= beta_window:
            continue

        betas = []
        dates = aligned.index
        for i in range(beta_window, len(dates)):
            window_data = aligned.iloc[i - beta_window:i]
            y = window_data[stock].values
            x = window_data[etf].values
            xx = x @ x
            if xx < 1e-10:
                b = 0.0
            else:
                b = float((x @ y) / xx)
            betas.append({"date": dates[i], "beta": b})

        if not betas:
            continue

        beta_series = pd.Series(
            [b["beta"] for b in betas],
            index=[b["date"] for b in betas],
            name="beta"
        )

        r_stock = aligned[stock].loc[beta_series.index]
        r_factor = aligned[etf].loc[beta_series.index]
        resid = r_stock - beta_series * r_factor
        residuals[stock] = resid

        for b in betas:
            beta_records.append({
                "ticker": stock,
                "etf": etf,
                "date": b["date"],
                "beta": b["beta"]
            })

    beta_df = pd.DataFrame(beta_records).set_index(["date", "ticker"]).sort_index()
    meta = {"mapping": sector_mapping, "betas": beta_df}
    return residuals, meta

# ---------------- OU / AR(1) ESTIMATION ----------------

def fit_ou_from_residuals(
    residuals: pd.Series,
    window: int,
    use_trading_time: bool = False,
    vol_series: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    Robust OU fit from residuals using closed-form AR(1) on cleaned data:
        x_t = a + b x_{t-1} + eps

    Then map to OU:
        kappa = -ln(b) / dt
        mu    = a / (1 - b)
        sigma_eq^2 = Var_eps / (1 - b^2)

    Returns dict with a, b, mu, kappa, sigma, eq_std.
    Never calls np.linalg.lstsq (so no SVD errors).
    """
    if residuals is None or len(residuals) < window + 2:
        return {}

    # 1) Clean raw residuals
    x_raw = residuals.astype(float)
    x_raw = x_raw.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x_raw) < window + 2:
        return {}

    x_raw = x_raw.iloc[-window:]  # last window observations

    # 2) Optional volume-based "trading time" scaling
    if use_trading_time and vol_series is not None:
        vol_hist = vol_series.loc[x_raw.index].astype(float)
        vol_hist = vol_hist.replace([np.inf, -np.inf], np.nan)
        vol_hist = vol_hist.fillna(method="ffill").fillna(method="bfill")

        # Avoid zeros or negatives in volume
        positive = vol_hist[vol_hist > 0]
        if len(positive) > 0:
            vol_hist[vol_hist <= 0] = positive.min()
            vol_norm = vol_hist / vol_hist.mean()
            scale = 1.0 / vol_norm.replace(0, np.nan)
            scale = scale.fillna(1.0)
            x = x_raw * scale
        else:
            x = x_raw.copy()
    else:
        x = x_raw.copy()

    # 3) Build lagged series and clean again
    x_vals = x.values
    if len(x_vals) < 5:
        return {}

    x_tm1 = x_vals[:-1]
    x_t = x_vals[1:]

    good = (
        np.isfinite(x_tm1) &
        np.isfinite(x_t)
    )
    x_tm1 = x_tm1[good]
    x_t = x_t[good]

    if len(x_tm1) < 5:
        return {}

    # 4) Closed-form AR(1) estimation
    xm1_mean = x_tm1.mean()
    xt_mean = x_t.mean()
    denom = np.sum((x_tm1 - xm1_mean) ** 2)
    if denom <= 0 or not np.isfinite(denom):
        return {}

    b = np.sum((x_tm1 - xm1_mean) * (x_t - xt_mean)) / denom
    a = xt_mean - b * xm1_mean

    if not np.isfinite(a) or not np.isfinite(b):
        return {}

    # Keep AR(1) in a reasonable, stationary band
    if b >= 0.9999:
        b = 0.9999
    if b <= -0.9999:
        b = -0.9999

    # 5) Innovations
    eps = x_t - (a + b * x_tm1)
    if len(eps) < 3:
        return {}

    sigma_eps = np.std(eps, ddof=1)
    if not np.isfinite(sigma_eps) or sigma_eps == 0:
        return {}

    # 6) Map AR(1) → OU parameters
    dt = 1.0 / 252.0

    # For OU, we want 0 < b < 1; if b <= 0, treat as very weak mean reversion
    if b <= 0:
        b = 1e-6

    kappa = -np.log(b) / dt
    if not np.isfinite(kappa) or kappa <= 0:
        return {}

    # Stationary variance of AR(1): sigma_eps^2 / (1 - b^2)
    var_stationary = sigma_eps**2 / (1.0 - b**2 + 1e-8)
    if var_stationary <= 0 or not np.isfinite(var_stationary):
        return {}

    sigma_eq = np.sqrt(var_stationary)
    sigma = sigma_eq * np.sqrt(2.0 * kappa)

    if not (np.isfinite(sigma_eq) and np.isfinite(sigma)):
        return {}

    mu = a / (1.0 - b)

    return {
        "a": float(a),
        "b": float(b),
        "mu": float(mu),
        "kappa": float(kappa),
        "sigma": float(sigma),
        "eq_std": float(sigma_eq),
    }

###############################################################
#  FULL PAPER MODEL: SPY + SECTOR ETF + SECTOR PCA (3-Stage)
###############################################################
def build_residuals_full_paper_model(
    returns: pd.DataFrame,
    sector_mapping: Dict[str, str],
    beta_window: int,
    window_corr_days: int,
    explained_target: float,
    market_ticker: str = "SPY",
) -> Tuple[Dict[str, pd.Series], Dict]:
    """
    Full Avellaneda–Lee factor decomposition:
      Stage 1: Remove global market (SPY)
      Stage 2: Remove sector ETF factor exposure
      Stage 3: Sector-level PCA → eigenportfolios
    Produces: final idiosyncratic residuals for OU.
    """

    # REQUIRE SPY
    if market_ticker not in returns.columns:
        raise ValueError("SPY not present in returns!")

    ############################################
    # STAGE 1 — Remove global market factor
    ############################################
    stage1_resid = {}
    for ticker in returns.columns:
        if ticker == market_ticker:
            continue

        aligned = returns[[ticker, market_ticker]].dropna()
        if len(aligned) <= beta_window:
            continue

        betas = []
        idx = aligned.index

        for i in range(beta_window, len(idx)):
            w = aligned.iloc[i-beta_window:i]
            y = w[ticker].values
            x = w[market_ticker].values
            xx = x @ x
            b = (x @ y) / xx if xx > 1e-10 else 0
            betas.append(b)

        beta_series = pd.Series(betas, index=idx[beta_window:])
        r_stock = aligned[ticker].loc[beta_series.index]
        r_mkt = aligned[market_ticker].loc[beta_series.index]
        resid = r_stock - beta_series * r_mkt

        stage1_resid[ticker] = resid

    stage1_resid_df = pd.DataFrame(stage1_resid).dropna(how="all")

    ############################################
    # STAGE 2 — Remove sector ETF factor
    ############################################
    stage2_resid = {}
    for stock, etf in sector_mapping.items():
        if stock not in stage1_resid_df.columns: continue
        if etf not in returns.columns: continue

        aligned = pd.concat(
            [stage1_resid_df[stock], returns[etf]], axis=1,
            keys=["resid1", etf]
        ).dropna()

        if len(aligned) <= beta_window:
            continue

        betas = []
        idx = aligned.index

        for i in range(beta_window, len(idx)):
            w = aligned.iloc[i-beta_window:i]
            y = w["resid1"].values
            x = w[etf].values
            xx = x @ x
            b = (x @ y) / xx if xx > 1e-10 else 0
            betas.append(b)

        beta_series = pd.Series(betas, index=idx[beta_window:])
        r1 = aligned["resid1"].loc[beta_series.index]
        r_etf = aligned[etf].loc[beta_series.index]
        resid2 = r1 - beta_series * r_etf

        stage2_resid[stock] = resid2

    stage2_resid_df = pd.DataFrame(stage2_resid).dropna(how="all")

    ############################################
    # STAGE 3 — Sector-level PCA (Eigenportfolios)
    ############################################
    final_residuals = {}
    for stock, etf in sector_mapping.items():
        sector_stocks = [s for s,e in sector_mapping.items() if e == etf]
        sector_resid = stage2_resid_df[sector_stocks].dropna(how="all")

        if sector_resid.shape[1] < 3: continue
        if len(sector_resid) <= window_corr_days: continue

        # ONLY last window for PCA (rolling PCA)
        window = sector_resid.iloc[-window_corr_days:]
        returns_w = window.pct_change().dropna()
        if returns_w.empty: continue

        resid_pca, eigvals, eigvecs = robust_pca_factors(returns_w)
        if resid_pca.empty: continue

        # final residual = last available day
        final_residuals[stock] = resid_pca[stock]

    return final_residuals, {
        "stage1": stage1_resid_df,
        "stage2": stage2_resid_df,
        "sector_mapping": sector_mapping,
    }

# ---------------- RESIDUALS DISPATCH ----------------

def build_residuals(
    returns: pd.DataFrame,
    factor_model_type: str,
    window_corr_days: int,
    pca_explained_var: float,
    beta_window: int,
    sector_mapping: Dict[str, str]
) -> Tuple[Dict[str, pd.Series], Dict]:
    if factor_model_type == "PCA":
        residuals_df, eigvals, eigvecs = build_residuals_pca(
            prices=returns,
            tickers=user_tickers,
            pca_lookback=window_corr_days
        )
        residuals = {
            col: residuals_df[col].dropna()
            for col in residuals_df.columns
        }
        return residuals, {"method": "robust_pca", "eigvals": eigvals, "eigvecs": eigvecs}

    elif factor_model_type == "Single Market Factor (SPY)":
        return build_residuals_single_factor_rolling(
            returns,
            market_ticker="SPY",
            beta_window=beta_window
        )

    elif factor_model_type == "Sector ETF factors (auto mapping)":
        return build_residuals_sector_etf_rolling(
            returns,
            sector_mapping=sector_mapping,
            beta_window=beta_window
        )

    elif factor_model_type == "SPY + Sector + PCA":
        return build_residuals_full_paper_model(
            returns=returns,
            sector_mapping=sector_mapping,
            beta_window=beta_window,
            window_corr_days=window_corr_days,
            explained_target=pca_explained_var,
            market_ticker="SPY",
        )

    else:
        raise ValueError(f"Unknown factor model type: {factor_model_type}")

# ---------------- COINTEGRATION HELPERS ----------------

def adf_pvalue(series: pd.Series) -> float:
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 30:
        return 1.0
    try:
        return float(adfuller(clean, autolag="AIC")[1])
    except Exception:
        return 1.0


def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> Tuple[float, float]:
    lr = LinearRegression(fit_intercept=True)
    lr.fit(x.values.reshape(-1, 1), y.values)
    return float(lr.coef_[0]), float(lr.intercept_)


def compute_half_life(spread: pd.Series) -> float:
    # AR(1) on spread changes for half-life proxy
    spread = pd.Series(spread).dropna()
    if len(spread) < 30:
        return np.inf
    y = spread.diff().dropna()
    x = spread.shift(1).reindex(y.index)
    if y.std() == 0 or x.std() == 0:
        return np.inf
    b = np.polyfit(x, y, 1)[0]
    if b >= 0:
        return np.inf
    return -np.log(2) / b


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    tickers: List[str],
    lookback: int,
    corr_floor: float,
    adf_alpha: float
) -> pd.DataFrame:
    window = prices[tickers].iloc[-lookback:]
    logp = np.log(window.replace(0, np.nan)).dropna(how="all")

    rows = []
    for i, t1 in enumerate(tickers):
        if t1 not in logp.columns:
            continue
        for t2 in tickers[i+1:]:
            if t2 not in logp.columns:
                continue
            s1, s2 = logp[t1].dropna(), logp[t2].dropna()
            common = s1.index.intersection(s2.index)
            if len(common) < 60:
                continue
            s1, s2 = s1.loc[common], s2.loc[common]
            corr = s1.corr(s2)
            if corr is None or np.isnan(corr) or corr < corr_floor:
                continue

            beta, intercept = estimate_hedge_ratio(s1, s2)
            spread = s1 - (beta * s2 + intercept)
            pval = adf_pvalue(spread)
            if pval > adf_alpha:
                continue
            hl = compute_half_life(spread)
            mu = spread.mean()
            std = spread.std()
            last_s = (spread.iloc[-1] - mu) / std if std and std > 0 else np.nan
            rows.append({
                "pair": f"{t1}/{t2}",
                "leg1": t1,
                "leg2": t2,
                "beta": beta,
                "intercept": intercept,
                "corr": corr,
                "adf_p": pval,
                "half_life": hl,
                "last_s": last_s,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["adf_p", "corr"])


def backtest_pairs(
    prices: pd.DataFrame,
    pairs_df: pd.DataFrame,
    ou_window_days: int,
    entry_z: float,
    exit_z: float,
    tc_bps: float,
    initial_equity: float,
    notional_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if pairs_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    tc = tc_bps / 10000.0
    dates = prices.index
    equity = initial_equity
    positions: Dict[str, Dict[str, float]] = {}
    equity_curve = []
    trade_records = []

    for i in range(1, len(dates)):
        d_prev, d = dates[i - 1], dates[i]
        price_prev, price_today = prices.loc[d_prev], prices.loc[d]

        # Mark to market
        pnl = 0.0
        for key, pos in positions.items():
            l1, l2 = pos["leg1"], pos["leg2"]
            if l1 not in price_today or l2 not in price_today:
                continue
            pnl += pos["q1"] * (price_today[l1] - price_prev[l1])
            pnl += pos["q2"] * (price_today[l2] - price_prev[l2])
        equity += pnl

        for _, row in pairs_df.iterrows():
            l1, l2 = row.leg1, row.leg2
            if l1 not in price_today or l2 not in price_today:
                continue
            series = np.log(prices[[l1, l2]].loc[:d].replace(0, np.nan)).dropna()
            if len(series) < ou_window_days:
                continue

            spread_hist = series[l1] - (row.beta * series[l2] + row.intercept)
            tail = spread_hist.iloc[-ou_window_days:]
            if tail.std() == 0 or np.isnan(tail.std()):
                continue
            s_score = (spread_hist.iloc[-1] - tail.mean()) / tail.std()

            key = f"{l1}/{l2}"
            active = key in positions

            # Exit logic
            if active and abs(s_score) <= exit_z:
                q1 = positions[key]["q1"]
                q2 = positions[key]["q2"]
                traded = abs(q1) * price_today[l1] + abs(q2) * price_today[l2]
                equity -= traded * tc
                trade_records.append({
                    "date": d,
                    "pair": key,
                    "leg1": l1,
                    "leg2": l2,
                    "action": "EXIT",
                    "s_score": s_score,
                    "price1": price_today[l1],
                    "price2": price_today[l2],
                    "equity": equity
                })
                positions.pop(key, None)
                continue

            # Entry logic
            if not active and abs(s_score) >= entry_z:
                notional = equity * notional_frac
                if notional <= 0:
                    continue
                qty1 = notional / price_today[l1]
                qty2 = -row.beta * qty1
                if s_score > 0:
                    qty1 *= -1.0
                    qty2 *= -1.0
                traded = abs(qty1) * price_today[l1] + abs(qty2) * price_today[l2]
                equity -= traded * tc
                positions[key] = {"q1": qty1, "q2": qty2, "leg1": l1, "leg2": l2}
                trade_records.append({
                    "date": d,
                    "pair": key,
                    "leg1": l1,
                    "leg2": l2,
                    "action": "ENTER",
                    "s_score": s_score,
                    "price1": price_today[l1],
                    "price2": price_today[l2],
                    "qty1": qty1,
                    "qty2": qty2,
                    "equity": equity
                })

        gross_exposure = 0.0
        for pos in positions.values():
            gross_exposure += abs(pos["q1"]) * price_today[pos["leg1"]]
            gross_exposure += abs(pos["q2"]) * price_today[pos["leg2"]]

        equity_curve.append({
            "date": d,
            "equity": equity,
            "gross_exposure": gross_exposure
        })

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trade_records)
    if not trades_df.empty:
        trades_df.set_index("date", inplace=True)
        trades_df.sort_index(inplace=True)
    return equity_df, trades_df

# ---------------- BACKTEST ENGINE ----------------

def generate_signals_and_backtest(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    residuals: Dict[str, pd.Series],
    ou_window_days: int,
    entry_z: float,
    exit_z: float,
    tc_bps: float,
    leverage_long: float,
    leverage_short: float,
    initial_equity: float,
    use_trading_time: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Backtest engine:
      - Rolling OU fit per stock
      - s-scores & contrarian entry/exit
      - Equal-notional long/short with leverage
      - Transaction costs in bps
      - Records per-ticker positions AND discrete trade events.
    """
    tc = tc_bps / 10000.0
    if not residuals:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_dates = sorted(set().union(*[s.index for s in residuals.values()]))
    all_dates = [d for d in all_dates if d in prices.index]

    equity = initial_equity
    positions = {ticker: 0.0 for ticker in prices.columns}

    equity_curve_records = []
    position_records = []
    trade_records = []

    for i, date in enumerate(all_dates):
        prev_date = all_dates[i - 1] if i > 0 else date
        price_today = prices.loc[date]
        price_prev = prices.loc[prev_date]

        # 1) PnL from existing positions (before new trades)
        if i > 0:
            pnl = 0.0
            for ticker, qty in positions.items():
                if qty == 0:
                    continue
                if ticker in price_today and ticker in price_prev:
                    pnl += qty * (price_today[ticker] - price_prev[ticker])
            equity += pnl
        else:
            pnl = 0.0

        equity_before_trades = equity

        # 2) s-scores via OU fit up to today (using last ou_window_days)
        s_scores: Dict[str, float] = {}
        for ticker, resid_series in residuals.items():
            resid_hist = resid_series[resid_series.index <= date]
            if len(resid_hist) <= ou_window_days:
                continue
            vol_series = volume[ticker] if ticker in volume.columns else None
            params = fit_ou_from_residuals(
                resid_hist,
                window=ou_window_days,
                use_trading_time=use_trading_time,
                vol_series=vol_series
            )
            if not params:
                continue

            # Build consistent x_t (scaled if needed)
            if use_trading_time and vol_series is not None:
                x_raw = resid_hist.iloc[-ou_window_days:]
                vol_hist = vol_series.loc[x_raw.index].astype(float)
                vol_hist = vol_hist.replace([np.inf, -np.inf], np.nan)
                vol_hist = vol_hist.fillna(method="ffill").fillna(method="bfill")
                positive = vol_hist[vol_hist > 0]
                if len(positive) > 0:
                    vol_hist[vol_hist <= 0] = positive.min()
                    vol_norm = vol_hist / vol_hist.mean()
                    scale = 1.0 / vol_norm.replace(0, np.nan)
                    scale = scale.fillna(1.0)
                    x_scaled = (x_raw * scale).iloc[-1]
                else:
                    x_scaled = x_raw.iloc[-1]
            else:
                x_scaled = resid_hist.iloc[-1]

            mu = params["mu"]
            eq_std = params["eq_std"]
            if eq_std <= 1e-8 or not np.isfinite(eq_std):
                continue
            s = (x_scaled - mu) / eq_std
            if not np.isfinite(s):
                continue
            s_scores[ticker] = s

        # 3) Determine entry/exit signals
        target_dollar_long = leverage_long * equity
        target_dollar_short = leverage_short * equity

        longs = []
        shorts = []
        exits = []

        for ticker, s in s_scores.items():
            if ticker not in price_today or price_today[ticker] <= 0:
                continue
            current_qty = positions.get(ticker, 0.0)
            current_dollar = current_qty * price_today[ticker]

            if abs(current_dollar) < 1e-8:
                # Flat -> entry
                if s <= -entry_z:
                    longs.append(ticker)
                elif s >= entry_z:
                    shorts.append(ticker)
            else:
                # Exit rule
                if abs(s) <= exit_z:
                    exits.append(ticker)

        # 4) Apply exits first
        traded_notional = 0.0
        for ticker in exits:
            if ticker not in price_today:
                continue
            price = price_today[ticker]
            old_qty = positions.get(ticker, 0.0)
            if old_qty == 0:
                continue
            traded_notional += abs(old_qty) * price
            positions[ticker] = 0.0

            trade_records.append({
                "date": date,
                "ticker": ticker,
                "action": "EXIT",
                "qty": 0.0,
                "price": price,
                "s_score": s_scores.get(ticker, np.nan),
                "equity_before": equity_before_trades,
                # equity_after filled after costs below
            })

        # 5) New entries: equal-notional allocation
        n_longs = len(longs)
        n_shorts = len(shorts)
        per_long = target_dollar_long / n_longs if n_longs > 0 else 0.0
        per_short = target_dollar_short / n_shorts if n_shorts > 0 else 0.0

        for ticker in longs:
            price = price_today[ticker]
            if price <= 0:
                continue

            old_qty = positions.get(ticker, 0.0)

            # PATCH 6 — Do NOT resize if we are already long
            if old_qty > 0:
                continue

            new_qty = per_long / price
            traded_notional += abs(new_qty - old_qty) * price
            positions[ticker] = new_qty

            trade_records.append({
                "date": date,
                "ticker": ticker,
                "action": "ENTER_LONG",
                "qty": new_qty,
                "price": price,
                "s_score": s_scores.get(ticker, np.nan),
                "equity_before": equity_before_trades,
            })


        for ticker in shorts:
            price = price_today[ticker]
            if price <= 0:
                continue

            old_qty = positions.get(ticker, 0.0)

            # PATCH 6 — Do NOT resize if we are already short
            if old_qty < 0:
                continue

            new_qty = -per_short / price
            traded_notional += abs(new_qty - old_qty) * price
            positions[ticker] = new_qty

            trade_records.append({
                "date": date,
                "ticker": ticker,
                "action": "ENTER_SHORT",
                "qty": new_qty,
                "price": price,
                "s_score": s_scores.get(ticker, np.nan),
                "equity_before": equity_before_trades,
            })

        # 6) Apply transaction costs
        if traded_notional > 0:
            equity -= traded_notional * tc

        equity_after_trades = equity

        # Fill equity_after in trade_records for this date
        for rec in trade_records:
            if rec["date"] == date and "equity_after" not in rec:
                rec["equity_after"] = equity_after_trades

        # 7) Exposure stats & per-ticker snapshot
        gross_exposure = 0.0
        n_long_active = 0
        n_short_active = 0
        for ticker, qty in positions.items():
            if ticker not in price_today:
                continue
            dollar = qty * price_today[ticker]
            gross_exposure += abs(dollar)
            if dollar > 0:
                n_long_active += 1
            elif dollar < 0:
                n_short_active += 1

        equity_curve_records.append({
            "date": date,
            "equity": equity,
            "pnl": pnl,
            "gross_exposure": gross_exposure,
            "n_longs": n_long_active,
            "n_shorts": n_short_active,
            "traded_notional": traded_notional
        })

        for ticker in prices.columns:
            qty = positions.get(ticker, 0.0)
            s = s_scores.get(ticker, np.nan)
            position_records.append({
                "date": date,
                "ticker": ticker,
                "price": price_today.get(ticker, np.nan),
                "qty": qty,
                "s_score": s
            })

    equity_df = pd.DataFrame(equity_curve_records).set_index("date")
    detailed_df = pd.DataFrame(position_records).set_index(["date", "ticker"])
    trades_df = pd.DataFrame(trade_records)
    if not trades_df.empty:
        trades_df.set_index("date", inplace=True)
        trades_df.sort_index(inplace=True)

    return equity_df, detailed_df, trades_df

# ---------------- LIVE SIGNALS ENGINE ----------------

def compute_live_signals(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    residuals: Dict[str, pd.Series],
    ou_window_days: int,
    entry_z: float,
    exit_z: float,
    use_trading_time: bool
) -> pd.DataFrame:
    """
    Live mode:
      - Use latest available date
      - Fit OU on last ou_window_days residuals
      - Compute s-score and suggested action per stock
    """
    if not residuals:
        return pd.DataFrame()

    all_dates = sorted(set().union(*[s.index for s in residuals.values()]))
    all_dates = [d for d in all_dates if d in prices.index]
    if not all_dates:
        return pd.DataFrame()
    last_date = all_dates[-1]

    records = []
    for ticker, resid_series in residuals.items():
        resid_hist = resid_series[resid_series.index <= last_date]
        if len(resid_hist) <= ou_window_days:
            continue
        vol_series = volume[ticker] if ticker in volume.columns else None
        params = fit_ou_from_residuals(
            resid_hist,
            window=ou_window_days,
            use_trading_time=use_trading_time,
            vol_series=vol_series
        )
        if not params:
            continue

        if use_trading_time and vol_series is not None:
            x_raw = resid_hist.iloc[-ou_window_days:]
            vol_hist = vol_series.loc[x_raw.index].copy()
            vol_hist = vol_hist.replace(0, np.nan).fillna(method="ffill").fillna(method="bfill")
            if (vol_hist > 0).any():
                vol_norm = vol_hist / vol_hist.mean()
                scale = 1.0 / vol_norm.replace(0, np.nan)
                scale = scale.fillna(1.0)
                x_scaled = (x_raw * scale).iloc[-1]
            else:
                x_scaled = x_raw.iloc[-1]
        else:
            x_scaled = resid_hist.iloc[-1]

        mu = params["mu"]
        eq_std = params["eq_std"]
        if eq_std <= 1e-8:
            continue
        s = (x_scaled - mu) / eq_std

        if s <= -entry_z:
            action = "Go LONG"
        elif s >= entry_z:
            action = "Go SHORT"
        elif abs(s) <= exit_z:
            action = "CLOSE (if any)"
        else:
            action = "NO TRADE"

        price_today = prices.loc[last_date, ticker] if ticker in prices.columns else np.nan

        records.append({
            "date": last_date,
            "ticker": ticker,
            "price": price_today,
            "residual_last": float(resid_hist.iloc[-1]),
            "mu_hat": mu,
            "eq_std": eq_std,
            "s_score": s,
            "action": action
        })

    df = pd.DataFrame(records).sort_values("s_score", ascending=False)
    df.set_index("ticker", inplace=True)
    return df

# =========================================================
# MAIN EXECUTION
# =========================================================

if st.session_state["run_backtest"]:
    if len(user_tickers) < 2:
        st.error("Please provide at least 2 tickers in the universe.")
    elif factor_model_type == "Cointegration (pairs)":
        with st.spinner("Loading prices for cointegration scan..."):
            prices, volume = load_price_and_volume(
                user_tickers,
                start_date,
                end_date
            )
        if prices.empty:
            st.error("No price data loaded – check tickers and dates.")
            st.stop()

        with st.spinner("Scanning for cointegrated pairs..."):
            pairs_df = find_cointegrated_pairs(
                prices=prices,
                tickers=user_tickers,
                lookback=window_corr_days,
                corr_floor=pair_corr_floor,
                adf_alpha=pair_adf_alpha
            )

        if pairs_df.empty:
            st.warning("No cointegrated pairs found with current filters.")
            st.stop()

        filtered_pairs = pairs_df[
            pairs_df["half_life"].between(pair_min_hl, pair_max_hl)
        ]
        if filtered_pairs.empty:
            st.warning("Pairs found, but none within the half-life band. Relax filters.")
            st.stop()

        st.subheader("Candidate Cointegrated Pairs")
        st.dataframe(filtered_pairs.assign(
            adf_p=lambda df: df["adf_p"].round(4),
            corr=lambda df: df["corr"].round(3),
            half_life=lambda df: df["half_life"].round(1),
            last_s=lambda df: df["last_s"].round(2),
        ))

        with st.spinner("Backtesting cointegration pairs..."):
            equity_df, trades_df = backtest_pairs(
                prices=prices[user_tickers],
                pairs_df=filtered_pairs,
                ou_window_days=ou_window_days,
                entry_z=entry_z,
                exit_z=exit_z,
                tc_bps=tc_bps,
                initial_equity=initial_equity,
                notional_frac=pair_notional_frac
            )

        if equity_df.empty:
            st.warning("Cointegration backtest produced no results.")
            st.stop()

        st.subheader("Equity Curve & Performance (Cointegration)")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.line_chart(equity_df[["equity"]])
        with col2:
            total_return = equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1
            daily_ret = equity_df["equity"].pct_change().dropna()
            ann_ret = (1 + daily_ret.mean()) ** 252 - 1
            ann_vol = daily_ret.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
            st.metric("Final Equity", f"${equity_df['equity'].iloc[-1]:,.0f}")
            st.metric("Total Return", f"{100 * total_return:,.1f}%")
            st.metric("Annualized Return", f"{100 * ann_ret:,.1f}%")
            st.metric("Annualized Vol", f"{100 * ann_vol:,.1f}%")
            st.metric("Sharpe (no RF)", f"{sharpe:,.2f}")

        # Additional cointegration diagnostics
        st.subheader("Cointegration Stats")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Pairs scanned", f"{len(pairs_df)}")
        c2.metric("Pairs kept", f"{len(filtered_pairs)}")
        c3.metric("Avg corr (kept)", f"{filtered_pairs['corr'].mean():.2f}")
        c4.metric("Med half-life (days)", f"{filtered_pairs['half_life'].median():.1f}")
        entries = int((trades_df["action"] == "ENTER").sum()) if not trades_df.empty else 0
        exits = int((trades_df["action"] == "EXIT").sum()) if not trades_df.empty else 0
        c5.metric("Entries", f"{entries}")
        c6.metric("Exits", f"{exits}")

        st.markdown("**Top signals (sorted by latest s-score)**")
        st.dataframe(
            filtered_pairs.sort_values("last_s", ascending=False)[
                ["pair", "corr", "adf_p", "half_life", "last_s", "beta"]
            ].head(10)
        )

        if not trades_df.empty:
            st.subheader("Pair Trade Log")
            st.dataframe(trades_df[
                ["pair", "action", "s_score", "price1", "price2", "equity"]
            ])

            pairs_list = sorted(trades_df["pair"].unique().tolist())
            chosen_pair = st.selectbox("Inspect pair:", pairs_list)
            pair_trades = trades_df[trades_df["pair"] == chosen_pair]
            l1 = pair_trades["leg1"].iloc[0]
            l2 = pair_trades["leg2"].iloc[0]

            log_prices = np.log(prices[[l1, l2]].replace(0, np.nan)).dropna()
            beta = filtered_pairs.loc[filtered_pairs["pair"] == chosen_pair, "beta"].iloc[0]
            intercept = filtered_pairs.loc[filtered_pairs["pair"] == chosen_pair, "intercept"].iloc[0]
            spread = log_prices[l1] - (beta * log_prices[l2] + intercept)

            fig_spread = go.Figure()
            fig_spread.add_trace(go.Scatter(x=spread.index, y=spread.values, name="Spread"))
            mu, std = spread.mean(), spread.std()
            fig_spread.add_hline(mu + entry_z * std, line=dict(color="red", dash="dash"), annotation_text="+entry")
            fig_spread.add_hline(mu - entry_z * std, line=dict(color="red", dash="dash"), annotation_text="-entry")
            fig_spread.add_hline(mu, line=dict(color="gray", dash="dot"), annotation_text="mean")
            for action, marker, color in [("ENTER", "triangle-up", "green"), ("EXIT", "circle", "orange")]:
                sub = pair_trades[pair_trades["action"] == action]
                if sub.empty:
                    continue
                fig_spread.add_trace(
                    go.Scatter(
                        x=sub.index,
                        y=spread.loc[sub.index],
                        mode="markers",
                        name=action,
                        marker=dict(symbol=marker, color=color, size=9, line=dict(width=1, color="black"))
                    )
                )
            fig_spread.update_layout(title=f"Spread with signals – {chosen_pair}", hovermode="x unified")
            st.plotly_chart(fig_spread, use_container_width=True)
        else:
            st.info("No trades triggered in cointegration backtest.")

    else:
        # Build sector mapping automatically if needed
        if factor_model_type in [
            "Sector ETF factors (auto mapping)",
            "SPY + Sector + PCA"
        ]:
            sector_mapping = auto_sector_etf_mapping(user_tickers)
            if not sector_mapping:
                st.error("Automatic sector ETF mapping failed for all tickers.")
                st.stop()
        else:
            sector_mapping = {}

        # Build final ticker list (include ETFs for sector model or SPY)
        extra_tickers = set()

        if factor_model_type == "Sector ETF factors (auto mapping)":
            extra_tickers |= set(sector_mapping.values())

        elif factor_model_type == "Single Market Factor (SPY)":
            extra_tickers.add("SPY")

        elif factor_model_type == "SPY + Sector + PCA":
            extra_tickers.add("SPY")
            extra_tickers |= set(sector_mapping.values())


        all_tickers = sorted(set(user_tickers) | extra_tickers)

        with st.spinner("Loading data and building factor/residual model..."):
            prices, volume = load_price_and_volume(
                all_tickers,
                start_date,
                end_date
            )
            if prices.empty:
                st.error("No price data loaded – check tickers and dates.")
            else:
                st.success(f"Loaded price data for {len(prices.columns)} tickers.")

                returns = compute_returns(prices)
                try:
                    residuals, meta = build_residuals(
                        returns,
                        factor_model_type=factor_model_type,
                        window_corr_days=window_corr_days,
                        pca_explained_var=pca_explained_var,
                        beta_window=beta_window_days,
                        sector_mapping=sector_mapping
                    )
                except Exception as e:
                    st.error(f"Error while building residuals: {e}")
                    residuals, meta = {}, {}

        if prices.empty:
            pass
        elif not residuals:
            st.error("Residuals could not be built – check factor model & mappings.")
        else:
            # Persist for diagnostics pages
            st.session_state["prices"] = prices
            st.session_state["volume"] = volume
            st.session_state["returns"] = returns
            st.session_state["residuals"] = residuals
            st.session_state["residuals_meta"] = meta
            st.session_state["factor_model_type"] = factor_model_type

            # Show mapping if using sector ETF model
            if factor_model_type == "SPY + Sector + PCA":
                st.subheader("Sector mapping used in full model")
                mapping_df = pd.DataFrame(
                    [{"Ticker": k, "Sector ETF": v} for k, v in sector_mapping.items()]
                ).set_index("Ticker")
                st.dataframe(mapping_df)

            if mode == "Backtest":
                with st.spinner("Running backtest..."):
                    equity_df, detailed_df, trades_df = generate_signals_and_backtest(
                        prices=prices,
                        volume=volume,
                        residuals=residuals,
                        ou_window_days=ou_window_days,
                        entry_z=entry_z,
                        exit_z=exit_z,
                        tc_bps=tc_bps,
                        leverage_long=leverage_long,
                        leverage_short=leverage_short,
                        initial_equity=initial_equity,
                        use_trading_time=use_trading_time
                    )

                if equity_df.empty:
                    st.warning("Backtest produced no results – possibly insufficient history.")
                else:
                    st.subheader("Equity Curve & Performance")

                    # Save for analytics pages
                    st.session_state["equity_df"] = equity_df
                    st.session_state["detailed_df"] = detailed_df
                    st.session_state["trades_df"] = trades_df

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.line_chart(equity_df[["equity"]])

                    with col2:
                        total_return = equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1
                        daily_ret = equity_df["equity"].pct_change().dropna()
                        ann_ret = (1 + daily_ret.mean()) ** 252 - 1
                        ann_vol = daily_ret.std() * np.sqrt(252)
                        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

                        st.metric("Final Equity", f"${equity_df['equity'].iloc[-1]:,.0f}")
                        st.metric("Total Return", f"{100 * total_return:,.1f}%")
                        st.metric("Annualized Return", f"{100 * ann_ret:,.1f}%")
                        st.metric("Annualized Vol", f"{100 * ann_vol:,.1f}%")
                        st.metric("Sharpe (no RF)", f"{sharpe:,.2f}")

                    st.subheader("Exposure & Turnover")
                    st.area_chart(equity_df[["gross_exposure"]])

                    st.write("Sample of daily stats:")
                    st.dataframe(equity_df.tail())

                    st.subheader("Per-stock s-scores & positions on last backtest day")
                    last_date = equity_df.index[-1]
                    last_positions = detailed_df.xs(last_date, level="date")
                    st.dataframe(
                        last_positions.sort_values("s_score", ascending=False)
                    )
                    if not trades_df.empty:
                        st.subheader("Per-Ticker Trade Log & Signal Charts")

                        # Only tickers that actually traded
                        traded_tickers = sorted(trades_df["ticker"].unique().tolist())
                        selected_ticker = st.selectbox(
                            "Select ticker to inspect trades:",
                            traded_tickers
                        )

                        # --- Trade log table for selected ticker ---
                        ticker_trades = trades_df[trades_df["ticker"] == selected_ticker].copy()
                        ticker_trades = ticker_trades.sort_index()
                        st.markdown(f"**Trade log for {selected_ticker}:**")
                        st.dataframe(
                            ticker_trades[
                                ["action", "qty", "price", "s_score", "equity_before", "equity_after"]
                            ]
                        )

                        # --- Price + signal markers chart ---
                        st.markdown(f"**Price with LONG / SHORT / EXIT signals – {selected_ticker}**")

                        # Restrict to backtest window where we have equity data
                        backtest_dates = equity_df.index
                        price_series = prices[selected_ticker].loc[
                            prices.index.intersection(backtest_dates)
                        ].dropna()

                        # --- Construct OHLC (yfinance has only Adj Close here, so fetch OHLC separately) ---
                        stock_ohlc = yf.download(
                            selected_ticker,
                            start=start_date,
                            end=end_date + timedelta(days=1),
                            progress=False
                        )

                        if isinstance(stock_ohlc.columns, pd.MultiIndex):
                            stock_ohlc = stock_ohlc.droplevel(1, axis=1)

                        stock_ohlc = stock_ohlc[["Open", "High", "Low", "Close"]].loc[price_series.index]

                        # ETF for overlay
                        selected_etf = None
                        if factor_model_type == "Sector ETF factors (auto mapping)":
                            if selected_ticker in sector_mapping:
                                selected_etf = sector_mapping[selected_ticker]

                        if factor_model_type == "Single Market Factor (SPY)":
                            selected_etf = "SPY"

                        fig_price = go.Figure()

                        # 1. Candlestick
                        fig_price.add_trace(
                            go.Candlestick(
                                x=stock_ohlc.index,
                                open=stock_ohlc["Open"],
                                high=stock_ohlc["High"],
                                low=stock_ohlc["Low"],
                                close=stock_ohlc["Close"],
                                name=f"{selected_ticker} OHLC"
                            )
                        )

                        # 2. ETF overlay
                        if selected_etf and selected_etf in prices.columns:
                            etf_series = prices[selected_etf].loc[stock_ohlc.index]
                            fig_price.add_trace(
                                go.Scatter(
                                    x=etf_series.index,
                                    y=etf_series.values,
                                    mode="lines",
                                    name=f"{selected_etf} ETF",
                                    line=dict(color="yellow", width=2),
                                    yaxis="y2"
                                )
                            )

                        fig_price.update_layout(
                            xaxis_rangeslider_visible=False,
                            yaxis=dict(title=f"{selected_ticker} Price"),
                            yaxis2=dict(
                                title="ETF",
                                overlaying="y",
                                side="right",
                                showgrid=False
                            ),
                            hovermode="x unified"
                        )


                        # Marker helper
                        def add_trade_markers(fig, df, action, marker_symbol, marker_color, name):
                            sub = df[df["action"] == action]
                            if sub.empty:
                                return
                            fig.add_trace(
                                go.Scatter(
                                    x=sub.index,
                                    y=sub["price"],
                                    mode="markers",
                                    name=name,
                                    marker=dict(
                                        symbol=marker_symbol,
                                        color=marker_color,
                                        size=10,
                                        line=dict(width=1, color="black"),
                                    )
                                )
                            )

                        add_trade_markers(fig_price, ticker_trades, "ENTER_LONG", "triangle-up", "green", "Enter LONG")
                        add_trade_markers(fig_price, ticker_trades, "ENTER_SHORT", "triangle-down", "red", "Enter SHORT")
                        add_trade_markers(fig_price, ticker_trades, "EXIT", "circle", "gold", "EXIT")

                        fig_price.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Price",
                            legend_title="Signals",
                            hovermode="x unified",
                        )

                        st.plotly_chart(fig_price, use_container_width=True)

                        # --- s-score chart with thresholds + markers ---
                        st.markdown(f"**s-score with entry/exit bands – {selected_ticker}**")

                        # Extract s-scores for this ticker from detailed_df
                        try:
                            ticker_detail = detailed_df.xs(selected_ticker, level="ticker").sort_index()
                        except KeyError:
                            ticker_detail = pd.DataFrame(columns=["s_score"])

                        s_series = ticker_detail["s_score"].dropna()

                        fig_s = go.Figure()
                        fig_s.add_trace(
                            go.Scatter(
                                x=s_series.index,
                                y=s_series.values,
                                mode="lines",
                                name="s-score"
                            )
                        )

                        # Threshold bands
                        if not s_series.empty:
                            x0 = s_series.index.min()
                            x1 = s_series.index.max()

                            def add_hline(y, name, color, dash="dash"):
                                fig_s.add_shape(
                                    type="line",
                                    x0=x0, x1=x1,
                                    y0=y, y1=y,
                                    line=dict(color=color, dash=dash),
                                )
                                fig_s.add_annotation(
                                    x=x1,
                                    y=y,
                                    xanchor="left",
                                    yanchor="middle",
                                    text=name,
                                    showarrow=False,
                                    font=dict(size=10, color=color),
                                )

                            add_hline(entry_z, f"+entry ({entry_z})", "red")
                            add_hline(-entry_z, f"-entry ({-entry_z})", "red")
                            add_hline(exit_z, f"+exit ({exit_z})", "orange", dash="dot")
                            add_hline(-exit_z, f"-exit ({-exit_z})", "orange", dash="dot")
                            add_hline(0.0, "mean", "gray", dash="dot")

                        # Markers for trades on s-score chart
                        if not ticker_trades.empty and "s_score" in ticker_trades.columns:
                            def add_s_markers(fig, df, action, marker_symbol, marker_color, name):
                                sub = df[df["action"] == action].copy()
                                sub = sub[sub["s_score"].notna()]
                                if sub.empty:
                                    return
                                fig.add_trace(
                                    go.Scatter(
                                        x=sub.index,
                                        y=sub["s_score"],
                                        mode="markers",
                                        name=name,
                                        marker=dict(
                                            symbol=marker_symbol,
                                            color=marker_color,
                                            size=9,
                                            line=dict(width=1, color="black"),
                                        )
                                    )
                                )

                            add_s_markers(fig_s, ticker_trades, "ENTER_LONG", "triangle-up", "green", "Enter LONG")
                            add_s_markers(fig_s, ticker_trades, "ENTER_SHORT", "triangle-down", "red", "Enter SHORT")
                            add_s_markers(fig_s, ticker_trades, "EXIT", "circle", "gold", "EXIT")

                        fig_s.update_layout(
                            xaxis_title="Date",
                            yaxis_title="s-score",
                            legend_title="Signals",
                            hovermode="x unified",
                        )

                        st.plotly_chart(fig_s, use_container_width=True)
                    else:
                        st.info("No trades recorded in backtest (no s-scores crossed entry thresholds).")


            else:  # Live Signals
                with st.spinner("Computing live s-scores and actions..."):
                    live_df = compute_live_signals(
                        prices=prices,
                        volume=volume,
                        residuals=residuals,
                        ou_window_days=ou_window_days,
                        entry_z=entry_z,
                        exit_z=exit_z,
                        use_trading_time=use_trading_time
                    )

                if live_df.empty:
                    st.warning("No live signals available – likely not enough history per stock.")
                else:
                    st.subheader("Live Signals (latest date)")

                    last_date = live_df["date"].iloc[0]
                    st.markdown(f"**Signal date:** `{last_date.date()}`")

                    st.dataframe(
                        live_df.sort_values("s_score", ascending=False)[
                            ["price", "residual_last", "mu_hat", "eq_std", "s_score", "action"]
                        ]
                    )

                    st.markdown(
                        """
                        - **Go LONG**: stock appears cheap vs factor; residual far below mean  
                        - **Go SHORT**: stock appears rich vs factor; residual far above mean  
                        - **CLOSE (if any)**: s-score near 0 ⇒ mean reversion mostly realized  
                        - **NO TRADE**: residual within neutral band
                        """
                    )
else:
    st.info("Configure parameters in the sidebar and click **Run** to backtest, or **Compute Live Signals** to see current trades.")

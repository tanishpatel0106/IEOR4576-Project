import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Factor Diagnostics", page_icon="🧮", layout="wide")

# =========================================================
# PAGE TITLE
# =========================================================
st.title("🧮 Factor Diagnostics")

# =========================================================
# CHECK FOR DATA
# =========================================================
if (
    "prices" not in st.session_state or
    "returns" not in st.session_state or
    "factor_model_type" not in st.session_state or
    "residuals_meta" not in st.session_state
):
    st.warning("Run Backtest or Live Signals first to generate factor diagnostics.")
    st.stop()

prices = st.session_state["prices"]
returns = st.session_state["returns"]
factor_model_type = st.session_state["factor_model_type"]
meta = st.session_state["residuals_meta"]

st.markdown(f"### Current Factor Model: **{factor_model_type}**")

# =========================================================
# 1. CORRELATION MATRIX — ETFs grouped separately
# =========================================================
st.header("1. Correlation Matrix (ETFs grouped separately)")

KNOWN_ETFS = {
    "SPY", "QQQ", "DIA", "IWM",
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLB", "XLY", "XLP", "XLU",
    "XLRE", "XLC"
}

tickers = list(returns.columns)
etf_list = [t for t in tickers if t in KNOWN_ETFS]
stock_list = [t for t in tickers if t not in KNOWN_ETFS]

ordered_cols = etf_list + stock_list
corr = returns[ordered_cols].corr()

fig_corr = px.imshow(
    corr,
    labels=dict(color="Correlation"),
    x=ordered_cols,
    y=ordered_cols,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
)
fig_corr.update_layout(height=800)

# Add separation line between ETFs and stocks
if len(etf_list) > 0 and len(stock_list) > 0:
    split = len(etf_list) - 0.5

    fig_corr.add_shape(
        type="line",
        x0=split, x1=split,
        y0=-0.5, y1=len(ordered_cols)-0.5,
        line=dict(color="black", width=2)
    )
    fig_corr.add_shape(
        type="line",
        x0=-0.5, x1=len(ordered_cols)-0.5,
        y0=split, y1=split,
        line=dict(color="black", width=2)
    )

st.plotly_chart(fig_corr, use_container_width=True)

st.markdown(f"""
**ETFs Detected:** `{etf_list}`  
**Stocks Detected:** `{stock_list}`
""")

# =========================================================
# 2. FACTOR EXPOSURE MAP (β heatmap)
# =========================================================
st.header("2. Factor Exposure Map (β Loadings)")

if factor_model_type == "PCA" and "eigvecs" in meta:
    eigvecs = meta["eigvecs"]

    num_factors = min(10, eigvecs.shape[1])
    loadings = pd.DataFrame(
        eigvecs[:, :num_factors],
        index=returns.columns,
        columns=[f"PC{i+1}" for i in range(num_factors)]
    )

    fig_load = px.imshow(
        loadings.T,
        labels=dict(color="Loading"),
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
    )
    fig_load.update_layout(height=600)

    st.plotly_chart(fig_load, use_container_width=True)
    st.dataframe(loadings)

else:
    st.info("β loadings available only for PCA model.")

# =========================================================
# 3. PCA Eigenportfolio Visualizer
# =========================================================
st.header("3. PCA Eigenportfolio Visualizer")

if factor_model_type == "PCA" and "eigvecs" in meta:
    eigvecs = meta["eigvecs"]
    tickers = returns.columns

    num_components = min(5, eigvecs.shape[1])
    selected_pc = st.selectbox(
        "Select PCA Component (Eigenportfolio)",
        [f"PC{i+1}" for i in range(num_components)]
    )
    idx = int(selected_pc[2:]) - 1

    eigenportfolio = pd.Series(eigvecs[:, idx], index=tickers)

    fig_ep = px.bar(
        eigenportfolio.sort_values(),
        title=f"Eigenportfolio Weights — {selected_pc}",
        labels={"value": "Weight", "index": "Ticker"}
    )

    st.plotly_chart(fig_ep, use_container_width=True)

else:
    st.info("Eigenportfolios available only for PCA model.")

# =========================================================
# 4. Residual Mean-Reversion Diagnostics (t-Statistics)
# =========================================================
st.header("4. Residual Mean-Reversion Diagnostics (t-Statistics)")

if "residuals" in st.session_state:
    residuals = st.session_state["residuals"]

    t_stats = {}
    for ticker, series in residuals.items():
        x = series.dropna()
        if len(x) < 20:
            continue
        mean = x.mean()
        std = x.std()
        t = mean / (std / np.sqrt(len(x)))
        t_stats[ticker] = t

    t_df = pd.DataFrame({"t_stat": t_stats}).sort_values("t_stat")

    fig_t = px.bar(
        t_df,
        x=t_df.index,
        y="t_stat",
        title="Cross-Sectional t-Statistics of Residual Means",
        labels={"index": "Ticker", "t_stat": "t-Statistic"}
    )
    st.plotly_chart(fig_t, use_container_width=True)

    st.dataframe(t_df)

else:
    st.info("Residuals not found.")

# =========================================================
# 5. Factor Return Time Series
# =========================================================
st.header("5. Factor Return Time Series")

if factor_model_type == "PCA" and "eigvecs" in meta:
    eigvecs = meta["eigvecs"]
    R = returns.values

    F = R @ eigvecs[:, :5]
    factor_df = pd.DataFrame(
        F,
        index=returns.index,
        columns=[f"PC{i+1}" for i in range(F.shape[1])]
    )

    st.subheader("PCA Factor Returns (PC1–PC5)")
    st.line_chart(factor_df)

elif factor_model_type == "Single Market Factor (SPY)":
    st.subheader("SPY Market Factor Return")
    st.line_chart(returns["SPY"])

elif factor_model_type == "Sector ETF factors (auto mapping)":
    etfs = sorted(set(meta["mapping"].values()))
    available = [e for e in etfs if e in returns.columns]

    st.subheader("Sector ETF Factor Returns")
    st.line_chart(returns[available])

else:
    st.info("No factor return data available.")

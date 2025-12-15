import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Trade Analytics", page_icon="📊", layout="wide")
st.title("📊 Trade Analytics Dashboard")

# =========================================================
# VALIDATION
# =========================================================
if (
    "equity_df" not in st.session_state or
    "trades_df" not in st.session_state or
    "detailed_df" not in st.session_state or
    "prices" not in st.session_state
):
    st.warning("Run the Backtest first.")
    st.stop()

equity_df = st.session_state["equity_df"]
trades_df = st.session_state["trades_df"].copy()
detailed_df = st.session_state["detailed_df"]
prices = st.session_state["prices"]

# =========================================================
# 1. BUILD TRADE PAIRS (ENTRY → EXIT)
# =========================================================
trade_pairs = []
active = {}

for idx, row in trades_df.sort_index().iterrows():
    t = row["ticker"]
    if row["action"].startswith("ENTER"):
        active[t] = {
            "side": row["action"],
            "price_in": row["price"],
            "time_in": idx,
        }
    elif row["action"] == "EXIT" and t in active:
        entry = active.pop(t)
        pnl = (row["price"] - entry["price_in"]) * (1 if "LONG" in entry["side"] else -1)
        ret_pct = pnl / entry["price_in"] * (1 if "LONG" in entry["side"] else -1)

        holding_days = (idx - entry["time_in"]).days
        trade_pairs.append({
            "ticker": t,
            "side": entry["side"],
            "entry_time": entry["time_in"],
            "exit_time": idx,
            "entry_price": entry["price_in"],
            "exit_price": row["price"],
            "pnl": pnl,
            "return_pct": ret_pct * 100,
            "holding_days": holding_days,
        })

tp = pd.DataFrame(trade_pairs)

if tp.empty:
    st.info("No closed trades in this backtest.")
    st.stop()

# Convenience time fields for heatmaps
tp["exit_date"] = tp["exit_time"].dt.date
tp["exit_dow"] = tp["exit_time"].dt.day_name()
tp["exit_hour"] = tp["exit_time"].dt.hour

# Weekday ordering for nicer heatmaps
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# =========================================================
# 2. KPI CARDS
# =========================================================
st.header("📌 Key Performance Indicators")

total_pnl = tp["pnl"].sum()
avg_pnl = tp["pnl"].mean()
best_trade = tp["pnl"].max()
worst_trade = tp["pnl"].min()
win_rate = (tp["pnl"] > 0).mean() * 100

long_trades = tp[tp["side"].str.contains("LONG")]
short_trades = tp[tp["side"].str.contains("SHORT")]
long_win_rate = (long_trades["pnl"] > 0).mean() * 100 if len(long_trades) > 0 else 0.0
short_win_rate = (short_trades["pnl"] > 0).mean() * 100 if len(short_trades) > 0 else 0.0

gross_wins = tp[tp["pnl"] > 0]["pnl"].sum()
gross_losses = tp[tp["pnl"] < 0]["pnl"].sum()
profit_factor = gross_wins / abs(gross_losses) if gross_losses != 0 else np.nan

expectancy = total_pnl / len(tp)

avg_hold = tp["holding_days"].replace([np.inf, -np.inf], np.nan).mean()
avg_daily_return = (
    tp["return_pct"] / tp["holding_days"].replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan).mean()

# --- Primary KPI row ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total PnL ($)", f"{total_pnl:,.2f}")
c2.metric("Win Rate", f"{win_rate:.1f}%")
c3.metric("Profit Factor", f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "N/A")
c4.metric("Expectancy ($/trade)", f"{expectancy:,.2f}")
c5.metric("Avg Hold (days)", f"{avg_hold:.1f}" if not np.isnan(avg_hold) else "N/A")

# --- Secondary KPI row ---
c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("Best Trade", f"{best_trade:,.2f}")
c7.metric("Worst Trade", f"{worst_trade:,.2f}")
c8.metric("Avg Daily Return %", f"{avg_daily_return:.2f}" if not np.isnan(avg_daily_return) else "N/A")
c9.metric("Long Win Rate", f"{long_win_rate:.1f}%")
c10.metric("Short Win Rate", f"{short_win_rate:.1f}%")

# =========================================================
# 3. ADDITIONAL RISK METRICS (Sharpe, Sortino, Streaks)
# =========================================================
st.subheader("⚠️ Additional Risk Metrics")

# Use % returns for risk metrics
r = (tp["return_pct"] / 100.0).replace([np.inf, -np.inf], np.nan).dropna()
if len(r) > 1:
    mean_r = r.mean()
    std_r = r.std(ddof=1)
    sharpe_trades = mean_r / std_r * np.sqrt(len(r)) if std_r > 0 else np.nan

    downside = r[r < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 0 else np.nan
    sortino_trades = (
        mean_r / downside_std * np.sqrt(len(r)) if downside_std and downside_std > 0 else np.nan
    )
else:
    sharpe_trades = np.nan
    sortino_trades = np.nan

# Max win/loss streaks (chronological)
tp_sorted_time = tp.sort_values("exit_time")
max_win_streak = 0
max_loss_streak = 0
cur_win = 0
cur_loss = 0

for pnl in tp_sorted_time["pnl"]:
    if pnl > 0:
        cur_win += 1
        cur_loss = 0
    elif pnl < 0:
        cur_loss += 1
        cur_win = 0
    else:
        # flat PnL resets both
        cur_win = 0
        cur_loss = 0
    max_win_streak = max(max_win_streak, cur_win)
    max_loss_streak = max(max_loss_streak, cur_loss)

r1, r2, r3, r4 = st.columns(4)
r1.metric("Trade Sharpe (on %)", f"{sharpe_trades:.2f}" if np.isfinite(sharpe_trades) else "N/A")
r2.metric("Trade Sortino (on %)", f"{sortino_trades:.2f}" if np.isfinite(sortino_trades) else "N/A")
r3.metric("Max Win Streak", f"{max_win_streak}")
r4.metric("Max Loss Streak", f"{max_loss_streak}")

# =========================================================
# 4. PNL DISTRIBUTIONS
# =========================================================
st.header("📈 Trade PnL Distribution")

fig_pnl = px.histogram(
    tp,
    x="pnl",
    nbins=30,
    title="Distribution of Trade PnL",
)
st.plotly_chart(fig_pnl, use_container_width=True)

fig_ret = px.histogram(
    tp,
    x="return_pct",
    nbins=30,
    title="Distribution of Trade Returns (%)",
)
st.plotly_chart(fig_ret, use_container_width=True)

# =========================================================
# 5. CUMULATIVE TRADE PNL
# =========================================================
st.header("📈 Cumulative PnL Over Trades")

tp_sorted = tp.sort_values("exit_time")
tp_sorted["cum_pnl"] = tp_sorted["pnl"].cumsum()

fig_cum = px.line(
    tp_sorted,
    x="exit_time",
    y="cum_pnl",
    title="Cumulative Realized PnL",
    labels={"exit_time": "Exit Time", "cum_pnl": "Cumulative PnL"},
)
st.plotly_chart(fig_cum, use_container_width=True)

# =========================================================
# 6. LONG VS SHORT PNL
# =========================================================
st.header("🔻 Long vs Short Performance")

tp_sorted["pnl_long"] = np.where(tp_sorted["side"].str.contains("LONG"), tp_sorted["pnl"], 0.0)
tp_sorted["pnl_short"] = np.where(tp_sorted["side"].str.contains("SHORT"), tp_sorted["pnl"], 0.0)

fig_ls = go.Figure()
fig_ls.add_trace(go.Scatter(
    x=tp_sorted["exit_time"],
    y=tp_sorted["pnl_long"].cumsum(),
    name="Long PnL",
    mode="lines",
))
fig_ls.add_trace(go.Scatter(
    x=tp_sorted["exit_time"],
    y=tp_sorted["pnl_short"].cumsum(),
    name="Short PnL",
    mode="lines",
))
fig_ls.update_layout(
    title="Cumulative Long vs Short PnL",
    xaxis_title="Exit Time",
    yaxis_title="Cumulative PnL",
)
st.plotly_chart(fig_ls, use_container_width=True)

# =========================================================
# 7. HOLDING PERIOD VS PNL (ALPHA TIMING)
# =========================================================
st.header("⏱ Holding Period vs PnL")

fig_hp = px.scatter(
    tp,
    x="holding_days",
    y="pnl",
    color="side",
    title="Holding Time vs Profitability",
    labels={"holding_days": "Holding Days", "pnl": "PnL"},
)
st.plotly_chart(fig_hp, use_container_width=True)

# =========================================================
# 8. TOP TICKER CONTRIBUTORS
# =========================================================
st.header("🏆 Ticker PnL Contribution")

ticker_contrib = tp.groupby("ticker")["pnl"].sum().sort_values()
fig_ticker = px.bar(
    ticker_contrib,
    title="PnL Contribution by Ticker",
    labels={"value": "Total PnL", "index": "Ticker"},
)
st.plotly_chart(fig_ticker, use_container_width=True)
st.dataframe(ticker_contrib.to_frame("Total_PnL"))

# =========================================================
# 9. TRADE HEATMAPS
# =========================================================
st.header("🔥 Trade Heatmaps")

# --- 9.1 Day-of-week × Hour-of-day heatmap (Avg PnL) ---
st.subheader("Heatmap: Day of Week × Hour of Day (Average PnL)")

dow_hour = tp.pivot_table(
    index="exit_dow",
    columns="exit_hour",
    values="pnl",
    aggfunc="mean",
)

# Reorder weekdays if present
dow_hour = dow_hour.reindex([d for d in WEEKDAY_ORDER if d in dow_hour.index])

if not dow_hour.empty and dow_hour.shape[1] > 1:
    fig_dh = px.imshow(
        dow_hour,
        aspect="auto",
        labels=dict(x="Exit Hour", y="Day of Week", color="Avg PnL"),
    )
    st.plotly_chart(fig_dh, use_container_width=True)
else:
    st.info("Not enough intraday variation (hours) to build a day×hour heatmap.")

# --- 9.2 Ticker × Day-of-week heatmap (Avg PnL) ---
st.subheader("Heatmap: Ticker × Day of Week (Average PnL)")

ticker_dow = tp.pivot_table(
    index="ticker",
    columns="exit_dow",
    values="pnl",
    aggfunc="mean",
)

ticker_dow = ticker_dow[[c for c in WEEKDAY_ORDER if c in ticker_dow.columns]]

if not ticker_dow.empty:
    fig_td = px.imshow(
        ticker_dow,
        aspect="auto",
        labels=dict(x="Day of Week", y="Ticker", color="Avg PnL"),
    )
    st.plotly_chart(fig_td, use_container_width=True)
else:
    st.info("Not enough variation across tickers/days to build a ticker×day heatmap.")

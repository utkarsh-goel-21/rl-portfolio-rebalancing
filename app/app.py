"""Streamlit dashboard: reads saved results only (no training here).
Run from the repo root:  streamlit run app/app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import TICKERS, TABLE_DIR
from src.metrics import compute_metrics

st.set_page_config(page_title="RL Portfolio Rebalancing", layout="wide")

PPO = "PPO ensemble"
REQUIRED = ["test_returns_all.csv", "test_ppo_ensemble_history.csv", "test_baseline_weights.csv",
            "test_metrics.csv", "test_ppo_seed_metrics.csv", "test_yearly_returns.csv"]


@st.cache_data
def load():
    rets = pd.read_csv(TABLE_DIR / "test_returns_all.csv", index_col=0, parse_dates=True)
    ens = pd.read_csv(TABLE_DIR / "test_ppo_ensemble_history.csv", index_col=0, parse_dates=True)
    bw = pd.read_csv(TABLE_DIR / "test_baseline_weights.csv", header=[0, 1], index_col=0, parse_dates=True)
    full = pd.read_csv(TABLE_DIR / "test_metrics.csv", index_col=0)
    seeds = pd.read_csv(TABLE_DIR / "test_ppo_seed_metrics.csv", index_col=0)
    yearly = pd.read_csv(TABLE_DIR / "test_yearly_returns.csv", index_col=0)
    val_path = TABLE_DIR / "exp_final_runs.csv"
    val = pd.read_csv(val_path) if val_path.exists() else None
    return rets, ens, bw, full, seeds, yearly, val


missing = [f for f in REQUIRED if not (TABLE_DIR / f).exists()]
if missing:
    st.error(f"Missing result files in results/tables: {missing}. Run `python -m src.evaluate` first.")
    st.stop()

rets, ens, bw, full, seeds, yearly, val = load()
all_names = list(rets.columns)

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Controls")
d0, d1 = rets.index.min().date(), rets.index.max().date()
start, end = st.sidebar.slider("Date range (test period)", min_value=d0, max_value=d1, value=(d0, d1))
chosen = st.sidebar.multiselect("Strategies", all_names, default=all_names)
if PPO not in chosen:
    chosen = [PPO] + chosen

sub = rets.loc[pd.Timestamp(start):pd.Timestamp(end), chosen]
if len(sub) < 30:
    st.warning("Pick a range of at least 30 trading days.")
    st.stop()

st.title("Deep RL for Dynamic Portfolio Allocation (PPO)")
st.caption("Assets: SPY, GLD, TLT, EEM | Held-out test period 2022-2025 | Cost 0.1% per unit turnover")

tab_perf, tab_alloc, tab_year, tab_seed, tab_about = st.tabs(
    ["Performance", "Allocation", "Calendar years", "Seeds and validation", "About"])

# ------------------------------------------------------------------ performance
with tab_perf:
    m = pd.DataFrame({c: compute_metrics(sub[c]) for c in chosen}).T
    m["AnnTurnover (full test)"] = full["AnnTurnover"].reindex(m.index)
    pct = ["CumReturn", "AnnReturn", "AnnVol", "MaxDD"]
    st.dataframe(
        m.style.format({**{c: "{:.1%}" for c in pct}, "Sharpe": "{:.2f}", "Calmar": "{:.2f}",
                        "AnnTurnover (full test)": "{:.1f}"}),
        use_container_width=True)

    best = m.drop(index=PPO, errors="ignore")["Sharpe"]
    if len(best):
        st.info(f"In this range PPO ensemble Sharpe = {m.loc[PPO, 'Sharpe']:.2f}; best selected baseline = "
                f"{best.idxmax()} at {best.max():.2f}.")

    wealth = (1 + sub).cumprod()
    wealth = pd.concat([pd.DataFrame([[1.0] * len(chosen)], columns=chosen, index=[sub.index[0] - pd.Timedelta(days=1)]),
                        wealth])
    dd = wealth / wealth.cummax() - 1

    def lines(df, title, yfmt):
        fig = go.Figure()
        for c in df.columns:
            ppo = c == PPO
            fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c, mode="lines",
                                     line=dict(width=4 if ppo else 1.6, color="black" if ppo else None)))
        fig.update_layout(title=title, yaxis_tickformat=yfmt, height=400, legend=dict(orientation="h"),
                          margin=dict(t=50, b=10))
        return fig

    st.plotly_chart(lines(wealth, "Growth of 1 (rebased at range start)", ".2f"), use_container_width=True)
    st.plotly_chart(lines(dd, "Drawdown", ".0%"), use_container_width=True)

# ------------------------------------------------------------------ allocation
with tab_alloc:
    pick = st.selectbox("Strategy", [PPO] + [c for c in all_names if c != PPO])
    w = ens[TICKERS] if pick == PPO else bw[pick][TICKERS]
    w = w.loc[pd.Timestamp(start):pd.Timestamp(end)]
    long = w.reset_index().melt(id_vars=w.index.name or "index", var_name="Asset", value_name="Weight")
    long = long.rename(columns={long.columns[0]: "Date"})
    fig = px.area(long, x="Date", y="Weight", color="Asset", title=f"{pick}: portfolio weights over time")
    fig.update_layout(yaxis_range=[0, 1], yaxis_tickformat=".0%", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Average weight by calendar year")
    yw = w.groupby(w.index.year).mean()
    yw.index.name = "Year"
    st.dataframe(yw.style.format("{:.1%}"), use_container_width=True)

# ------------------------------------------------------------------ calendar years
with tab_year:
    y = yearly[[c for c in chosen if c in yearly.columns]].copy()
    y.index.name = "Year"
    long = y.reset_index().melt(id_vars="Year", var_name="Strategy", value_name="Return")
    fig = px.bar(long, x="Year", y="Return", color="Strategy", barmode="group",
                 title="Calendar-year returns (full test period)")
    fig.update_layout(yaxis_tickformat=".0%", height=450)
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(y.T.style.format("{:.1%}"), use_container_width=True)

# ------------------------------------------------------------------ seeds
with tab_seed:
    st.subheader("PPO test metrics by seed")
    pct = ["CumReturn", "AnnReturn", "AnnVol", "MaxDD"]
    st.dataframe(seeds.style.format({**{c: "{:.1%}" for c in pct}, "Sharpe": "{:.2f}",
                                     "Calmar": "{:.2f}", "AnnTurnover": "{:.1f}"}), use_container_width=True)
    if val is not None and "best_Sharpe" in val.columns:
        t = seeds.copy()
        t["seed"] = t.index.str.extract(r"(\d+)$")[0].astype(int)
        cmp = val[["seed", "best_Sharpe"]].merge(t[["seed", "Sharpe"]], on="seed")
        cmp.columns = ["Seed", "Validation Sharpe (best checkpoint)", "Test Sharpe"]
        fig = px.scatter(cmp, x="Validation Sharpe (best checkpoint)", y="Test Sharpe", text="Seed",
                         title="Validation rank did not predict test rank")
        fig.update_traces(textposition="top center", marker_size=11)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Validation scores are optimistic because each checkpoint was selected on the validation window itself.")

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown("""
**Problem.** Learn how to split capital across four assets from different classes (US equities, gold,
long-term US bonds, emerging markets) and rebalance daily, judged on risk-adjusted performance.

**Method.** Custom Gymnasium environment; PPO (stable-baselines3). State = 16 causal features (returns,
volatility, momentum) plus current weights. Action = softmax weights, long-only. Reward = net return
(after 0.1% costs) minus a variance penalty (lambda = 20) and a turnover penalty, with partial-adjustment
smoothing of trades. Features standardised on the training period only.

**Protocol.** Train 2010-2019, validation 2020-2021 (design choices and checkpoint selection), test
2022-2025 (evaluated once, plan fixed beforehand). Baselines: equal-weight (buy-and-hold and monthly),
Markowitz max-Sharpe, minimum variance, momentum, SPY. The environment was checked by replaying baselines
through it with zero numerical difference.

**Result in one line.** On the held-out period the PPO agent did not beat the classical baselines, and
results varied widely across seeds.

**Limitations.** One market history, a short validation window, five seeds, no slippage or liquidity
model, zero risk-free rate, and a training period (a long bond rally) that differs from the test period
(2022 bond crash). Historical backtests do not guarantee future performance. This is a course project,
not investment advice.
""")
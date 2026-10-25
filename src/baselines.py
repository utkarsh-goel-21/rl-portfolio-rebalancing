"""Baseline strategies + cost-aware backtester (same conventions as the RL env)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from src.config import (
    TICKERS, N_ASSETS, TRADING_DAYS, RISK_FREE_RATE, TRANSACTION_COST,
    REBALANCE_EVERY, MARKOWITZ_LOOKBACK, MOMENTUM_WINDOW, MOMENTUM_TOP_K,
    FIG_DIR, TABLE_DIR,
)
from src.data import load_clean_prices, compute_returns, SPLITS
from src.metrics import metrics_table, wealth_curve

EQUAL = np.ones(N_ASSETS) / N_ASSETS
ONCE = 10**9  # rebalance only on the first day (buy-and-hold)


# ---------------- strategies: history DataFrame (up to and incl. day t) -> weights ----------------
def equal_weight(hist):
    return EQUAL.copy()


def spy_only(hist):
    w = np.zeros(N_ASSETS)
    w[TICKERS.index("SPY")] = 1.0
    return w


def _moments(hist):
    h = hist.iloc[-MARKOWITZ_LOOKBACK:]
    if len(h) < 60:
        return None, None
    return h.mean().values * TRADING_DAYS, h.cov().values * TRADING_DAYS


def _optimise(objective):
    res = minimize(
        objective, EQUAL, method="SLSQP",
        bounds=[(0.0, 1.0)] * N_ASSETS,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
    )
    if not res.success:
        return EQUAL.copy()
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def min_variance(hist):
    _, cov = _moments(hist)
    if cov is None:
        return EQUAL.copy()
    return _optimise(lambda w: w @ cov @ w)


def markowitz_max_sharpe(hist):
    mu, cov = _moments(hist)
    if mu is None:
        return EQUAL.copy()
    if mu.max() <= RISK_FREE_RATE:           # nothing worth holding on a risk-adjusted basis
        return min_variance(hist)
    return _optimise(lambda w: -(w @ mu - RISK_FREE_RATE) / np.sqrt(w @ cov @ w))


def momentum_topk(hist):
    h = hist.iloc[-MOMENTUM_WINDOW:]
    if len(h) < MOMENTUM_WINDOW:
        return EQUAL.copy()
    mom = (1 + h).prod().values - 1
    w = np.zeros(N_ASSETS)
    w[np.argsort(mom)[-MOMENTUM_TOP_K:]] = 1.0 / MOMENTUM_TOP_K
    return w


STRATEGIES = {
    "EqualWeight_BuyHold": (equal_weight, ONCE),
    "EqualWeight_Monthly": (equal_weight, REBALANCE_EVERY),
    "Markowitz_MaxSharpe": (markowitz_max_sharpe, REBALANCE_EVERY),
    "MinVariance": (min_variance, REBALANCE_EVERY),
    "Momentum_TopK": (momentum_topk, REBALANCE_EVERY),
    "SPY_BuyHold": (spy_only, ONCE),
}


# ---------------- backtester ----------------
def backtest(returns, strategy, start, end, rebalance_every=REBALANCE_EVERY, cost=TRANSACTION_COST):
    """Simulate a strategy on [start, end]. `returns` is the FULL return history, so
    look-back windows can reach before `start` (past data only)."""
    dates = returns.loc[start:end].index
    pos0 = returns.index.get_loc(dates[0])
    w = EQUAL.copy()                       # current (drifted) holdings
    rows = []
    for k in range(len(dates) - 1):
        pos = pos0 + k
        if k % rebalance_every == 0:
            target = np.asarray(strategy(returns.iloc[: pos + 1]), dtype=float)
        else:
            target = w
        turnover = np.abs(target - w).sum()
        r = returns.iloc[pos + 1].values   # realised next-day returns
        gross = float(target @ r)
        net = gross - cost * turnover
        w = target * (1 + r) / (1 + gross)  # drift
        rows.append((dates[k + 1], net, turnover, *target))
    cols = ["date", "ret", "turnover"] + TICKERS
    return pd.DataFrame(rows, columns=cols).set_index("date")


def run_all(returns, start, end):
    return {name: backtest(returns, fn, start, end, every) for name, (fn, every) in STRATEGIES.items()}


# ---------------- main: TRAIN + VAL only ----------------
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    returns = compute_returns(load_clean_prices())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, split in zip(axes, ["train", "val"]):     # test deliberately NOT touched until Step 9
        start, end = SPLITS[split]
        results = run_all(returns, start, end)

        table = metrics_table(
            {n: r["ret"] for n, r in results.items()},
            {n: r["turnover"] for n, r in results.items()},
        )
        print(f"\n=== {split.upper()} ({start} -> {end}) ===")
        print(table.round(3))
        table.round(4).to_csv(TABLE_DIR / f"baselines_{split}.csv")

        for n, r in results.items():
            wealth_curve(r["ret"]).plot(ax=ax, label=n, lw=1.3)
        ax.set_title(f"Baseline wealth curves: {split}")
        ax.set_ylabel("Growth of 1"); ax.legend(fontsize=7)

        # ---- sanity checks ----
        for n, r in results.items():
            w = r[TICKERS].values
            assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6), f"{n}: weights don't sum to 1"
            assert (w >= -1e-9).all(), f"{n}: negative weights"
        bh = results["EqualWeight_BuyHold"]
        expected = (1 + returns.loc[bh.index]).prod().mean() - 1
        assert np.isclose((1 + bh["ret"]).prod() - 1, expected, atol=1e-9), "Buy-and-hold mismatch"

    plt.tight_layout()
    plt.savefig(FIG_DIR / "baselines_wealth.png", dpi=200)
    plt.close()
    print("\nSanity checks passed. Saved tables and results/figures/baselines_wealth.png")


if __name__ == "__main__":
    main()
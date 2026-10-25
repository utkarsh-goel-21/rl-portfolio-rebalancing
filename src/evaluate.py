"""Step 9: ONE-SHOT evaluation on the held-out TEST split (2022-2025).
Run once:  python -m src.evaluate"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from src.baselines import run_all
from src.config import TICKERS, MODELS_DIR, FIG_DIR, TABLE_DIR
from src.data import SPLITS, compute_returns, load_clean_prices
from src.metrics import compute_metrics, metrics_table, wealth_curve
from src.portfolio_env import make_env

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
KEYS = ["CumReturn", "AnnReturn", "AnnVol", "Sharpe", "MaxDD", "Calmar", "AnnTurnover"]


def run_policy(act_fn, kw):
    env = make_env("test", random_start=False, **kw)
    obs, _ = env.reset()
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(act_fn(obs))
        done = term or trunc
    return env.get_history()


def drawdown(ret):
    w = wealth_curve(ret)
    return w / w.cummax() - 1


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(TABLE_DIR / "final_models.json"))
    kw, seeds, names = cfg["env_kwargs"], cfg["seeds"], cfg["models"]
    print("Env settings used for PPO:", kw)

    models = [PPO.load(str(MODELS_DIR / n), device="cpu") for n in names]

    # ---------- PPO: per-seed + ensemble ----------
    hist = {}
    for s, m in zip(seeds, models):
        hist[f"PPO seed {s}"] = run_policy(lambda o, m=m: m.predict(o, deterministic=True)[0], kw)
    ens = run_policy(lambda o: np.mean([m.predict(o, deterministic=True)[0] for m in models], axis=0), kw)

    # ---------- Baselines ----------
    start, end = SPLITS["test"]
    returns = compute_returns(load_clean_prices())
    base = run_all(returns, start, end)

    # sanity: identical dates and length everywhere
    ref = base["EqualWeight_Monthly"].index
    for n, h in {**hist, "ensemble": ens}.items():
        assert len(h) == len(ref) and (h.index == ref).all(), f"{n}: date mismatch"
    print(f"Test window: {ref[0].date()} -> {ref[-1].date()} ({len(ref)} steps)\n")

    # ---------- Metrics ----------
    seed_tab = pd.DataFrame({n: compute_metrics(h["ret"], h["turnover"]) for n, h in hist.items()}).T[KEYS]
    base_tab = metrics_table({n: r["ret"] for n, r in base.items()},
                             {n: r["turnover"] for n, r in base.items()})[KEYS]
    ens_row = pd.Series(compute_metrics(ens["ret"], ens["turnover"]))[KEYS]

    print("=== PPO per seed (TEST) ===")
    print(seed_tab.round(3))
    print("\n=== PPO mean ± std over 5 seeds (TEST) ===")
    print(pd.DataFrame({k: [f"{seed_tab[k].mean():.3f} ± {seed_tab[k].std():.3f}"] for k in KEYS},
                       index=["PPO"]))

    full = pd.concat([base_tab,
                      pd.DataFrame([seed_tab.mean()], index=["PPO (seed mean)"]),
                      pd.DataFrame([ens_row], index=["PPO ensemble"])])
    print("\n=== ALL STRATEGIES (TEST) ===")
    print(full.round(3))
    seed_tab.round(4).to_csv(TABLE_DIR / "test_ppo_seed_metrics.csv")
    full.round(4).to_csv(TABLE_DIR / "test_metrics.csv")

    # ---------- Calendar years + weights ----------
    yearly = {n: r["ret"] for n, r in base.items()}
    yearly["PPO ensemble"] = ens["ret"]
    ytab = pd.DataFrame({n: (1 + r).groupby(r.index.year).prod() - 1 for n, r in yearly.items()})
    print("\n=== Calendar-year returns (TEST) ===")
    print((ytab * 100).round(1).T)
    ytab.round(4).to_csv(TABLE_DIR / "test_yearly_returns.csv")

    wy = ens[TICKERS].groupby(ens.index.year).mean()
    print("\n=== PPO ensemble: average weights per year ===")
    print(wy.round(3))

    # ---------- Save histories for the dashboard (Step 10) ----------
    pd.DataFrame({n: r["ret"] for n, r in base.items()} | {"PPO ensemble": ens["ret"]}
                 ).to_csv(TABLE_DIR / "test_returns_all.csv")
    ens.to_csv(TABLE_DIR / "test_ppo_ensemble_history.csv")
    pd.concat({n: r[TICKERS] for n, r in base.items()}, axis=1).to_csv(TABLE_DIR / "test_baseline_weights.csv")

    # ---------- Figures ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    for n, r in base.items():
        wealth_curve(r["ret"]).plot(ax=ax, lw=1.2, label=n)
    for n, h in hist.items():
        wealth_curve(h["ret"]).plot(ax=ax, lw=0.7, alpha=0.35, color="grey")
    wealth_curve(ens["ret"]).plot(ax=ax, lw=2.5, color="black", label="PPO ensemble")
    ax.set_title("Test period wealth curves (grey = individual PPO seeds)")
    ax.set_ylabel("Growth of 1"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG_DIR / "test_wealth.png", dpi=200); plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    for n, r in [("PPO ensemble", ens["ret"]), ("EqualWeight_Monthly", base["EqualWeight_Monthly"]["ret"]),
                 ("SPY_BuyHold", base["SPY_BuyHold"]["ret"]), ("MinVariance", base["MinVariance"]["ret"])]:
        drawdown(r).plot(ax=ax, lw=1.4, label=n)
    ax.set_title("Test period drawdowns"); ax.set_ylabel("Drawdown"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR / "test_drawdown.png", dpi=200); plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(ens.index, ens[TICKERS].T.values, labels=TICKERS, alpha=0.85)
    ax.set_ylim(0, 1); ax.set_title("PPO ensemble: allocation over the test period")
    ax.set_ylabel("Weight"); ax.legend(loc="upper right", fontsize=8, ncol=4)
    plt.tight_layout(); plt.savefig(FIG_DIR / "test_allocation.png", dpi=200); plt.close()

    print("\nSaved: results/tables/test_*.csv and results/figures/test_{wealth,drawdown,allocation}.png")


if __name__ == "__main__":
    main()
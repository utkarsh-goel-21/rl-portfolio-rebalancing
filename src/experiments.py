"""Step 8 experiments (TRAIN + VAL only).
   python -m src.experiments --stage turnover | lambda | cost | final  [--seeds 42,43,44]"""
import argparse
import json
import time

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.baselines import STRATEGIES, backtest
from src.config import (
    TICKERS, N_ENVS, EVAL_FREQ, TOTAL_TIMESTEPS, PPO_PARAMS, POLICY_KWARGS,
    MODELS_DIR, FIG_DIR, TABLE_DIR, RISK_AVERSION, TRANSACTION_COST,
)
from src.data import SPLITS, compute_returns, load_clean_prices
from src.metrics import compute_metrics
from src.portfolio_env import make_env
from src.train import ValSharpeCallback, rollout

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

CHOSEN = TABLE_DIR / "chosen_config.json"
DEFAULT = dict(turnover_penalty=0.0, smoothing=1.0, risk_aversion=RISK_AVERSION, cost=TRANSACTION_COST)
METRIC_KEYS = ["Sharpe", "AnnReturn", "MaxDD", "AnnTurnover"]

TURNOVER_CONFIGS = {
    "v1_baseline": dict(turnover_penalty=0.0, smoothing=1.0),
    "penalty":     dict(turnover_penalty=0.003, smoothing=1.0),
    "smoothing":   dict(turnover_penalty=0.0, smoothing=0.2),
    "both":        dict(turnover_penalty=0.003, smoothing=0.2),
}
LAMBDAS = [0, 5, 20, 50]
COSTS = [0.0, 0.0005, 0.001, 0.002]
DEFAULT_SEEDS = {"turnover": [42, 43, 44], "lambda": [42, 43, 44], "cost": [42, 43, 44],
                 "final": [42, 43, 44, 45, 46]}


# ------------------------------------------------------------------ helpers
def train_one(tag, kw, seed, timesteps=TOTAL_TIMESTEPS):
    venv = DummyVecEnv([lambda: make_env("train", **kw) for _ in range(N_ENVS)])
    model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0,
                policy_kwargs=POLICY_KWARGS, **PPO_PARAMS)
    path = MODELS_DIR / f"exp_{tag}_seed{seed}_best"
    cb = ValSharpeCallback(EVAL_FREQ, str(path), kw)
    model.learn(total_timesteps=timesteps, callback=cb)

    rec = pd.DataFrame(cb.records)
    best, last = rec.loc[rec["Sharpe"].idxmax()], rec.iloc[-1]
    row = {"tag": tag, "seed": seed, "model": path.name}
    for k in METRIC_KEYS:
        row[f"best_{k}"], row[f"final_{k}"] = best[k], last[k]
    w = rollout(PPO.load(str(path), device="cpu"), "val", kw)[TICKERS].mean()
    row.update({f"w_{t}": w[t] for t in TICKERS})
    return row


def run_group(configs, seeds, stage):
    rows = []
    for name, kw in configs.items():
        full = {**DEFAULT, **kw}
        for s in seeds:
            t0 = time.time()
            row = train_one(f"{stage}_{name}", full, s)
            row["config"] = name
            rows.append(row)
            print(f"  {name:<12} seed {s}: final Sharpe={row['final_Sharpe']:.3f} "
                  f"turnover={row['final_AnnTurnover']:.1f} (best {row['best_Sharpe']:.3f}) "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / f"exp_{stage}_runs.csv", index=False)
    return df


def summary(df, cols):
    rows = {}
    for cfg, g in df.groupby("config", sort=False):
        rows[cfg] = {c: f"{g[c].mean():.3f} ± {g[c].std():.3f}" for c in cols}
    return pd.DataFrame(rows).T


def baseline_table(cost):
    returns = compute_returns(load_clean_prices())
    start, end = SPLITS["val"]
    rows = {}
    for name, (fn, every) in STRATEGIES.items():
        r = backtest(returns, fn, start, end, every, cost)
        rows[name] = compute_metrics(r["ret"], r["turnover"])
    return pd.DataFrame(rows).T


def load_chosen():
    return json.load(open(CHOSEN))


def base_kw(chosen):
    return {k: chosen[k] for k in ["turnover_penalty", "smoothing", "risk_aversion", "cost"]}


# ------------------------------------------------------------------ stages
def stage_turnover(seeds):
    df = run_group(TURNOVER_CONFIGS, seeds, "turnover")
    print("\n=== A. Turnover control (validation, mean ± std over seeds) ===")
    print(summary(df, ["final_Sharpe", "best_Sharpe", "final_AnnReturn", "final_MaxDD", "final_AnnTurnover"]))
    winner = df.groupby("config")["final_Sharpe"].mean().idxmax()
    chosen = {**DEFAULT, **TURNOVER_CONFIGS[winner], "name": winner}
    json.dump(chosen, open(CHOSEN, "w"), indent=2)
    print(f"\nChosen config (highest mean final val Sharpe): {winner}\n{chosen}")


def stage_lambda(seeds):
    base = base_kw(load_chosen())
    cfgs = {f"lam_{l}": {**base, "risk_aversion": float(l)} for l in LAMBDAS}
    df = run_group(cfgs, seeds, "lambda")
    print("\n=== B. Risk-penalty ablation (validation) ===")
    print(summary(df, ["final_Sharpe", "final_AnnReturn", "final_MaxDD", "final_AnnTurnover",
                       "w_SPY", "w_GLD", "w_TLT", "w_EEM"]))


def stage_cost(seeds):
    base = base_kw(load_chosen())
    cfgs = {f"cost_{int(round(c * 1e4))}bp": {**base, "cost": c} for c in COSTS}
    df = run_group(cfgs, seeds, "cost")
    rows = {}
    for (name, _), c in zip(cfgs.items(), COSTS):
        g = df[df.config == name]
        bt = baseline_table(c)["Sharpe"]
        rows[f"{c * 100:.2f}%"] = {
            "PPO final Sharpe": g.final_Sharpe.mean(), "PPO turnover": g.final_AnnTurnover.mean(),
            "EW_Monthly": bt["EqualWeight_Monthly"], "Momentum": bt["Momentum_TopK"],
            "SPY": bt["SPY_BuyHold"], "Markowitz": bt["Markowitz_MaxSharpe"],
        }
    table = pd.DataFrame(rows).T
    table.round(4).to_csv(TABLE_DIR / "exp_cost_summary.csv")
    print("\n=== C. Cost sensitivity: validation Sharpe, PPO retrained at each cost ===")
    print(table.round(3))


def stage_final(seeds):
    chosen = load_chosen()
    kw = base_kw(chosen)
    df = run_group({"final": kw}, seeds, "final")
    print(f"\n=== D. Final config '{chosen['name']}' | per-seed validation results ===")
    print(df[["seed", "best_Sharpe", "final_Sharpe", "best_AnnReturn", "best_MaxDD", "best_AnnTurnover"]].round(3))
    print("\nMean ± std over seeds:")
    print(summary(df, ["best_Sharpe", "final_Sharpe", "best_AnnReturn", "best_MaxDD", "best_AnnTurnover"]))
    base = baseline_table(kw["cost"])
    print("\nBaselines on validation (same cost):")
    print(base.round(3))

    json.dump({"env_kwargs": kw, "seeds": seeds, "models": list(df["model"])},
              open(TABLE_DIR / "final_models.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(s) for s in df.seed], df.best_Sharpe, color="tab:blue", label="PPO best ckpt (per seed)")
    for name in ["EqualWeight_Monthly", "SPY_BuyHold", "Momentum_TopK"]:
        ax.axhline(base.loc[name, "Sharpe"], ls="--", lw=1, label=f"{name} ({base.loc[name, 'Sharpe']:.2f})")
    ax.set_xlabel("Seed"); ax.set_ylabel("Validation Sharpe"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG_DIR / "final_seeds_val_sharpe.png", dpi=200); plt.close()
    print("\nSaved results/tables/final_models.json and results/figures/final_seeds_val_sharpe.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["turnover", "lambda", "cost", "final"])
    ap.add_argument("--seeds", default=None, help="comma-separated, e.g. 42,43")
    a = ap.parse_args()
    MODELS_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True); TABLE_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else DEFAULT_SEEDS[a.stage]
    {"turnover": stage_turnover, "lambda": stage_lambda, "cost": stage_cost, "final": stage_final}[a.stage](seeds)
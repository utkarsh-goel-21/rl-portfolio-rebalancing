"""PPO training with validation-based checkpoint selection.  Run:  python -m src.train --seed 42"""
import argparse
import shutil
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import (
    TICKERS, SEED, TOTAL_TIMESTEPS, N_ENVS, EVAL_FREQ, LOG_DIR, TB_DIR,
    PPO_PARAMS, POLICY_KWARGS, MODELS_DIR, FIG_DIR, TABLE_DIR,
)
from src.baselines import run_all
from src.data import SPLITS, compute_returns, load_clean_prices
from src.metrics import compute_metrics, metrics_table
from src.portfolio_env import make_env

torch.set_num_threads(1)   # tiny MLPs run faster single-threaded on CPU


def rollout(model, split):
    """Run the policy deterministically over a full split; return the episode history."""
    env = make_env(split, random_start=False)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.get_history()


class ValSharpeCallback(BaseCallback):
    """Every `eval_freq` steps: evaluate on validation, save the best-Sharpe checkpoint."""

    def __init__(self, eval_freq, save_path):
        super().__init__(verbose=0)
        self.eval_freq, self.save_path = eval_freq, save_path
        self.next_eval, self.best, self.records = eval_freq, -np.inf, []

    def _evaluate(self):
        h = rollout(self.model, "val")
        m = compute_metrics(h["ret"], h["turnover"])
        is_best = bool(m["Sharpe"] > self.best)
        if is_best:
            self.best = m["Sharpe"]
            self.model.save(self.save_path)
        self.records.append({"timesteps": self.num_timesteps, **m, "best": is_best})
        print(f"[{self.num_timesteps:>7}] val Sharpe={m['Sharpe']:.3f}  ret={m['AnnReturn']:.3f}  "
              f"MaxDD={m['MaxDD']:.3f}  turnover={m['AnnTurnover']:.2f}" + ("   <- best" if is_best else ""))

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self._evaluate()
            self.next_eval += self.eval_freq
        return True

    def _on_training_end(self):
        if not self.records or self.records[-1]["timesteps"] != self.num_timesteps:
            self._evaluate()


def plot_training(log_dir, curve, baseline_sharpe, seed):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: smoothed training episode reward (from Monitor files)
    for k, f in enumerate(sorted(log_dir.glob("env*.monitor.csv"))):
        m = pd.read_csv(f, skiprows=1)
        steps = m["l"].cumsum() * N_ENVS              # approx. global timesteps
        axes[0].plot(steps, m["r"].rolling(10, min_periods=1).mean(), alpha=0.8, label=f"env {k}")
    axes[0].set_title("Training episode reward (10-episode rolling mean)")
    axes[0].set_xlabel("Timesteps"); axes[0].set_ylabel("Episode reward"); axes[0].legend(fontsize=7)

    # Right: validation Sharpe over training vs baselines
    axes[1].plot(curve["timesteps"], curve["Sharpe"], marker="o", ms=3, label="PPO (val Sharpe)")
    for name, s in baseline_sharpe.items():
        axes[1].axhline(s, ls="--", lw=1, label=f"{name} ({s:.2f})")
    axes[1].set_title("Validation Sharpe during training")
    axes[1].set_xlabel("Timesteps"); axes[1].set_ylabel("Sharpe"); axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"training_curves_seed{seed}.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = ap.parse_args()

    for d in (MODELS_DIR, FIG_DIR, TABLE_DIR, TB_DIR):
        d.mkdir(parents=True, exist_ok=True)
    log_dir = LOG_DIR / f"seed{args.seed}"
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True)

    def make_train_env(rank):
        return lambda: Monitor(make_env("train"), str(log_dir / f"env{rank}"))

    venv = DummyVecEnv([make_train_env(i) for i in range(N_ENVS)])
    model = PPO("MlpPolicy", venv, seed=args.seed, device="cpu", verbose=0,
                tensorboard_log=str(TB_DIR), policy_kwargs=POLICY_KWARGS, **PPO_PARAMS)

    best_path = MODELS_DIR / f"ppo_seed{args.seed}_best"
    cb = ValSharpeCallback(EVAL_FREQ, str(best_path))

    print(f"Training PPO | seed={args.seed} | timesteps={args.timesteps:,} | envs={N_ENVS}")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb, tb_log_name=f"ppo_seed{args.seed}")
    print(f"\nTraining finished in {(time.time() - t0) / 60:.1f} min")
    model.save(MODELS_DIR / f"ppo_seed{args.seed}_final")

    curve = pd.DataFrame(cb.records)
    curve.to_csv(log_dir / "val_curve.csv", index=False)

    # ---- Best checkpoint vs baselines on VALIDATION ----
    best = PPO.load(str(best_path), device="cpu")
    h = rollout(best, "val")
    h.to_csv(TABLE_DIR / f"ppo_val_history_seed{args.seed}.csv")

    start, end = SPLITS["val"]
    returns = compute_returns(load_clean_prices())
    base = run_all(returns, start, end)
    rets = {n: r["ret"] for n, r in base.items()}
    turns = {n: r["turnover"] for n, r in base.items()}
    rets["PPO (best ckpt)"], turns["PPO (best ckpt)"] = h["ret"], h["turnover"]
    table = metrics_table(rets, turns)
    table.round(4).to_csv(TABLE_DIR / f"val_comparison_seed{args.seed}.csv")
    print("\n=== VALIDATION: PPO best checkpoint vs baselines ===")
    print(table.round(3))
    print("\nPPO average weights on validation:")
    print(h[TICKERS].mean().round(3).to_string())

    plot_training(log_dir, curve,
                  {n: table.loc[n, "Sharpe"] for n in ["EqualWeight_Monthly", "SPY_BuyHold", "Momentum_TopK"]},
                  args.seed)
    print(f"\nSaved: {best_path}.zip, curves and tables in results/")


if __name__ == "__main__":
    main()
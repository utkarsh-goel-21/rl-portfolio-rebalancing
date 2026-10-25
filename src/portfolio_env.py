"""Custom Gymnasium environment for multi-asset portfolio rebalancing.

Conventions (identical to src/baselines.py backtest):
  * the agent observes features known at the close of day t,
  * chooses target weights at that close,
  * earns the realised return of day t+1,
  * pays  cost * sum|target - drifted_weights|  as transaction cost,
  * weights then drift:  w <- target * (1 + r) / (1 + gross_return).
"""
from functools import lru_cache

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.config import (
    TICKERS, N_ASSETS, TRANSACTION_COST, EPISODE_LENGTH, SEED,
    ACTION_TEMPERATURE, RISK_AVERSION, COV_WINDOW, REWARD_SCALE,
)
from src.data import load_clean_prices, compute_returns
from src.features import load_dataset


def softmax_weights(action, temperature=ACTION_TEMPERATURE):
    """Map a raw action in [-1, 1]^N to long-only weights that sum to 1."""
    z = np.asarray(action, dtype=np.float64) * temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def rolling_cov_array(returns: pd.DataFrame, window: int = COV_WINDOW) -> np.ndarray:
    """cov[t] = covariance of the `window` daily returns ending at day t (causal)."""
    R = returns.values
    cov = np.zeros((len(R), N_ASSETS, N_ASSETS))
    for t in range(window - 1, len(R)):
        cov[t] = np.cov(R[t - window + 1: t + 1].T)
    return cov


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, features, returns, cov, episode_length=EPISODE_LENGTH,
                 random_start=True, cost=TRANSACTION_COST, risk_aversion=RISK_AVERSION,
                 temperature=ACTION_TEMPERATURE, reward_scale=REWARD_SCALE, turnover_penalty=0.0, smoothing=1.0):
        super().__init__()
        assert len(features) == len(returns) == len(cov), "inputs must be aligned"
        self.features = features.values.astype(np.float32)
        self.returns = returns.values.astype(np.float64)
        self.cov = cov
        self.dates = returns.index
        self.T = len(returns)

        self.random_start = random_start
        # steps need returns[i+1], so at most T-1 steps are possible
        self.episode_length = min(episode_length, self.T - 1) if random_start else self.T - 1
        self.cost = cost
        self.risk_aversion = risk_aversion
        self.temperature = temperature
        self.reward_scale = reward_scale
        self.turnover_penalty = turnover_penalty   # kappa: extra reward penalty per unit turnover
        self.smoothing = smoothing                 # alpha: 1.0 = jump straight to target weights


        self.n_features = self.features.shape[1]
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.n_features + N_ASSETS,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_ASSETS,), dtype=np.float32)

    # ------------------------------------------------------------------
    def _obs(self):
        return np.concatenate([self.features[self.i], self.w.astype(np.float32)])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start:
            max_start = self.T - 1 - self.episode_length
            self.start = int(self.np_random.integers(0, max_start + 1))
        else:
            self.start = 0
        self.i = self.start
        self.steps = 0
        self.w = np.ones(N_ASSETS) / N_ASSETS      # drifted holdings; start at 1/N
        self.value = 1.0
        self._log = []
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = softmax_weights(action, self.temperature)
        if self.smoothing < 1.0:                   # partial adjustment toward desired weights
            target = self.w + self.smoothing * (target - self.w)

        turnover = float(np.abs(target - self.w).sum())
        r = self.returns[self.i + 1]                      # realised NEXT-day returns
        gross = float(target @ r)
        net = gross - self.cost * turnover
        risk = float(target @ self.cov[self.i] @ target)  # ex-ante daily variance (past data)

        reward = self.reward_scale * (net - self.risk_aversion * risk - self.turnover_penalty * turnover)

        self.w = target * (1.0 + r) / (1.0 + gross)       # drift to next close
        self.value *= (1.0 + net)
        self._log.append((self.dates[self.i + 1], net, turnover, self.value, *target))

        self.i += 1
        self.steps += 1
        truncated = self.steps >= self.episode_length
        info = {"net_return": net, "turnover": turnover, "portfolio_value": self.value}
        return self._obs(), float(reward), False, truncated, info

    def get_history(self) -> pd.DataFrame:
        """Episode log in the same layout as the baseline backtests (usable by metrics.py)."""
        cols = ["date", "ret", "turnover", "value"] + TICKERS
        return pd.DataFrame(self._log, columns=cols).set_index("date")


# ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def _cached_data():
    data, scaler = load_dataset()
    full_returns = compute_returns(load_clean_prices())
    return data, full_returns, rolling_cov_array(full_returns)


def make_env(split: str, random_start: bool = None, **kwargs) -> PortfolioEnv:
    """Build an environment for 'train', 'val' or 'test'."""
    data, full_returns, cov_full = _cached_data()
    d = data[split]
    idx = full_returns.index.get_indexer(d["returns"].index)
    assert (idx >= 0).all(), "split dates missing from the full return history"
    if random_start is None:
        random_start = (split == "train")
    return PortfolioEnv(d["features"], d["returns"], cov_full[idx],
                        random_start=random_start, **kwargs)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    # 1) Random-agent episode on a training window
    env = make_env("train")
    obs, _ = env.reset(seed=SEED)
    print("Observation shape:", obs.shape, "| action space:", env.action_space)
    total_reward, done = 0.0, False
    while not done:
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += rew
        done = terminated or truncated
    print(f"[train/random] steps={env.steps} start={env.dates[env.start].date()} "
          f"final value={env.value:.3f} total reward={total_reward:.2f}")

    # 2) Uniform-action pass over the full validation split (deterministic)
    env = make_env("val")
    obs, _ = env.reset()
    done = False
    while not done:
        obs, rew, terminated, truncated, info = env.step(np.zeros(N_ASSETS))
        done = terminated or truncated
    h = env.get_history()
    print(f"[val/uniform ] steps={env.steps} ({h.index[0].date()} -> {h.index[-1].date()}) "
          f"final value={env.value:.3f} mean turnover={h['turnover'].mean():.4f}")
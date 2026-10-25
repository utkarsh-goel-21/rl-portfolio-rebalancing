"""Environment sanity checks (Step 6).  Run:  python -m src.test_env
Uses TRAIN and VAL only; the test split is not touched."""
import numpy as np
from stable_baselines3.common.env_checker import check_env

import src.portfolio_env as pe
from src.portfolio_env import (
    make_env, softmax_weights, rolling_cov_array, _cached_data, PortfolioEnv,
)
from src.config import (
    N_ASSETS, TICKERS, TRANSACTION_COST, RISK_AVERSION, REWARD_SCALE,
    EPISODE_LENGTH, TRADING_DAYS, SEED,
)
from src.data import compute_returns, load_clean_prices, SPLITS
from src.baselines import STRATEGIES, backtest


# ---------------------------------------------------------------- 1
def test_gym_api():
    for split in ["train", "val"]:
        check_env(make_env(split), warn=True)
    print("PASS 1  SB3 check_env on train and val envs")


# ---------------------------------------------------------------- 2
def test_softmax():
    assert np.allclose(softmax_weights(np.zeros(N_ASSETS)), 1 / N_ASSETS)
    rng = np.random.default_rng(SEED)
    for _ in range(2000):
        w = softmax_weights(rng.uniform(-1, 1, N_ASSETS))
        assert abs(w.sum() - 1) < 1e-12 and (w > 0).all()
    w_max = softmax_weights(np.array([1.0, -1.0, -1.0, -1.0])).max()
    assert w_max > 0.99
    print(f"PASS 2  softmax: weights valid; max reachable single-asset weight = {w_max:.5f}")


# ---------------------------------------------------------------- 3
class _Replay:
    """Stands in for softmax_weights and returns prescribed target weights."""
    def __init__(self, targets):
        self.targets, self.k = targets, 0

    def __call__(self, action, temperature=None):
        w = self.targets[self.k]
        self.k += 1
        return w


def replay_through_env(name, split="val"):
    start, end = SPLITS[split]
    returns = compute_returns(load_clean_prices())
    fn, every = STRATEGIES[name]
    bt = backtest(returns, fn, start, end, every)

    env = make_env(split, random_start=False)
    original = pe.softmax_weights
    pe.softmax_weights = _Replay(bt[TICKERS].values)
    try:
        env.reset()
        done = False
        while not done:
            _, _, _, truncated, _ = env.step(np.zeros(N_ASSETS))
            done = truncated
    finally:
        pe.softmax_weights = original

    h = env.get_history()
    assert len(h) == len(bt) and (h.index == bt.index).all(), "date alignment mismatch"
    err_ret = np.abs(h["ret"].values - bt["ret"].values).max()
    err_turn = np.abs(h["turnover"].values - bt["turnover"].values).max()
    assert err_ret < 1e-10 and err_turn < 1e-10, f"{name}: env != backtest ({err_ret}, {err_turn})"
    return err_ret, err_turn


def test_replay():
    for name in ["EqualWeight_Monthly", "Markowitz_MaxSharpe", "Momentum_TopK", "SPY_BuyHold"]:
        e_ret, e_turn = replay_through_env(name)
        print(f"PASS 3  replay {name:<20} max|ret err|={e_ret:.1e}  max|turnover err|={e_turn:.1e}")


# ---------------------------------------------------------------- 4
def test_reward():
    env = make_env("val", random_start=False)
    env.reset()
    a = np.array([0.5, -0.3, 0.2, -0.8], dtype=np.float32)
    w0, i = env.w.copy(), env.i
    _, rew, terminated, truncated, info = env.step(a)

    target = softmax_weights(a.astype(np.float64))
    r = env.returns[i + 1]
    turnover = np.abs(target - w0).sum()
    net = target @ r - TRANSACTION_COST * turnover
    risk = target @ env.cov[i] @ target
    assert risk > 0, "covariance is empty, so the risk term is not active"
    expected = REWARD_SCALE * (net - RISK_AVERSION * risk)
    assert np.isclose(rew, expected, rtol=1e-9), (rew, expected)
    assert np.isclose(info["net_return"], net) and np.isclose(info["turnover"], turnover)
    assert not terminated
    print(f"PASS 4  reward matches manual calculation ({rew:.6f}); "
          f"first-step turnover from 1/N = {turnover:.3f}")


# ---------------------------------------------------------------- 5
def test_no_lookahead():
    data, full_returns, cov_full = _cached_data()

    # (a) covariance at day t must not change if everything after t is altered
    t = 2000
    perturbed = full_returns.copy()
    perturbed.iloc[t + 1:] = perturbed.iloc[t + 1:] * 3.0
    cov_p = rolling_cov_array(perturbed)
    assert np.allclose(cov_full[t], cov_p[t]), "covariance uses future data"
    assert not np.allclose(cov_full[t + 30], cov_p[t + 30]), "perturbation had no effect, so the test is vacuous"

    # (b) the first observation must not depend on any future return
    d = data["val"]
    env_a = make_env("val", random_start=False)
    r2 = d["returns"].copy()
    r2.iloc[1:] = r2.iloc[1:] * 3.0
    env_b = PortfolioEnv(d["features"], r2, env_a.cov, random_start=False)
    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()
    assert np.allclose(obs_a, obs_b), "observation depends on future returns"
    print("PASS 5  no lookahead in covariance or initial observation (and the test is sensitive)")


# ---------------------------------------------------------------- 6
def test_episodes():
    env = make_env("train")
    env.reset(seed=1)
    s1 = env.start
    env.reset(seed=1)
    assert env.start == s1, "seeded reset is not reproducible"
    assert len({(env.reset(seed=s), env.start)[1] for s in range(20)}) > 1, "random starts not varying"

    env.reset(seed=3)
    n, term_any, trunc_flags = 0, False, []
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        n += 1
        term_any |= term
        trunc_flags.append(trunc)
        done = term or trunc
    assert n == EPISODE_LENGTH and not term_any
    assert trunc_flags[-1] and not any(trunc_flags[:-1])
    assert env.i <= env.T - 1, "episode ran past the end of the data"

    v1, v2 = make_env("val"), make_env("val")
    v1.reset(); v2.reset()
    assert v1.start == v2.start == 0 and v1.episode_length == v1.T - 1
    print(f"PASS 6  episodes: train={n} steps (truncated, never terminated); "
          f"val is deterministic and covers the full split ({v1.episode_length} steps)")


# ---------------------------------------------------------------- 7
def _run(env, seed, policy):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(policy(env))
        done = term or trunc
    h = env.get_history()
    return env.value, h["turnover"].mean()


def test_behaviour_profile():
    policies = {
        "random":  lambda e: e.action_space.sample(),
        "uniform": lambda e: np.zeros(N_ASSETS, dtype=np.float32),
        "all-SPY": lambda e: np.array([1, -1, -1, -1], dtype=np.float32),
    }
    stats = {}
    for name, pol in policies.items():
        env = make_env("train")
        out = np.array([_run(env, s, pol) for s in range(20)])
        stats[name] = out.mean(axis=0)
    print("PASS 7  behaviour over 20 training episodes (mean final value | mean daily turnover | cost drag/yr)")
    for name, (val, turn) in stats.items():
        print(f"          {name:>8}: {val:.3f} | {turn:.4f} | {turn * TRANSACTION_COST * TRADING_DAYS:.1%}")
    assert stats["random"][1] > 10 * stats["uniform"][1], "random churn should dwarf uniform churn"


if __name__ == "__main__":
    test_gym_api()
    test_softmax()
    test_replay()
    test_reward()
    test_no_lookahead()
    test_episodes()
    test_behaviour_profile()
    print("\nALL ENVIRONMENT CHECKS PASSED")
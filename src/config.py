"""Central configuration for the RL portfolio project."""
from pathlib import Path

# ---- Paths ----
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

# ---- Assets ----
TICKERS = ["SPY", "GLD", "TLT", "EEM"]
N_ASSETS = len(TICKERS)

# ---- Date ranges ----
DATA_START = "2009-01-01"   # warm-up for rolling features
DATA_END = "2025-12-31"
TRAIN_START, TRAIN_END = "2010-01-01", "2019-12-31"
VAL_START, VAL_END = "2020-01-01", "2021-12-31"
TEST_START, TEST_END = "2022-01-01", "2025-12-31"

# ---- Environment ----
TRANSACTION_COST = 0.001    # 0.1% of traded value
EPISODE_LENGTH = 252        # ~1 trading year
TRADING_DAYS = 252
RISK_FREE_RATE = 0.0        # simplification; stated in report

# ---- Features (finalized in Step 3) ----
RETURN_WINDOW = 20
VOL_WINDOW = 20
MOMENTUM_WINDOW = 60

# ---- Reproducibility ----
SEED = 42

FEATURE_CLIP = 5.0          # clip standardised features to +/- this value

# ---- Baselines ----
REBALANCE_EVERY = 21        # trading days (~monthly)
MARKOWITZ_LOOKBACK = 252
MOMENTUM_TOP_K = 2

# ---- RL environment ----
ACTION_TEMPERATURE = 5.0    # softmax(action * T); action in [-1, 1]
RISK_AVERSION = 5.0         # lambda in reward = net_ret - lambda * w' Cov w
COV_WINDOW = 20             # trailing days for the covariance estimate
REWARD_SCALE = 100.0        # scale reward for stable PPO training

# ---- PPO training ----
TOTAL_TIMESTEPS = 300_000
N_ENVS = 4
EVAL_FREQ = 10_000          # run a validation rollout every N timesteps
LOG_DIR = ROOT / "results" / "logs"
TB_DIR = ROOT / "tensorboard_logs"

PPO_PARAMS = dict(
    learning_rate=3e-4, n_steps=256, batch_size=128, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
    vf_coef=0.5, max_grad_norm=0.5,
)
POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-1.0)
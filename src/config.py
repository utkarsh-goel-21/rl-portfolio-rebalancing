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
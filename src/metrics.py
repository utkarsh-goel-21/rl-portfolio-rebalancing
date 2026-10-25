"""Performance metrics computed from a series of daily net returns."""
import numpy as np
import pandas as pd

from src.config import TRADING_DAYS, RISK_FREE_RATE


def wealth_curve(ret: pd.Series) -> pd.Series:
    return (1 + ret).cumprod()


def max_drawdown(ret: pd.Series) -> float:
    w = np.concatenate([[1.0], (1 + ret.values).cumprod()])
    return float((w / np.maximum.accumulate(w) - 1).min())


def compute_metrics(ret: pd.Series, turnover: pd.Series = None) -> dict:
    n = len(ret)
    cum = float((1 + ret).prod() - 1)
    ann_ret = (1 + cum) ** (TRADING_DAYS / n) - 1
    std = ret.std()
    vol = std * np.sqrt(TRADING_DAYS)
    sharpe = (ret.mean() - RISK_FREE_RATE / TRADING_DAYS) / std * np.sqrt(TRADING_DAYS) if std > 0 else np.nan
    mdd = max_drawdown(ret)
    out = {
        "CumReturn": cum,
        "AnnReturn": ann_ret,
        "AnnVol": vol,
        "Sharpe": sharpe,
        "MaxDD": mdd,
        "Calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,
    }
    if turnover is not None:
        out["AnnTurnover"] = float(turnover.mean() * TRADING_DAYS)  # 2.0 = 200% of portfolio per year
    return out


def metrics_table(returns_by_name: dict, turnover_by_name: dict = None) -> pd.DataFrame:
    rows = {
        name: compute_metrics(r, None if turnover_by_name is None else turnover_by_name[name])
        for name, r in returns_by_name.items()
    }
    return pd.DataFrame(rows).T
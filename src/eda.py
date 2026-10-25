"""Exploratory analysis on TRAIN + VAL only (test set deliberately untouched)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    TRADING_DAYS, RISK_FREE_RATE, FIG_DIR, TABLE_DIR,
    VAL_START, VAL_END, TRAIN_START,
)
from src.data import load_clean_prices, compute_returns, get_split


def ann_stats(ret: pd.DataFrame) -> pd.DataFrame:
    mean = ret.mean() * TRADING_DAYS
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    wealth = (1 + ret).cumprod()
    mdd = (wealth / wealth.cummax() - 1).min()
    return pd.DataFrame({
        "AnnReturn": mean, "AnnVol": vol,
        "Sharpe": (mean - RISK_FREE_RATE) / vol, "MaxDD": mdd,
    })


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_clean_prices()
    returns = compute_returns(prices)
    train_r, val_r = get_split(returns, "train"), get_split(returns, "val")
    dev_prices = prices.loc[TRAIN_START:VAL_END]

    # 1) Normalised price paths (train + val), validation shaded
    norm = dev_prices / dev_prices.iloc[0] * 100
    ax = norm.plot(figsize=(10, 5), lw=1.4)
    ax.axvspan(pd.Timestamp(VAL_START), pd.Timestamp(VAL_END), color="grey", alpha=0.15, label="Validation")
    ax.set_title("Normalised prices (base = 100), train + validation")
    ax.set_ylabel("Index level"); ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "eda_prices.png", dpi=200); plt.close()

    # 2) Correlation heatmaps: train vs validation
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for a, (name, r) in zip(axes, [("Train", train_r), ("Validation", val_r)]):
        sns.heatmap(r.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=a, cbar=False)
        a.set_title(f"Return correlation: {name}")
    plt.tight_layout(); plt.savefig(FIG_DIR / "eda_correlation.png", dpi=200); plt.close()

    # 3) Rolling 60-day annualised volatility
    roll_vol = (returns.loc[TRAIN_START:VAL_END].rolling(60).std() * np.sqrt(TRADING_DAYS)).dropna()
    ax = roll_vol.plot(figsize=(10, 4), lw=1.2)
    ax.set_title("Rolling 60-day annualised volatility"); ax.set_ylabel("Volatility")
    plt.tight_layout(); plt.savefig(FIG_DIR / "eda_rolling_vol.png", dpi=200); plt.close()

    # 4) Per-asset stats table
    table = pd.concat({"train": ann_stats(train_r), "val": ann_stats(val_r)})
    table.round(3).to_csv(TABLE_DIR / "asset_stats.csv")
    print(table.round(3))


if __name__ == "__main__":
    main()
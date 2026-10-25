"""Causal feature engineering + train-only standardisation + dataset loader."""
import json
import numpy as np
import pandas as pd

from src.config import (
    TICKERS, RETURN_WINDOW, VOL_WINDOW, MOMENTUM_WINDOW,
    TRADING_DAYS, DATA_PROCESSED, FEATURE_CLIP,
)
from src.data import load_clean_prices, compute_returns, get_split, SPLITS

FEATURE_NAMES = [
    "ret_1d",
    f"ret_{RETURN_WINDOW}d",
    f"vol_{VOL_WINDOW}d",
    f"mom_{MOMENTUM_WINDOW}d",
]


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-asset features using only past/current data (causal)."""
    daily = prices.pct_change()
    frames = {
        "ret_1d": daily,
        f"ret_{RETURN_WINDOW}d": prices.pct_change(RETURN_WINDOW),
        f"vol_{VOL_WINDOW}d": daily.rolling(VOL_WINDOW).std() * np.sqrt(TRADING_DAYS),
        f"mom_{MOMENTUM_WINDOW}d": prices.pct_change(MOMENTUM_WINDOW),
    }
    cols = {}
    for t in TICKERS:
        for name in FEATURE_NAMES:
            cols[f"{t}_{name}"] = frames[name][t]
    return pd.DataFrame(cols).dropna()


def fit_scaler(train_feats: pd.DataFrame) -> dict:
    std = train_feats.std().replace(0, 1.0)
    return {"mean": train_feats.mean().to_dict(), "std": std.to_dict()}


def apply_scaler(feats: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    mean = pd.Series(scaler["mean"])[feats.columns]
    std = pd.Series(scaler["std"])[feats.columns]
    return ((feats - mean) / std).clip(-FEATURE_CLIP, FEATURE_CLIP)


def load_dataset():
    """Return ({split: {prices, returns, features}}, scaler).

    Row t holds: close prices at t, the return realised from t-1 to t,
    and standardised features known at close t. In the env, an action
    taken at t is scored on returns.iloc[t+1].
    """
    prices = load_clean_prices()
    returns = compute_returns(prices)
    feats = build_features(prices)

    idx = feats.index.intersection(returns.index)
    prices, returns, feats = prices.loc[idx], returns.loc[idx], feats.loc[idx]

    scaler = fit_scaler(get_split(feats, "train"))   # TRAIN ONLY
    feats_z = apply_scaler(feats, scaler)

    data = {
        name: {
            "prices": get_split(prices, name),
            "returns": get_split(returns, name),
            "features": get_split(feats_z, name),
        }
        for name in SPLITS
    }
    return data, scaler


if __name__ == "__main__":
    data, scaler = load_dataset()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(DATA_PROCESSED / "scaler.json", "w") as f:
        json.dump(scaler, f, indent=2)
    for name, d in data.items():
        d["features"].to_csv(DATA_PROCESSED / f"features_{name}.csv")

    # ---- Sanity checks ----
    print("Feature count:", data["train"]["features"].shape[1])
    for name, d in data.items():
        f = d["features"]
        assert not f.isna().any().any(), f"NaNs in {name} features"
        assert len(f) == len(d["returns"]) == len(d["prices"])
        clipped = (f.abs() >= FEATURE_CLIP - 1e-9).mean().mean()
        print(f"{name:>5}: {len(f)} rows | mean|mean|={f.mean().abs().mean():.3f} "
              f"| mean std={f.std().mean():.3f} | clipped={clipped:.2%}")

    # Causality test: features at date d must not change if the future is removed
    prices = load_clean_prices()
    full = build_features(prices)
    d = full.index[1500]
    trunc = build_features(prices.loc[:d])
    assert np.allclose(full.loc[d].values, trunc.loc[d].values), "LOOKAHEAD DETECTED"
    print(f"Causality check passed (tested at {d.date()})")
    print("Saved scaler + feature CSVs to data/processed/")
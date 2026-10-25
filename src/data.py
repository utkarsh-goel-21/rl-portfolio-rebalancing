"""Data download, cleaning, caching and time-based splits."""
import pandas as pd
import yfinance as yf

from src.config import (
    TICKERS, DATA_START, DATA_END, DATA_RAW, DATA_PROCESSED,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
)

SPLITS = {
    "train": (TRAIN_START, TRAIN_END),
    "val": (VAL_START, VAL_END),
    "test": (TEST_START, TEST_END),
}


def download_prices(force: bool = False) -> pd.DataFrame:
    """Download adjusted close prices (cached in data/raw/prices.csv)."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    path = DATA_RAW / "prices.csv"
    if path.exists() and not force:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    end = (pd.Timestamp(DATA_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # end is exclusive
    raw = yf.download(TICKERS, start=DATA_START, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices[TICKERS]
    prices.index.name = "Date"
    prices.to_csv(path)
    return prices


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Sort, forward-fill tiny gaps, drop leftovers, and validate."""
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.ffill(limit=3).dropna()
    assert not prices.isna().any().any(), "NaNs remain after cleaning"
    assert (prices > 0).all().all(), "Non-positive prices found"
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns."""
    return prices.pct_change().dropna()


def load_clean_prices(force: bool = False) -> pd.DataFrame:
    return clean_prices(download_prices(force))


def get_split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = SPLITS[name]
    return df.loc[start:end]


if __name__ == "__main__":
    prices = load_clean_prices()
    returns = compute_returns(prices)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    prices.to_csv(DATA_PROCESSED / "prices_clean.csv")
    returns.to_csv(DATA_PROCESSED / "returns.csv")

    print("Shape:", prices.shape)
    print("Range:", prices.index.min().date(), "->", prices.index.max().date())
    print("Missing values:", int(prices.isna().sum().sum()))
    for name in SPLITS:
        print(f"{name:>5}: {len(get_split(prices, name))} trading days")
    print(prices.tail())
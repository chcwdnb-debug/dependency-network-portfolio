"""Universe, price download, and daily returns."""
from pathlib import Path

import pandas as pd

# Tickers that only changed names after 2019 (old -> Yahoo Finance today).
RENAMED = {"FB": "META", "BRK.B": "BRK-B", "UTX": "RTX", "BK": "BNY"}


def load_universe(path: Path) -> list[str]:
    """Ticker list as of late 2019, with renamed tickers mapped to today's symbols."""
    tickers = pd.read_csv(path)["ticker"].tolist()
    return [RENAMED.get(t, t) for t in tickers]


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Daily adjusted close from Yahoo Finance. Columns that are all NaN are dropped."""
    import yfinance as yf  # only needed for this step

    prices = yf.download(tickers=tickers, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]
    prices = prices.dropna(axis=1, how="all")
    if prices.columns.duplicated().any():
        raise ValueError("duplicate tickers in price data")
    return prices


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns. The first row has no previous day and is dropped."""
    return prices.pct_change().iloc[1:]


def load_returns(path: Path) -> pd.DataFrame:
    """Load returns and refuse files with duplicated tickers (pandas renames them X.1, X.2)."""
    returns = pd.read_csv(path, index_col=0, parse_dates=True)
    if returns.columns.str.contains(r"\.\d+$").any():
        raise ValueError(f"{path} has duplicated ticker columns")
    return returns

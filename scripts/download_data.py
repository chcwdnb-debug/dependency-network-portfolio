"""Download prices for the late-2019 S&P 100 list and save daily returns.

Usage:  python scripts/download_data.py
Output: data/returns.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netport import config
from netport.data import download_prices, load_universe, to_returns


def main() -> None:
    tickers = load_universe(config.UNIVERSE_FILE)
    prices = download_prices(tickers, config.START, config.END)
    missing = sorted(set(tickers) - set(prices.columns))
    returns = to_returns(prices)

    print(f"Universe: {len(tickers)}   Downloaded: {prices.shape[1]}   Missing: {missing}")
    print(f"Returns: {returns.shape[0]} days x {returns.shape[1]} stocks, "
          f"{returns.index[0].date()} to {returns.index[-1].date()}, "
          f"{int(returns.isna().sum().sum())} missing values")
    returns.to_csv(config.RETURNS_FILE)


if __name__ == "__main__":
    main()

"""Randomization test: the same strategy with randomly picked stocks."""
import numpy as np
import pandas as pd

from netport.backtest import apply_costs, run_strategy, sharpe


def random_picks(rng: np.random.Generator, tickers: list[str], k: int = 5) -> list[str]:
    return list(rng.choice(tickers, size=k, replace=False))


def percentile_rank(value: float, samples) -> float:
    """Fraction of samples strictly below `value`."""
    return float((np.asarray(samples) < value).mean())


def random_benchmark(returns: pd.DataFrame, dates: list, window: int, weigh,
                     n: int, cost: float, k: int = 5, seed: int = 2026,
                     verbose: bool = True) -> pd.DataFrame:
    """Run the strategy `n` times with k random stocks each month.

    Same universe, dates, weighting, and costs as the MST strategy.
    Returns one row per run: gross Sharpe, net Sharpe, average monthly turnover.
    """
    rng = np.random.default_rng(seed)
    tickers = list(returns.columns)
    rows = []
    for b in range(n):
        gross, turns = run_strategy(returns, dates, window,
                                    pick=lambda d, win: random_picks(rng, tickers, k), weigh=weigh)
        rows.append({
            "sharpe_gross": sharpe(gross),
            "sharpe_net": sharpe(apply_costs(gross, turns, cost)),
            "avg_turnover": turns.iloc[1:].mean(),
        })
        if verbose and (b + 1) % 100 == 0:
            print(f"  random portfolios: {b + 1}/{n}")
    return pd.DataFrame(rows)

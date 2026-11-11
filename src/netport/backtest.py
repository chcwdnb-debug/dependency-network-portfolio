"""Monthly rebalancing backtest.

Timeline on each rebalance date d (last trading day of a month):
- estimation window = the `window` trading days ending ON d (inclusive)
- holding period    = the trading day AFTER d through the next rebalance date
"""
from typing import Callable

import numpy as np
import pandas as pd

from netport.network import cost_matrix, top_k_by_degree
from netport.portfolio import end_weights, holding_returns, turnover, var_weights

PickRule = Callable[[pd.Timestamp, pd.DataFrame], list]
WeightRule = Callable[[pd.DataFrame, list], pd.Series]


def rebalance_dates(returns: pd.DataFrame, window: int) -> list[pd.Timestamp]:
    """Last trading day of each month that has at least `window` days of history."""
    month_ends = returns.groupby(returns.index.to_period("M")).tail(1).index
    return [d for d in month_ends if returns.index.get_loc(d) + 1 >= window]


def estimation_window(returns: pd.DataFrame, d: pd.Timestamp, window: int) -> pd.DataFrame:
    return returns.loc[:d].iloc[-window:]


def holding_period(returns: pd.DataFrame, d: pd.Timestamp, d_next: pd.Timestamp) -> pd.DataFrame:
    return returns[(returns.index > d) & (returns.index <= d_next)]


def select_stocks(returns: pd.DataFrame, dates: list, window: int, k: int,
                  horizon: int, verbose: bool = False) -> pd.DataFrame:
    """Top-k MST-degree stocks at each rebalance date except the last."""
    rows = {}
    for d in dates[:-1]:
        win = estimation_window(returns, d, window)
        rows[d] = list(top_k_by_degree(cost_matrix(win, horizon), win.columns, k).index)
        if verbose:
            print(d.date(), rows[d])
    return pd.DataFrame.from_dict(rows, orient="index", columns=[f"s{i + 1}" for i in range(k)])


def var_rule(alpha: float = 0.05, floor: float = 0.001) -> WeightRule:
    return lambda win, picks: var_weights(win, picks, alpha, floor)


def equal_rule() -> WeightRule:
    return lambda win, picks: pd.Series(1 / len(picks), index=list(picks))


def run_strategy(returns: pd.DataFrame, dates: list, window: int,
                 pick: PickRule, weigh: WeightRule) -> tuple[pd.Series, pd.Series]:
    """Daily gross returns, and turnover indexed by the first day of each holding period."""
    daily, turns = [], {}
    w_prev = pd.Series(dtype=float)                      # start in cash
    for d, d_next in zip(dates[:-1], dates[1:]):
        win = estimation_window(returns, d, window)
        hold = holding_period(returns, d, d_next)
        w = weigh(win, pick(d, win))
        turns[hold.index[0]] = turnover(w_prev, w)
        daily.append(holding_returns(w, hold))
        w_prev = end_weights(w, hold)
    return pd.concat(daily), pd.Series(turns)


def apply_costs(gross: pd.Series, turns: pd.Series, cost: float) -> pd.Series:
    """Subtract turnover x cost on the first day of each holding period."""
    net = gross.copy()
    net.loc[turns.index] -= turns * cost
    return net


def sharpe(ret: pd.Series, periods: int = 252) -> float:
    """Annualized Sharpe ratio with a zero risk-free rate (paper, Sec. 4.4)."""
    return ret.mean() / ret.std() * np.sqrt(periods)


def performance_stats(ret: pd.Series, periods: int = 252) -> pd.Series:
    nav = (1 + ret).cumprod()
    years = len(ret) / periods
    return pd.Series({
        "total_return": nav.iloc[-1] - 1,
        "annual_return": nav.iloc[-1] ** (1 / years) - 1,
        "annual_vol": ret.std() * np.sqrt(periods),
        "sharpe": sharpe(ret, periods),
        "max_drawdown": (nav / nav.cummax() - 1).min(),
    })

"""Weights, buy-and-hold returns within a period, and turnover."""
import pandas as pd


def var_weights(window: pd.DataFrame, tickers: list[str],
                alpha: float = 0.05, floor: float = 0.001) -> pd.Series:
    """Weights proportional to 1 / historical VaR (paper, eq. 11-12).

    VaR = -(alpha quantile of daily returns). Non-positive VaR is replaced by `floor`.
    """
    var = -window[tickers].quantile(alpha)
    raw = 1 / var.clip(lower=floor)
    return raw / raw.sum()


def holding_returns(weights: pd.Series, hold: pd.DataFrame) -> pd.Series:
    """Daily portfolio returns when buying `weights` at the start and not trading after.

    Weights drift with prices during the period.
    """
    growth = (1 + hold[weights.index]).cumprod()
    value = (growth * weights).sum(axis=1)
    prev = value.shift(1, fill_value=1.0)
    return value / prev - 1


def end_weights(weights: pd.Series, hold: pd.DataFrame) -> pd.Series:
    """Weights at the end of the holding period, after drift."""
    value = weights * (1 + hold[weights.index]).prod()
    return value / value.sum()


def turnover(w_old: pd.Series, w_new: pd.Series) -> float:
    """Sum of |new - old| over all stocks. A stock missing on one side counts as 0.

    Full swap = 2.0, buying from cash = 1.0.
    """
    return w_new.sub(w_old, fill_value=0).abs().sum()

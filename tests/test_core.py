"""Unit tests. Most of these are the CHECK cells from the original notebooks."""
import numpy as np
import pandas as pd
import pytest

from netport.backtest import apply_costs, holding_period, rebalance_dates
from netport.fevd import fevd_pair, fit_var1
from netport.network import cost_matrix, mst_degree, top_k_by_degree
from netport.portfolio import end_weights, holding_returns, turnover, var_weights
from netport.random_test import percentile_rank, random_picks


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    market = rng.normal(0, 0.01, (300, 1))
    data = market * rng.uniform(0.5, 1.5, 6) + rng.normal(0, 0.01, (300, 6))
    return pd.DataFrame(data, index=idx, columns=list("ABCDEF"))


# ---------- FEVD ----------
def test_fevd_rows_sum_to_one(returns):
    theta = fevd_pair(returns.iloc[:120, :2])
    assert np.allclose(theta.sum(axis=1), 1)
    assert ((theta >= 0) & (theta <= 1)).all()


def test_var1_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    A_true = np.array([[0.3, -0.2], [0.1, 0.4]])
    y = np.zeros((5000, 2))
    for t in range(1, 5000):
        y[t] = A_true @ y[t - 1] + rng.normal(0, 1, 2)
    A, _ = fit_var1(pd.DataFrame(y))
    assert np.allclose(A, A_true, atol=0.05)


def test_fevd_no_spillover_when_independent():
    rng = np.random.default_rng(2)
    r2 = pd.DataFrame(rng.normal(0, 1, (5000, 2)))
    theta = fevd_pair(r2)
    assert theta[0, 1] < 0.01 and theta[1, 0] < 0.01


# ---------- network ----------
def test_cost_matrix_shape_and_symmetry(returns):
    w = returns.iloc[:120]
    C = cost_matrix(w)
    assert C.shape == (6, 6)
    assert np.allclose(C, C.T)
    assert np.allclose(np.diag(C), 0)
    off = C[~np.eye(6, dtype=bool)]
    assert ((off > 0) & (off <= 1)).all()
    th = fevd_pair(w.iloc[:, [2, 4]])
    assert np.isclose(C[2, 4], min(1 - th[1, 0], 1 - th[0, 1]))


def test_star_network():
    C = np.full((5, 5), 0.9)
    C[2, :] = C[:, 2] = 0.1
    np.fill_diagonal(C, 0)
    names = ["A", "B", "HUB", "D", "E"]
    top = top_k_by_degree(C, names, k=5)
    assert top.index[0] == "HUB" and top.iloc[0] == 4
    assert mst_degree(C, names).sum() == 2 * (5 - 1)
    assert len(top_k_by_degree(C, names, k=2)) == 2


# ---------- portfolio ----------
def test_var_weights_inverse_risk():
    x = np.linspace(-0.05, 0.05, 120)
    toy = pd.DataFrame({"A": x, "B": 2 * x, "C": x})
    w = var_weights(toy, ["A", "B"])
    assert list(w.index) == ["A", "B"]
    assert np.isclose(w.sum(), 1)
    assert np.isclose(w["A"], 2 / 3) and np.isclose(w["B"], 1 / 3)


def test_var_weights_floor():
    x = np.linspace(-0.05, 0.05, 120)
    toy = pd.DataFrame({"A": x, "UP": np.abs(x) + 0.01})   # never loses -> VaR < 0
    w = var_weights(toy, ["A", "UP"])
    assert np.isfinite(w).all() and (w > 0).all()


def test_holding_returns_with_drift():
    hold = pd.DataFrame({"A": [0.10, -0.10], "B": [0.0, 0.0], "Z": [9.9, 9.9]})
    out = holding_returns(pd.Series({"A": 0.5, "B": 0.5}), hold)
    assert np.isclose(out.iloc[0], 0.05)
    assert np.isclose(out.iloc[1], 0.995 / 1.05 - 1)
    assert np.isclose((1 + out).prod(), 0.995)


def test_end_weights_drift():
    w = end_weights(pd.Series({"A": 0.5, "B": 0.5}), pd.DataFrame({"A": [0.10], "B": [0.0]}))
    assert np.isclose(w["A"], 0.55 / 1.05)


@pytest.mark.parametrize("old,new,expected", [
    ({"A": .5, "B": .5}, {"A": .5, "B": .5}, 0.0),
    ({"A": .5, "B": .5}, {"C": .5, "D": .5}, 2.0),
    ({"A": .5, "B": .5}, {"A": .5, "C": .5}, 1.0),
    ({}, {"A": .5, "B": .5}, 1.0),
])
def test_turnover(old, new, expected):
    assert np.isclose(turnover(pd.Series(old, dtype=float), pd.Series(new, dtype=float)), expected)


# ---------- backtest timing (no look-ahead) ----------
def test_rebalance_and_holding_have_no_overlap(returns):
    dates = rebalance_dates(returns, window=120)
    assert returns.index.get_loc(dates[0]) + 1 >= 120
    for d, d_next in zip(dates[:-1], dates[1:]):
        hold = holding_period(returns, d, d_next)
        assert hold.index[0] > d                  # starts the day AFTER d
        assert hold.index[-1] == d_next


def test_apply_costs_first_day_only():
    gross = pd.Series([0.01, 0.01, 0.01], index=pd.bdate_range("2020-01-01", periods=3))
    net = apply_costs(gross, pd.Series([2.0], index=[gross.index[0]]), cost=0.001)
    assert np.isclose(net.iloc[0], 0.01 - 0.002) and np.isclose(net.iloc[1], 0.01)


# ---------- random benchmark ----------
def test_random_picks():
    names = list("ABCDEFGHIJ")
    p = random_picks(np.random.default_rng(0), names, k=5)
    assert len(set(p)) == 5 and set(p) <= set(names)
    assert p == random_picks(np.random.default_rng(0), names, k=5)


def test_percentile_rank():
    s = [0.1, 0.2, 0.3, 0.4]
    assert percentile_rank(0.25, s) == 0.5
    assert percentile_rank(1.0, s) == 1.0
    assert percentile_rank(0.0, pd.Series(s)) == 0.0

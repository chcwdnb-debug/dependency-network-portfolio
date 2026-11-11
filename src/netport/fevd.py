"""Bivariate VAR(1) and generalized forecast error variance decomposition."""
import numpy as np
import pandas as pd


def fit_var1(r2: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit y_t = c + A y_{t-1} + u_t by least squares.

    Parameters
    ----------
    r2 : DataFrame with exactly 2 columns of returns.

    Returns
    -------
    A : (2, 2) VAR matrix, one row per equation.
    Sigma : (2, 2) covariance of the residuals u_t.
    """
    today = r2.iloc[1:].to_numpy()
    yesterday = r2.shift(1).iloc[1:].to_numpy()        # lag by one day: no look-ahead
    X = np.column_stack([np.ones(len(today)), yesterday])
    B, *_ = np.linalg.lstsq(X, today, rcond=None)       # (3, 2), one column per equation
    resid = today - X @ B
    A = B[1:].T
    Sigma = np.cov(resid.T)
    return A, Sigma


def generalized_fevd(A: np.ndarray, Sigma: np.ndarray, horizon: int) -> np.ndarray:
    """Generalized FEVD (Diebold & Yilmaz, 2012), rows normalized to sum to 1.

    theta[j, i] = share of variable j's h-step forecast error variance
    that comes from shocks to variable i.
    """
    n = A.shape[0]
    num = np.zeros((n, n))
    den = np.zeros(n)
    for s in range(horizon):
        Phi = np.linalg.matrix_power(A, s)              # impulse response at step s
        PS = Phi @ Sigma
        num += PS ** 2 / np.diag(Sigma)[None, :]
        den += np.diag(PS @ Phi.T)
    theta = num / den[:, None]
    return theta / theta.sum(axis=1, keepdims=True)


def fevd_pair(r2: pd.DataFrame, horizon: int = 10) -> np.ndarray:
    """2x2 FEVD matrix for a pair of stocks. Each row sums to 1."""
    A, Sigma = fit_var1(r2)
    return generalized_fevd(A, Sigma, horizon)

"""Cost matrix, minimum spanning tree, and degree-based selection."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree

from netport.fevd import fevd_pair


def cost_matrix(window: pd.DataFrame, horizon: int = 10) -> np.ndarray:
    """Symmetric cost matrix from pairwise FEVD.

    C[i, j] = min(1 - theta_{j<-i}, 1 - theta_{i<-j})  (paper, eq. 7-8).
    Stronger influence -> lower cost. Diagonal is 0.
    """
    n = window.shape[1]
    C = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        theta = fevd_pair(window.iloc[:, [i, j]], horizon)
        c_ij = 1 - theta[1, 0]      # i -> j
        c_ji = 1 - theta[0, 1]      # j -> i
        C[i, j] = C[j, i] = min(c_ij, c_ji)
    return C


def mst_degree(C: np.ndarray, tickers) -> pd.Series:
    """Number of MST edges at each node."""
    T = minimum_spanning_tree(C).toarray()   # nonzero = edge; only one triangle is filled
    T = T + T.T
    return pd.Series((T > 0).sum(axis=1), index=tickers)


def top_k_by_degree(C: np.ndarray, tickers, k: int = 5) -> pd.Series:
    """The k nodes with the highest MST degree."""
    return mst_degree(C, tickers).sort_values(ascending=False).head(k)

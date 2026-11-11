"""Where does FEVD dependence come from? A worked example on one stock pair.

Fits a VAR(1) to AAPL and MSFT over the first 120-day window and computes the
FEVD share theta three ways:
  1. full: horizon 10, as in the backtest
  2. same-day only: horizon 1, so only s = 0 (the residual covariance) counts
  3. no same-day correlation: horizon 10, with the off-diagonal of Sigma set to 0

Usage:  python scripts/pair_example.py [--a AAPL] [--b MSFT]
Output: results/pair_example.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from netport import config
from netport.data import load_returns
from netport.fevd import fit_var1, generalized_fevd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="AAPL")
    p.add_argument("--b", default="MSFT")
    args = p.parse_args()

    returns = load_returns(config.RETURNS_FILE)
    r2 = returns[[args.a, args.b]].iloc[:config.WINDOW]
    A, Sigma = fit_var1(r2)
    resid_corr = Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])

    cases = {
        "full (h = 10)": generalized_fevd(A, Sigma, config.HORIZON),
        "same-day term only (h = 1)": generalized_fevd(A, Sigma, 1),
        "residual correlation removed": generalized_fevd(A, np.diag(np.diag(Sigma)), config.HORIZON),
    }
    table = pd.DataFrame({
        f"theta {args.b} -> {args.a}": {k: v[0, 1] for k, v in cases.items()},
        f"theta {args.a} -> {args.b}": {k: v[1, 0] for k, v in cases.items()},
    })
    table.to_csv(config.RESULTS_DIR / "pair_example.csv")

    print(f"{args.a}-{args.b}, {r2.index[0].date()} to {r2.index[-1].date()}")
    print("VAR(1) matrix A (rows = equations):")
    print(pd.DataFrame(A, index=[args.a, args.b], columns=[f"{args.a} lag", f"{args.b} lag"]).round(3))
    print(f"Residual correlation: {resid_corr:.3f}\n")
    print(table.round(3))


if __name__ == "__main__":
    main()

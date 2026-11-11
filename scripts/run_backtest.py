"""Run the full study: selection, backtest, transaction costs, random benchmark.

Usage:  python scripts/run_backtest.py [--n-random 1000] [--recompute]
Output: results/*.csv and results/figures/*.png
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from netport import config
from netport.backtest import (
    apply_costs, equal_rule, performance_stats, rebalance_dates, run_strategy,
    select_stocks, sharpe, var_rule,
)
from netport.data import load_returns
from netport.plots import plot_nav, plot_random_test
from netport.random_test import percentile_rank, random_benchmark

COSTS_BP = [0, 10, 30]
RANDOM_COST_BP = 10


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-random", type=int, default=1000, help="number of random portfolios")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--recompute", action="store_true", help="ignore cached selections")
    args = p.parse_args()

    out, fig = config.RESULTS_DIR, config.FIGURES_DIR
    fig.mkdir(parents=True, exist_ok=True)

    returns = load_returns(config.RETURNS_FILE)
    dates = rebalance_dates(returns, config.WINDOW)
    print(f"Data: {returns.shape[1]} stocks, {len(dates) - 1} monthly rebalances "
          f"({dates[0].date()} to {dates[-2].date()})")

    # 1. Stock selection (slow part, cached)
    sel_file = out / "selections.csv"
    if sel_file.exists() and not args.recompute:
        sel = pd.read_csv(sel_file, index_col=0, parse_dates=True)
        print("Loaded cached selections")
    else:
        t0 = time.time()
        sel = select_stocks(returns, dates, config.WINDOW, config.TOP_K, config.HORIZON)
        sel.to_csv(sel_file)
        print(f"Selected stocks in {time.time() - t0:.0f}s")

    mst_pick = lambda d, win: list(sel.loc[d])
    all_pick = lambda d, win: list(win.columns)
    var_w = var_rule(config.VAR_ALPHA, config.VAR_FLOOR)

    # 2. Backtest without costs
    mst_var, mst_turns = run_strategy(returns, dates, config.WINDOW, mst_pick, var_w)
    mst_eq, _ = run_strategy(returns, dates, config.WINDOW, mst_pick, equal_rule())
    all_eq, _ = run_strategy(returns, dates, config.WINDOW, all_pick, equal_rule())
    perf = pd.DataFrame({
        "MST + VaR": performance_stats(mst_var),
        "MST + equal weight": performance_stats(mst_eq),
        "All stocks, equal weight": performance_stats(all_eq),
    }).T
    perf.to_csv(out / "performance.csv")
    plot_nav({"MST + VaR": mst_var, "MST + equal": mst_eq, "All stocks equal": all_eq},
             fig / "nav_no_cost.png", "Net value, monthly rebalance, no transaction costs")

    # 3. Transaction costs
    costs = pd.DataFrame({f"{bp} bp": performance_stats(apply_costs(mst_var, mst_turns, bp / 1e4))
                          for bp in COSTS_BP}).T
    costs["avg_turnover"] = mst_turns.iloc[1:].mean()
    costs.to_csv(out / "costs.csv")

    # 4. Random benchmark
    t0 = time.time()
    rand = random_benchmark(returns, dates, config.WINDOW, var_w, n=args.n_random,
                            cost=RANDOM_COST_BP / 1e4, k=config.TOP_K, seed=args.seed)
    rand.to_csv(out / "random_sharpes.csv", index=False)
    mst_gross = sharpe(mst_var)
    mst_net = sharpe(apply_costs(mst_var, mst_turns, RANDOM_COST_BP / 1e4))
    summary = pd.DataFrame({
        "mst_sharpe": [mst_gross, mst_net],
        "random_median": [rand.sharpe_gross.median(), rand.sharpe_net.median()],
        "mst_percentile": [percentile_rank(mst_gross, rand.sharpe_gross),
                           percentile_rank(mst_net, rand.sharpe_net)],
        "mst_turnover": mst_turns.iloc[1:].mean(),
        "random_turnover": rand.avg_turnover.mean(),
    }, index=["no cost", f"{RANDOM_COST_BP} bp"])
    summary.to_csv(out / "random_test.csv")
    plot_random_test(rand.sharpe_net, mst_net, percentile_rank(mst_net, rand.sharpe_net),
                     fig / "random_test.png",
                     f"Sharpe ratio after {RANDOM_COST_BP} bp costs: MST vs random picks")
    print(f"Random benchmark ({args.n_random} runs) in {time.time() - t0:.0f}s")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\nPerformance (no costs)\n", perf.round(3))
    print("\nMST + VaR with costs\n", costs.round(3))
    print("\nRandom benchmark\n", summary.round(3))


if __name__ == "__main__":
    main()

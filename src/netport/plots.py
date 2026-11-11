"""Figures for the README."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_nav(series: dict[str, pd.Series], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, ret in series.items():
        ax.plot((1 + ret).cumprod(), label=name)
    ax.set_title(title)
    ax.set_ylabel("Value of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_random_test(random_sharpes, mst_sharpe: float, pct: float, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(random_sharpes, bins=30, color="#9aa5b1", edgecolor="white",
            label=f"Random 5 stocks (N={len(random_sharpes)})")
    ax.axvline(mst_sharpe, color="#c2410c", lw=2.5, label=f"MST + VaR (percentile {pct:.0%})")
    ax.set_title(title)
    ax.set_xlabel("Annualized Sharpe ratio")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

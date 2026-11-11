"""Default settings and paths.

Values follow Lin, Liu & Rojas (2025) unless noted.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

UNIVERSE_FILE = DATA_DIR / "sp100_20191231.csv"
RETURNS_FILE = DATA_DIR / "returns.csv"

START = "2019-07-01"     # ~6 months before 2020 for the first window
END = "2025-01-01"       # yfinance `end` is exclusive

WINDOW = 120             # trading days per estimation window (paper, Sec. 2)
HORIZON = 10             # FEVD forecast horizon in days (Sec. 3.2)
TOP_K = 5                # stocks per portfolio (Sec. 4.2)
VAR_ALPHA = 0.05         # 1-day historical VaR at 95% (Sec. 4.3)
VAR_FLOOR = 0.001        # replaces non-positive VaR (Sec. 4.3)

TRADING_DAYS = 252

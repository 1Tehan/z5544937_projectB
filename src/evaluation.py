"""Small evaluation helpers used by the Part B robustness extension.

The block bootstrap deliberately resamples the *same* date blocks for every
selected fund in a draw.  This preserves cross-fund dependence and makes paired
Sharpe differences meaningful.  It uses the same geometric annualised-return
Sharpe definition as ``src.portfolios.performance_metrics``.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def geometric_sharpe(daily: np.ndarray, periods_per_year: int = 252) -> float:
    """Sharpe matching the project's fact-sheet definition (rf = 0)."""
    x = np.asarray(daily, dtype=float)
    growth = float(np.prod(1.0 + x))
    ann_ret = growth ** (periods_per_year / len(x)) - 1.0
    ann_vol = float(np.std(x, ddof=1)) * np.sqrt(periods_per_year)
    return ann_ret / ann_vol if ann_vol > 0 else np.nan


def moving_block_bootstrap_sharpe(
    returns: pd.DataFrame,
    funds: Iterable[str],
    *,
    periods_per_year: int = 252,
    block_length: int = 21,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return individual 90% Sharpe intervals and paired diagnostics.

    ``returns`` must contain the selected equity-calendar funds. Rows where any
    selected fund is missing are dropped. Each bootstrap draw samples contiguous
    21-day blocks with replacement, concatenates them to the original sample
    length, and applies that identical resampled date path to every selected fund.
    """
    funds = list(funds)
    panel = returns[funds].dropna(how="any")
    arr = panel.to_numpy(dtype=float)
    n = len(panel)
    if n < block_length:
        raise ValueError("sample shorter than block length")

    rng = np.random.default_rng(seed)
    blocks_per_draw = math.ceil(n / block_length)
    boot = np.empty((n_boot, len(funds)), dtype=float)
    max_start = n - block_length + 1

    for b in range(n_boot):
        starts = rng.integers(0, max_start, size=blocks_per_draw)
        idx = np.concatenate(
            [np.arange(s, s + block_length, dtype=int) for s in starts]
        )[:n]
        sample = arr[idx]
        for j in range(len(funds)):
            boot[b, j] = geometric_sharpe(sample[:, j], periods_per_year)

    rows = []
    for j, fund in enumerate(funds):
        point = geometric_sharpe(arr[:, j], periods_per_year)
        lo, hi = np.quantile(boot[:, j], [0.05, 0.95])
        rows.append({"fund_id": fund, "sharpe": point,
                     "ci_5pct": lo, "ci_95pct": hi})
    individual = pd.DataFrame(rows).set_index("fund_id")

    pairs = [
        ("combined_risk_parity", "combined_max_sharpe", "risk_parity_minus_max_sharpe"),
        ("equity_max_sharpe_tilt_finvader", "equity_max_sharpe", "finvader_tilt_minus_base"),
    ]
    paired_rows = []
    col = {f: i for i, f in enumerate(funds)}
    for a, b, label in pairs:
        if a not in col or b not in col:
            continue
        diff = boot[:, col[a]] - boot[:, col[b]]
        point = individual.loc[a, "sharpe"] - individual.loc[b, "sharpe"]
        lo, hi = np.quantile(diff, [0.05, 0.95])
        paired_rows.append({
            "comparison": label,
            "point_difference": point,
            "ci_5pct": lo,
            "ci_95pct": hi,
            "share_first_better": float(np.mean(diff > 0)),
        })
    paired = pd.DataFrame(paired_rows).set_index("comparison")
    return individual, paired

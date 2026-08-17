"""Station 3 (fusion) - fold the sector sentiment into the equity fund.

Rule (a transparent tilt, stated in the report):
  at each rebalance date t,
    1. LAG the daily sector index by one trading day (shift(1)), so day t's
       decision uses only sentiment from t-1 or earlier - a Saturday/Monday
       headline (both mapped to Monday) is first usable for Tuesday;
    2. take the trailing 21-day mean of the lagged index (a monthly tone
       signal, matching the monthly rebalance);
    3. z-score it CROSS-SECTIONALLY across the 10 sectors on that date;
    4. multiply each stock's base weight by clip(1 + gamma * z_sector, 0.6, 1.4)
       with gamma = 0.25, then renormalise to fully invested.

The tilt reuses the SAME walk-forward engine dates and estimation windows as
the base fund, so base vs tilted is a clean like-for-like comparison. Applied
to equities only - crypto has no news in this data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GAMMA = 0.25
TILT_FLOOR, TILT_CAP = 0.6, 1.4
SIGNAL_WINDOW = 21
LAG_DAYS = 1


def sector_signal(sector_index: pd.DataFrame) -> pd.DataFrame:
    """Lagged, smoothed, cross-sectionally z-scored sector signal (date x sector)."""
    sectors = [c for c in sector_index.columns if c != "ALL"]
    sig = (sector_index[sectors]
           .shift(LAG_DAYS)                       # look-ahead safety
           .rolling(SIGNAL_WINDOW, min_periods=5).mean())
    mu = sig.mean(axis=1)
    sd = sig.std(axis=1).replace(0, np.nan)
    return sig.sub(mu, axis=0).div(sd, axis=0)


def tilt_weights(base_weights: pd.DataFrame, signal: pd.DataFrame,
                 sector_map: dict[str, str],
                 gamma: float = GAMMA) -> pd.DataFrame:
    """Apply the sentiment tilt to each rebalance-date weight vector."""
    tilted = {}
    for t, w in base_weights.iterrows():
        # last signal STRICTLY BEFORE the rebalance date (already lagged too)
        past = signal.loc[:t]
        if len(past) and past.index[-1] == t:
            past = past.iloc[:-1]
        if past.empty or past.iloc[-1].isna().all():
            tilted[t] = w.values
            continue
        z = past.iloc[-1]
        mult = np.array([np.clip(1 + gamma * z.get(sector_map.get(a, ""), 0.0),
                                 TILT_FLOOR, TILT_CAP)
                         if a in sector_map else 1.0
                         for a in base_weights.columns])
        mult = np.nan_to_num(mult, nan=1.0)
        wt = np.maximum(w.values * mult, 0)
        tilted[t] = wt / wt.sum()
    out = pd.DataFrame(tilted, index=base_weights.columns).T
    out.index.name = "date"
    return out


def run_weight_path(returns: pd.DataFrame, weights: pd.DataFrame) -> dict:
    """Replay a rebalance-date weight schedule through the daily-drift engine.

    Same mechanics as portfolios.walk_forward_backtest, but with the weight
    schedule supplied - used so the tilted fund is backtested identically to
    its base fund.
    """
    rets = returns.dropna(how="all").fillna(0.0)[weights.columns]
    rebs = weights.index
    fund_ret, turn = [], {}
    w_drift = None
    for k, t in enumerate(rebs):
        w_tgt = weights.loc[t].values
        turn[t] = 0.0 if w_drift is None else 0.5 * np.abs(w_tgt - w_drift).sum()
        t_end = rebs[k + 1] if k + 1 < len(rebs) else None
        window = rets.loc[t:t_end]
        if t_end is not None:
            window = window.iloc[:-1]
        w = w_tgt.copy()
        for _, r in window.iterrows():
            rp = float(w @ r.values)
            fund_ret.append(rp)
            w = w * (1 + r.values) / (1 + rp) if (1 + rp) != 0 else w
        w_drift = w
    idx = rets.loc[rebs[0]:].index
    return {"daily": pd.Series(fund_ret, index=idx, name="ret"),
            "weights": weights,
            "turnover": pd.Series(turn, name="turnover")}

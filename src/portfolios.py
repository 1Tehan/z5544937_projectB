"""Station 3 - the funds: optimal portfolios + walk-forward OOS backtest.

Design choices (all stated in the report):
  * Long-only, fully invested. Per-asset weight cap so max-Sharpe cannot
    collapse into one name (20% for 50-60 asset funds, 40% for crypto-only).
  * Mean/covariance are ANNUALISED before optimisation (tiny daily-return
    covariances can stall SLSQP below its tolerance - a known trap), and the
    covariance is shrunk 10% toward its diagonal for numerical stability.
  * Walk-forward: weights at rebalance date t use ONLY returns up to t-1
    (a strict `< t` slice), so there is no look-ahead by construction.
  * Between rebalances weights DRIFT with returns (buy-and-hold within the
    month); turnover is measured against the drifted weights, which feeds the
    transaction-cost extension.
  * Risk-free rate: 0 (stated). Rebalance: first trading day of each month.
  * Annualisation: 252 for equity-calendar funds, 365 for the crypto-only
    funds that live on the 7-day crypto calendar.

Methods: equal_weight, min_variance, max_sharpe, risk_parity (exact ERC via the
Spinu log-barrier formulation), and hrp (hierarchical risk parity, Lopez de
Prado 2016) - HRP is the 'newer method' innovation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import minimize

# ----------------------------------------------------------------- optimisers

def _shrink(cov: np.ndarray, delta: float = 0.10) -> np.ndarray:
    """Shrink the covariance 10% toward its diagonal (stabilises inversion)."""
    return (1 - delta) * cov + delta * np.diag(np.diag(cov))


def equal_weight(n: int, **_) -> np.ndarray:
    return np.full(n, 1.0 / n)


def min_variance(mu: np.ndarray, cov: np.ndarray, cap: float) -> np.ndarray:
    n = len(mu)
    cov = _shrink(cov)
    res = minimize(lambda w: w @ cov @ w, equal_weight(n),
                   jac=lambda w: 2 * cov @ w,
                   bounds=[(0.0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-10})
    return res.x if res.success else equal_weight(n)


def max_sharpe(mu: np.ndarray, cov: np.ndarray, cap: float) -> np.ndarray:
    """Maximise (w'mu)/sqrt(w'Sigma w), rf = 0, long-only, capped."""
    n = len(mu)
    cov = _shrink(cov)

    def neg_sharpe(w):
        vol = np.sqrt(max(w @ cov @ w, 1e-12))
        return -(w @ mu) / vol

    res = minimize(neg_sharpe, equal_weight(n),
                   bounds=[(0.0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-10})
    return res.x if res.success else min_variance(mu, cov, cap)


def risk_parity(mu: np.ndarray, cov: np.ndarray, cap: float) -> np.ndarray:
    """Exact equal-risk-contribution weights (Spinu 2013 log-barrier form).

    Minimise 0.5 x'Sigma x - (1/n) sum log(x) over x > 0; the normalised
    solution equalises risk contributions. No cap applied - ERC is naturally
    diversified and a cap would break exactness (stated in the report).
    """
    n = len(mu)
    cov = _shrink(cov)

    def obj(x):
        return 0.5 * x @ cov @ x - np.log(np.maximum(x, 1e-12)).sum() / n

    def grad(x):
        return cov @ x - 1.0 / (n * np.maximum(x, 1e-12))

    res = minimize(obj, equal_weight(n), jac=grad,
                   bounds=[(1e-9, None)] * n,
                   method="L-BFGS-B", options={"maxiter": 1000})
    x = res.x if res.success else equal_weight(n)
    return x / x.sum()


def hrp(mu: np.ndarray, cov: np.ndarray, cap: float,
        returns_window: pd.DataFrame | None = None) -> np.ndarray:
    """Hierarchical Risk Parity (Lopez de Prado 2016) - the 'newer method'.

    Cluster assets on correlation distance, quasi-diagonalise, then allocate
    top-down by inverse cluster variance. Needs no matrix inversion, so it is
    robust when the covariance is noisy (60 assets, 252-day window).
    """
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1, 1)
    dist = np.sqrt(0.5 * (1 - corr))
    # condensed distance for linkage
    iu = np.triu_indices_from(dist, k=1)
    order = leaves_list(linkage(dist[iu], method="single"))

    w = pd.Series(1.0, index=order)
    clusters = [order.tolist()]
    while clusters:
        clusters = [c[j:k] for c in clusters
                    for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
                    if len(c) > 1]
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                continue
            left, right = clusters[i], clusters[i + 1]

            def cluster_var(idx):
                sub = cov[np.ix_(idx, idx)]
                ivp = 1.0 / np.diag(sub)
                ivp /= ivp.sum()
                return ivp @ sub @ ivp

            vl, vr = cluster_var(left), cluster_var(right)
            alpha = 1 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1 - alpha
    out = np.zeros(len(mu))
    out[w.index.values] = w.values
    return out / out.sum()


OPTIMISERS = {
    "equal_weight": lambda mu, cov, cap: equal_weight(len(mu)),
    "min_variance": min_variance,
    "max_sharpe": max_sharpe,
    "risk_parity": risk_parity,
    "hrp": hrp,
}

METHOD_LABEL = {
    "equal_weight": "Equal Weight",
    "min_variance": "Minimum Variance",
    "max_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk Parity (ERC)",
    "hrp": "Hierarchical Risk Parity",
}

# ------------------------------------------------------------ backtest engine

def month_start_rebalances(index: pd.DatetimeIndex,
                           first_live: pd.Timestamp) -> pd.DatetimeIndex:
    """First trading day of each month at or after `first_live`."""
    s = pd.Series(index, index=index)
    firsts = s.groupby([index.year, index.month]).first()
    return pd.DatetimeIndex([d for d in firsts if d >= first_live])


def walk_forward_backtest(returns: pd.DataFrame, method: str,
                          lookback: int, cap: float,
                          ann_factor: int) -> dict:
    """Walk-forward OOS backtest of one (universe, method) fund.

    At each month-start rebalance date t, weights are estimated from the
    `lookback` most recent rows STRICTLY BEFORE t, then held (drifting with
    returns) until the next rebalance. Returns daily fund returns, the target
    weights at every rebalance, and per-rebalance turnover.
    """
    rets = returns.dropna(how="all").fillna(0.0)
    idx = rets.index
    first_live = idx[lookback]
    rebs = month_start_rebalances(idx, first_live)

    opt = OPTIMISERS[method]
    fund_ret, w_hist, turn = [], {}, {}
    w_drift = None
    for k, t in enumerate(rebs):
        past = rets.loc[:t].iloc[:-1].tail(lookback)      # strictly before t
        mu = past.mean().values * ann_factor
        cov = past.cov().values * ann_factor
        w_tgt = opt(mu, cov, cap)
        w_tgt = np.maximum(w_tgt, 0)
        w_tgt = w_tgt / w_tgt.sum()
        w_hist[t] = w_tgt
        turn[t] = 0.0 if w_drift is None else 0.5 * np.abs(w_tgt - w_drift).sum()

        t_end = rebs[k + 1] if k + 1 < len(rebs) else None
        window = rets.loc[t:t_end]
        if t_end is not None:
            window = window.iloc[:-1]                     # next reb starts next block
        w = w_tgt.copy()
        for _, r in window.iterrows():
            rp = float(w @ r.values)
            fund_ret.append(rp)
            w = w * (1 + r.values) / (1 + rp) if (1 + rp) != 0 else w
        w_drift = w

    live_idx = idx[idx >= rebs[0]]
    daily = pd.Series(fund_ret, index=live_idx, name="ret")
    weights = pd.DataFrame(w_hist, index=rets.columns).T
    weights.index.name = "date"
    turnover = pd.Series(turn, name="turnover")
    return {"daily": daily, "weights": weights, "turnover": turnover}


# ---------------------------------------------------------------- fact sheet

def drawdown_series(daily: pd.Series) -> pd.Series:
    g = (1 + daily).cumprod()
    return g / g.cummax() - 1


def performance_metrics(daily: pd.Series, periods_per_year: int,
                        rf: float = 0.0) -> dict:
    """Annualised return (geometric), volatility, Sharpe (rf=0), max drawdown."""
    n = len(daily)
    growth = float((1 + daily).prod())
    ann_ret = growth ** (periods_per_year / n) - 1
    ann_vol = float(daily.std(ddof=1)) * np.sqrt(periods_per_year)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    max_dd = float(drawdown_series(daily).min())
    return {"ann_return": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "max_drawdown": max_dd,
            "growth_of_1": growth, "n_obs": n}


def net_of_costs(daily: pd.Series, turnover: pd.Series,
                 cost_bps: float = 10.0) -> pd.Series:
    """Transaction-cost extension: charge `cost_bps` x turnover on rebalance days."""
    fees = pd.Series(0.0, index=daily.index)
    hit = turnover.index.intersection(daily.index)
    fees.loc[hit] = turnover.loc[hit].values * cost_bps / 10_000.0
    return daily - fees

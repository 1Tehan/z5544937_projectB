"""Reproduce ALL Part B results. Run from the project root:

    python scripts/run_part_b.py

Writes (exact required names first):
    results/data/fund_returns.csv            daily OOS returns, one column per fund
    results/data/fund_weights.csv            long: fund_id, date, asset, weight
    results/data/sector_sentiment_index.csv  daily plain-VADER sector index (+ ALL)
    results/tables/performance_metrics.csv   fact-sheet metrics for every fund
plus additional artifacts (fund_catalog, Fin-VADER index, coverage, fusion
comparison, lexicon stats, reproducible block-bootstrap diagnostics) and every
required report figure under
results/figures/. Ends with a key-numbers dump for the report narrative.

Runtime: a few minutes (VADER scores ~147k headlines twice). The deployed app
never runs this - it only READS results/.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import etl, evaluation, features, fusion, portfolios, sentiment  # noqa: E402
from src.plotstyle import CYCLE, PALETTE, fundx_theme, save_fig        # noqa: E402

DATA = ROOT / "results" / "data"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
for p in (DATA, TABLES, FIGS):
    p.mkdir(parents=True, exist_ok=True)

LOOKBACK_EQ, LOOKBACK_CR = 252, 365
CAP_WIDE, CAP_CRYPTO = 0.20, 0.40
COST_BPS = 10.0
METHODS = ["equal_weight", "min_variance", "max_sharpe", "risk_parity", "hrp"]

FAMILY_LABEL = {"combined": "Combined (equity + crypto)",
                "equity": "Equity-only", "crypto": "Crypto-only"}


def fund_label(family: str, method: str) -> str:
    return f"{FAMILY_LABEL[family].split(' (')[0]} {portfolios.METHOD_LABEL[method]}"


def main() -> None:
    fundx_theme()

    # ---------------------------------------------------------------- data
    # Station 1-2 reuse my Part A implementation unchanged; only Part B provenance notes were added
    # (cap 2023-12-31, dedup ticker+date+title, tz-naive news), same
    # compute-returns-then-merge order, same trading calendar.
    print("[1/6] loading + cleaning through the Part A foundation (src/etl.py) ...")
    equity = etl.load_clean_equities()
    crypto = etl.load_clean_crypto()          # drops the 10 stray 2024-01-01 rows
    news = etl.load_clean_news()              # dedup ticker+date+title, tz stripped
    sector_map = dict(zip(equity["ticker"], equity["sector"]))
    calendar = etl.trading_calendar(equity)   # 1,006 equity trading days

    eq_ret_long = features.daily_returns(equity)
    cr_ret_long = features.daily_returns(crypto)
    combined = features.combined_returns_panel(eq_ret_long, cr_ret_long, calendar)
    eq_only = features.to_wide(eq_ret_long).reindex(calendar)
    cr_only = features.to_wide(cr_ret_long)   # crypto's own 365-day calendar
    # first row of each panel is all-NaN (no prior day) - drop for the backtests
    combined, eq_only, cr_only = combined.iloc[1:], eq_only.iloc[1:], cr_only.iloc[1:]
    trading_days = eq_only.index

    # ------------------------------------------------------------- funds
    print("[2/6] walk-forward out-of-sample backtests ...")
    runs: dict[str, dict] = {}
    catalog = []
    spec = [("combined", combined, LOOKBACK_EQ, CAP_WIDE, 252),
            ("equity", eq_only, LOOKBACK_EQ, CAP_WIDE, 252),
            ("crypto", cr_only, LOOKBACK_CR, CAP_CRYPTO, 365)]
    for family, panel, lookback, cap, ann in spec:
        for method in METHODS:
            fid = f"{family}_{method}"
            print(f"    {fid}")
            runs[fid] = portfolios.walk_forward_backtest(
                panel, method, lookback=lookback, cap=cap, ann_factor=ann)
            runs[fid]["ann"] = ann
            explicit_cap = cap if method in {"min_variance", "max_sharpe"} else np.nan
            catalog.append({"fund_id": fid, "label": fund_label(family, method),
                            "family": family, "method": method,
                            "ann_factor": ann, "weight_cap": explicit_cap,
                            "lookback": lookback, "rebalance": "monthly",
                            "oos_start": str(runs[fid]["daily"].index[0].date()),
                            "oos_end": str(runs[fid]["daily"].index[-1].date())})

    # sanity: methods must genuinely differ (solver-stall trap from the brief)
    w_chk = {m: runs[f"combined_{m}"]["weights"].iloc[-1] for m in METHODS}
    spread = pd.DataFrame(w_chk).std(axis=1).sum()
    assert spread > 0.01, "optimiser weights identical across methods - check scaling"

    # -------------------------------------------------------- sentiment
    print("[3/6] sentiment: scoring headlines with VADER and Fin-VADER ...")
    panel = features.assemble_headline_panel(news, calendar)   # Part A Station 2
    per_headline = sentiment.explode_panel(panel)              # the ' || ' round trip
    assert len(per_headline) == len(news), "panel explode lost/invented headlines"
    scored = sentiment.score_headlines(per_headline)
    stats = sentiment.neutral_stats(scored)

    td_vader = sentiment.ticker_day_sentiment(scored, "compound_vader")
    td_fin = sentiment.ticker_day_sentiment(scored, "compound_finvader")
    idx_vader = sentiment.sector_sentiment_index(td_vader)
    idx_fin = sentiment.sector_sentiment_index(td_fin)
    coverage = sentiment.coverage_series(td_vader)

    # ----------------------------------------------------------- fusion
    print("[4/6] fusion: sentiment tilt on the Equity Maximum-Sharpe fund ...")
    base_id = "equity_max_sharpe"
    base = runs[base_id]
    for tag, idx in [("vader", idx_vader), ("finvader", idx_fin)]:
        sig = fusion.sector_signal(idx)
        w_tilt = fusion.tilt_weights(base["weights"], sig, sector_map)
        fid = f"equity_max_sharpe_tilt_{tag}"
        runs[fid] = fusion.run_weight_path(eq_only, w_tilt)
        runs[fid]["ann"] = 252
        catalog.append({"fund_id": fid,
                        "label": ("Equity Max Sharpe + Sentiment Tilt "
                                  + ("(Fin-VADER)" if tag == "finvader" else "(VADER)")),
                        "family": "equity", "method": f"max_sharpe_tilt_{tag}",
                        "ann_factor": 252, "weight_cap": np.nan,
                        "lookback": LOOKBACK_EQ, "rebalance": "monthly",
                        "oos_start": str(runs[fid]["daily"].index[0].date()),
                        "oos_end": str(runs[fid]["daily"].index[-1].date())})

    # ------------------------------------------------------- artifacts
    print("[5/6] writing app artifacts + tables ...")
    cat = pd.DataFrame(catalog)
    cat.to_csv(DATA / "fund_catalog.csv", index=False)

    fund_returns = pd.DataFrame({fid: r["daily"] for fid, r in runs.items()})
    fund_returns.index.name = "date"
    fund_returns.to_csv(DATA / "fund_returns.csv")

    fw = [r["weights"].reset_index().melt("date", var_name="asset",
                                          value_name="weight").assign(fund_id=fid)
          for fid, r in runs.items()]
    fund_weights = pd.concat(fw, ignore_index=True)
    fund_weights = fund_weights[fund_weights["weight"] > 1e-6]
    fund_weights = fund_weights[["fund_id", "date", "asset", "weight"]]
    fund_weights.to_csv(DATA / "fund_weights.csv", index=False)

    idx_vader.to_csv(DATA / "sector_sentiment_index.csv")
    idx_fin.to_csv(DATA / "sector_sentiment_index_finlex.csv")
    coverage.to_csv(DATA / "sentiment_coverage.csv")
    stats.to_csv(TABLES / "sentiment_model_stats.csv", index=False)

    rows = []
    for fid, r in runs.items():
        m = portfolios.performance_metrics(r["daily"], r["ann"])
        net = portfolios.performance_metrics(
            portfolios.net_of_costs(r["daily"], r["turnover"], COST_BPS), r["ann"])
        info = cat.set_index("fund_id").loc[fid]
        rows.append({"fund_id": fid, "label": info["label"],
                     "family": info["family"], "method": info["method"],
                     "ann_factor": r["ann"],
                     "ann_return": m["ann_return"], "ann_vol": m["ann_vol"],
                     "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                     "growth_of_1": m["growth_of_1"],
                     "avg_turnover": float(r["turnover"].mean()),
                     "net_sharpe_10bps": net["sharpe"],
                     "oos_start": info["oos_start"], "oos_end": info["oos_end"]})
    perf = pd.DataFrame(rows)
    perf.to_csv(TABLES / "performance_metrics.csv", index=False)

    fusion_ids = [base_id, "equity_max_sharpe_tilt_vader",
                  "equity_max_sharpe_tilt_finvader"]
    fus = perf[perf["fund_id"].isin(fusion_ids)].copy()
    fus.to_csv(TABLES / "fusion_before_after.csv", index=False)

    # Reproducible robustness extension: same 21-day date blocks are resampled
    # for every selected equity-calendar fund in a draw, preserving paired
    # dependence. This is intentionally separate from the core OOS backtest.
    boot_funds = [
        "combined_risk_parity", "combined_max_sharpe",
        "combined_equal_weight", "equity_equal_weight",
        "equity_max_sharpe", "equity_max_sharpe_tilt_finvader",
    ]
    boot, boot_paired = evaluation.moving_block_bootstrap_sharpe(
        fund_returns, boot_funds, periods_per_year=252,
        block_length=21, n_boot=2000, seed=42)
    boot.to_csv(TABLES / "sharpe_bootstrap.csv")
    boot_paired.to_csv(TABLES / "sharpe_bootstrap_paired.csv")

    # --------------------------------------------------------- figures
    print("[6/6] figures ...")
    period = (f"OOS {runs['combined_max_sharpe']['daily'].index[0].date()} to "
              f"{runs['combined_max_sharpe']['daily'].index[-1].date()}")

    # 1. growth of $1 - combined family across methods (required)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for m in METHODS:
        d = runs[f"combined_{m}"]["daily"]
        ax.plot((1 + d).cumprod(), label=portfolios.METHOD_LABEL[m])
    ax.set_title("Growth of $1 - Combined equity+crypto fund, five optimisation methods")
    ax.set_xlabel("Date"); ax.set_ylabel("Value of $1 invested (USD)")
    ax.legend(ncols=2)
    save_fig(fig, FIGS / "growth_of_1_combined.png", period)

    # 2. growth of $1 by family (max-Sharpe funds)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for fam, col in [("equity", PALETTE["navy"]), ("crypto", PALETTE["gold"]),
                     ("combined", PALETTE["teal"])]:
        d = runs[f"{fam}_max_sharpe"]["daily"]
        ax.plot((1 + d).cumprod(), label=fund_label(fam, "max_sharpe"), color=col)
    ax.set_title("Growth of $1 - Maximum-Sharpe fund by asset family")
    ax.set_xlabel("Date"); ax.set_ylabel("Value of $1 invested (USD)")
    ax.legend()
    save_fig(fig, FIGS / "growth_of_1_families.png", period)

    # 3. drawdown (required, >= 1 fund)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    dd = portfolios.drawdown_series(runs["combined_max_sharpe"]["daily"])
    ax.fill_between(dd.index, dd * 100, 0, color=PALETTE["coral"], alpha=0.45)
    ax.plot(dd * 100, color=PALETTE["wine"], lw=1.1)
    ax.set_title("Drawdown - Combined Maximum-Sharpe fund")
    ax.set_xlabel("Date"); ax.set_ylabel("Drawdown (%)")
    save_fig(fig, FIGS / "drawdown_combined_max_sharpe.png", period)

    # 4. weights over time (required, >= 1 fund) - top 12 assets + Other
    fig, ax = plt.subplots(figsize=(9, 4.8))
    W = runs["combined_max_sharpe"]["weights"]
    top = W.mean().nlargest(12).index
    plot_w = W[top].copy()
    plot_w["Other"] = 1 - plot_w.sum(axis=1)
    ax.stackplot(plot_w.index, plot_w.T.values, labels=plot_w.columns,
                 colors=(CYCLE + [PALETTE["grid"]] * 4)[:len(plot_w.columns)],
                 alpha=0.92)
    ax.set_title("Portfolio weights over time - Combined Maximum-Sharpe fund "
                 "(monthly rebalances)")
    ax.set_xlabel("Rebalance date"); ax.set_ylabel("Weight (share of fund)")
    ax.set_ylim(0, 1)
    ax.legend(ncols=4, fontsize=7.2, loc="upper center",
              bbox_to_anchor=(0.5, -0.14))
    save_fig(fig, FIGS / "weights_over_time_combined_max_sharpe.png", period)

    # 5. Sharpe barplot across funds and methods (required)
    # Horizontal grouped bars keep the long method labels fully readable in the report.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    base15 = perf[perf["method"].isin(METHODS)]
    piv = base15.pivot(index="method", columns="family", values="sharpe").loc[METHODS]
    y = np.arange(len(piv))
    for i, fam in enumerate(["equity", "crypto", "combined"]):
        ax.barh(y + (i - 1) * 0.22, piv[fam], height=0.20,
                label=FAMILY_LABEL[fam],
                color=[PALETTE["navy"], PALETTE["gold"], PALETTE["teal"]][i])
    ax.set_yticks(y, [portfolios.METHOD_LABEL[m] for m in METHODS])
    ax.invert_yaxis()
    ax.axvline(0, color=PALETTE["ink"], lw=0.8)
    ax.set_title("Out-of-sample Sharpe ratio across funds and methods (rf = 0)")
    ax.set_xlabel("Sharpe ratio (annualised)")
    ax.legend(ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.20))
    fig.subplots_adjust(left=0.24, bottom=0.22)
    save_fig(fig, FIGS / "sharpe_barplot.png", period)

    # 6. sector sentiment index (required)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    smooth = idx_vader.drop(columns="ALL").rolling(21, min_periods=5).mean()
    for c in smooth.columns:
        ax.plot(smooth[c], label=c, lw=1.2)
    ax.axhline(0, color=PALETTE["ink"], lw=0.8)
    ax.set_title("News-sentiment index by equity sector "
                 "(plain VADER, 21-day mean of daily index)")
    ax.set_xlabel("Date"); ax.set_ylabel("Mean compound score (-1 to +1)")
    ax.legend(ncols=5, fontsize=7.2, loc="upper center",
              bbox_to_anchor=(0.5, -0.14))
    save_fig(fig, FIGS / "sentiment_index.png", "2020-2023, headlines only")

    # 7. fusion before vs after (required)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for fid, col, lab in [
            (base_id, PALETTE["navy"], "Base: Equity Max Sharpe"),
            ("equity_max_sharpe_tilt_vader", PALETTE["teal"], "+ Sentiment tilt (VADER)"),
            ("equity_max_sharpe_tilt_finvader", PALETTE["gold"], "+ Sentiment tilt (Fin-VADER)")]:
        ax.plot((1 + runs[fid]["daily"]).cumprod(), color=col, label=lab)
    ax.set_title("Fusion: growth of $1 before vs after the sentiment tilt")
    ax.set_xlabel("Date"); ax.set_ylabel("Value of $1 invested (USD)")
    ax.legend()
    save_fig(fig, FIGS / "fusion_before_after.png", period)

    # 8. lexicon extension evidence (innovation exhibit)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(["Plain VADER", "Fin-VADER"], stats["neutral_share"] * 100,
                color=[PALETTE["steel"], PALETTE["teal"]], width=0.55)
    axes[0].set_title("Headlines scored neutral")
    axes[0].set_ylabel("% of headlines with |compound| < 0.05")
    for i, v in enumerate(stats["neutral_share"] * 100):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    axes[1].hist(scored["compound_vader"], bins=41, alpha=0.65,
                 label="Plain VADER", color=PALETTE["steel"])
    axes[1].hist(scored["compound_finvader"], bins=41, alpha=0.55,
                 label="Fin-VADER", color=PALETTE["teal"])
    axes[1].set_title("Distribution of compound scores")
    axes[1].set_xlabel("Compound score"); axes[1].set_ylabel("Headlines")
    axes[1].legend()
    fig.suptitle("Extending VADER with a finance lexicon unlocks direction "
                 "plain VADER misses", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save_fig(fig, FIGS / "lexicon_effect.png", "all deduplicated headlines, 2020-2023")

    # 9. turnover & transaction costs (innovation exhibit)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    d = runs["combined_max_sharpe"]["daily"]
    dn = portfolios.net_of_costs(d, runs["combined_max_sharpe"]["turnover"], COST_BPS)
    ax.plot((1 + d).cumprod(), color=PALETTE["navy"], label="Gross")
    ax.plot((1 + dn).cumprod(), color=PALETTE["coral"],
            label=f"Net of {COST_BPS:.0f} bps x turnover")
    ax.set_title("Transaction-cost model - Combined Maximum-Sharpe fund")
    ax.set_xlabel("Date"); ax.set_ylabel("Value of $1 invested (USD)")
    ax.legend()
    save_fig(fig, FIGS / "turnover_costs.png", period)

    # ------------------------------------------- key numbers for the report
    lines = ["KEY NUMBERS FOR THE REPORT (auto-generated - rerun after any change)",
             "=" * 72, "",
             perf.round(3).to_string(index=False), "",
             "Sentiment model comparison:", stats.round(3).to_string(index=False), "",
             f"Deduplicated headlines scored: {len(scored):,}",
             f"OOS window: {period}",
             f"Avg monthly turnover, combined max-Sharpe: "
             f"{runs['combined_max_sharpe']['turnover'].mean():.1%}"]
    (TABLES / "report_key_numbers.txt").write_text("\n".join(lines))
    print("\nDone. Artifacts in results/. Key numbers:",
          TABLES / "report_key_numbers.txt")


if __name__ == "__main__":
    main()

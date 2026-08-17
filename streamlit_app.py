"""FundX - systematic multi-asset funds with news-sentiment analytics.

FINS3645 FinTech Project 2026, Part B (z5544937). Entry point for Streamlit
Community Cloud.

The deployed app is a READER: every fund, metric and sentiment series is
precomputed by `python scripts/run_part_b.py` into results/ (committed), so
the app never runs an optimiser or VADER (the free tier cannot). The one
network touch is an optional expander on the Methodology page that loads the
hosted price data through src/data_access.py, demonstrating the provided
helper end to end.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------ brand
FX = {"navy": "#12283F", "teal": "#1F8A8C", "gold": "#E3A82B",
      "coral": "#E2593B", "sky": "#5FA8D3", "moss": "#6B9080",
      "plum": "#8D5A97", "steel": "#64748B", "sand": "#C8B08A",
      "wine": "#9E2B25", "paper": "#FBFAF7", "ink": "#1E293B"}
CYCLE = [FX[k] for k in ("navy", "teal", "gold", "coral", "sky",
                         "moss", "plum", "steel", "sand", "wine")]
MGMT_FEE = 0.0075  # illustrative 0.75% p.a. management fee FundX charges

st.set_page_config(page_title="FundX - Systematic Funds", page_icon="chart_with_upwards_trend",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  .stApp {{ background: {FX['paper']}; }}
  h1, h2, h3 {{ color: {FX['navy']}; font-weight: 700; }}
  section[data-testid="stSidebar"] {{ background: {FX['navy']}; }}
  section[data-testid="stSidebar"] * {{ color: #EDF2F7 !important; }}
  .fx-hero {{ background: linear-gradient(120deg, {FX['navy']} 0%, {FX['teal']} 100%);
             color: #fff; padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 0.8rem; }}
  .fx-hero h1 {{ color: #fff; margin: 0; font-size: 1.55rem; }}
  .fx-hero p {{ margin: 0.25rem 0 0 0; color: #E6EFEA; font-size: 0.92rem; }}
  .fx-card {{ background: #fff; border: 1px solid #E7E2D9; border-radius: 12px;
             padding: 0.9rem 1rem; box-shadow: 0 1px 4px rgba(18,40,63,0.06); }}
  .fx-kpi {{ font-size: 1.45rem; font-weight: 700; color: {FX['navy']}; }}
  .fx-kpi-label {{ font-size: 0.78rem; color: {FX['steel']}; text-transform: uppercase;
                  letter-spacing: 0.04em; }}
  .fx-pill {{ display: inline-block; background: {FX['teal']}22; color: {FX['teal']};
             border-radius: 999px; padding: 0.1rem 0.6rem; font-size: 0.75rem;
             font-weight: 600; margin-right: 0.3rem; }}
  .fx-note {{ color: {FX['steel']}; font-size: 0.8rem; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def load_artifacts():
    d = ROOT / "results" / "data"
    t = ROOT / "results" / "tables"
    art = {
        "returns": pd.read_csv(d / "fund_returns.csv", parse_dates=["date"]).set_index("date"),
        "weights": pd.read_csv(d / "fund_weights.csv", parse_dates=["date"]),
        "senti": pd.read_csv(d / "sector_sentiment_index.csv", parse_dates=["date"]).set_index("date"),
        "catalog": pd.read_csv(d / "fund_catalog.csv").set_index("fund_id"),
        "perf": pd.read_csv(t / "performance_metrics.csv").set_index("fund_id"),
    }
    p = d / "sector_sentiment_index_finlex.csv"
    art["senti_fin"] = (pd.read_csv(p, parse_dates=["date"]).set_index("date")
                        if p.exists() else None)
    p = d / "sentiment_coverage.csv"
    art["coverage"] = (pd.read_csv(p, parse_dates=["date"]).set_index("date")
                       if p.exists() else None)
    return art


def growth(series: pd.Series) -> pd.Series:
    return (1 + series.dropna()).cumprod()


def drawdown(series: pd.Series) -> pd.Series:
    g = growth(series)
    return g / g.cummax() - 1


def line_fig(title: str, ytitle: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(color=FX["navy"], size=16)),
        paper_bgcolor=FX["paper"], plot_bgcolor="#FFFFFF",
        font=dict(color=FX["ink"], family="Helvetica, Arial, sans-serif"),
        yaxis_title=ytitle, hovermode="x unified", height=420,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=40, r=20, t=55, b=40))
    fig.update_xaxes(gridcolor="#EDEAE2")
    fig.update_yaxes(gridcolor="#EDEAE2")
    return fig


def kpi_row(items: list[tuple[str, str]]):
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.markdown(f"<div class='fx-card'><div class='fx-kpi-label'>{label}</div>"
                   f"<div class='fx-kpi'>{value}</div></div>", unsafe_allow_html=True)


try:
    A = load_artifacts()
except FileNotFoundError as e:
    st.error("App artifacts are missing under results/. Run "
             "`python scripts/run_part_b.py` first, then restart the app. "
             f"Missing: {e}")
    st.stop()

PERF, CAT, RET = A["perf"], A["catalog"], A["returns"]
label_of = CAT["label"].to_dict()
id_of = {v: k for k, v in label_of.items()}

# ------------------------------------------------------------------ nav
st.sidebar.markdown("## FundX")
st.sidebar.caption("Systematic funds, transparent rules.")
page = st.sidebar.radio("Navigate", [
    "Compare funds", "Fund fact sheet", "Build your allocation",
    "Sentiment analytics", "Methodology & data"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown(f"<span class='fx-note'>Out-of-sample: "
                    f"{PERF['oos_start'].min()} to {PERF['oos_end'].max()}<br>"
                    f"Rebalanced monthly - weights from past data only.<br>"
                    f"Prototype for FINS3645; not investment advice.</span>",
                    unsafe_allow_html=True)

st.markdown("<div class='fx-hero'><h1>FundX</h1>"
            "<p>Rules-based multi-asset funds with a news-sentiment lens - "
            "compare, read the fact sheet, allocate.</p></div>",
            unsafe_allow_html=True)

# =================================================================== page 1
if page == "Compare funds":
    st.subheader("Compare the FundX range")
    fam = st.multiselect("Asset family",
                         ["combined", "equity", "crypto"],
                         default=["combined", "equity", "crypto"],
                         format_func=str.title)
    view = PERF[PERF["family"].isin(fam)].copy()

    best = view.sort_values("sharpe", ascending=False).iloc[0]
    kpi_row([("Funds on the shelf", f"{len(view)}"),
             ("Best OOS Sharpe", f"{best['sharpe']:.2f}"),
             ("Its fund", best["label"]),
             ("Its growth of $1", f"${best['growth_of_1']:.2f}")])

    st.markdown("#### Fact-sheet metrics (out-of-sample)")
    tbl = view[["label", "family", "ann_return", "ann_vol", "sharpe",
                "max_drawdown", "growth_of_1", "avg_turnover", "ann_factor"]].copy()
    tbl.columns = ["Fund", "Family", "Ann. return", "Ann. vol", "Sharpe",
                   "Max drawdown", "Growth of $1", "Avg monthly turnover",
                   "Ann. factor"]
    st.dataframe(
        tbl.sort_values("Sharpe", ascending=False).style.format({
            "Ann. return": "{:.1%}", "Ann. vol": "{:.1%}", "Sharpe": "{:.2f}",
            "Max drawdown": "{:.1%}", "Growth of $1": "${:.2f}",
            "Avg monthly turnover": "{:.1%}"}),
        width="stretch", height=430)

    picks = st.multiselect(
        "Plot growth of $1", [label_of[i] for i in view.index],
        default=[label_of[i] for i in
                 view.sort_values("sharpe", ascending=False).index[:4]])
    fig = line_fig("Growth of $1 invested (out-of-sample)", "Value of $1 (USD)")
    for i, lab in enumerate(picks):
        g = growth(RET[id_of[lab]])
        fig.add_trace(go.Scatter(x=g.index, y=g, name=lab,
                                 line=dict(color=CYCLE[i % 10], width=2)))
    st.plotly_chart(fig, width="stretch")

    bar = view.sort_values("sharpe")
    bfig = px.bar(bar, x="sharpe", y="label", orientation="h",
                  color="family",
                  color_discrete_map={"equity": FX["navy"],
                                      "crypto": FX["gold"],
                                      "combined": FX["teal"]},
                  labels={"sharpe": "Sharpe ratio (rf = 0)", "label": ""})
    bfig.update_layout(title="Out-of-sample Sharpe by fund",
                       paper_bgcolor=FX["paper"], plot_bgcolor="#FFFFFF",
                       height=480, font=dict(color=FX["ink"]))
    st.plotly_chart(bfig, width="stretch")

# =================================================================== page 2
elif page == "Fund fact sheet":
    lab = st.selectbox("Choose a fund", [label_of[i] for i in PERF.index])
    fid = id_of[lab]
    p = PERF.loc[fid]
    r = RET[fid].dropna()

    st.subheader(lab)
    st.markdown(f"<span class='fx-pill'>{p['family'].title()}</span>"
                f"<span class='fx-pill'>Monthly rebalance</span>"
                f"<span class='fx-pill'>Annualised x{int(p['ann_factor'])}</span>"
                f"<span class='fx-pill'>OOS {p['oos_start']} to {p['oos_end']}</span>",
                unsafe_allow_html=True)
    st.write("")
    kpi_row([("Growth of $1", f"${p['growth_of_1']:.2f}"),
             ("Ann. return", f"{p['ann_return']:.1%}"),
             ("Ann. volatility", f"{p['ann_vol']:.1%}"),
             ("Sharpe (rf=0)", f"{p['sharpe']:.2f}"),
             ("Max drawdown", f"{p['max_drawdown']:.1%}")])

    c1, c2 = st.columns(2)
    g = growth(r)
    fig = line_fig("Growth of $1 (cumulative return)", "Value of $1 (USD)")
    fig.add_trace(go.Scatter(x=g.index, y=g, line=dict(color=FX["teal"], width=2.3),
                             name=lab))
    c1.plotly_chart(fig, width="stretch")

    dd = drawdown(r) * 100
    dfig = line_fig("Drawdown from peak", "Drawdown (%)")
    dfig.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy",
                              line=dict(color=FX["coral"]), name="Drawdown"))
    c2.plotly_chart(dfig, width="stretch")

    W = A["weights"]
    w_fund = W[W["fund_id"] == fid]
    latest_date = w_fund["date"].max()
    latest = (w_fund[w_fund["date"] == latest_date]
              .sort_values("weight", ascending=False))

    c3, c4 = st.columns(2)
    hfig = px.bar(latest.head(15), x="weight", y="asset", orientation="h",
                  labels={"weight": "Target weight", "asset": ""},
                  color_discrete_sequence=[FX["navy"]])
    hfig.update_layout(title=f"Current holdings - top 15 (rebalance "
                             f"{latest_date.date()})",
                       yaxis=dict(autorange="reversed"),
                       paper_bgcolor=FX["paper"], plot_bgcolor="#FFFFFF",
                       height=460, font=dict(color=FX["ink"]))
    hfig.update_xaxes(tickformat=".0%")
    c3.plotly_chart(hfig, width="stretch")

    top_assets = w_fund.groupby("asset")["weight"].mean().nlargest(10).index
    area = (w_fund[w_fund["asset"].isin(top_assets)]
            .pivot_table(index="date", columns="asset", values="weight",
                         aggfunc="sum").fillna(0))
    area["Other"] = (1 - area.sum(axis=1)).clip(lower=0)
    afig = go.Figure()
    for i, ccol in enumerate(area.columns):
        afig.add_trace(go.Scatter(x=area.index, y=area[ccol], stackgroup="w",
                                  name=ccol,
                                  line=dict(width=0.4,
                                            color=(CYCLE + ["#D8D3C8"])[i % 11])))
    afig.update_layout(title="Weights over time (top 10 assets + other)",
                       paper_bgcolor=FX["paper"], plot_bgcolor="#FFFFFF",
                       height=460, font=dict(color=FX["ink"]),
                       yaxis=dict(tickformat=".0%", range=[0, 1]),
                       legend=dict(orientation="h", y=-0.2))
    c4.plotly_chart(afig, width="stretch")

    st.markdown(f"<span class='fx-note'>Fact-sheet metrics are out-of-sample: "
                f"weights are re-estimated each month from the prior "
                f"{int(CAT.loc[fid, 'lookback'])} days only, then held. "
                f"Gross of the {MGMT_FEE:.2%} p.a. FundX management fee and "
                f"trading costs (see Methodology).</span>", unsafe_allow_html=True)

# =================================================================== page 3
elif page == "Build your allocation":
    st.subheader("Split your money across FundX funds")
    st.caption("Pick funds, set sliders, and see the blended out-of-sample "
               "result. Weights are normalised to 100% automatically.")
    default = [label_of.get("combined_max_sharpe"),
               label_of.get("combined_risk_parity"),
               label_of.get("equity_min_variance")]
    chosen = st.multiselect("Funds in your mix",
                            [label_of[i] for i in PERF.index],
                            default=[d for d in default if d])
    if not chosen:
        st.info("Choose at least one fund to build a mix.")
        st.stop()

    amount = st.number_input("Amount to invest (USD)", 100, 10_000_000, 10_000,
                             step=500)
    cols = st.columns(len(chosen))
    raw = [c.slider(lab, 0, 100, 100 // len(chosen), key=f"w_{lab}")
           for c, lab in zip(cols, chosen)]
    if sum(raw) == 0:
        st.warning("Give at least one fund a positive weight.")
        st.stop()
    w = np.array(raw, dtype=float) / sum(raw)

    ids = [id_of[lab] for lab in chosen]
    mix_ret = (RET[ids].dropna(how="all").fillna(0) @ w)
    ppy = 252 if any(CAT.loc[i, "family"] != "crypto" for i in ids) else 365
    n = len(mix_ret)
    g = float((1 + mix_ret).prod())
    ann_ret = g ** (ppy / n) - 1
    ann_vol = float(mix_ret.std(ddof=1)) * np.sqrt(ppy)
    sharpe = ann_ret / ann_vol if ann_vol else float("nan")
    mdd = float(drawdown(mix_ret).min())
    fee_dollars = amount * MGMT_FEE

    kpi_row([("Your mix, growth of $1", f"${g:.2f}"),
             ("Ann. return", f"{ann_ret:.1%}"),
             ("Ann. vol", f"{ann_vol:.1%}"),
             ("Sharpe", f"{sharpe:.2f}"),
             ("Max drawdown", f"{mdd:.1%}")])
    st.markdown(f"<span class='fx-note'>Allocation: " +
                ", ".join(f"{lab} {wi:.0%}" for lab, wi in zip(chosen, w)) +
                f". FundX earns an illustrative {MGMT_FEE:.2%} p.a. management "
                f"fee = ${fee_dollars:,.0f}/yr on ${amount:,.0f} "
                f"(returns above are gross of fees).</span>",
                unsafe_allow_html=True)

    fig = line_fig("Your blended portfolio vs its building blocks",
                   "Value of $1 (USD)")
    gmix = growth(mix_ret)
    fig.add_trace(go.Scatter(x=gmix.index, y=gmix, name="Your mix",
                             line=dict(color=FX["gold"], width=3)))
    for i, (lab, fid) in enumerate(zip(chosen, ids)):
        gf = growth(RET[fid])
        fig.add_trace(go.Scatter(x=gf.index, y=gf, name=lab,
                                 line=dict(color=CYCLE[i % 10], width=1.3,
                                           dash="dot")))
    st.plotly_chart(fig, width="stretch")

    proj = amount * g
    st.markdown(f"<div class='fx-card'>Had you invested "
                f"<b>${amount:,.0f}</b> at the start of the out-of-sample "
                f"window, this mix would have grown to <b>${proj:,.0f}</b> "
                f"(gross). Past out-of-sample performance still does not "
                f"guarantee future results.</div>", unsafe_allow_html=True)

# =================================================================== page 4
elif page == "Sentiment analytics":
    st.subheader("News-sentiment index across equity sectors")
    st.caption("Built from ~147k deduplicated headlines for the 50 stocks, "
               "scored per headline, averaged per ticker-day, then "
               "equal-weighted within each sector. Headlines are a noisy "
               "proxy - read levels as tone, not certainty.")

    senti = A["senti"]
    use_fin = False
    if A["senti_fin"] is not None:
        use_fin = st.toggle("Use Fin-VADER (finance-lexicon extension)", value=True,
                            help="Plain VADER leaves ~half of finance headlines "
                                 "neutral; the extended lexicon adds market terms "
                                 "like 'downgrade' or 'all-time high'.")
    S = A["senti_fin"] if use_fin else senti
    sectors = [c for c in S.columns if c != "ALL"]
    win = st.slider("Smoothing window (trading days)", 1, 63, 21)
    pick = st.multiselect("Sectors", sectors, default=sectors[:5])

    fig = line_fig("Sector sentiment over time "
                   f"({'Fin-VADER' if use_fin else 'plain VADER'}, "
                   f"{win}-day mean)", "Mean compound score (-1 to +1)")
    for i, ccol in enumerate(pick):
        sm = S[ccol].rolling(win, min_periods=max(2, win // 4)).mean()
        fig.add_trace(go.Scatter(x=sm.index, y=sm, name=ccol,
                                 line=dict(color=CYCLE[i % 10], width=1.7)))
    if "ALL" in S.columns:
        sm = S["ALL"].rolling(win, min_periods=max(2, win // 4)).mean()
        fig.add_trace(go.Scatter(x=sm.index, y=sm, name="All 50 stocks",
                                 line=dict(color=FX["ink"], width=2.6, dash="dash")))
    fig.add_hline(y=0, line_color=FX["steel"], line_width=1)
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    latest = S[sectors].rolling(win, min_periods=2).mean().iloc[-1].sort_values()
    lfig = px.bar(latest, orientation="h",
                  labels={"value": "Sentiment", "index": ""},
                  color=latest.values, color_continuous_scale=[
                      FX["coral"], "#EDEAE2", FX["teal"]])
    lfig.update_layout(title=f"Latest sector tone ({win}-day mean)",
                       coloraxis_showscale=False, height=420,
                       paper_bgcolor=FX["paper"], plot_bgcolor="#FFFFFF",
                       font=dict(color=FX["ink"]))
    c1.plotly_chart(lfig, width="stretch")

    if A["coverage"] is not None:
        cov = A["coverage"]["headlines"].rolling(21, min_periods=5).mean()
        cfig = line_fig("News flow (21-day mean headlines/day)", "Headlines per day")
        cfig.add_trace(go.Scatter(x=cov.index, y=cov, fill="tozeroy",
                                  line=dict(color=FX["sky"]), name="Headlines"))
        c2.plotly_chart(cfig, width="stretch")

    st.markdown("<span class='fx-note'>In the funds, this signal is lagged at "
                "least one trading day before it is used, so a decision on day "
                "t only ever sees news from day t-1 or earlier.</span>",
                unsafe_allow_html=True)

# =================================================================== page 5
else:
    st.subheader("Methodology & data")
    st.markdown(f"""
<div class='fx-card'>

**The product.** FundX offers systematically managed funds. Each fund is one
(asset family, optimisation rule) pair - for example *Combined Minimum
Variance* - built from daily adjusted-close returns of 50 US large-caps
(10 sectors) and 10 major cryptocurrencies, 2020-2023.

**The rules.** Five methods: Equal Weight, Minimum Variance, Maximum Sharpe,
Risk Parity (equal risk contribution), and Hierarchical Risk Parity. Long-only,
fully invested, per-asset cap 20% (40% for crypto-only). Risk-free rate 0.

**Out-of-sample discipline.** Walk-forward backtest: on the first trading day
of each month, weights are re-estimated from the previous 252 trading days
(365 calendar days for crypto-only funds) - strictly before that day - then
held, drifting with returns, until the next rebalance. Equity-calendar funds
annualise with 252, crypto-only funds with 365. What you see is what the rule
would have earned on data it had not seen.

**Sentiment.** ~147k deduplicated headlines are scored per headline (VADER, and
a finance-lexicon extension), averaged per ticker-day, and equal-weighted into
sector indices. The sentiment-tilt funds multiply each stock's weight by
1 + 0.25 x its sector's lagged, cross-sectionally z-scored 21-day tone
(clipped to [0.6, 1.4]) and renormalise.

**Costs & fees.** Fact sheets show gross returns; the report also nets a
10 bps x turnover transaction-cost model. FundX's business model is an
illustrative {MGMT_FEE:.2%} p.a. management fee on invested balances.

**Honesty box.** Prices and headlines end 2023-12-31; nothing here is
investment advice; out-of-sample is not a guarantee - it is just a fair test.
</div>
""", unsafe_allow_html=True)

    st.markdown("#### Underlying data (loads live through the provided helper)")
    with st.expander("Preview the hosted equity data via src/data_access.py"):
        try:
            from src import data_access
            eq = data_access.load_equity_prices()
            st.write(f"equity_prices: {eq.shape[0]:,} rows, "
                     f"{eq['ticker'].nunique()} tickers, "
                     f"{eq['date'].min().date()} to {eq['date'].max().date()}")
            st.dataframe(eq.head(15), width="stretch")
        except Exception as e:
            st.warning(f"Hosted data not reachable right now ({e}). The funds "
                       "and analytics above are unaffected - they read "
                       "precomputed artifacts.")

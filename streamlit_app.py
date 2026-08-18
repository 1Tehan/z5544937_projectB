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
import plotly.graph_objects as go
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------ brand
FX = {"navy": "#0B1C2C", "panel": "#12283F", "panel2": "#16324D",
      "teal": "#2BC4C4", "gold": "#E3A82B", "coral": "#E2593B",
      "sky": "#5FA8D3", "moss": "#6B9080", "plum": "#B07FC7",
      "steel": "#7C8DA3", "sand": "#C8B08A", "wine": "#D2564B",
      "ink": "#EAF1F8", "mute": "#93A6BC", "line": "#20374F"}
CYCLE = [FX[k] for k in ("teal", "gold", "sky", "coral", "moss",
                         "plum", "sand", "wine", "steel", "navy")]
MGMT_FEE = 0.0075  # illustrative 0.75% p.a. management fee FundX charges

st.set_page_config(page_title="FundX", page_icon="chart_with_upwards_trend",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  :root {{ --navy:{FX['navy']}; --panel:{FX['panel']}; --teal:{FX['teal']};
           --gold:{FX['gold']}; --ink:{FX['ink']}; --mute:{FX['mute']};
           --line:{FX['line']}; }}
  .stApp {{ background:
      radial-gradient(1200px 600px at 15% -10%, #14304A 0%, {FX['navy']} 55%) fixed; }}
  #MainMenu, footer {{ visibility: hidden; }}
  .block-container {{ padding-top: 1.4rem; max-width: 1250px; }}
  h1,h2,h3,h4 {{ color: {FX['ink']}; font-weight: 700;
      letter-spacing: -0.01em; font-family: 'Helvetica Neue', Arial, sans-serif; }}
  p, span, label, .stMarkdown {{ color: {FX['ink']}; }}
  section[data-testid="stSidebar"] {{ background: {FX['navy']};
      border-right: 1px solid {FX['line']}; }}
  section[data-testid="stSidebar"] * {{ color: {FX['ink']} !important; }}

  .fx-hero {{ background:
      linear-gradient(120deg, {FX['panel']} 0%, #0E4C4C 100%);
      border: 1px solid {FX['line']}; border-radius: 18px;
      padding: 1.5rem 1.7rem; margin-bottom: 1.1rem;
      box-shadow: 0 10px 40px rgba(0,0,0,0.35); }}
  .fx-hero h1 {{ margin: 0; font-size: 2.0rem;
      background: linear-gradient(90deg,#fff,{FX['teal']});
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .fx-hero p {{ margin: 0.35rem 0 0 0; color: {FX['ink']}; opacity: 0.85;
      font-size: 1.0rem; }}
  .fx-eyebrow {{ color: {FX['teal']}; font-size: 0.72rem; font-weight: 700;
      letter-spacing: 0.18em; text-transform: uppercase; }}

  .fx-card {{ background: {FX['panel']}; border: 1px solid {FX['line']};
      border-radius: 14px; padding: 1.0rem 1.15rem;
      box-shadow: 0 4px 18px rgba(0,0,0,0.25); color: {FX['ink']}; }}
  .fx-card p, .fx-card li, .fx-card strong, .fx-card em {{ color: {FX['ink']}; }}
  .fx-kpi {{ font-size: 1.9rem; font-weight: 800; color: #fff;
      font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }}
  .fx-kpi.sm {{ font-size: 1.45rem; }}
  .fx-kpi-label {{ font-size: 0.72rem; color: {FX['mute']};
      text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }}
  .fx-delta-up {{ color: {FX['teal']}; font-weight: 700; }}
  .fx-delta-dn {{ color: {FX['coral']}; font-weight: 700; }}
  .fx-pill {{ display: inline-block; background: {FX['teal']}1f;
      color: {FX['teal']}; border: 1px solid {FX['teal']}55;
      border-radius: 999px; padding: 0.12rem 0.7rem; font-size: 0.74rem;
      font-weight: 600; margin-right: 0.35rem; margin-bottom: 0.3rem; }}
  .fx-note {{ color: {FX['mute']}; font-size: 0.82rem; line-height: 1.5; }}
  .fx-big {{ font-size: 2.6rem; font-weight: 800; color: #fff;
      font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }}
  .fx-divider {{ height: 1px; background: {FX['line']}; border: 0;
      margin: 1.4rem 0 1.1rem 0; }}

  .stButton>button, .stDownloadButton>button {{
      background: linear-gradient(90deg,{FX['teal']},#1FA9A9);
      color: #05202A; border: 0; border-radius: 10px; font-weight: 700;
      padding: 0.5rem 1.1rem; }}
  .stButton>button:hover {{ filter: brightness(1.08); color:#05202A; }}
  div[data-testid="stMetric"] {{ background: {FX['panel']};
      border: 1px solid {FX['line']}; border-radius: 12px; padding: 0.7rem 0.9rem; }}
  div[data-baseweb="slider"] [role="slider"] {{ background: {FX['teal']}; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
  .stTabs [data-baseweb="tab"] {{ background: {FX['panel']};
      border-radius: 10px 10px 0 0; color: {FX['mute']}; }}
  .stTabs [aria-selected="true"] {{ background: {FX['panel2']};
      color: {FX['ink']}; }}
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


def metrics_from_returns(r: pd.Series, ppy: int) -> dict:
    r = r.dropna()
    n = len(r)
    g = float((1 + r).prod())
    ann_ret = g ** (ppy / n) - 1 if n else float("nan")
    ann_vol = float(r.std(ddof=1)) * np.sqrt(ppy) if n > 1 else float("nan")
    return {"growth": g, "ann_ret": ann_ret, "ann_vol": ann_vol,
            "sharpe": ann_ret / ann_vol if ann_vol else float("nan"),
            "mdd": float(drawdown(r).min()) if n else float("nan")}


def base_fig(title: str = "", ytitle: str = "", height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(color=FX["ink"], size=15)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=FX["ink"], family="Helvetica Neue, Arial, sans-serif"),
        yaxis_title=ytitle, hovermode="x unified", height=height,
        legend=dict(orientation="h", y=-0.2, font=dict(color=FX["mute"])),
        margin=dict(l=45, r=20, t=50, b=40))
    fig.update_xaxes(gridcolor=FX["line"], zeroline=False,
                     tickfont=dict(color=FX["mute"]))
    fig.update_yaxes(gridcolor=FX["line"], zeroline=False,
                     tickfont=dict(color=FX["mute"]))
    return fig


def kpi_card(col, label, value, sub=None, sub_class="fx-note", big=False):
    cls = "fx-kpi" if big else "fx-kpi sm"
    html = (f"<div class='fx-card'><div class='fx-kpi-label'>{label}</div>"
            f"<div class='{cls}'>{value}</div>")
    if sub is not None:
        html += f"<div class='{sub_class}'>{sub}</div>"
    html += "</div>"
    col.markdown(html, unsafe_allow_html=True)


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


def ppy_for(fid: str) -> int:
    return 365 if CAT.loc[fid, "family"] == "crypto" else 252


# ------------------------------------------------------------------ nav
st.sidebar.markdown("<div style='font-size:1.5rem;font-weight:800;"
                    f"background:linear-gradient(90deg,#fff,{FX['teal']});"
                    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>"
                    "FundX</div>", unsafe_allow_html=True)
st.sidebar.caption("Systematic funds, transparent rules.")
page = st.sidebar.radio("Navigate", [
    "Overview", "Compare funds", "Fund fact sheet", "Find my mix",
    "Build your allocation", "Sentiment analytics", "Methodology & data"],
    label_visibility="collapsed")
st.sidebar.markdown("<hr style='border-color:#20374F'>", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='fx-note'>Out-of-sample "
                    f"{PERF['oos_start'].min()} to {PERF['oos_end'].max()}<br>"
                    f"Monthly rebalance; weights from past data only.<br>"
                    f"Prototype for FINS3645; not investment advice.</span>",
                    unsafe_allow_html=True)

st.markdown("<div class='fx-hero'><div class='fx-eyebrow'>Systematic multi-asset funds</div>"
            "<h1>FundX</h1><p>Rules-based equity &amp; crypto funds with a "
            "news-sentiment lens - compare, read the fact sheet, and build a mix.</p></div>",
            unsafe_allow_html=True)

# =================================================================== OVERVIEW
if page == "Overview":
    combined = PERF[PERF["family"] == "combined"]
    best = PERF.sort_values("sharpe", ascending=False).iloc[0]
    safest = PERF.loc[PERF["max_drawdown"].idxmax()]
    grow = PERF.loc[PERF["growth_of_1"].idxmax()]

    c = st.columns(4)
    kpi_card(c[0], "Funds on the shelf", f"{len(PERF)}",
             f"{PERF['family'].nunique()} asset families", big=True)
    kpi_card(c[1], "Best risk-adjusted", best["label"],
             f"Sharpe {best['sharpe']:.2f}", big=False)
    kpi_card(c[2], "Steadiest ride", safest["label"],
             f"Max drawdown {safest['max_drawdown']:.1%}", big=False)
    kpi_card(c[3], "Biggest grower", grow["label"],
             f"${grow['growth_of_1']:.2f} per $1", big=False)

    st.markdown("<hr class='fx-divider'>", unsafe_allow_html=True)
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("<div class='fx-eyebrow'>Growth of $1, out-of-sample</div>",
                    unsafe_allow_html=True)
        show = ["combined_risk_parity", "combined_max_sharpe",
                "equity_equal_weight", "crypto_min_variance"]
        fig = base_fig("", "Value of $1 (USD)", height=430)
        for i, fid in enumerate(show):
            g = growth(RET[fid])
            fig.add_trace(go.Scatter(x=g.index, y=g, name=label_of[fid],
                                     line=dict(color=CYCLE[i], width=2.4)))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("<div class='fx-eyebrow'>How FundX works</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div class='fx-card'>"
            "<p><b>1 &nbsp;Pick a rule.</b> Each fund follows one transparent "
            "optimisation rule across one asset family.</p>"
            "<p><b>2 &nbsp;See the evidence.</b> Every metric is out-of-sample - "
            "weights come only from past data, rebalanced monthly.</p>"
            "<p><b>3 &nbsp;Build a mix.</b> Blend funds to your risk appetite; "
            f"FundX charges an illustrative {MGMT_FEE:.2%} p.a. fee.</p>"
            "</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='fx-card'><div class='fx-kpi-label'>Data window</div>"
                    f"<div class='fx-kpi sm'>2020 - 2023</div>"
                    f"<div class='fx-note'>50 US equities, 10 cryptocurrencies, "
                    f"~147k news headlines.</div></div>", unsafe_allow_html=True)

# =================================================================== COMPARE
elif page == "Compare funds":
    st.subheader("Compare the FundX range")
    fam = st.multiselect("Asset family", ["combined", "equity", "crypto"],
                         default=["combined", "equity", "crypto"],
                         format_func=str.title)
    view = PERF[PERF["family"].isin(fam)].copy()
    if view.empty:
        st.info("Pick at least one asset family.")
        st.stop()

    best = view.sort_values("sharpe", ascending=False).iloc[0]
    c = st.columns(4)
    kpi_card(c[0], "Funds shown", f"{len(view)}", big=True)
    kpi_card(c[1], "Best OOS Sharpe", f"{best['sharpe']:.2f}", best["label"])
    kpi_card(c[2], "Its growth of $1", f"${best['growth_of_1']:.2f}")
    kpi_card(c[3], "Its max drawdown", f"{best['max_drawdown']:.1%}")

    st.markdown("<hr class='fx-divider'>", unsafe_allow_html=True)
    st.markdown("#### Fact-sheet metrics (out-of-sample)")
    tbl = view[["label", "family", "ann_return", "ann_vol", "sharpe",
                "max_drawdown", "growth_of_1", "avg_turnover"]].copy()
    tbl.columns = ["Fund", "Family", "Ann. return", "Ann. vol", "Sharpe",
                   "Max drawdown", "Growth of $1", "Avg turnover"]
    st.dataframe(
        tbl.sort_values("Sharpe", ascending=False).style.format({
            "Ann. return": "{:.1%}", "Ann. vol": "{:.1%}", "Sharpe": "{:.2f}",
            "Max drawdown": "{:.1%}", "Growth of $1": "${:.2f}",
            "Avg turnover": "{:.1%}"}).background_gradient(
            subset=["Sharpe"], cmap="BuGn"),
        width="stretch", height=430)

    picks = st.multiselect(
        "Plot growth of $1", [label_of[i] for i in view.index],
        default=[label_of[i] for i in
                 view.sort_values("sharpe", ascending=False).index[:4]])
    fig = base_fig("Growth of $1 invested (out-of-sample)", "Value of $1 (USD)")
    for i, lab in enumerate(picks):
        g = growth(RET[id_of[lab]])
        fig.add_trace(go.Scatter(x=g.index, y=g, name=lab,
                                 line=dict(color=CYCLE[i % 10], width=2)))
    st.plotly_chart(fig, width="stretch")

    bar = view.sort_values("sharpe")
    bfig = base_fig("Out-of-sample Sharpe by fund", "", height=480)
    cmap = {"equity": FX["sky"], "crypto": FX["gold"], "combined": FX["teal"]}
    bfig.add_trace(go.Bar(x=bar["sharpe"], y=bar["label"], orientation="h",
                          marker_color=[cmap[f] for f in bar["family"]]))
    bfig.update_xaxes(title="Sharpe ratio (rf = 0)")
    st.plotly_chart(bfig, width="stretch")

# =================================================================== FACT SHEET
elif page == "Fund fact sheet":
    lab = st.selectbox("Choose a fund", [label_of[i] for i in PERF.index])
    fid = id_of[lab]
    p = PERF.loc[fid]
    r_all = RET[fid].dropna()

    st.subheader(lab)
    st.markdown(f"<span class='fx-pill'>{p['family'].title()}</span>"
                f"<span class='fx-pill'>Monthly rebalance</span>"
                f"<span class='fx-pill'>Annualised x{int(p['ann_factor'])}</span>"
                f"<span class='fx-pill'>OOS {p['oos_start']} to {p['oos_end']}</span>",
                unsafe_allow_html=True)

    yrs = sorted({d.year for d in r_all.index})
    lo, hi = st.select_slider("Zoom the backtest window",
                              options=yrs, value=(yrs[0], yrs[-1]))
    r = r_all[(r_all.index.year >= lo) & (r_all.index.year <= hi)]
    m = metrics_from_returns(r, int(p["ann_factor"]))

    c = st.columns(5)
    kpi_card(c[0], "Growth of $1", f"${m['growth']:.2f}", big=True)
    kpi_card(c[1], "Ann. return", f"{m['ann_ret']:.1%}")
    kpi_card(c[2], "Ann. volatility", f"{m['ann_vol']:.1%}")
    kpi_card(c[3], "Sharpe (rf=0)", f"{m['sharpe']:.2f}")
    kpi_card(c[4], "Max drawdown", f"{m['mdd']:.1%}")

    c1, c2 = st.columns(2)
    g = growth(r)
    fig = base_fig("Growth of $1 (cumulative return)", "Value of $1 (USD)")
    fig.add_trace(go.Scatter(x=g.index, y=g, fill="tozeroy",
                             fillcolor="rgba(43,196,196,0.10)",
                             line=dict(color=FX["teal"], width=2.4), name=lab))
    c1.plotly_chart(fig, width="stretch")

    dd = drawdown(r) * 100
    dfig = base_fig("Drawdown from peak", "Drawdown (%)")
    dfig.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy",
                              fillcolor="rgba(226,89,59,0.18)",
                              line=dict(color=FX["coral"]), name="Drawdown"))
    c2.plotly_chart(dfig, width="stretch")

    W = A["weights"]
    w_fund = W[W["fund_id"] == fid]
    latest_date = w_fund["date"].max()
    latest = (w_fund[w_fund["date"] == latest_date]
              .sort_values("weight", ascending=False))
    c3, c4 = st.columns(2)
    hfig = base_fig(f"Current holdings - top 15 (rebalance {latest_date.date()})",
                    "", height=460)
    hh = latest.head(15).iloc[::-1]
    hfig.add_trace(go.Bar(x=hh["weight"], y=hh["asset"], orientation="h",
                          marker_color=FX["teal"]))
    hfig.update_xaxes(tickformat=".0%", title="Target weight")
    c3.plotly_chart(hfig, width="stretch")

    top_assets = w_fund.groupby("asset")["weight"].mean().nlargest(10).index
    area = (w_fund[w_fund["asset"].isin(top_assets)]
            .pivot_table(index="date", columns="asset", values="weight",
                         aggfunc="sum").fillna(0))
    area["Other"] = (1 - area.sum(axis=1)).clip(lower=0)
    afig = base_fig("Weights over time (top 10 + other)", "", height=460)
    for i, ccol in enumerate(area.columns):
        afig.add_trace(go.Scatter(x=area.index, y=area[ccol], stackgroup="w",
                                  name=ccol, line=dict(width=0.4,
                                  color=(CYCLE + ["#3A4C60"])[i % 11])))
    afig.update_yaxes(tickformat=".0%", range=[0, 1])
    c4.plotly_chart(afig, width="stretch")

    st.markdown(f"<span class='fx-note'>Metrics recompute live for the "
                f"{lo}-{hi} window you selected. Weights are re-estimated each "
                f"month from the prior {int(CAT.loc[fid, 'lookback'])} days only, "
                f"then held. Gross of the {MGMT_FEE:.2%} p.a. fee and trading "
                f"costs (see Methodology).</span>", unsafe_allow_html=True)

# =================================================================== FIND MY MIX
elif page == "Find my mix":
    st.subheader("Find my mix")
    st.caption("Answer three quick questions and FundX suggests a starting "
               "blend from its combined funds. Illustrative only, from "
               "out-of-sample history - not personal advice.")

    q1 = st.radio("1. If your portfolio fell 20% in a month, you would...",
                  ["Sell to stop the bleeding", "Feel uneasy but hold",
                   "Hold", "Buy more - it's on sale"], index=2)
    q2 = st.radio("2. When do you expect to need this money?",
                  ["Within 2 years", "3-5 years", "7+ years"], index=1)
    q3 = st.radio("3. What matters more to you right now?",
                  ["Protecting what I have", "A balance", "Growing as fast as possible"],
                  index=1)

    score = (["Sell to stop the bleeding", "Feel uneasy but hold", "Hold",
              "Buy more - it's on sale"].index(q1)
             + ["Within 2 years", "3-5 years", "7+ years"].index(q2)
             + ["Protecting what I have", "A balance",
                "Growing as fast as possible"].index(q3))

    if score <= 2:
        profile, blend = "Cautious", {"combined_min_variance": 0.6,
                                      "combined_risk_parity": 0.4}
    elif score <= 5:
        profile, blend = "Balanced", {"combined_risk_parity": 0.5,
                                      "combined_equal_weight": 0.3,
                                      "combined_min_variance": 0.2}
    else:
        profile, blend = "Growth", {"combined_max_sharpe": 0.5,
                                    "combined_risk_parity": 0.3,
                                    "combined_equal_weight": 0.2}

    ids = list(blend)
    w = np.array([blend[i] for i in ids])
    ppy = 252
    mix_ret = (RET[ids].dropna(how="all").fillna(0) @ w)
    m = metrics_from_returns(mix_ret, ppy)

    st.markdown("<hr class='fx-divider'>", unsafe_allow_html=True)
    st.markdown(f"<div class='fx-eyebrow'>Your profile</div>"
                f"<div class='fx-big'>{profile}</div>", unsafe_allow_html=True)
    c = st.columns(4)
    kpi_card(c[0], "Suggested growth of $1", f"${m['growth']:.2f}", big=True)
    kpi_card(c[1], "Ann. return", f"{m['ann_ret']:.1%}")
    kpi_card(c[2], "Ann. vol", f"{m['ann_vol']:.1%}")
    kpi_card(c[3], "Max drawdown", f"{m['mdd']:.1%}")

    st.markdown("<br>", unsafe_allow_html=True)
    dfig = base_fig("Suggested blend", "", height=300)
    dfig.add_trace(go.Pie(labels=[label_of[i] for i in ids], values=w,
                          hole=0.62, marker=dict(colors=CYCLE[:len(ids)]),
                          textinfo="label+percent"))
    dfig.update_layout(showlegend=False)
    st.plotly_chart(dfig, width="stretch")
    st.markdown("<span class='fx-note'>Want to fine-tune these weights and add "
                "a dollar amount? Head to <b>Build your allocation</b>.</span>",
                unsafe_allow_html=True)

# =================================================================== ALLOCATION
elif page == "Build your allocation":
    st.subheader("Build your allocation")
    st.caption("Pick funds, set the mix, and project a real dollar amount over "
               "your horizon. Weights normalise to 100% automatically.")

    default = [label_of.get("combined_max_sharpe"),
               label_of.get("combined_risk_parity"),
               label_of.get("equity_min_variance")]
    chosen = st.multiselect("Funds in your mix",
                            [label_of[i] for i in PERF.index],
                            default=[d for d in default if d])
    if not chosen:
        st.info("Choose at least one fund to build a mix.")
        st.stop()

    a, b = st.columns(2)
    amount = a.number_input("Amount to invest (USD)", 100, 10_000_000, 10_000,
                            step=500)
    horizon = b.slider("Investment horizon (years)", 1, 30, 10)

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
    m = metrics_from_returns(mix_ret, ppy)
    fee_dollars = amount * MGMT_FEE
    projected = amount * (1 + m["ann_ret"]) ** horizon

    c = st.columns(5)
    kpi_card(c[0], "Mix growth of $1", f"${m['growth']:.2f}", big=True)
    kpi_card(c[1], "Ann. return", f"{m['ann_ret']:.1%}")
    kpi_card(c[2], "Ann. vol", f"{m['ann_vol']:.1%}")
    kpi_card(c[3], "Sharpe", f"{m['sharpe']:.2f}")
    kpi_card(c[4], "Max drawdown", f"{m['mdd']:.1%}")

    st.markdown("<hr class='fx-divider'>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.3])
    with left:
        st.markdown(f"<div class='fx-card'>"
                    f"<div class='fx-kpi-label'>Projected value in {horizon} years</div>"
                    f"<div class='fx-big'>${projected:,.0f}</div>"
                    f"<div class='fx-note'>From ${amount:,.0f} at the mix's "
                    f"out-of-sample annual return of {m['ann_ret']:.1%}, "
                    f"compounded. FundX fee ~${fee_dollars:,.0f}/yr "
                    f"({MGMT_FEE:.2%} p.a.). Illustrative; returns are gross of "
                    f"fees and past performance is not a guarantee.</div></div>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<span class='fx-note'>Allocation: " +
                    ", ".join(f"{lab} {wi:.0%}" for lab, wi in zip(chosen, w)) +
                    "</span>", unsafe_allow_html=True)
    with right:
        years = np.arange(0, horizon + 1)
        proj_path = amount * (1 + m["ann_ret"]) ** years
        pfig = base_fig(f"Projected growth of ${amount:,.0f}", "Value (USD)",
                        height=340)
        pfig.add_trace(go.Scatter(x=years, y=proj_path, fill="tozeroy",
                                  fillcolor="rgba(227,168,43,0.12)",
                                  line=dict(color=FX["gold"], width=3),
                                  name="Projected"))
        pfig.update_xaxes(title="Years from now")
        st.plotly_chart(pfig, width="stretch")

    fig = base_fig("Your blended portfolio vs its building blocks",
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

# =================================================================== SENTIMENT
elif page == "Sentiment analytics":
    st.subheader("News-sentiment index across equity sectors")
    st.caption("Built from ~147k deduplicated headlines for the 50 stocks, "
               "scored per headline, averaged per ticker-day, then "
               "equal-weighted within each sector. Read levels as tone, not "
               "certainty.")

    senti = A["senti"]
    use_fin = False
    if A["senti_fin"] is not None:
        use_fin = st.toggle("Use Fin-VADER (finance-lexicon extension)",
                            value=True,
                            help="Plain VADER leaves ~half of finance headlines "
                                 "neutral; the extended lexicon adds market "
                                 "terms like 'downgrade' or 'all-time high'.")
    S = A["senti_fin"] if use_fin else senti
    sectors = [c for c in S.columns if c != "ALL"]
    win = st.slider("Smoothing window (trading days)", 1, 63, 21)
    pick = st.multiselect("Sectors", sectors, default=sectors[:5])

    fig = base_fig(f"Sector sentiment over time "
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
    lfig = base_fig(f"Latest sector tone ({win}-day mean)", "Sentiment",
                    height=420)
    lcolors = [FX["coral"] if v < latest.median() else FX["teal"]
               for v in latest.values]
    lfig.add_trace(go.Bar(x=latest.values, y=latest.index, orientation="h",
                          marker_color=lcolors))
    c1.plotly_chart(lfig, width="stretch")

    if A["coverage"] is not None:
        cov = A["coverage"]["headlines"].rolling(21, min_periods=5).mean()
        cfig = base_fig("News flow (21-day mean headlines/day)",
                        "Headlines per day")
        cfig.add_trace(go.Scatter(x=cov.index, y=cov, fill="tozeroy",
                                  fillcolor="rgba(95,168,211,0.15)",
                                  line=dict(color=FX["sky"]), name="Headlines"))
        c2.plotly_chart(cfig, width="stretch")

    st.markdown("<span class='fx-note'>In the funds, this signal is lagged at "
                "least one trading day before use, so a decision on day t only "
                "ever sees news from day t-1 or earlier.</span>",
                unsafe_allow_html=True)

# =================================================================== METHODOLOGY
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

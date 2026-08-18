"""FundX - systematic multi-asset funds with news-sentiment analytics.

FINS3645 FinTech Project 2026, Part B (z5544937). Entry point for Streamlit
Community Cloud.

The deployed app is a READER: every fund, metric and sentiment series is
precomputed by `python scripts/run_part_b.py` into results/ (committed), so
the app never runs an optimiser or VADER (the free tier cannot). The one
network touch is an optional expander on the Methodology page that loads the
hosted price data through src/data_access.py, demonstrating the provided
helper end to end.

Design system: the FundX visual language is my own (src/plotstyle.py evolved
into the dark product palette below) - one brand across the report figures
and the app. No external templates or component libraries; the sparklines in
the KPI cards are hand-built inline SVG.
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

PAGES = ["Overview", "Compare funds", "Fund fact sheet", "Find my mix",
         "Build your allocation", "Sentiment analytics", "Methodology & data"]
GLYPH = {"Overview": "◈", "Compare funds": "⇄", "Fund fact sheet": "▤",
         "Find my mix": "✦", "Build your allocation": "◔",
         "Sentiment analytics": "◍", "Methodology & data": "☰"}

# One plain-words line per optimisation rule (shown on the fact sheet).
METHOD_BLURB = {
    "equal_weight": "Splits money equally across every asset - the "
                    "no-estimation benchmark the optimised rules must beat.",
    "min_variance": "Chooses the weights with the lowest historical "
                    "volatility - the defensive rule.",
    "max_sharpe": "Maximises historical return per unit of risk - the "
                  "classic optimiser, and the hungriest for estimates.",
    "risk_parity": "Sizes every position so each contributes the same risk - "
                   "no return forecasts needed.",
    "hrp": "Hierarchical Risk Parity: clusters similar assets first, then "
           "splits risk down the hierarchy - robust to estimation noise.",
    "max_sharpe_tilt_vader": "The Max Sharpe rule, with each stock's weight "
                             "tilted by its sector's lagged news tone "
                             "(plain-VADER scoring).",
    "max_sharpe_tilt_finvader": "The Max Sharpe rule, tilted by lagged "
                                "sector news tone scored with the extended "
                                "finance lexicon.",
}

st.set_page_config(page_title="FundX", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

  :root {{ --navy:{FX['navy']}; --panel:{FX['panel']}; --teal:{FX['teal']};
           --gold:{FX['gold']}; --ink:{FX['ink']}; --mute:{FX['mute']};
           --line:{FX['line']}; }}
  .stApp {{ background:
      radial-gradient(1100px 520px at 12% -8%, #14304A 0%, {FX['navy']} 58%) fixed; }}
  #MainMenu, footer {{ visibility: hidden; }}
  .block-container {{ padding-top: 1.2rem; max-width: 1250px; }}

  html, body, p, span, label, li, .stMarkdown {{
      font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; }}
  h1,h2,h3,h4 {{ color: {FX['ink']}; font-weight: 700; letter-spacing: -0.015em;
      font-family: 'Space Grotesk', 'Helvetica Neue', Arial, sans-serif; }}
  p, span, label, .stMarkdown {{ color: {FX['ink']}; }}

  /* ------- sidebar & nav (styled native radio, no external components) */
  section[data-testid="stSidebar"] {{ background: {FX['navy']};
      border-right: 1px solid {FX['line']}; }}
  section[data-testid="stSidebar"] label[data-baseweb="radio"] {{
      display: flex; width: 100%; padding: 0.5rem 0.7rem; margin: 2px 0;
      border-radius: 10px; border: 1px solid transparent;
      transition: background 0.15s ease; cursor: pointer; }}
  section[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {{
      display: none; }}  /* hide the native radio circle */
  section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {{
      background: {FX['panel']}; }}
  section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {{
      background: {FX['teal']}1c; border: 1px solid {FX['teal']}44; }}
  section[data-testid="stSidebar"] label[data-baseweb="radio"] p {{
      font-size: 0.92rem; font-weight: 500; }}

  /* ------- hero + page headers */
  .fx-hero {{ position: relative; overflow: hidden;
      background: radial-gradient(130% 150% at 0% 0%, #0E4C4C 0%,
          {FX['panel']} 52%, {FX['navy']} 100%);
      border: 1px solid {FX['line']}; border-radius: 18px;
      padding: 1.6rem 1.8rem 1.5rem; margin-bottom: 1.0rem;
      box-shadow: 0 10px 40px rgba(0,0,0,0.35); }}
  .fx-hero::after {{ content: ""; position: absolute; inset: 0;
      background-image: radial-gradient({FX['teal']}26 1px, transparent 1.4px);
      background-size: 22px 22px; opacity: 0.5; pointer-events: none; }}
  .fx-hero h1 {{ margin: 0; font-size: 2.3rem; line-height: 1.1;
      background: linear-gradient(90deg,#fff,{FX['teal']});
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .fx-hero p {{ margin: 0.4rem 0 0 0; color: {FX['ink']}; opacity: 0.85;
      font-size: 1.02rem; max-width: 46rem; }}
  .fx-eyebrow {{ color: {FX['teal']}; font-size: 0.72rem; font-weight: 700;
      letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 0.3rem; }}
  .fx-pagehead {{ margin: 0 0 0.9rem 0; }}
  .fx-pagehead h2 {{ margin: 0; font-size: 1.55rem; }}
  .fx-pagehead p {{ margin: 0.25rem 0 0 0; color: {FX['mute']};
      font-size: 0.92rem; max-width: 52rem; }}

  /* ------- cards, KPIs, pills */
  .fx-card {{ background: {FX['panel']}; border: 1px solid {FX['line']};
      border-radius: 14px; padding: 1.0rem 1.15rem; height: 100%;
      box-shadow: 0 6px 22px rgba(0,0,0,0.30); color: {FX['ink']};
      transition: transform 0.15s ease, border-color 0.15s ease,
          box-shadow 0.15s ease; }}
  .fx-card:hover {{ transform: translateY(-2px); border-color: {FX['teal']}66;
      box-shadow: 0 12px 30px rgba(0,0,0,0.38), 0 0 0 1px {FX['teal']}22; }}
  .fx-card p, .fx-card li, .fx-card strong, .fx-card em {{ color: {FX['ink']}; }}
  .fx-card.acc-teal  {{ border-top: 3px solid {FX['teal']}; }}
  .fx-card.acc-gold  {{ border-top: 3px solid {FX['gold']}; }}
  .fx-card.acc-coral {{ border-top: 3px solid {FX['coral']}; }}
  .fx-card.acc-sky   {{ border-top: 3px solid {FX['sky']}; }}
  .fx-kpi {{ font-size: 1.9rem; font-weight: 700; color: #fff;
      font-family: 'Space Grotesk', sans-serif;
      font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }}
  .fx-kpi.sm {{ font-size: 1.4rem; }}
  .fx-kpi-label {{ font-size: 0.68rem; color: {FX['mute']};
      text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }}
  .fx-big {{ font-size: 2.6rem; font-weight: 700; color: #fff;
      font-family: 'Space Grotesk', sans-serif;
      font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }}
  .fx-pill {{ display: inline-block; background: {FX['teal']}1f;
      color: {FX['teal']}; border: 1px solid {FX['teal']}55;
      border-radius: 999px; padding: 0.12rem 0.7rem; font-size: 0.74rem;
      font-weight: 600; margin: 0 0.35rem 0.3rem 0; }}
  .fx-pill.gold {{ background: {FX['gold']}1f; color: {FX['gold']};
      border-color: {FX['gold']}55; }}
  .fx-note {{ color: {FX['mute']}; font-size: 0.82rem; line-height: 1.5; }}
  .fx-divider {{ height: 1px; border: 0; margin: 1.3rem 0 1.0rem 0;
      background: linear-gradient(90deg, transparent, {FX['line']} 15%,
          {FX['line']} 85%, transparent); }}
  .fx-spark {{ margin-top: 0.35rem; }}

  /* ------- journey steps (Overview) */
  .fx-step {{ display: flex; gap: 0.7rem; align-items: flex-start; }}
  .fx-step-n {{ flex: 0 0 auto; width: 1.7rem; height: 1.7rem;
      border-radius: 8px; background: {FX['teal']}22; color: {FX['teal']};
      border: 1px solid {FX['teal']}55; font-weight: 700; text-align: center;
      line-height: 1.6rem; font-family: 'Space Grotesk', sans-serif; }}

  /* ------- widgets */
  .stButton>button, .stDownloadButton>button {{
      background: linear-gradient(90deg,{FX['teal']},#1FA9A9);
      color: #05202A; border: 0; border-radius: 10px; font-weight: 700;
      padding: 0.48rem 1.05rem; }}
  .stButton>button:hover, .stDownloadButton>button:hover {{
      filter: brightness(1.08); color:#05202A; }}
  .stButton>button[kind="secondary"], .stDownloadButton>button {{
      background: {FX['panel2']}; color: {FX['ink']};
      border: 1px solid {FX['line']}; }}
  div[data-testid="stMetric"] {{ background: {FX['panel']};
      border: 1px solid {FX['line']}; border-radius: 12px;
      padding: 0.7rem 0.9rem; }}
  div[data-baseweb="slider"] [role="slider"] {{ background: {FX['teal']}; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {FX['line']}; }}
  .stTabs [data-baseweb="tab"] {{ background: transparent;
      border-radius: 10px 10px 0 0; color: {FX['mute']}; padding: 0.4rem 0.9rem; }}
  .stTabs [aria-selected="true"] {{ background: {FX['panel']};
      color: {FX['ink']}; }}
  /* ------- risk quiz */
  .fx-crow {{ display:flex; align-items:center; gap:0.6rem; margin:0.5rem 0; }}
  .fx-crow .q {{ flex:0 0 44%; color:{FX['mute']}; font-size:0.82rem; }}
  .fx-crow .pts {{ flex:0 0 2.2rem; text-align:right; color:{FX['ink']};
      font-weight:700; font-variant-numeric:tabular-nums; }}
  .fx-bar {{ flex:1; height:8px; background:{FX['line']}; border-radius:999px;
      overflow:hidden; }}
  .fx-bar-fill {{ height:100%; border-radius:999px;
      background:linear-gradient(90deg,{FX['teal']},{FX['gold']}); }}
  .fx-live {{ background:{FX['panel']}; border:1px solid {FX['line']};
      border-top:3px solid {FX['teal']}; border-radius:14px;
      padding:1.1rem 1.2rem; box-shadow:0 4px 18px rgba(0,0,0,0.25); }}
  .fx-live-kpis {{ display:flex; gap:1.1rem; margin-top:0.7rem; flex-wrap:wrap; }}
  .fx-live-kpis .kl {{ font-size:0.62rem; color:{FX['mute']};
      text-transform:uppercase; letter-spacing:0.08em; }}
  .fx-live-kpis .kv {{ font-size:1.25rem; font-weight:700; color:#fff;
      font-family:'Space Grotesk',sans-serif; font-variant-numeric:tabular-nums; }}
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
    # Optional artifacts - the app degrades gracefully if any is absent.
    def _opt(path, **kw):
        try:
            return pd.read_csv(path, **kw) if path.exists() else None
        except Exception:
            return None
    art["senti_fin"] = _opt(d / "sector_sentiment_index_finlex.csv",
                            parse_dates=["date"])
    if art["senti_fin"] is not None:
        art["senti_fin"] = art["senti_fin"].set_index("date")
    art["coverage"] = _opt(d / "sentiment_coverage.csv", parse_dates=["date"])
    if art["coverage"] is not None:
        art["coverage"] = art["coverage"].set_index("date")
    art["bootstrap"] = _opt(t / "sharpe_bootstrap.csv", index_col=0)
    art["fusion"] = _opt(t / "fusion_before_after.csv")
    art["lexstats"] = _opt(t / "sentiment_model_stats.csv")
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


# ------------------------------------------------------------------ ui helpers
def base_fig(title: str = "", ytitle: str = "", height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(color=FX["ink"], size=15,
                                         family="Space Grotesk, sans-serif")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=FX["ink"], family="Inter, Helvetica Neue, sans-serif"),
        yaxis_title=ytitle, hovermode="x unified", height=height,
        hoverlabel=dict(bgcolor=FX["panel2"], bordercolor=FX["line"],
                        font=dict(color=FX["ink"])),
        legend=dict(orientation="h", y=-0.2, font=dict(color=FX["mute"])),
        margin=dict(l=45, r=20, t=48 if title else 22, b=40))
    fig.update_xaxes(gridcolor=FX["line"], zeroline=False, showspikes=True,
                     spikecolor=FX["steel"], spikethickness=1,
                     tickfont=dict(color=FX["mute"]))
    fig.update_yaxes(gridcolor=FX["line"], zeroline=False,
                     tickfont=dict(color=FX["mute"]))
    return fig


def spark_svg(series: pd.Series, color: str, width: int = 150,
              height: int = 38) -> str:
    """Tiny inline-SVG sparkline (own design system - no chart lib)."""
    s = pd.Series(series).dropna().astype(float)
    if len(s) < 2:
        return ""
    step = max(1, len(s) // 70)          # downsample to <=70 points
    v = s.iloc[::step].to_numpy()
    lo, hi = float(v.min()), float(v.max())
    rng = (hi - lo) or 1.0
    xs = np.linspace(2, width - 2, len(v))
    ys = height - 3 - (v - lo) / rng * (height - 8)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"2,{height-2} {pts} {width-2},{height-2}"
    return (f"<svg class='fx-spark' width='{width}' height='{height}' "
            f"viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"
            f"<polygon points='{area}' fill='{color}' opacity='0.14'/>"
            f"<polyline points='{pts}' fill='none' stroke='{color}' "
            f"stroke-width='1.8' stroke-linejoin='round'/></svg>")


STRESS_EVENTS = [
    ("2022-06-13", "2022 selloff", "macro"),
    ("2023-03-10", "SVB", "macro"),
    ("2022-05-09", "Terra", "crypto"),
    ("2022-11-08", "FTX", "crypto"),
]


def add_events(fig, family, x0, x1):
    """Mark in-range stress periods with a faint dashed line + label."""
    try:
        x0, x1 = pd.Timestamp(x0), pd.Timestamp(x1)
        scope = "crypto" if family == "crypto" else "macro"
        for ds, lab, sc in STRESS_EVENTS:
            if sc != scope:
                continue
            d = pd.Timestamp(ds)
            if x0 <= d <= x1:
                fig.add_vline(x=d, line_width=1, line_dash="dot",
                              line_color=FX["steel"])
                fig.add_annotation(x=d, y=1, yref="paper", yanchor="bottom",
                                   text=lab, showarrow=False,
                                   font=dict(color=FX["mute"], size=10))
    except Exception:
        pass
    return fig


def kpi_card(col, label, value, sub=None, sub_class="fx-note", big=False,
             accent=None, spark=None, help=None):
    cls = "fx-kpi" if big else "fx-kpi sm"
    card = "fx-card" + (f" acc-{accent}" if accent else "")
    if help:
        lab_html = (f"<div class='fx-kpi-label' title=\"{help}\">{label} "
                    f"<span style='opacity:0.5'>&#9432;</span></div>")
    else:
        lab_html = f"<div class='fx-kpi-label'>{label}</div>"
    html = (f"<div class='{card}'>{lab_html}"
            f"<div class='{cls}'>{value}</div>")
    if sub is not None:
        html += f"<div class='{sub_class}'>{sub}</div>"
    if spark:
        html += spark
    html += "</div>"
    col.markdown(html, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, sub: str = ""):
    st.markdown(f"<div class='fx-pagehead'>"
                f"<div class='fx-eyebrow'>{eyebrow}</div><h2>{title}</h2>"
                + (f"<p>{sub}</p>" if sub else "") + "</div>",
                unsafe_allow_html=True)


def goto(page_name: str, **state):
    """Callback: jump to another page (and optionally pre-set widget state)."""
    st.session_state["nav"] = page_name
    for k, v in state.items():
        st.session_state[k] = v


def apply_mix(labels: list[str], weights: list[int]):
    """Callback: preload the allocation builder with a given mix."""
    st.session_state["nav"] = "Build your allocation"
    st.session_state["alloc_funds"] = labels
    for lab, w in zip(labels, weights):
        st.session_state[f"w_{lab}"] = int(w)


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


# ------------------------------------------------------------------ sidebar
LOGO = f"""
<div style='display:flex;align-items:center;gap:0.6rem;margin:0.2rem 0 0.1rem;'>
  <svg width='34' height='34' viewBox='0 0 34 34' xmlns='http://www.w3.org/2000/svg'>
    <rect x='1' y='1' width='32' height='32' rx='9'
          fill='{FX['panel']}' stroke='{FX['teal']}' stroke-width='1.4'/>
    <rect x='8'  y='18' width='4.5' height='8'  rx='1.5' fill='{FX['sky']}'/>
    <rect x='15' y='13' width='4.5' height='13' rx='1.5' fill='{FX['gold']}'/>
    <rect x='22' y='8'  width='4.5' height='18' rx='1.5' fill='{FX['teal']}'/>
  </svg>
  <div>
    <div style='font-family:Space Grotesk,sans-serif;font-size:1.35rem;
                font-weight:700;line-height:1;
                background:linear-gradient(90deg,#fff,{FX['teal']});
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
      FundX</div>
    <div style='font-size:0.7rem;color:{FX['mute']};letter-spacing:0.06em;'>
      Systematic funds, transparent rules</div>
  </div>
</div>"""
st.sidebar.markdown(LOGO, unsafe_allow_html=True)
st.sidebar.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", PAGES, key="nav",
                        format_func=lambda p: f"{GLYPH[p]}\u2002{p}",
                        label_visibility="collapsed")

st.sidebar.markdown("<hr style='border-color:#20374F;margin:0.8rem 0'>",
                    unsafe_allow_html=True)
st.sidebar.markdown(
    f"<div class='fx-card' style='padding:0.7rem 0.85rem'>"
    f"<div class='fx-kpi-label'>Out-of-sample window</div>"
    f"<div style='font-weight:600;font-size:0.9rem'>"
    f"{PERF['oos_start'].min()} → {PERF['oos_end'].max()}</div>"
    f"<div class='fx-note'>Monthly rebalance; weights from past data only."
    f"</div></div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<div class='fx-note' style='margin-top:0.7rem'>"
                    f"Prototype for FINS3645 · not investment advice.</div>",
                    unsafe_allow_html=True)

# =================================================================== OVERVIEW
if page == "Overview":
    st.markdown("<div class='fx-hero'>"
                "<div class='fx-eyebrow'>Systematic multi-asset funds</div>"
                "<h1>Invest by rules you can read.</h1>"
                "<p>FundX runs pre-built equity &amp; crypto portfolios, each "
                "following one named, repeatable rule - backtested strictly "
                "out-of-sample, with a news-sentiment lens on top. Compare the "
                "shelf, read a fact sheet, and build your own mix.</p></div>",
                unsafe_allow_html=True)

    best = PERF.sort_values("sharpe", ascending=False).iloc[0]
    safest = PERF.loc[PERF["max_drawdown"].idxmax()]
    grow = PERF.loc[PERF["growth_of_1"].idxmax()]

    c = st.columns(4)
    kpi_card(c[0], "Funds on the shelf", f"{len(PERF)}",
             f"{PERF['family'].nunique()} asset families · 5 rules",
             big=True, accent="teal")
    kpi_card(c[1], "Best risk-adjusted", best["label"],
             f"Sharpe {best['sharpe']:.2f} out-of-sample", accent="gold",
             spark=spark_svg(growth(RET[best.name]), FX["gold"]))
    kpi_card(c[2], "Steadiest ride", safest["label"],
             f"Max drawdown {safest['max_drawdown']:.1%}", accent="sky",
             spark=spark_svg(growth(RET[safest.name]), FX["sky"]))
    kpi_card(c[3], "Biggest grower", grow["label"],
             f"${grow['growth_of_1']:.2f} per $1 (max DD "
             f"{grow['max_drawdown']:.0%})", accent="coral",
             spark=spark_svg(growth(RET[grow.name]), FX["coral"]))

    st.markdown("<hr class='fx-divider'>", unsafe_allow_html=True)
    left, right = st.columns([1.45, 1])
    with left:
        st.markdown("<div class='fx-eyebrow'>Growth of $1, out-of-sample</div>",
                    unsafe_allow_html=True)
        show = ["combined_risk_parity", "combined_max_sharpe",
                "equity_equal_weight", "crypto_min_variance"]
        fig = base_fig("", "Value of $1 (USD)", height=430)
        for i, fid in enumerate(show):
            g = growth(RET[fid])
            fig.add_trace(go.Scatter(
                x=g.index, y=g, name=label_of[fid],
                line=dict(color=CYCLE[i], width=2.4),
                hovertemplate="%{y:$.2f}<extra>" + label_of[fid] + "</extra>"))
        st.plotly_chart(fig, width="stretch")
        st.markdown("<span class='fx-note'>Four flavours of the shelf: the "
                    "risk-budgeted all-rounder, the return-seeker, the simple "
                    "benchmark, and the wildest ride. Full range on "
                    "<b>Compare funds</b>.</span>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='fx-eyebrow'>Your journey</div>",
                    unsafe_allow_html=True)
        steps = [("Compare the shelf",
                  "Every fund's out-of-sample record side by side.",
                  "Compare funds"),
                 ("Read a fact sheet",
                  "Growth, drawdown, holdings and calendar returns per fund.",
                  "Fund fact sheet"),
                 ("Build your mix",
                  "Blend funds, project a dollar amount, see the fee.",
                  "Build your allocation")]
        for i, (t, s, target) in enumerate(steps, start=1):
            st.markdown(f"<div class='fx-card' style='margin-bottom:0.5rem'>"
                        f"<div class='fx-step'><div class='fx-step-n'>{i}</div>"
                        f"<div><b>{t}</b><div class='fx-note'>{s}</div></div>"
                        f"</div></div>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        b1.button("Compare →", key="go_cmp", on_click=goto,
                  args=("Compare funds",))
        b2.button("Fact sheets →", key="go_fs", on_click=goto,
                  args=("Fund fact sheet",))
        b3.button("Build a mix →", key="go_mix", on_click=goto,
                  args=("Build your allocation",))
        st.markdown(f"<div class='fx-card' style='margin-top:0.6rem'>"
                    f"<div class='fx-kpi-label'>Data window</div>"
                    f"<div class='fx-kpi sm'>2020 – 2023</div>"
                    f"<div class='fx-note'>50 US equities, 10 cryptocurrencies, "
                    f"~147k news headlines.</div></div>",
                    unsafe_allow_html=True)

# =================================================================== COMPARE
elif page == "Compare funds":
    page_header("The shelf", "Compare the FundX range",
                "Filter by asset family and rule; every number is "
                "out-of-sample. Toggle costs to see the net-of-trading view.")

    f1, f2, f3 = st.columns([1.1, 1.6, 1])
    fam = f1.segmented_control("Asset family",
                               ["combined", "equity", "crypto"],
                               selection_mode="multi",
                               default=["combined", "equity", "crypto"],
                               format_func=str.title)
    methods = list(dict.fromkeys(PERF["method"]))
    meth = f2.multiselect("Rule", methods, default=methods,
                          format_func=lambda m: m.replace("_", " ").title())
    net = f3.toggle("Net of 10 bps costs", value=False,
                    help="Sharpe after a 10 bps × turnover transaction-cost "
                         "model - the funds that trade most pay most.")
    scol = "net_sharpe_10bps" if net else "sharpe"
    stitle = "Net Sharpe (10 bps)" if net else "Sharpe"

    view = PERF[PERF["family"].isin(fam or []) &
                PERF["method"].isin(meth or [])].copy()
    if view.empty:
        st.info("Pick at least one asset family and one rule.")
        st.stop()

    best = view.sort_values(scol, ascending=False).iloc[0]
    c = st.columns(4)
    kpi_card(c[0], "Funds shown", f"{len(view)}", big=True, accent="teal")
    kpi_card(c[1], f"Best OOS {stitle.lower()}", f"{best[scol]:.2f}",
             best["label"], accent="gold")
    kpi_card(c[2], "Its growth of $1", f"${best['growth_of_1']:.2f}")
    kpi_card(c[3], "Its max drawdown", f"{best['max_drawdown']:.1%}")

    st.markdown("<hr class='fx-divider'>", unsafe_allow_html=True)
    tab_tbl, tab_map, tab_gro, tab_corr, tab_conf = st.tabs(
        ["Fact-sheet table", "Risk vs return", "Growth of $1",
         "Diversification", "How sure are we?"])

    with tab_tbl:
        tbl = view[["label", "family", "ann_return", "ann_vol", scol,
                    "max_drawdown", "growth_of_1", "avg_turnover"]].copy()
        tbl.columns = ["Fund", "Family", "Ann. return", "Ann. vol", stitle,
                       "Max drawdown", "Growth of $1", "Avg turnover"]
        st.dataframe(
            tbl.sort_values(stitle, ascending=False).reset_index(drop=True),
            width="stretch", height=430, hide_index=True,
            column_config={
                "Family": st.column_config.TextColumn(width="small"),
                "Ann. return": st.column_config.NumberColumn(format="percent"),
                "Ann. vol": st.column_config.NumberColumn(format="percent"),
                stitle: st.column_config.ProgressColumn(
                    format="%.2f", min_value=-0.1,
                    max_value=float(PERF[scol].max()) + 0.05),
                "Max drawdown": st.column_config.NumberColumn(format="percent"),
                "Growth of $1": st.column_config.NumberColumn(format="dollar"),
                "Avg turnover": st.column_config.NumberColumn(format="percent"),
            })
        st.download_button("Download metrics (CSV)",
                           PERF.to_csv().encode(),
                           file_name="fundx_performance_metrics.csv",
                           mime="text/csv", key="dl_perf")

    with tab_map:
        mfig = base_fig("", "Annualised return", height=470)
        cmap = {"equity": FX["sky"], "crypto": FX["gold"],
                "combined": FX["teal"]}
        # Faint Sharpe reference rays (return = sharpe × vol, rf = 0).
        vmax = float(view["ann_vol"].max()) * 1.08
        for s_ref in (0.25, 0.5, 0.75, 1.0):
            mfig.add_trace(go.Scatter(
                x=[0, vmax], y=[0, s_ref * vmax], mode="lines",
                line=dict(color=FX["line"], width=1, dash="dot"),
                hoverinfo="skip", showlegend=False))
            mfig.add_annotation(x=vmax, y=s_ref * vmax, text=f"S={s_ref:g}",
                                showarrow=False, xanchor="left",
                                font=dict(color=FX["steel"], size=10))
        for famname, gdf in view.groupby("family"):
            mfig.add_trace(go.Scatter(
                x=gdf["ann_vol"], y=gdf["ann_return"], mode="markers",
                name=famname.title(),
                marker=dict(color=cmap.get(famname, FX["steel"]),
                            size=np.clip(gdf["growth_of_1"] * 7, 8, 30),
                            line=dict(color=FX["navy"], width=1),
                            opacity=0.9),
                customdata=np.stack([gdf["label"], gdf[scol],
                                     gdf["max_drawdown"]], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>"
                              "Vol %{x:.1%} · Return %{y:.1%}<br>"
                              f"{stitle} %{{customdata[1]:.2f}} · "
                              "Max DD %{customdata[2]:.1%}<extra></extra>"))
        mfig.update_xaxes(title="Annualised volatility", tickformat=".0%")
        mfig.update_yaxes(tickformat=".0%")
        mfig.update_layout(hovermode="closest")
        st.plotly_chart(mfig, width="stretch")
        st.markdown("<span class='fx-note'>Up and to the left is better; "
                    "bubble size is growth of $1; dotted rays are constant-"
                    "Sharpe lines. The crypto family lives far right - big "
                    "returns bought with much bigger risk.</span>",
                    unsafe_allow_html=True)

    with tab_gro:
        picks = st.multiselect(
            "Funds to plot", [label_of[i] for i in view.index],
            default=[label_of[i] for i in
                     view.sort_values(scol, ascending=False).index[:4]])
        fig = base_fig("", "Value of $1 (USD)")
        for i, lab in enumerate(picks):
            g = growth(RET[id_of[lab]])
            fig.add_trace(go.Scatter(
                x=g.index, y=g, name=lab,
                line=dict(color=CYCLE[i % 10], width=2),
                hovertemplate="%{y:$.2f}<extra>" + lab + "</extra>"))
        st.plotly_chart(fig, width="stretch")

        bar = view.sort_values(scol)
        bfig = base_fig("", "", height=480)
        bfig.add_trace(go.Bar(
            x=bar[scol], y=bar["label"], orientation="h",
            marker_color=[{"equity": FX["sky"], "crypto": FX["gold"],
                           "combined": FX["teal"]}.get(f, FX["steel"])
                          for f in bar["family"]],
            hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        bfig.update_xaxes(title=f"{stitle} (rf = 0)")
        st.plotly_chart(bfig, width="stretch")

    with tab_corr:
        cids = list(view.index)
        if len(cids) < 2:
            st.info("Pick at least two funds to see how they co-move.")
        else:
            corr = RET[cids].dropna(how="all").corr()
            labels = [label_of[i] for i in corr.columns]
            off = corr.values[~np.eye(len(corr), dtype=bool)]
            avg_off = float(np.nanmean(off))
            hfig = base_fig("Return correlation across the selected funds", "",
                            height=140 + 40 * len(cids))
            hfig.add_trace(go.Heatmap(
                z=corr.values, x=labels, y=labels,
                colorscale=[[0.0, FX["teal"]], [0.5, FX["panel2"]],
                            [1.0, FX["coral"]]],
                zmin=-1, zmax=1, xgap=2, ygap=2,
                texttemplate="%{z:.2f}", textfont=dict(size=10),
                hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
                colorbar=dict(title="ρ", outlinewidth=0,
                              tickfont=dict(color=FX["mute"]))))
            hfig.update_xaxes(tickangle=-40)
            st.plotly_chart(hfig, width="stretch")
            st.markdown(
                "<span class='fx-note'>Coral cells move together; teal cells "
                "pull in opposite directions. The average off-diagonal "
                f"correlation here is <b>{avg_off:.2f}</b> - the lower it is, "
                "the more these funds diversify each other, which is exactly "
                "what lets a blend cut risk without giving up much return."
                "</span>", unsafe_allow_html=True)

    with tab_conf:
        bs = A["bootstrap"]
        if bs is None:
            st.info("Bootstrap table not found (results/tables/"
                    "sharpe_bootstrap.csv).")
        else:
            st.markdown("<span class='fx-note'>Block-bootstrap 90% confidence "
                        "intervals for the Sharpe ratio (21-day blocks, "
                        "B = 2,000). Three years of daily data leaves wide "
                        "bands: every interval spans zero, so the leaders are "
                        "not statistically separable - a reason FundX ranks "
                        "its lead funds on drawdown and turnover too, not "
                        "Sharpe alone.</span>", unsafe_allow_html=True)
            bsv = bs.copy()
            bsv["label"] = [label_of.get(i, i) for i in bsv.index]
            bsv = bsv.sort_values("sharpe")
            cfig = base_fig("", "", height=90 + 52 * len(bsv))
            cfig.add_vline(x=0, line_color=FX["steel"], line_width=1)
            cfig.add_trace(go.Scatter(
                x=bsv["sharpe"], y=bsv["label"], mode="markers",
                marker=dict(color=FX["gold"], size=10,
                            line=dict(color=FX["navy"], width=1)),
                error_x=dict(type="data", symmetric=False,
                             array=bsv["ci_95pct"] - bsv["sharpe"],
                             arrayminus=bsv["sharpe"] - bsv["ci_5pct"],
                             color=FX["teal"], thickness=2, width=6),
                hovertemplate="<b>%{y}</b><br>Sharpe %{x:.2f}"
                              "<extra></extra>", showlegend=False))
            cfig.update_xaxes(title="Sharpe ratio, point estimate with 90% CI")
            st.plotly_chart(cfig, width="stretch")

# =================================================================== FACT SHEET
elif page == "Fund fact sheet":
    page_header("Fact sheet", "One fund, the whole story",
                "Pick a fund; zoom the window and every metric recomputes "
                "live from its daily out-of-sample returns.")

    top_l, top_r = st.columns([2.2, 1])
    lab = top_l.selectbox("Choose a fund", [label_of[i] for i in PERF.index])
    fid = id_of[lab]
    p = PERF.loc[fid]
    r_all = RET[fid].dropna()
    top_r.download_button("Download daily returns (CSV)",
                          r_all.rename(fid).to_csv().encode(),
                          file_name=f"{fid}_returns.csv", mime="text/csv",
                          key="dl_ret")

    pills = (f"<span class='fx-pill'>{p['family'].title()}</span>"
             f"<span class='fx-pill'>Monthly rebalance</span>"
             f"<span class='fx-pill'>Annualised ×{int(p['ann_factor'])}</span>"
             f"<span class='fx-pill'>OOS {p['oos_start']} → {p['oos_end']}</span>")
    bs = A["bootstrap"]
    if bs is not None and fid in bs.index:
        pills += (f"<span class='fx-pill gold'>Sharpe 90% CI "
                  f"[{bs.loc[fid, 'ci_5pct']:.2f}, "
                  f"{bs.loc[fid, 'ci_95pct']:.2f}]</span>")
    st.markdown(pills, unsafe_allow_html=True)
    st.markdown(f"<span class='fx-note'>"
                f"{METHOD_BLURB.get(p['method'], '')}</span>",
                unsafe_allow_html=True)

    yrs = sorted({d.year for d in r_all.index})
    if len(yrs) > 1:
        lo, hi = st.select_slider("Zoom the backtest window", options=yrs,
                                  value=(yrs[0], yrs[-1]))
    else:
        lo = hi = yrs[0]
    r = r_all[(r_all.index.year >= lo) & (r_all.index.year <= hi)]
    m = metrics_from_returns(r, int(p["ann_factor"]))

    c = st.columns(5)
    kpi_card(c[0], "Growth of $1", f"${m['growth']:.2f}", big=True,
             accent="teal")
    kpi_card(c[1], "Ann. return", f"{m['ann_ret']:.1%}",
             help="Average yearly growth rate over the window.")
    kpi_card(c[2], "Ann. volatility", f"{m['ann_vol']:.1%}",
             help="How much returns swing year to year - higher means a "
                  "bumpier ride.")
    kpi_card(c[3], "Sharpe (rf=0)", f"{m['sharpe']:.2f}",
             help="Return per unit of risk (return divided by volatility). "
                  "Higher is better; above ~1 is strong.")
    kpi_card(c[4], "Max drawdown", f"{m['mdd']:.1%}", accent="coral",
             help="The worst peak-to-trough fall over the window. Closer to "
                  "zero is safer.")

    bench_id = f"{p['family']}_equal_weight"
    can_bench = bench_id in RET.columns and bench_id != fid

    tab_perf, tab_risk, tab_hold, tab_cal = st.tabs(
        ["Performance", "Risk over time", "Holdings", "Calendar returns"])

    with tab_perf:
        show_bench = st.toggle(
            f"Overlay benchmark ({label_of.get(bench_id, 'Equal Weight')})",
            value=False, disabled=not can_bench,
            help="The same family's Equal Weight fund - the no-estimation "
                 "benchmark every optimised rule must beat.") if can_bench \
            else False
        c1, c2 = st.columns(2)
        g = growth(r)
        fig = base_fig("Growth of $1 (cumulative return)", "Value of $1 (USD)")
        fig.add_trace(go.Scatter(
            x=g.index, y=g, fill="tozeroy",
            fillcolor="rgba(43,196,196,0.10)",
            line=dict(color=FX["teal"], width=2.4), name=lab,
            hovertemplate="%{y:$.2f}<extra>" + lab + "</extra>"))
        if show_bench:
            rb = RET[bench_id].dropna()
            rb = rb[(rb.index.year >= lo) & (rb.index.year <= hi)]
            gb = growth(rb)
            fig.add_trace(go.Scatter(
                x=gb.index, y=gb, name=label_of[bench_id],
                line=dict(color=FX["gold"], width=1.8, dash="dash"),
                hovertemplate="%{y:$.2f}<extra>"
                              + label_of[bench_id] + "</extra>"))
        add_events(fig, p["family"], g.index.min(), g.index.max())
        c1.plotly_chart(fig, width="stretch")

        dd = drawdown(r) * 100
        dfig = base_fig("Drawdown from peak", "Drawdown (%)")
        dfig.add_trace(go.Scatter(
            x=dd.index, y=dd, fill="tozeroy",
            fillcolor="rgba(226,89,59,0.18)",
            line=dict(color=FX["coral"]), name="Drawdown",
            hovertemplate="%{y:.1f}%<extra></extra>"))
        c2.plotly_chart(dfig, width="stretch")
        if show_bench:
            mb = metrics_from_returns(rb, int(p["ann_factor"]))
            diff = m["sharpe"] - mb["sharpe"]
            word = "ahead of" if diff >= 0 else "behind"
            st.markdown(f"<span class='fx-note'>Over {lo}–{hi} this fund's "
                        f"Sharpe is {m['sharpe']:.2f} vs the benchmark's "
                        f"{mb['sharpe']:.2f} - {abs(diff):.2f} {word} the "
                        f"no-estimation rule.</span>", unsafe_allow_html=True)

    with tab_risk:
        af = int(p["ann_factor"])
        win_v = max(21, af // 4)          # ~ one quarter
        win_s = af                        # ~ one year
        if len(r) < win_v + 5:
            st.info("Zoom to a wider window to see rolling risk "
                    "(needs at least about a quarter of data).")
        else:
            c5, c6 = st.columns(2)
            roll_vol = (r.rolling(win_v, min_periods=max(10, win_v // 2))
                        .std(ddof=1) * np.sqrt(af) * 100)
            vfig = base_fig(f"Rolling volatility ({win_v}-day, annualised)",
                            "Volatility (%)")
            vfig.add_trace(go.Scatter(
                x=roll_vol.index, y=roll_vol, fill="tozeroy",
                fillcolor="rgba(227,168,43,0.12)",
                line=dict(color=FX["gold"], width=2), name="Vol",
                hovertemplate="%{y:.1f}%<extra></extra>"))
            vi = roll_vol.dropna().index
            if len(vi):
                add_events(vfig, p["family"], vi.min(), vi.max())
            c5.plotly_chart(vfig, width="stretch")

            rmean = r.rolling(win_s, min_periods=max(30, win_s // 2)).mean() * af
            rstd = (r.rolling(win_s, min_periods=max(30, win_s // 2))
                    .std(ddof=1) * np.sqrt(af))
            roll_sh = rmean / rstd
            sfig = base_fig(f"Rolling Sharpe ({win_s}-day, rf = 0)", "Sharpe")
            sfig.add_hline(y=0, line_color=FX["steel"], line_width=1)
            sfig.add_trace(go.Scatter(
                x=roll_sh.index, y=roll_sh,
                line=dict(color=FX["teal"], width=2), name="Sharpe",
                hovertemplate="%{y:.2f}<extra></extra>"))
            si = roll_sh.dropna().index
            if len(si):
                add_events(sfig, p["family"], si.min(), si.max())
            c6.plotly_chart(sfig, width="stretch")
            st.markdown(
                "<span class='fx-note'>A single headline Sharpe hides how much "
                "it wandered. Where the rolling line dips below zero the fund "
                "was losing money over that window; the dotted markers flag "
                "the period's main stress events.</span>",
                unsafe_allow_html=True)

    with tab_hold:
        W = A["weights"]
        w_fund = W[W["fund_id"] == fid]
        latest_date = w_fund["date"].max()
        latest = (w_fund[w_fund["date"] == latest_date]
                  .sort_values("weight", ascending=False))
        c3, c4 = st.columns(2)
        hfig = base_fig(f"Latest rebalance - top 15 "
                        f"({latest_date.date()})", "", height=460)
        hh = latest.head(15).iloc[::-1]
        hfig.add_trace(go.Bar(x=hh["weight"], y=hh["asset"], orientation="h",
                              marker_color=FX["teal"],
                              hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        hfig.update_xaxes(tickformat=".0%", title="Target weight")
        c3.plotly_chart(hfig, width="stretch")
        c3.download_button("Download latest weights (CSV)",
                           latest[["asset", "weight"]].to_csv(index=False)
                           .encode(),
                           file_name=f"{fid}_latest_weights.csv",
                           mime="text/csv", key="dl_w")

        top_assets = (w_fund.groupby("asset")["weight"].mean()
                      .nlargest(10).index)
        area = (w_fund[w_fund["asset"].isin(top_assets)]
                .pivot_table(index="date", columns="asset", values="weight",
                             aggfunc="sum").fillna(0))
        area["Other"] = (1 - area.sum(axis=1)).clip(lower=0)
        afig = base_fig("Weights over time (top 10 + other)", "", height=460)
        for i, ccol in enumerate(area.columns):
            afig.add_trace(go.Scatter(
                x=area.index, y=area[ccol], stackgroup="w", name=ccol,
                line=dict(width=0.4, color=(CYCLE + ["#3A4C60"])[i % 11]),
                hovertemplate="%{y:.1%}<extra>" + str(ccol) + "</extra>"))
        afig.update_yaxes(tickformat=".0%", range=[0, 1])
        c4.plotly_chart(afig, width="stretch")

    with tab_cal:
        mret = r.resample("ME").apply(lambda x: float((1 + x).prod() - 1))
        cal = pd.DataFrame({"year": mret.index.year,
                            "month": mret.index.month, "ret": mret.values})
        piv = cal.pivot_table(index="year", columns="month", values="ret")
        piv = piv.reindex(columns=range(1, 13))
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        zmax = float(np.nanmax(np.abs(piv.values))) or 0.01
        hfig2 = base_fig("", "", height=110 + 62 * len(piv))
        hfig2.add_trace(go.Heatmap(
            z=piv.values, x=months, y=piv.index.astype(str),
            colorscale=[[0, FX["coral"]], [0.5, FX["panel2"]],
                        [1, FX["teal"]]],
            zmin=-zmax, zmax=zmax, xgap=3, ygap=3,
            texttemplate="%{z:.1%}", textfont=dict(size=11),
            hovertemplate="%{y} %{x}: %{z:.1%}<extra></extra>",
            colorbar=dict(tickformat=".0%", outlinewidth=0,
                          tickfont=dict(color=FX["mute"]))))
        hfig2.update_yaxes(autorange="reversed")
        st.plotly_chart(hfig2, width="stretch")
        st.markdown("<span class='fx-note'>Monthly compounded returns for "
                    "the zoomed window - the 2022 row is where every fund "
                    "earned its drawdown.</span>", unsafe_allow_html=True)

    st.markdown(f"<span class='fx-note'>Metrics recompute live for the "
                f"{lo}–{hi} window you selected. Weights are re-estimated "
                f"each month from the prior {int(CAT.loc[fid, 'lookback'])} "
                f"days only, then held. Gross of the {MGMT_FEE:.2%} p.a. fee "
                f"and trading costs (see Methodology).</span>",
                unsafe_allow_html=True)

# =================================================================== FIND MY MIX
elif page == "Find my mix":
    page_header("Risk quiz", "Find my mix",
                "Answer three questions - the suggested blend and its numbers "
                "update as you go, so you can see exactly how each choice moves "
                "the returns. Illustrative only, from out-of-sample history.")

    # --- continuous risk model: three anchors, linearly interpolated ----------
    ANCHORS = [
        (0.0, {"combined_min_variance": 0.60, "combined_risk_parity": 0.40}),
        (0.5, {"combined_risk_parity": 0.50, "combined_equal_weight": 0.30,
               "combined_min_variance": 0.20}),
        (1.0, {"combined_max_sharpe": 0.50, "combined_risk_parity": 0.30,
               "combined_equal_weight": 0.20}),
    ]

    def mix_from_risk(pv):
        pv = min(1.0, max(0.0, float(pv)))
        (pa, wa), (pb, wb) = (ANCHORS[0], ANCHORS[1]) if pv <= 0.5 \
            else (ANCHORS[1], ANCHORS[2])
        t = 0.0 if pb == pa else (pv - pa) / (pb - pa)
        keys = sorted(set(wa) | set(wb))
        wv = np.array([(1 - t) * wa.get(k, 0) + t * wb.get(k, 0) for k in keys])
        keep = wv > 1e-9
        keys = [k for k, mk in zip(keys, keep) if mk]
        wv = wv[keep]
        return keys, wv / wv.sum()

    def mix_series(keys, wv):
        return RET[keys].dropna(how="all").fillna(0) @ wv

    def band_of(pv):
        return "Cautious" if pv < 2.5 / 7 else ("Balanced" if pv < 5.5 / 7
                                                else "Growth")

    def risk_meter_svg(pv, wd=460, ht=58):
        mg = 12
        xx = mg + pv * (wd - 2 * mg)
        b1 = mg + (2.5 / 7) * (wd - 2 * mg)
        b2 = mg + (5.5 / 7) * (wd - 2 * mg)
        y0, bh = 30, 12
        seg = (f"<rect x='{mg}' y='{y0}' width='{b1-mg:.1f}' height='{bh}' "
               f"rx='6' fill='{FX['teal']}' opacity='0.30'/>"
               f"<rect x='{b1:.1f}' y='{y0}' width='{b2-b1:.1f}' height='{bh}' "
               f"fill='{FX['gold']}' opacity='0.30'/>"
               f"<rect x='{b2:.1f}' y='{y0}' width='{wd-mg-b2:.1f}' height='{bh}' "
               f"rx='6' fill='{FX['coral']}' opacity='0.30'/>")
        labels = (f"<text x='{mg}' y='20' fill='{FX['mute']}' font-size='10' "
                  f"font-family='Inter'>Cautious</text>"
                  f"<text x='{wd/2:.0f}' y='20' fill='{FX['mute']}' "
                  f"font-size='10' text-anchor='middle' "
                  f"font-family='Inter'>Balanced</text>"
                  f"<text x='{wd-mg}' y='20' fill='{FX['mute']}' font-size='10' "
                  f"text-anchor='end' font-family='Inter'>Growth</text>")
        marker = (f"<polygon points='{xx:.1f},{y0-4} {xx-5:.1f},{y0-12} "
                  f"{xx+5:.1f},{y0-12}' fill='#fff'/>"
                  f"<rect x='{xx-1.5:.1f}' y='{y0-2}' width='3' height='{bh+4}' "
                  f"rx='1.5' fill='#fff'/>")
        return (f"<svg width='100%' viewBox='0 0 {wd} {ht}' "
                f"xmlns='http://www.w3.org/2000/svg'>{seg}{labels}{marker}</svg>")

    QS = [
        ("1. If your portfolio fell 20% in a month, you would...",
         ["Sell to stop the bleeding", "Feel uneasy but hold", "Hold",
          "Buy more - it's on sale"], "Reaction to a 20% drop"),
        ("2. When do you expect to need this money?",
         ["Within 2 years", "3-5 years", "7+ years"], "Time horizon"),
        ("3. What matters more to you right now?",
         ["Protecting what I have", "A balance", "Growing as fast as possible"],
         "What matters now"),
    ]
    MAXS = 7

    ask, live = st.columns([1.1, 1], gap="large")
    with ask:
        a1 = st.radio(QS[0][0], QS[0][1], index=2, key="q1")
        a2 = st.radio(QS[1][0], QS[1][1], index=1, key="q2")
        a3 = st.radio(QS[2][0], QS[2][1], index=1, key="q3")

    pts = [QS[0][1].index(a1), QS[1][1].index(a2), QS[2][1].index(a3)]
    score = sum(pts)
    p = score / MAXS
    profile = band_of(p)
    ids, w = mix_from_risk(p)
    r = mix_series(ids, w)
    m = metrics_from_returns(r, 252)

    with ask:
        rows = ""
        for (q, opts, short), pt in zip(QS, pts):
            pct = 100 * pt / (len(opts) - 1)
            rows += (f"<div class='fx-crow'><span class='q'>{short}</span>"
                     f"<div class='fx-bar'><div class='fx-bar-fill' "
                     f"style='width:{pct:.0f}%'></div></div>"
                     f"<span class='pts'>+{pt}</span></div>")
        st.markdown("<div class='fx-note' style='margin-top:0.5rem'>"
                    "How your answers add up (each is worth 0-3 risk points):"
                    "</div>" + rows, unsafe_allow_html=True)

    with live:
        kv = (f"<div class='k'><div class='kl'>Growth of $1</div>"
              f"<div class='kv'>${m['growth']:.2f}</div></div>"
              f"<div class='k'><div class='kl'>Ann. return</div>"
              f"<div class='kv'>{m['ann_ret']:.1%}</div></div>"
              f"<div class='k'><div class='kl'>Ann. vol</div>"
              f"<div class='kv'>{m['ann_vol']:.1%}</div></div>"
              f"<div class='k'><div class='kl'>Max drawdown</div>"
              f"<div class='kv'>{m['mdd']:.1%}</div></div>")
        st.markdown(
            f"<div class='fx-live'><div class='fx-eyebrow'>Live suggestion</div>"
            f"<div class='fx-big' style='font-size:2rem'>{profile}</div>"
            f"{risk_meter_svg(p)}"
            f"<div class='fx-live-kpis'>{kv}</div></div>",
            unsafe_allow_html=True)
        st.button("Use this mix in the builder →", key="use_mix",
                  on_click=apply_mix,
                  args=([label_of[i] for i in ids],
                        [round(x * 100) for x in w]))

    st.markdown("<hr class='fx-divider'>", unsafe_allow_html=True)

    # --- how one notch changes the numbers ------------------------------------
    st.markdown("<div class='fx-eyebrow'>How one answer moves your returns</div>"
                "<div class='fx-note'>Every answer shifts your risk score by one "
                "point. Here is what a single point does to the blend, priced on "
                "the same out-of-sample history.</div>", unsafe_allow_html=True)
    s_lo, s_hi = max(0, score - 1), min(MAXS, score + 1)
    scen = []
    for sc, tag, acc in [(s_lo, "One notch safer", "sky"),
                         (score, "Your answers", "teal"),
                         (s_hi, "One notch bolder", "gold")]:
        i2, w2 = mix_from_risk(sc / MAXS)
        mm = metrics_from_returns(mix_series(i2, w2), 252)
        scen.append((tag, acc, mm, band_of(sc / MAXS)))
    cc = st.columns(3)
    base = scen[1][2]
    for col, (tag, acc, mm, bnd) in zip(cc, scen):
        dg = mm['growth'] - base['growth']
        dr = (mm['ann_ret'] - base['ann_ret']) * 100
        delta = ("<div class='fx-note'>&nbsp;</div>" if abs(dg) < 1e-9 else
                 f"<div class='fx-note'>{'+' if dg >= 0 else ''}${dg:.2f} growth "
                 f"· {'+' if dr >= 0 else ''}{dr:.1f} pp return</div>")
        col.markdown(
            f"<div class='fx-card acc-{acc}'><div class='fx-kpi-label'>{tag}</div>"
            f"<div class='fx-kpi sm'>${mm['growth']:.2f}</div>"
            f"<div class='fx-note'>{bnd} · {mm['ann_ret']:.1%} return · "
            f"{mm['mdd']:.1%} drawdown</div>{delta}</div>",
            unsafe_allow_html=True)

    st.markdown("<hr class='fx-divider'>", unsafe_allow_html=True)

    # --- donut + spectrum chart -----------------------------------------------
    c1, c2 = st.columns([1, 1.4])
    with c1:
        dfig = base_fig("Your suggested mix", "", height=330)
        dfig.add_trace(go.Pie(labels=[label_of[i] for i in ids], values=w,
                              hole=0.62, marker=dict(colors=CYCLE[:len(ids)]),
                              textinfo="label+percent",
                              textfont=dict(color=FX["ink"])))
        dfig.update_layout(showlegend=False,
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(dfig, width="stretch")
    with c2:
        pfig = base_fig("The full risk spectrum, out-of-sample", "Value of $1")
        for i, (pp, _wa) in enumerate(ANCHORS):
            gi = growth(mix_series(*mix_from_risk(pp)))
            nm = band_of(pp)
            pfig.add_trace(go.Scatter(
                x=gi.index, y=gi, name=nm,
                line=dict(color=CYCLE[i], width=1.2, dash="dot"),
                opacity=0.7,
                hovertemplate="%{y:$.2f}<extra>" + nm + "</extra>"))
        gme = growth(r)
        pfig.add_trace(go.Scatter(
            x=gme.index, y=gme, name="You",
            line=dict(color="#ffffff", width=3),
            hovertemplate="%{y:$.2f}<extra>You</extra>"))
        st.plotly_chart(pfig, width="stretch")
        st.markdown("<span class='fx-note'>The dotted lines are the three "
                    "anchor profiles; the white line is your blend. As you "
                    "change answers it slides between them.</span>",
                    unsafe_allow_html=True)

    # --- fine-tune ------------------------------------------------------------
    score_pct = round(p * 100)
    with st.expander("Fine-tune the mix yourself  ·  go beyond the quiz"):
        st.markdown("<span class='fx-note'>Your answers put you at "
                    f"<b>{score_pct}%</b> on the risk dial. Drag to explore any "
                    "level - every notch re-blends the funds and re-prices the "
                    "mix from the same out-of-sample history.</span>",
                    unsafe_allow_html=True)
        if "mix_dial" not in st.session_state:
            st.session_state["mix_dial"] = score_pct
        st.button("↻ Reset to my answers", key="dial_reset",
                  on_click=lambda v=score_pct: st.session_state.update(
                      mix_dial=v))
        dial = st.slider("Risk dial (0 = fully cautious, 100 = full growth)",
                         0, 100, key="mix_dial")
        i3, w3 = mix_from_risk(dial / 100)
        r3 = mix_series(i3, w3)
        m3 = metrics_from_returns(r3, 252)
        k = st.columns(5)
        kpi_card(k[0], "Profile", band_of(dial / 100))
        kpi_card(k[1], "Growth of $1", f"${m3['growth']:.2f}", accent="teal",
                 spark=spark_svg(growth(r3), FX["teal"]))
        kpi_card(k[2], "Ann. return", f"{m3['ann_ret']:.1%}")
        kpi_card(k[3], "Ann. vol", f"{m3['ann_vol']:.1%}")
        kpi_card(k[4], "Max drawdown", f"{m3['mdd']:.1%}", accent="coral")
        st.button("Use this fine-tuned mix in the builder →", key="use_mix2",
                  on_click=apply_mix,
                  args=([label_of[i] for i in i3],
                        [round(x * 100) for x in w3]))

# =================================================================== ALLOCATION
elif page == "Build your allocation":
    page_header("Allocation builder", "Build your allocation",
                "Pick funds, set the mix, and project a real dollar amount "
                "over your horizon. Weights normalise to 100% automatically.")

    pc = st.columns([0.8, 1, 1, 1, 2.2])
    pc[0].markdown("<div class='fx-kpi-label' style='margin-top:0.7rem'>"
                   "Presets</div>", unsafe_allow_html=True)
    pc[1].button("Cautious", key="pre_c", on_click=apply_mix, args=(
        [label_of["combined_min_variance"], label_of["combined_risk_parity"]],
        [60, 40]))
    pc[2].button("Balanced", key="pre_b", on_click=apply_mix, args=(
        [label_of["combined_risk_parity"], label_of["combined_equal_weight"],
         label_of["combined_min_variance"]], [50, 30, 20]))
    pc[3].button("Growth", key="pre_g", on_click=apply_mix, args=(
        [label_of["combined_max_sharpe"], label_of["combined_risk_parity"],
         label_of["combined_equal_weight"]], [50, 30, 20]))

    default = [label_of.get("combined_max_sharpe"),
               label_of.get("combined_risk_parity"),
               label_of.get("equity_min_variance")]
    if "alloc_funds" in st.session_state:
        chosen = st.multiselect("Funds in your mix",
                                [label_of[i] for i in PERF.index],
                                key="alloc_funds")
    else:
        chosen = st.multiselect("Funds in your mix",
                                [label_of[i] for i in PERF.index],
                                default=[d for d in default if d],
                                key="alloc_funds")
    if not chosen:
        st.info("Choose at least one fund to build a mix.")
        st.stop()

    a, b = st.columns(2)
    amount = a.number_input("Amount to invest (USD)", 100, 10_000_000, 10_000,
                            step=500)
    horizon = b.slider("Investment horizon (years)", 1, 30, 10)

    cols = st.columns(len(chosen))
    raw = []
    for c_, lab_ in zip(cols, chosen):
        key = f"w_{lab_}"
        if key in st.session_state:
            raw.append(c_.slider(lab_, 0, 100, key=key))
        else:
            raw.append(c_.slider(lab_, 0, 100, value=100 // len(chosen),
                                 key=key))
    if sum(raw) == 0:
        st.warning("Give at least one fund a positive weight.")
        st.stop()
    w = np.array(raw, dtype=float) / sum(raw)

    ids = [id_of[lab_] for lab_ in chosen]
    mix_ret = (RET[ids].dropna(how="all").fillna(0) @ w)
    ppy = 252 if any(CAT.loc[i, "family"] != "crypto" for i in ids) else 365
    m = metrics_from_returns(mix_ret, ppy)
    fee_dollars = amount * MGMT_FEE
    projected = amount * (1 + m["ann_ret"]) ** horizon
    projected_net = amount * max(1 + m["ann_ret"] - MGMT_FEE, 0.0) ** horizon

    c = st.columns(5)
    kpi_card(c[0], "Mix growth of $1", f"${m['growth']:.2f}", big=True,
             accent="teal", spark=spark_svg(growth(mix_ret), FX["teal"]))
    kpi_card(c[1], "Ann. return", f"{m['ann_ret']:.1%}")
    kpi_card(c[2], "Ann. vol", f"{m['ann_vol']:.1%}")
    kpi_card(c[3], "Sharpe", f"{m['sharpe']:.2f}")
    kpi_card(c[4], "Max drawdown", f"{m['mdd']:.1%}", accent="coral")

    st.markdown("<hr class='fx-divider'>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.35])
    with left:
        st.markdown(
            f"<div class='fx-card acc-gold'>"
            f"<div class='fx-kpi-label'>Projected value in {horizon} years"
            f"</div><div class='fx-big'>${projected_net:,.0f}</div>"
            f"<div class='fx-note'>Net of FundX's {MGMT_FEE:.2%} p.a. fee "
            f"(~${fee_dollars:,.0f}/yr on ${amount:,.0f}). Gross of fee it "
            f"would be ${projected:,.0f} - the fee costs you about "
            f"${projected - projected_net:,.0f} over the horizon. Based on "
            f"the mix's out-of-sample annual return of {m['ann_ret']:.1%}, "
            f"compounded; illustrative, and past performance is not a "
            f"guarantee.</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:0.6rem'></div>",
                    unsafe_allow_html=True)
        dfig = base_fig("", "", height=280)
        dfig.add_trace(go.Pie(labels=chosen, values=w, hole=0.62,
                              marker=dict(colors=CYCLE[:len(chosen)]),
                              textinfo="percent",
                              textfont=dict(color=FX["ink"])))
        dfig.update_layout(showlegend=True,
                           legend=dict(orientation="h", y=-0.08),
                           margin=dict(l=10, r=10, t=6, b=6))
        st.plotly_chart(dfig, width="stretch")
        st.download_button("Download mix daily returns (CSV)",
                           mix_ret.rename("mix_return").to_csv().encode(),
                           file_name="fundx_my_mix_returns.csv",
                           mime="text/csv", key="dl_mix")
    with right:
        years = np.arange(0, horizon + 1)
        gross_path = amount * (1 + m["ann_ret"]) ** years
        net_path = amount * np.maximum(1 + m["ann_ret"] - MGMT_FEE,
                                       0.0) ** years
        pfig = base_fig(f"Projected growth of ${amount:,.0f}", "Value (USD)",
                        height=340)
        pfig.add_trace(go.Scatter(
            x=years, y=net_path, fill="tozeroy",
            fillcolor="rgba(227,168,43,0.12)",
            line=dict(color=FX["gold"], width=3), name="Net of fee",
            hovertemplate="Year %{x}: %{y:$,.0f}<extra>Net of fee</extra>"))
        pfig.add_trace(go.Scatter(
            x=years, y=gross_path, name="Gross of fee",
            line=dict(color=FX["teal"], width=1.6, dash="dash"),
            hovertemplate="Year %{x}: %{y:$,.0f}<extra>Gross</extra>"))
        pfig.update_xaxes(title="Years from now")
        st.plotly_chart(pfig, width="stretch")

    hist_g, hist_d = st.tabs(["Mix vs building blocks", "Mix drawdown"])
    with hist_g:
        fig = base_fig("", "Value of $1 (USD)")
        gmix = growth(mix_ret)
        fig.add_trace(go.Scatter(
            x=gmix.index, y=gmix, name="Your mix",
            line=dict(color=FX["gold"], width=3),
            hovertemplate="%{y:$.2f}<extra>Your mix</extra>"))
        for i, (lab_, fid_) in enumerate(zip(chosen, ids)):
            gf = growth(RET[fid_])
            fig.add_trace(go.Scatter(
                x=gf.index, y=gf, name=lab_,
                line=dict(color=CYCLE[i % 10], width=1.3, dash="dot"),
                hovertemplate="%{y:$.2f}<extra>" + lab_ + "</extra>"))
        st.plotly_chart(fig, width="stretch")
    with hist_d:
        ddm = drawdown(mix_ret) * 100
        dfig2 = base_fig("", "Drawdown (%)")
        dfig2.add_trace(go.Scatter(
            x=ddm.index, y=ddm, fill="tozeroy",
            fillcolor="rgba(226,89,59,0.18)", line=dict(color=FX["coral"]),
            name="Mix drawdown",
            hovertemplate="%{y:.1f}%<extra></extra>"))
        st.plotly_chart(dfig2, width="stretch")
        st.markdown(f"<span class='fx-note'>The blended mix's worst "
                    f"peak-to-trough fall over the window was "
                    f"{m['mdd']:.1%} - the number to sit with before the "
                    f"projection above.</span>", unsafe_allow_html=True)

# =================================================================== SENTIMENT
elif page == "Sentiment analytics":
    page_header("News lens", "Sentiment across equity sectors",
                "~147k deduplicated headlines for the 50 stocks, scored per "
                "headline, averaged per ticker-day, then equal-weighted "
                "within each sector. Read levels as tone, not certainty.")

    senti = A["senti"]
    use_fin = False
    if A["senti_fin"] is not None:
        use_fin = st.toggle("Use Fin-VADER (finance-lexicon extension)",
                            value=True,
                            help="Plain VADER leaves ~half of finance "
                                 "headlines neutral; the extended lexicon "
                                 "adds market terms like 'downgrade' or "
                                 "'all-time high'.")
    S = A["senti_fin"] if use_fin else senti
    sectors = [c for c in S.columns if c != "ALL"]

    tab_idx, tab_heat, tab_lex, tab_fus = st.tabs(
        ["Sector index", "Sector heatmap", "Lexicon effect",
         "Does it trade better?"])

    with tab_idx:
        win = st.slider("Smoothing window (trading days)", 1, 63, 21)
        pick = st.multiselect("Sectors", sectors, default=sectors[:5])
        fig = base_fig("", "Mean compound score (−1 to +1)")
        for i, ccol in enumerate(pick):
            sm = S[ccol].rolling(win, min_periods=min(win, max(2, win // 4))).mean()
            fig.add_trace(go.Scatter(
                x=sm.index, y=sm, name=ccol,
                line=dict(color=CYCLE[i % 10], width=1.7),
                hovertemplate="%{y:.3f}<extra>" + str(ccol) + "</extra>"))
        if "ALL" in S.columns:
            sm = S["ALL"].rolling(win, min_periods=min(win, max(2, win // 4))).mean()
            fig.add_trace(go.Scatter(
                x=sm.index, y=sm, name="All 50 stocks",
                line=dict(color=FX["ink"], width=2.6, dash="dash"),
                hovertemplate="%{y:.3f}<extra>All 50 stocks</extra>"))
        fig.add_hline(y=0, line_color=FX["steel"], line_width=1)
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        latest = (S[sectors].rolling(win, min_periods=min(win, 2)).mean()
                  .iloc[-1].sort_values())
        lfig = base_fig(f"Latest sector tone ({win}-day mean)", "Sentiment",
                        height=420)
        lcolors = [FX["coral"] if v < latest.median() else FX["teal"]
                   for v in latest.values]
        lfig.add_trace(go.Bar(x=latest.values, y=latest.index,
                              orientation="h", marker_color=lcolors,
                              hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        c1.plotly_chart(lfig, width="stretch")
        if A["coverage"] is not None:
            cov = A["coverage"]["headlines"].rolling(21, min_periods=5).mean()
            cfig = base_fig("News flow (21-day mean headlines/day)",
                            "Headlines per day")
            cfig.add_trace(go.Scatter(
                x=cov.index, y=cov, fill="tozeroy",
                fillcolor="rgba(95,168,211,0.15)",
                line=dict(color=FX["sky"]), name="Headlines",
                hovertemplate="%{y:.0f}/day<extra></extra>"))
            c2.plotly_chart(cfig, width="stretch")
        st.markdown("<span class='fx-note'>The index sits above zero almost "
                    "the whole sample - headlines skew promotional. In the "
                    "funds, this signal is lagged at least one trading day "
                    "before use, so a decision on day t only ever sees news "
                    "from day t−1 or earlier.</span>",
                    unsafe_allow_html=True)

    with tab_heat:
        Sm = S[sectors].resample("ME").mean()
        zmax = float(np.nanmax(np.abs(Sm.values))) or 0.01
        hm = base_fig("", "", height=460)
        hm.add_trace(go.Heatmap(
            z=Sm.T.values, x=Sm.index, y=Sm.columns,
            colorscale=[[0, FX["coral"]], [0.5, FX["panel2"]],
                        [1, FX["teal"]]],
            zmin=-zmax, zmax=zmax, ygap=2,
            hovertemplate="%{y} · %{x|%b %Y}: %{z:.3f}<extra></extra>",
            colorbar=dict(outlinewidth=0, tickfont=dict(color=FX["mute"]))))
        st.plotly_chart(hm, width="stretch")
        st.markdown("<span class='fx-note'>Monthly mean tone by sector - "
                    "coral is cooler, teal warmer. The one broad cold patch "
                    "is early 2020 (COVID); thin-coverage sectors like "
                    "Utilities run persistently warm.</span>",
                    unsafe_allow_html=True)

    with tab_lex:
        ls = A["lexstats"]
        if ls is None:
            st.info("Lexicon comparison table not found (results/tables/"
                    "sentiment_model_stats.csv).")
        else:
            rows = {r["model"]: r for _, r in ls.iterrows()}
            plain = next((v for k, v in rows.items() if "Plain" in k), None)
            fin = next((v for k, v in rows.items() if "Fin" in k), None)
            if plain is not None and fin is not None:
                d_neu = (fin["neutral_share"] - plain["neutral_share"]) * 100
                c = st.columns(4)
                kpi_card(c[0], "Neutral share, plain VADER",
                         f"{plain['neutral_share']:.1%}",
                         "Part A predicted 'roughly half' - "
                         "the forecast held.", accent="gold")
                kpi_card(c[1], "Neutral share, Fin-VADER",
                         f"{fin['neutral_share']:.1%}",
                         f"{d_neu:+.1f} pts once ~90 finance terms are "
                         f"scored", accent="teal")
                kpi_card(c[2], "Mean tone",
                         f"{plain['mean']:.3f} → {fin['mean']:.3f}",
                         "Slightly warmer on average")
                kpi_card(c[3], "Tone dispersion (std)",
                         f"{plain['std']:.3f} → {fin['std']:.3f}",
                         "More headlines take a side")
                shares = pd.DataFrame({
                    "Plain VADER": [plain["neg_share"],
                                    plain["neutral_share"],
                                    plain["pos_share"]],
                    "Fin-VADER": [fin["neg_share"], fin["neutral_share"],
                                  fin["pos_share"]]},
                    index=["Negative", "Neutral", "Positive"])
                sfig = base_fig("", "Share of ~147k headlines", height=360)
                for i, colname in enumerate(shares.columns):
                    sfig.add_trace(go.Bar(
                        x=shares.index, y=shares[colname], name=colname,
                        marker_color=[FX["steel"], FX["teal"]][i],
                        hovertemplate="%{x}: %{y:.1%}<extra>"
                                      + colname + "</extra>"))
                sfig.update_yaxes(tickformat=".0%")
                sfig.update_layout(barmode="group")
                st.plotly_chart(sfig, width="stretch")
                st.markdown("<span class='fx-note'>The extended lexicon "
                            "changes the <i>measurement</i>: several "
                            "thousand headlines move out of neutral, mostly "
                            "to positive. Whether that changes the "
                            "<i>trading</i> is the next tab.</span>",
                            unsafe_allow_html=True)

    with tab_fus:
        fu = A["fusion"]
        if fu is None:
            st.info("Fusion table not found (results/tables/"
                    "fusion_before_after.csv).")
        else:
            st.markdown(
                "<div class='fx-card acc-coral'><b>An honest negative "
                "result.</b> Tilting the Equity Max Sharpe fund by lagged "
                "sector tone costs about 0.04 of Sharpe and ~2 pts of "
                "turnover, for a slightly shallower drawdown. On this "
                "evidence FundX does not ship the sentiment tilt as a "
                "product claim - the index stays an analytic.</div>",
                unsafe_allow_html=True)
            ft = fu[["label", "sharpe", "net_sharpe_10bps", "max_drawdown",
                     "avg_turnover"]].copy()
            ft.columns = ["Fund", "Sharpe", "Net Sharpe (10 bps)",
                          "Max drawdown", "Avg turnover"]
            st.dataframe(ft, width="stretch", hide_index=True,
                         column_config={
                             "Sharpe": st.column_config.NumberColumn(
                                 format="%.3f"),
                             "Net Sharpe (10 bps)":
                                 st.column_config.NumberColumn(format="%.3f"),
                             "Max drawdown": st.column_config.NumberColumn(
                                 format="percent"),
                             "Avg turnover": st.column_config.NumberColumn(
                                 format="percent")})
            over = [("equity_max_sharpe", "Base (no tilt)", FX["teal"],
                     None, 2.4),
                    ("equity_max_sharpe_tilt_finvader", "Tilt (Fin-VADER)",
                     FX["gold"], "dash", 1.7),
                    ("equity_max_sharpe_tilt_vader", "Tilt (plain VADER)",
                     FX["sky"], "dot", 1.4)]
            ofig = base_fig("", "Value of $1 (USD)", height=380)
            for fid_, name, colr, dash, wdt in over:
                if fid_ in RET.columns:
                    gg = growth(RET[fid_])
                    ofig.add_trace(go.Scatter(
                        x=gg.index, y=gg, name=name,
                        line=dict(color=colr, width=wdt, dash=dash),
                        hovertemplate="%{y:$.2f}<extra>" + name +
                                      "</extra>"))
            st.plotly_chart(ofig, width="stretch")
            st.markdown("<span class='fx-note'>Three lines, one story: the "
                        "tilt's entire effect is smaller than the line "
                        "width.</span>", unsafe_allow_html=True)

# =================================================================== METHODOLOGY
else:
    page_header("Under the hood", "Methodology & data",
                "Everything the funds and analytics rest on - and the "
                "honesty box.")

    steps = [("Ingest", "50 US equities + 10 cryptos + ~147k headlines "
              "(2020–2023), via the provided data helper."),
             ("Score", "Each headline scored (VADER + a finance-lexicon "
              "extension), averaged to sector-day tone."),
             ("Optimise", "Five rules per family, re-estimated monthly from "
              "strictly past data."),
             ("Backtest", "Walk-forward, out-of-sample, costs modelled at "
              "10 bps × turnover."),
             ("Serve", "This app reads the precomputed results/ artifacts - "
              "nothing is re-optimised at runtime.")]
    cols = st.columns(len(steps))
    for c_, (i, (t, s)) in zip(cols, enumerate(steps, start=1)):
        c_.markdown(f"<div class='fx-card' style='min-height:9.5rem'>"
                    f"<div class='fx-step-n'>{i}</div>"
                    f"<div style='margin-top:0.4rem'><b>{t}</b></div>"
                    f"<div class='fx-note'>{s}</div></div>",
                    unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""
<div class='fx-card'>

**The product.** FundX offers systematically managed funds. Each fund is one
(asset family, optimisation rule) pair - for example *Combined Minimum
Variance* - built from daily adjusted-close returns of 50 US large-caps
(10 sectors) and 10 major cryptocurrencies, 2020-2023.

**The rules.** Five methods: Equal Weight, Minimum Variance, Maximum Sharpe,
Risk Parity (equal risk contribution), and Hierarchical Risk Parity. All are
long-only and fully invested. Minimum Variance and Maximum Sharpe explicitly
enforce a 20% per-asset cap (40% for crypto-only); Equal Weight is below those
thresholds by construction, while ERC and HRP are left unclipped so their
defining rules are not distorted (their realised weights remain below the same
thresholds in this sample). Min Variance, Max Sharpe and ERC use 10% diagonal
covariance shrinkage for numerical stability. Risk-free rate 0.

**Out-of-sample discipline.** Walk-forward backtest: on the first trading day
of each month, weights are re-estimated from the previous 252 trading days
(365 calendar days for crypto-only funds) - strictly before that day - then
held, drifting with returns, until the next rebalance. Equity-calendar funds
annualise with 252, crypto-only funds with 365. What you see is what the rule
would have earned on data it had not seen.

**Sentiment.** ~147k deduplicated headlines are scored per headline (VADER, and
a finance-lexicon extension), averaged per ticker-day, and equal-weighted into
sector indices. The sentiment-tilt funds multiply each stock's weight by
1 + 0.25 × its sector's lagged, cross-sectionally z-scored 21-day tone
(clipped to [0.6, 1.4]) and renormalise.

**Costs & fees.** Fact sheets show gross returns; the report also nets a
10 bps × turnover transaction-cost model. FundX's business model is an
illustrative {MGMT_FEE:.2%} p.a. management fee on invested balances.

**Honesty box.** Prices and headlines end 2023-12-31; nothing here is
investment advice; out-of-sample is not a guarantee - it is just a fair test.
</div>
""", unsafe_allow_html=True)

    st.markdown("#### Take the artifacts with you")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Performance metrics (CSV)", PERF.to_csv().encode(),
                       file_name="performance_metrics.csv", mime="text/csv",
                       key="dl_m1")
    d2.download_button("Daily fund returns (CSV)", RET.to_csv().encode(),
                       file_name="fund_returns.csv", mime="text/csv",
                       key="dl_m2")
    d3.download_button("Sector sentiment index (CSV)",
                       A["senti"].to_csv().encode(),
                       file_name="sector_sentiment_index.csv",
                       mime="text/csv", key="dl_m3")

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

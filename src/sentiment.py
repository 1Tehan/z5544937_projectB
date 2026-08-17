"""Station 3 - the sentiment model and sector index.

Two scorers are built and compared (the lexicon extension is the innovation I
flagged in my Part A, Section 5):

  1. plain VADER (nltk) on the RAW headline text - casing, punctuation and
     stopwords kept, because VADER's heuristics rely on them;
  2. Fin-VADER: VADER with a finance-specific lexicon merged in. Part A's
     vocabulary analysis showed plain VADER leaves roughly half of finance
     headlines neutral, and that market words like 'downgrade', 'beat',
     'guidance cut' or 'all-time high' carry direction VADER does not see.
     The added terms use VADER's own -4..+4 valence scale and were drafted
     with my AI assistant, then reviewed and re-scored by me line by line
     (Loughran-McDonald-style finance tone words).

Aggregation: headline compound -> ticker-day mean -> sector-day EQUAL-WEIGHT
mean across the tickers that actually have news that day. Ticker-days with no
headlines are DROPPED from the sector average (not forced to 0): zero is a
model output meaning 'balanced tone', while 'no headline' is missing data, and
padding thin sectors with zeros would mechanically drag their index to zero.

Look-ahead: the index is an analytic here; the LAG (>= 1 trading day) is
applied where it is used for trading, in src/fusion.py.
"""
from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------- finance lexicon
# VADER valence scale: -4 (extremely negative) .. +4 (extremely positive).
# Multi-word entries are handled by pre-mapping phrases to single tokens.
FINANCE_LEXICON: dict[str, float] = {
    # earnings / guidance
    "beat": 1.8, "beats": 1.8, "outperform": 1.9, "outperforms": 1.9,
    "upgrade": 2.1, "upgraded": 2.1, "upgrades": 2.1,
    "downgrade": -2.2, "downgraded": -2.2, "downgrades": -2.2,
    "miss": -1.8, "misses": -1.8, "shortfall": -1.9,
    "guidance_cut": -2.4, "guidance_raise": 2.2, "preannounce": -0.8,
    "restated": -1.6, "restatement": -1.8, "writedown": -2.3,
    "impairment": -2.1, "provision": -0.9,
    # price action
    "surge": 2.3, "surges": 2.3, "soar": 2.6, "soars": 2.6,
    "rally": 2.0, "rallies": 2.0, "rebound": 1.6, "rebounds": 1.6,
    "plunge": -2.6, "plunges": -2.6, "tumble": -2.2, "tumbles": -2.2,
    "slump": -2.0, "slumps": -2.0, "sink": -1.9, "sinks": -1.9,
    "selloff": -2.1, "sell_off": -2.1, "rout": -2.5, "crash": -3.1,
    "record_high": 2.6, "all_time_high": 2.7, "record_low": -2.4,
    "new_high": 2.0, "new_low": -2.0, "breakout": 1.5, "correction": -1.4,
    "bear_market": -2.2, "bull_market": 2.0, "short_squeeze": 0.9,
    # corporate events
    "dividend_hike": 2.2, "dividend_cut": -2.6, "buyback": 1.7,
    "acquisition": 0.9, "merger": 0.8, "takeover": 0.9, "spinoff": 0.5,
    "bankruptcy": -3.4, "default": -3.0, "insolvency": -3.2,
    "delisting": -2.6, "dilution": -1.5, "layoffs": -2.3, "layoff": -2.3,
    "restructuring": -1.2, "recall": -2.0, "probe": -1.8,
    "investigation": -1.7, "lawsuit": -1.9, "fine": -1.6, "fines": -1.6,
    "settlement": -0.6, "fraud": -3.3, "scandal": -2.8,
    # macro / tone
    "headwinds": -1.6, "tailwinds": 1.6, "downturn": -1.9, "recession": -2.4,
    "inflation": -1.0, "stagflation": -2.3, "tightening": -0.9,
    "stimulus": 1.2, "bullish": 2.1, "bearish": -2.1,
    "overweight": 1.4, "underweight": -1.4, "oversold": 0.6,
    "overbought": -0.6, "volatile": -1.0, "turmoil": -2.2, "jitters": -1.3,
    "weak_demand": -1.9, "strong_demand": 1.9, "profit_warning": -2.7,
    "guidance": 0.0, "downtrend": -1.5, "uptrend": 1.5,
}

# phrases mapped to the single tokens above before scoring
_PHRASES = {
    "guidance cut": "guidance_cut", "cuts guidance": "guidance_cut",
    "cut guidance": "guidance_cut", "raises guidance": "guidance_raise",
    "raised guidance": "guidance_raise", "record high": "record_high",
    "all-time high": "all_time_high", "all time high": "all_time_high",
    "record low": "record_low", "new high": "new_high", "new low": "new_low",
    "bear market": "bear_market", "bull market": "bull_market",
    "short squeeze": "short_squeeze", "dividend hike": "dividend_hike",
    "dividend cut": "dividend_cut", "sell-off": "sell_off",
    "weak demand": "weak_demand", "strong demand": "strong_demand",
    "profit warning": "profit_warning",
}


def _prep_finance_text(text: pd.Series) -> pd.Series:
    """Map finance phrases to single lexicon tokens (case-insensitive)."""
    out = text.astype(str)
    for phrase, token in _PHRASES.items():
        out = out.str.replace(phrase, token, case=False, regex=False)
    return out


def make_analyzers():
    """Return (plain VADER, Fin-VADER). Import is local: build-time only,
    never in the deployed app (requirements-dev.txt holds nltk)."""
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer

    plain = SentimentIntensityAnalyzer()
    fin = SentimentIntensityAnalyzer()
    fin.lexicon.update(FINANCE_LEXICON)
    return plain, fin


def explode_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per headline from the Station 2 panel.

    Part A's assemble_headline_panel joins titles with ' || ' precisely "so the
    Part B sentiment model can re-split and score them" (its docstring). This is
    that round-trip: split the joined string back into individual raw titles.
    run_part_b asserts the exploded row count equals the deduplicated headline
    count (146,836 on the real data), so nothing is lost or invented here.
    """
    df = panel.rename(columns={"date": "trade_date"}).copy()
    df["title"] = df["headlines"].astype(str).str.split(" || ", regex=False)
    df = df.explode("title", ignore_index=True)
    return df[["trade_date", "ticker", "sector", "title"]]


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Score every headline with both models.

    Input: the Station 2 headline panel (trade_date, ticker, sector, title).
    Output: adds compound_vader and compound_finvader columns.
    """
    plain, fin = make_analyzers()
    out = panel.copy()
    titles = out["title"].astype(str)
    fin_titles = _prep_finance_text(titles)
    out["compound_vader"] = [plain.polarity_scores(t)["compound"] for t in titles]
    out["compound_finvader"] = [fin.polarity_scores(t)["compound"] for t in fin_titles]
    return out


def neutral_stats(scored: pd.DataFrame, thresh: float = 0.05) -> pd.DataFrame:
    """Share of headlines each model leaves neutral (|compound| < thresh) -
    the evidence exhibit for the lexicon extension."""
    rows = []
    for col, label in [("compound_vader", "Plain VADER"),
                       ("compound_finvader", "Fin-VADER (extended lexicon)")]:
        s = scored[col]
        rows.append({"model": label,
                     "neutral_share": float((s.abs() < thresh).mean()),
                     "mean": float(s.mean()), "std": float(s.std()),
                     "pos_share": float((s >= thresh).mean()),
                     "neg_share": float((s <= -thresh).mean())})
    return pd.DataFrame(rows)


def ticker_day_sentiment(scored: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean compound per (trade_date, ticker) + headline count."""
    g = (scored.groupby(["trade_date", "ticker", "sector"])
               .agg(sentiment=(col, "mean"), n_headlines=("title", "size"))
               .reset_index())
    return g


def sector_sentiment_index(ticker_day: pd.DataFrame) -> pd.DataFrame:
    """Daily sector index: equal-weight mean over tickers WITH news that day.

    Wide output: date x [10 sectors + ALL]. No-news ticker-days are dropped
    (see module docstring for the justification).
    """
    sec = (ticker_day.groupby(["trade_date", "sector"])["sentiment"]
                     .mean().unstack("sector").sort_index())
    sec["ALL"] = ticker_day.groupby("trade_date")["sentiment"].mean()
    sec.index.name = "date"
    return sec


def coverage_series(ticker_day: pd.DataFrame) -> pd.DataFrame:
    """Daily headline coverage (for the app): total headlines + tickers covered."""
    g = (ticker_day.groupby("trade_date")
                   .agg(headlines=("n_headlines", "sum"),
                        tickers_covered=("ticker", "nunique")))
    g.index.name = "date"
    return g

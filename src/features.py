"""Part A Station 2 logic is reused unchanged in Part B; this line is only a provenance note.

Station 2 - Feature engineering + headline text assembly.

Returns are the one feature the Part B optimiser actually needs. We also assemble
the headlines into a daily panel aligned to the equity trading calendar, and do a
DESCRIPTIVE look at the text (counts only). We do NOT score sentiment here - that
is the Station 3 model in Part B.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

# Small bundled word lists so Part A needs no nltk. Stopwords are used ONLY for the
# top-terms figure; the stored headline panel keeps the raw text untouched.
STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or
that the to was were will with this these those they you your we our us he she his
her their them then than but not no can could would should may might over under up
down out about after before more most new says say said amid via inc corp co ltd""".split())

# A small sentiment vocabulary for the DESCRIPTIVE word count (not scoring). Swap in
# nltk's VADER lexicon keys if you prefer a bigger vocabulary (see note in report).
SENTIMENT_WORDS = set("""gain gains gained rise rises rising rose surge surges soar
soars jump jumps rally beat beats strong growth profit profits up higher record high
outperform bullish upgrade boost win wins positive top fall falls fell drop drops
plunge plunges slump sink sinks tumble crash miss misses weak loss losses down lower
cut cuts decline declines slide fear fears risk risks warn warns bearish downgrade
lawsuit probe negative low worst slump concern concerns""".split())


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Simple daily returns per ticker (long format): [ticker, date, ret].

    Uses adjClose (splits/dividends adjusted) and pct_change WITHIN each ticker,
    so the first observation per ticker is NaN (there is no prior day to compare).
    """
    p = prices.sort_values(["ticker", "date"]).copy()
    p["ret"] = p.groupby("ticker")[price_col].pct_change()
    return p[["ticker", "date", "ret"]]


def to_wide(returns_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long returns to a wide date x ticker matrix."""
    return returns_long.pivot(index="date", columns="ticker", values="ret").sort_index()


def combined_returns_panel(eq_returns_long, cr_returns_long,
                           calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Left-merge crypto returns onto the equity trading calendar.

    CRITICAL ORDER: returns are computed inside each panel FIRST (on their own
    calendar), THEN crypto is reindexed to equity dates. This intentionally drops
    weekend-only crypto moves - a fund trading on equity days could not act on them.
    We never merge price *levels* across calendars and difference afterwards (that
    would fabricate returns across the weekend gap).
    """
    eq_wide = to_wide(eq_returns_long).reindex(calendar)
    cr_wide = to_wide(cr_returns_long).reindex(calendar)  # left-join onto equity days
    combined = eq_wide.join(cr_wide, how="left")
    combined.index.name = "date"
    return combined


def descriptive_stats_returns(eq_returns_long, cr_returns_long) -> pd.DataFrame:
    """Return descriptive stats by asset class (required exhibit).

    Reports daily mean/vol plus ANNUALISED figures using the correct factor
    (252 for equities, 365 for crypto), and a shape measure (skew, kurtosis).
    """
    def block(df, name, ann):
        r = df["ret"].dropna()
        return {
            "asset_class": name, "n_obs": len(r),
            "mean_daily": r.mean(), "vol_daily": r.std(),
            "ann_return": r.mean() * ann, "ann_vol": r.std() * np.sqrt(ann),
            "min": r.min(), "max": r.max(),
            "skew": r.skew(), "kurtosis": r.kurtosis(),
        }
    return pd.DataFrame([
        block(eq_returns_long, "Equity", 252),
        block(cr_returns_long, "Crypto", 365),
    ])


# --------------------------------------------------------------------------- #
# Headline text assembly
# --------------------------------------------------------------------------- #
def _map_to_trading_day(dates: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Map each date to the same trading day, else the NEXT trading day."""
    cal = np.array(calendar.values, dtype="datetime64[ns]")
    idx = np.searchsorted(cal, dates.values.astype("datetime64[ns]"), side="left")
    idx = np.clip(idx, 0, len(cal) - 1)          # dates past the last trading day -> last
    mapped = cal[idx]
    # searchsorted 'left' already gives same-day if it exists, else the next day.
    return pd.Series(mapped, index=dates.index)


def assemble_headline_panel(news: pd.DataFrame,
                            calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily headline panel per (trading_day, ticker, sector).

    Each headline is aligned to its equity trading day (same day if it trades, else
    the next trading day). Raw titles are preserved and joined with ' || ' so the
    Part B sentiment model can re-split and score them. n_headlines counts the flow.
    """
    df = news.copy()
    df["trading_day"] = _map_to_trading_day(df["date"], calendar)
    panel = (df.groupby(["trading_day", "ticker", "sector"])
               .agg(n_headlines=("title", "size"),
                    headlines=("title", lambda s: " || ".join(s.astype(str))))
               .reset_index()
               .rename(columns={"trading_day": "date"}))
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Descriptive text exploration (counts only - NOT sentiment scoring)
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text).lower())


def articles_over_time(news: pd.DataFrame, freq: str = "MS") -> pd.DataFrame:
    """Headline count over time (default monthly)."""
    s = news.set_index("date").resample(freq).size()
    return s.rename("n_headlines").reset_index()


def top_terms(news: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Most frequent content words across all headlines (stopwords removed).

    Stopword removal is for THIS descriptive figure only - the stored panel keeps
    raw text for VADER.
    """
    c = Counter()
    for t in news["title"]:
        c.update(w for w in _tokenize(t) if w not in STOPWORDS and len(w) > 2)
    return pd.DataFrame(c.most_common(n), columns=["term", "count"])


def sentiment_word_count(news: pd.DataFrame) -> pd.DataFrame:
    """Count occurrences of sentiment-bearing vocabulary (descriptive, not scoring).

    This is vocabulary counting - explicitly a Part A task. It measures how much
    sentiment-relevant language the headlines carry; it assigns NO score.
    """
    total_tokens = 0
    sent_hits = Counter()
    for t in news["title"]:
        for w in _tokenize(t):
            total_tokens += 1
            if w in SENTIMENT_WORDS:
                sent_hits[w] += 1
    sent_total = sum(sent_hits.values())
    top = "; ".join(f"{w}({n})" for w, n in sent_hits.most_common(10))
    return pd.DataFrame([{
        "n_headlines": len(news),
        "n_tokens": total_tokens,
        "n_sentiment_words": sent_total,
        "pct_sentiment_words": round(100 * sent_total / max(total_tokens, 1), 2),
        "top_sentiment_terms": top,
    }])

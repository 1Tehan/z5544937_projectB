"""Part A Station 1 logic is reused unchanged in Part B; this line is only a provenance note.

Station 1 - ETL: load, clean, and integrity-check the data.

All raw data enters through src.data_access (the provided, frozen helper). Nothing
here touches the network directly. Every function returns a *clean* frame plus,
where relevant, a small report you can drop straight into results/ as an exhibit.

Design choices that matter for the marks (see PROJECT_BRIEF.md, Station 1):
  * Prices should be unique by (ticker, date). News has many rows per (ticker,
    date), so its duplicate check is on (ticker, date, title) instead.
  * The sample is capped at 2023-12-31 (crypto has 10 stray rows dated 2024-01-01).
  * News dates are timezone-aware UTC; price dates are tz-naive. We normalise the
    news timezone here so later merges don't error.
  * Genuine extreme returns are real market events - we FLAG and KEEP them, we do
    not delete them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import data_access

SAMPLE_END = pd.Timestamp("2023-12-31")  # coverage cap: data ends in 2023


# --------------------------------------------------------------------------- #
# Loaders + cleaning
# --------------------------------------------------------------------------- #
def load_clean_equities() -> pd.DataFrame:
    """50 US equities, daily OHLCV + adjClose + sector. Cleaned and capped."""
    df = data_access.load_equity_prices().copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= SAMPLE_END]
    # Prices must be unique by (ticker, date). Keep the first if any exact dup.
    df = df.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"])
    return df.reset_index(drop=True)


def load_clean_crypto() -> pd.DataFrame:
    """10 cryptos, daily OHLCV + adjClose (365-day calendar). Cleaned and capped."""
    df = data_access.load_crypto_prices().copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= SAMPLE_END]  # drops the 10 stray 2024-01-01 rows
    df = df.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"])
    return df.reset_index(drop=True)


def load_clean_news() -> pd.DataFrame:
    """Headlines only. De-duplicated on (ticker, date, title); timezone normalised.

    The raw `date` is tz-aware UTC. We convert to a tz-naive daily date so it can
    align to the (tz-naive) price calendar later. We KEEP the raw `title` text
    exactly as-is - do not strip stopwords, VADER (Part B) relies on them.
    """
    df = data_access.load_news_headlines().copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)         # ensure UTC-aware
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()  # -> tz-naive midnight
    df = df[df["date"] <= SAMPLE_END]
    # News legitimately has many rows per (ticker, date); only EXACT repeats of the
    # same headline are duplicates.
    df = df.drop_duplicates(subset=["ticker", "date", "title"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def trading_calendar(equities: pd.DataFrame) -> pd.DatetimeIndex:
    """The equity trading days - the calendar everything else aligns to."""
    return pd.DatetimeIndex(sorted(equities["date"].unique()))


# --------------------------------------------------------------------------- #
# Integrity checks (missing dates, duplicates, outliers)
# --------------------------------------------------------------------------- #
def _missing_date_audit(prices: pd.DataFrame, label: str) -> dict:
    """Per-ticker coverage vs the union of dates in the panel."""
    all_dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    gaps = {}
    for tkr, g in prices.groupby("ticker"):
        have = pd.DatetimeIndex(sorted(g["date"].unique()))
        missing = all_dates.difference(have)
        if len(missing):
            gaps[tkr] = len(missing)
    return {
        "dataset": label,
        "n_tickers": prices["ticker"].nunique(),
        "n_dates_union": len(all_dates),
        "tickers_with_gaps": len(gaps),
        "max_gap_days": max(gaps.values()) if gaps else 0,
    }


def _duplicate_report(equities_raw, crypto_raw, news_raw) -> pd.DataFrame:
    rows = [
        {"dataset": "equity_prices", "duplicate_key": "ticker+date",
         "n_duplicates": int(equities_raw.duplicated(subset=["ticker", "date"]).sum()),
         "resolution": "dropped exact duplicates (prices must be unique per ticker-date)"},
        {"dataset": "crypto_prices", "duplicate_key": "ticker+date",
         "n_duplicates": int(crypto_raw.duplicated(subset=["ticker", "date"]).sum()),
         "resolution": "dropped exact duplicates"},
        {"dataset": "news_headlines", "duplicate_key": "ticker+date+title",
         "n_duplicates": int(news_raw.duplicated(subset=["ticker", "date", "title"]).sum()),
         "resolution": "dropped exact duplicate headlines (many rows per ticker-date is normal)"},
    ]
    return pd.DataFrame(rows)


def outlier_screen(returns_long: pd.DataFrame, z_thresh: float = 8.0) -> pd.DataFrame:
    """Flag extreme daily returns (|z| > z_thresh within each ticker). KEEP them.

    We report the flagged observations as evidence but do NOT delete them: the
    genuine extremes here are real events (e.g. crypto crashes, single-stock news).
    """
    out = returns_long.dropna(subset=["ret"]).copy()
    stats = out.groupby("ticker")["ret"].agg(["mean", "std"]).rename(
        columns={"mean": "mu", "std": "sd"})
    out = out.merge(stats, on="ticker", how="left")
    out["z"] = (out["ret"] - out["mu"]) / out["sd"]
    flagged = out[out["z"].abs() > z_thresh].copy()
    flagged = flagged.reindex(flagged["z"].abs().sort_values(ascending=False).index)
    return flagged[["ticker", "date", "ret", "z"]].reset_index(drop=True)


def dataset_inventory(equities, crypto, news) -> pd.DataFrame:
    """Rows, date span, frequency, coverage for each dataset (required exhibit)."""
    def span(df):
        return f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}"
    rows = [
        {"dataset": "equity_prices", "rows": len(equities), "date_span": span(equities),
         "frequency": "trading-daily (~252/yr)", "n_tickers": equities["ticker"].nunique(),
         "n_sectors": equities["sector"].nunique()},
        {"dataset": "crypto_prices", "rows": len(crypto), "date_span": span(crypto),
         "frequency": "daily (365/yr)", "n_tickers": crypto["ticker"].nunique(),
         "n_sectors": 0},
        {"dataset": "news_headlines", "rows": len(news), "date_span": span(news),
         "frequency": "event/daily", "n_tickers": news["ticker"].nunique(),
         "n_sectors": news["sector"].nunique()},
    ]
    return pd.DataFrame(rows)

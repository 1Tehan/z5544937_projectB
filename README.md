# FundX - FinTech Project Part B (z5544937)

FundX is my prototype investment app (named in Part A): systematically managed
multi-asset funds with a news-sentiment lens. This folder is the complete
Part B deliverable (DFF Stations 3-4): the out-of-sample funds, the sentiment
model + sector index, the fusion extension, the deployed Streamlit app, my
report, and my AI workflow pack. It is also the GitHub repository the app
deploys from (entrypoint `streamlit_app.py` at the root).

## How to run

    pip install -r requirements.txt -r requirements-dev.txt   # dev adds nltk (VADER)
    python scripts/run_part_b.py        # reproduces EVERYTHING into results/ (a few minutes)
    streamlit run streamlit_app.py      # the FundX app (reads precomputed results/)
    python scripts/check_handin.py      # pre-hand-in checks
    git status                          # nothing raw/secret staged

Raw data always loads through `src/data_access.py` (hosted ZIP, cached); no raw
data is ever committed. The deployed app reads only the committed artifacts in
`results/` - it never runs VADER or an optimiser (nltk lives in
requirements-dev.txt only).

## What I built

- **17 funds, walk-forward out-of-sample** (`src/portfolios.py`): equity-only,
  crypto-only and combined families x five methods (equal weight, minimum
  variance, maximum Sharpe, risk parity/ERC, hierarchical risk parity), monthly
  rebalance, weights strictly from past data, drifting between rebalances, with
  turnover and a 10 bps transaction-cost model. Crypto-only funds live on the
  365-day calendar (annualised x365); equity-calendar funds use x252.
- **Sentiment** (`src/sentiment.py`): ~147k deduplicated headlines scored with
  plain VADER AND my Fin-VADER (finance lexicon extension flagged in Part A),
  aggregated to daily equal-weight sector indices.
- **Fusion** (`src/fusion.py`): a look-ahead-safe sector sentiment tilt on the
  Equity Max-Sharpe fund, compared before vs after with both lexicons.
- **The app** (`streamlit_app.py`): compare funds, per-fund fact sheets
  (growth of $1, drawdown, Sharpe, holdings, weights over time), an allocation
  builder with a fee illustration, sentiment analytics, and methodology - all
  in the FundX design system (`src/plotstyle.py` for the report figures).
- **Artifacts** the markers and app rely on (exact names):
  `results/data/fund_returns.csv`, `results/data/fund_weights.csv`,
  `results/data/sector_sentiment_index.csv`,
  `results/tables/performance_metrics.csv` (+ catalog, Fin-VADER index,
  coverage, fusion table, lexicon stats, and all report figures).

## Deploy + hand in

See `docs/STUDENT_DEPLOY.md` and PROJECT_BRIEF.md Appendix D. In short: commit
this folder (results/ included) to a NEW private GitHub repo, connect it on
share.streamlit.io with entrypoint `streamlit_app.py`, make the repo PUBLIC at
hand-in, and submit the zip + repo link + live URL.

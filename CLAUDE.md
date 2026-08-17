# CLAUDE.md
Notes on how i used AI on part b. Tehan, z5544937. Same app as part a - FundX, pre-made rule-based funds with clear fact sheets instead of stock picking. Part b builds the actual funds (out of sample), the sentiment index, the fusion tilt and the deployed app.

Part a's code is the foundation here, literally: src/etl.py and src/features.py in this folder are my part a files carried over verbatim (cleaning rules, merge order, the headline panel). Everything new sits on top of them. As with part a, i used claude to draft most of the code and scaffolding, but i decided the design, reviewed everything, and the report/interpretation is mine.

## rules i gave it (several exist because of mistakes i caught in part a or here):
- reuse my part a modules as-is, don't rewrite station 1-2. It originally rebuilt its own versions from just my part a report; once i gave it the actual zip i had it swap in my real files
- no look-ahead anywhere. Weights at a rebalance date only use returns strictly before that date, and sentiment used for trading is lagged at least 1 trading day. Show me where the strict before-t slice is
- same calendar rule as part a: returns inside each dataset first, then crypto joined onto the stock calendar. Never merge prices across calendars and difference after (part a lesson - fake weekend returns)
- annualise the mean/covariance before running the optimiser and shrink the covariance a bit. Raw daily numbers are so small the solver just gives up and returns equal weights for every method, which looks fine until you compare them
- assert the five methods actually produce different weights so a silent solver stall can't slip through
- equity funds annualise x252, crypto-only funds x365 and live on their own calendar
- the deployed app only READS results/. it never imports nltk, never runs an optimiser. Heavy stuff happens once in scripts/run_part_b.py
- exact output filenames from the brief - fund_returns.csv, fund_weights.csv, sector_sentiment_index.csv, performance_metrics.csv - the app and the marking depend on them
- headline text stays raw for VADER (part a rule), and days where a sector has no news are missing data, not zero sentiment
- never commit raw data or secrets, everything loads through data_access.py

## decisions that are mine (it should state them, not quietly change them):
- long-only, fully invested, 20% cap (40% crypto-only), rf = 0, monthly rebalance, 252-day lookback (365 crypto)
- five methods: equal weight, min variance, max sharpe, risk parity, HRP (the newer one)
- the fin-vader lexicon is the extension i flagged in my part a section 5. AI drafted candidate words on vader's -4..+4 scale, i went through the list and re-scored/cut them
- tilt rule: lag 1 day, 21-day mean, z-score across sectors, weight x clip(1 + 0.25z, 0.6, 1.4), renormalise. If it underperforms i report that honestly, the rubric doesn't require the tilt to win
- costs: 10bps x turnover, turnover measured against drifted weights

## how i review its work:
- rerun python scripts/run_part_b.py myself on the real data and read every table and figure it writes
- check the anchors against my own part a numbers: 146,836 headlines after dedup (2,847 dupes), 50,300 equity rows, 1,006 trading days
- spot check one rebalance by hand (estimation window must end the day before) and recompute one fund's sharpe from fund_returns.csv
- click through all five app pages locally and on the deployed url, then python scripts/check_handin.py
- report writing and interpretation is mine alone

## honest reflection:
Big time saver again - 17 backtests, two sentiment models over ~147k headlines and a five page app is a lot of boilerplate i didn't have to type. The same weakness as part a showed up in new clothes though: in part a it invented reasons for outlier returns; here the equivalent risk is that everything it built was first validated on synthetic stand-in data (its sandbox can't reach the course data hosts), so every number was provisional until i reran on the real data myself. It also initially reconstructed "my" part a conventions from my report instead of my code, which was close but not the real thing. Same lesson as the KPMG/EY notes: fast drafts, but accuracy and the final call stay with me.

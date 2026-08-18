# CLAUDE.md
Notes on how i used AI on part b. Tehan, z5544937. Same app as part a - FundX, pre-made rule-based funds with clear fact sheets instead of stock picking. Part b builds the actual funds (out of sample), the sentiment index, the fusion tilt and the deployed app.

Part a's code is the foundation here: src/etl.py and src/features.py preserve my Part A implementation (cleaning rules, merge order and the headline panel); I only added a short provenance note at the top of each file for Part B. Everything new sits on top of them. As with part a, i used Claude to draft much of the code and scaffolding. I decided the design and remain responsible for the final interpretation. ChatGPT also assisted with late-stage report drafting, formatting and consistency checks; that use is explicitly logged in `ai/prompt_log_02_final_report_handover.md` and the final audit log, rather than being presented as unaided writing.

## rules i gave it (several exist because of mistakes i caught in part a or here):
- reuse my Part A Station 1-2 logic unchanged; provenance comments are fine, but do not rewrite the implementation. It originally rebuilt its own versions from just my part a report; once i gave it the actual zip i had it swap in my real files
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
- long-only and fully invested. Min-variance and max-Sharpe explicitly use the 20% cap (40% crypto-only); Equal Weight is below it by construction, while ERC/HRP are left unclipped so their defining rules are preserved. In the realised OOS weights both remain below the same thresholds. The sentiment tilt clips its multiplier rather than imposing a final stock-weight cap. rf = 0, monthly rebalance, 252-day lookback (365 crypto)
- five methods: equal weight, min variance, max sharpe, risk parity, HRP (the newer one)
- the fin-vader lexicon is the extension i flagged in my part a section 5. AI drafted candidate words on vader's -4..+4 scale, i went through the list and re-scored/cut them
- tilt rule: lag 1 day, 21-day mean, z-score across sectors, weight x clip(1 + 0.25z, 0.6, 1.4), renormalise. If it underperforms i report that honestly, the rubric doesn't require the tilt to win
- costs: 10bps x turnover, turnover measured against drifted weights

## how i review its work:
- rerun python scripts/run_part_b.py myself on the real data and read every table and figure it writes
- check the anchors against my own part a numbers: 146,836 headlines after dedup (2,847 dupes), 50,300 equity rows, 1,006 trading days
- spot check one rebalance by hand (estimation window must end the day before) and recompute one fund's sharpe from fund_returns.csv
- click through all seven app pages locally and on the deployed url, then python scripts/check_handin.py
- final economic interpretation decisions are mine; AI may help organise or edit wording, and any late-stage drafting assistance is logged in ai/

## honest reflection:
Big time saver again - 17 backtests, two sentiment models over ~147k headlines and a seven page app is a lot of boilerplate i didn't have to type. The same weakness as part a showed up in new clothes though: in part a it invented reasons for outlier returns; here the equivalent risk is that everything it built was first validated on synthetic stand-in data (its sandbox can't reach the course data hosts), so every number was provisional until i reran on the real data myself. It also initially reconstructed "my" part a conventions from my report instead of my code, which was close but not the real thing. Same lesson as the KPMG/EY notes: fast drafts, but accuracy and the final call stay with me.


## late-stage report handover
On 18 Aug I also used ChatGPT as a second assistant to inspect the current zip, check the saved outputs and assemble/format a report draft from the claims already fixed by the real-data run. I logged that separately in `ai/prompt_log_02_final_report_handover.md` so the report-stage AI use is explicit rather than contradicting this file.

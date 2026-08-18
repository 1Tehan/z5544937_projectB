# Prompt log -- Part B end-to-end build

## What I wanted to do
Build the whole of part b in one pass on top of my part a foundation: the walk-forward out-of-sample funds (equity / crypto / combined, five methods), the VADER and Fin-VADER sentiment index, the fusion tilt, every required artifact and figure, and the FundX streamlit app.

## What I asked for
I gave claude the brief, the starter, my part a report and later my actual part a zip, and asked it to finish the assignment following my rules in CLAUDE.md (AI use is allowed and graded in this course). The extensions were the ones i flagged in my part a section 5 - the finance lexicon, and treating the extreme-move screen as a data-quality feature.

## What went wrong / what I fixed
- Its sandbox could not reach the course data hosts, so it built and tested everything on a synthetic stand-in with the same shape (same tickers, calendars, tz quirk, 2,847 injected duplicate headlines). That means every number it produced was provisional. The fix is procedural: i rerun `python scripts/run_part_b.py` on the real data myself and only those regenerated results get submitted.
- Before it had my part a zip it reconstructed station 1-2 from my report alone - close to my rules but not my code. Once i uploaded the actual zip it replaced src/etl.py and src/features.py with my actual Part A implementations (logic unchanged apart from a Part B provenance note) and rewired run_part_b to my function signatures (my daily_returns is long format, its version was wide). It also added the ' || ' re-split my part a headline panel was designed for, with an assert that the exploded row count equals the deduplicated headline count.
- It flagged its own solver trap: feeding raw daily means/covariances to SLSQP stalls it into near-identical weights for every method. The pipeline annualises and shrinks the covariance first and asserts the five methods genuinely differ.
- The lexicon valences are hand-set, not estimated. It drafted the word list; the scores that ship are the ones i reviewed. Evidence is reported two ways (neutral-rate change AND the downstream fund effect) so the extension isn't oversold.
- The tilt parameters (0.25 gamma, clip 0.6-1.4) are design choices, not optimised, and the tilt may underperform the base fund - that gets reported honestly either way.
- Small one: a deprecated streamlit argument (use_container_width) surfaced in testing and was replaced before it becomes a breakage on streamlit cloud.

## How I checked
1. I reran/checked the real-data outputs and the Part A anchors reappeared exactly: 146,836 headlines after deduplication, 2,847 duplicates removed, 50,300 equity rows and 1,006 trading days.
2. My Part A report predicted that plain VADER would leave roughly half of the headlines neutral. The real result is 49.6% neutral for plain VADER and 45.2% for Fin-VADER.
3. I tested look-ahead behaviour by planting a +60% return on a rebalance date. The target weights at that same date did not change, which confirms that date t is excluded from its own estimation window. The first diagnostic for the following rebalance was inconclusive because the 0.90 test cap was already binding, so I did not use that print as evidence.
4. I hand-recomputed Combined Risk Parity from `fund_returns.csv` using the same geometric annualised return definition as the project. I obtained annualised return 0.140, volatility 0.162, Sharpe 0.862, max drawdown -0.195 and growth of $1 1.478 - the same values as `performance_metrics.csv`.
5. The real-data sentiment effect was much smaller than the synthetic stand-in suggested. Fin-VADER clearly changes measurement (neutral 49.6% -> 45.2%; standard deviation 0.288 -> 0.312) but not performance (Fin-VADER tilt Sharpe 0.342 versus 0.344 for the plain tilt and 0.380 for the base). I changed the report story to an evidenced negative result rather than trying to tune it away.
6. I added a 21-day block-bootstrap robustness check (2,000 draws, fixed seed). The wide Sharpe intervals changed my wording from "risk parity wins" to "risk parity is the preferred lead fund on the point estimate, drawdown and turnover, but three years does not statistically separate the top funds."
7. I ran `python scripts/check_handin.py` and it returned 22 checks passed, with only the non-failing cleanup reminder.
8. I also checked the predictive content of the lagged sector sentiment signal directly: across 9,800 sector-day observations, Pearson IC was -0.043 and Spearman IC was -0.023. I treat both as near zero and economically negligible; I did not convert them into a formal significance claim.

## Lesson
Same as part a but bigger: the AI is a fast drafter with no access to ground truth. Everything it produced here was structurally right and numerically provisional, and it stayed provisional until i ran it on the real data myself.

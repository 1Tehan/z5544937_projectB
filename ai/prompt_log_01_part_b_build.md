# Prompt log -- Part B end-to-end build

## What I wanted to do
Build the whole of part b in one pass on top of my part a foundation: the walk-forward out-of-sample funds (equity / crypto / combined, five methods), the VADER and Fin-VADER sentiment index, the fusion tilt, every required artifact and figure, and the FundX streamlit app.

## What I asked for
I gave claude the brief, the starter, my part a report and later my actual part a zip, and asked it to finish the assignment following my rules in CLAUDE.md (AI use is allowed and graded in this course). The extensions were the ones i flagged in my part a section 5 - the finance lexicon, and treating the extreme-move screen as a data-quality feature.

## What went wrong / what I fixed
- Its sandbox could not reach the course data hosts, so it built and tested everything on a synthetic stand-in with the same shape (same tickers, calendars, tz quirk, 2,847 injected duplicate headlines). That means every number it produced was provisional. The fix is procedural: i rerun `python scripts/run_part_b.py` on the real data myself and only those regenerated results get submitted.
- Before it had my part a zip it reconstructed station 1-2 from my report alone - close to my rules but not my code. Once i uploaded the actual zip it replaced src/etl.py and src/features.py with my part a files verbatim and rewired run_part_b to my function signatures (my daily_returns is long format, its version was wide). It also added the ' || ' re-split my part a headline panel was designed for, with an assert that the exploded row count equals the deduplicated headline count.
- It flagged its own solver trap: feeding raw daily means/covariances to SLSQP stalls it into near-identical weights for every method. The pipeline annualises and shrinks the covariance first and asserts the five methods genuinely differ.
- The lexicon valences are hand-set, not estimated. It drafted the word list; the scores that ship are the ones i reviewed. Evidence is reported two ways (neutral-rate change AND the downstream fund effect) so the extension isn't oversold.
- The tilt parameters (0.25 gamma, clip 0.6-1.4) are design choices, not optimised, and the tilt may underperform the base fund - that gets reported honestly either way.
- Small one: a deprecated streamlit argument (use_container_width) surfaced in testing and was replaced before it becomes a breakage on streamlit cloud.

## How I checked
[FILL THIS IN AFTER YOUR OWN RUN - at minimum:
 (1) rerun on the real data and confirm my part a anchors reappear: 146,836 headlines after dedup / 2,847 duplicates, 50,300 equity rows, 1,006 trading days;
 (2) my part a report predicted plain VADER would leave roughly half the headlines neutral - record the actual neutral share for both models from sentiment_model_stats.csv;
 (3) print the estimation window at one rebalance date and confirm it ends the day before t;
 (4) recompute one fund's sharpe and max drawdown from fund_returns.csv in a notebook and compare to performance_metrics.csv;
 (5) note anything that behaved differently on the real data vs the synthetic stand-in and what i did about it.]

## Lesson
Same as part a but bigger: the AI is a fast drafter with no access to ground truth. Everything it produced here was structurally right and numerically provisional, and it stayed provisional until i ran it on the real data myself.

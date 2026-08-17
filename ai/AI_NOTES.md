# AI notes - how I directed and checked AI in Part B (z5544937)

> Write every section below in your own words before hand-in. The prompt logs
> (prompt_log_*.md) hold the task-by-task evidence; this file is the candid
> reflective account the rubric asks for.

## My workflow in one paragraph
[YOUR WORDS. The honest shape of it: I set the rules first (CLAUDE.md), gave
Claude the brief + my Part A as the source of conventions, had it draft the
pipeline and app, then ran everything myself on the real data, checked the
mechanics I care most about (look-ahead slice, calendar/annualisation, dedup
counts, exact output filenames), and rewrote all narrative myself.]

## Where AI genuinely helped
[YOUR WORDS - e.g. speed on scaffolding 17 fund backtests + a 5-page app;
knowing the SLSQP scaling trap; wiring the exact required filenames; the
design-system consistency between figures and app.]

## Where AI was wrong or risky, and what I did
[YOUR WORDS - pull the concrete items from prompt_log_01 section 4 after you
verify locally. Include at least one thing YOU caught that the assistant did
not flag itself.]

## What I deliberately did NOT delegate
[YOUR WORDS - the economic interpretation, the final lexicon valences, the
choice of gamma/caps/rebalance frequency, and the report narrative.]

## Verification checklist I ran (tick + date them)
- [ ] `python scripts/run_part_b.py` on the REAL data completes; results/ regenerated
- [ ] Spot-checked one rebalance: estimation window ends the day before t
- [ ] Recomputed one fund's Sharpe/max-drawdown by hand from fund_returns.csv
- [ ] Confirmed dedup gives 146,836 headlines on the real data (matches Part A)
- [ ] Clicked through all 5 app pages locally, then on the deployed URL
- [ ] `python scripts/check_handin.py` all green before zipping

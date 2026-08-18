# Prompt log -- final report handover and second-assistant check

## What I wanted to do
Finish the parts of Part B that were still incomplete on 18 August: the written report, the AI transparency fill-ins and the final hand-in package, without changing the verified fund/sentiment results or inventing numbers.

## What I gave the assistant
I supplied a handover containing the real-data results and the current `z5544937_projectB.zip`. I asked ChatGPT to inspect the actual project rather than rely on the handover alone.

## What the assistant did
- unpacked the current project and read the brief, report skeleton, AI files and result tables;
- ran `python scripts/check_handin.py` on the supplied project and confirmed 22 mechanical checks passed, with only the non-failing cleanup reminder before the report work;
- independently recalculated the Combined Risk Parity metrics from `fund_returns.csv` and checked the reported maximum drawdown/holdings against the saved outputs;
- checked the public GitHub repository was visible;
- assembled a full Word report draft using the existing FundX design language, the nine real-data figures, the 17-fund metrics table and the real fusion table;
- used the existing project results for every numerical claim and added academic references for VADER, HRP, risk parity and the 1/N comparison;
- filled the unfinished AI notes and this log so the late-stage AI use is not hidden.

## What I need to be careful about
The course explicitly says that the economic interpretation must be my own and penalises verbatim AI prose. The final report therefore needs my read-through: I should rewrite any sentence that does not sound like me and only keep interpretations I can defend from the displayed figure/table. I should not present the assistant's wording as independent evidence. The role of this final pass was organisation, checking and drafting from my existing results, not creating a new result.

## Useful corrections/checks from this pass
The report skeleton still referred to a five-page app even though the current app has seven pages. The final description uses the actual navigation: Overview, Compare funds, Fund fact sheet, Find my mix, Build your allocation, Sentiment analytics, and Methodology & data. The latest Combined Maximum-Sharpe target weights were also checked directly: GE 20.0%, NVDA 19.4%, SO 16.3%, ADBE 12.4% and BTC 10.3%, with the top five representing about 78.4% of the portfolio.

## Final lesson
A second AI pass is useful for consistency and packaging, but it creates a transparency issue if it is hidden. Logging it is part of the assignment. The safest workflow is still results first, independent checks second, report last, and then a final human read-through before submission.

## Follow-up rubric audit and report redesign (18 Aug, late evening)
I then asked ChatGPT to audit the report against the Part B marking rubric rather than just proofread it. The second pass kept the verified numbers unchanged but reorganised the report around the marking evidence: a short executive product decision, a clearer fund-method table, a selected-results table, a separate innovation evidence map, an explicit limitations paragraph, and three recommendations tied directly to measured results. The required figures were consolidated into a more readable appendix instead of one mostly-empty page per chart. The Sharpe comparison was redrawn horizontally because the long method label was colliding with the figure footer, and the already-saved block-bootstrap table was turned into a new uncertainty figure so the robustness extension is demonstrated visually rather than mentioned only in prose. I still need to read the final narrative myself and rewrite any sentence I would not naturally defend in an interview or class discussion.

## Final numerical verification after report review
A final review flagged the holdings paragraph as the easiest claim for a marker to reproduce. I checked `results/data/fund_weights.csv` directly at the final Combined Maximum Sharpe rebalance (1 Dec 2023): GE 20.00%, NVDA 19.36%, SO 16.35%, ADBE 12.44% and BTC-USD 10.28%. These five sum to 78.43%, so the rounded report sentence is supported by the saved weights rather than an estimate from the chart. I also checked Crypto Maximum Sharpe in `performance_metrics.csv`: gross Sharpe is -0.03963 (reported -0.040) and net Sharpe after the 10 bps turnover model is -0.04297 (reported -0.043). I added explicit in-text attribution for the Spinu (2013) ERC formulation and López de Prado (2016) HRP method.
## Final interpretation audit (19 Aug)
A last rubric-focused review added one predictive-content diagnostic to the fusion discussion: across 9,800 sector-day observations, Pearson IC = -0.043 and Spearman IC = -0.023. I use this only as evidence that the signal is near zero and economically negligible at this horizon, not as a formal significance result. I also added the structural caveat that Equal Weight estimates nothing and has the lowest equity turnover (2.6%), so part of its OOS advantage is avoiding both estimation error and trading drag; a stricter method comparison would hold turnover fixed. The second recommendation is now operational rather than generic: estimate term-presence coefficients on 2020-2022 next-day abnormal returns, test them on held-out 2023 data, and only retain terms whose direction/evidence survives out of sample before rerunning the unchanged fusion rule.

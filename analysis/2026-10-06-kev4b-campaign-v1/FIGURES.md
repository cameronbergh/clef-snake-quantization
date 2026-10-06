# Kev prospective publication figures

These additive PNG/SVG desktop and portrait exports show all 30 paired seed scores and the two frozen paired comparisons with marginal 95% whole-seed bootstrap intervals and Holm-adjusted sign-flip p-values. Mean diamonds summarize food; faint lines connect each seed. All 90 games collided before the cap. Neither condition met the improvement criterion; equivalence is not established and CLEF Q2 is not refuted.

- [Desktop PNG](kev-snake-results.png) / [SVG](kev-snake-results.svg)
- [Portrait PNG](kev-snake-results-mobile.png) / [SVG](kev-snake-results-mobile.svg)
- [All paired rows/statistics](uncertainty.json)
- [Input, source, output hashes and environment](provenance.json)

Reproduction uses `requirements-kev-analysis.txt` and `python -B -m experiments.kev4b_publication_v1.analysis data/2026-10-06-kev4b-campaign-v1 runs/new-kev-analysis`. Output must be new; the frozen analysis uses 100,000 common paired draws for each procedure and the recorded PCG64 seeds. [Full methods and limits](../../docs/KEV_CAMPAIGN_RESULTS.md).

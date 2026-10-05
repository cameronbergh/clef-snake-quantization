# Exploratory seed-level uncertainty

75 executions, **15 paired seed units**. Score is food collected before collision or the 500-move cap.

| Model | Mean score | Difference vs BF16 | Descriptive paired-bootstrap 95% interval for difference | Wins / ties / losses |
|---|---:|---:|---|---|
| BF16 | 12.20 | — | — | — |
| Q6_K_L | 12.33 | +0.13 | [-0.20, +0.60] | 1 / 13 / 1 |
| Q4_K_M | 12.73 | +0.53 | [-1.93, +3.27] | 3 / 8 / 4 |
| IQ2_M | 18.53 | +6.33 | [+0.93, +11.27] | 11 / 2 / 2 |
| Q2_K | 17.13 | +4.93 | [+1.60, +8.47] | 10 / 4 / 1 |

## Interpretation

These data are a promising task-specific observation, not evidence that quantization universally or monotonically improves models. The IQ2 gain is larger than the Q6/Q4 differences on these seeds. Fresh, prospectively specified held-out trials are needed before a confirmatory claim.

## Methods and caveats

- Exploratory, post-observation analyses. Models were added adaptively; intervals are unadjusted, descriptive percentile-bootstrap intervals, not confirmatory evidence.
- Independent sample size is 15 paired environment seeds, not 60/75 games or thousands of correlated move decisions.
- 100,000 nonparametric paired-seed bootstrap resamples; NumPy PCG64 default_rng seed 20261005; 2.5th/97.5th percentile intervals. Resample entire seed rows, retaining cross-model pairing.
- Sign-flip p-values assume symmetry/exchangeability of paired differences under the null. Model labels were not randomized treatments; these are not causal randomization tests. Holm correction covers included quant-versus-BF16 tests only, not the whole adaptive analysis process.
- Score is food collected by collision or the 500-successful-move cap. Capped runs remain in score summaries; no uncensored survival or ultimate-score claim is made.
- BF16 versus GGUF variants changes both precision and runtime. IQ2 and Q2 ran later. Paired seed/event priorities may produce different food coordinates once bodies diverge.
- Selected same-runtime IQ2-versus-Q6 comparison is also exploratory, with unadjusted interval.

## Exploratory tests

P-values are supplied for transparency, not as a headline. The symmetry assumption, adaptive model selection, small sample, and runtime/order confounds limit their interpretation.

| Variant vs BF16 | Two-sided sign test, uncorrected | Two-sided sign-flip, uncorrected | Holm-adjusted sign-flip (included variants) |
|---|---:|---:|---:|
| Q6_K_L | 1.0000 | 1.0000 | 1.0000 |
| Q4_K_M | 1.0000 | 0.7656 | 1.0000 |
| IQ2_M | 0.0225 | 0.0405 | 0.1216 |
| Q2_K | 0.0117 | 0.0205 | 0.0820 |

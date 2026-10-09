# First-divergence mechanism note

*Research note, 2026-10-08. Additive; published evidence under `data/` and `analysis/` untouched. Complete per-pair records and the full report are posted to issue #4.*

## Question

Do lower-bit CLEF-Flash GGUF backbones make *better* Snake decisions than BF16 at identical board states, or *different* ones?

## Method (offline, read-only)

- Data: `data/2026-10-05-fiveway` — 75 games (BF16, Q6_K_L, Q4_K_M, IQ2_M, Q2_K) on 15 paired seeds; 10,747 audited decision calls.
- For each (seed, recipe) pair vs BF16: trajectories walked in lockstep on canonicalized request payloads (sorted-keys JSON, SHA-256); the first index with identical bytes but a different head-argmax action is that pair's *first divergence*.
- Prefix invariant (identical choices so far imply identical next requests, given the deterministic food protocol) held with zero violations across all 60 pairs.
- Tie-break frequency: all 10,747 decisions ranked by each model's own recorded probabilities; top-2 checked for feature-equivalence on (safe, manhattan_to_food, open_space, reduces_distance).
- Only identical states are ever compared; post-divergence trajectories are never treated as counterfactuals.

## Findings

1. Divergence counts monotonic in quantization aggressiveness — and all four recipes share the llama.cpp/Metal runtime, so this gradient is not confounded by the BF16 runtime difference: Q6_K_L 4/15 pairs, Q4_K_M 11/15, IQ2_M 15/15, Q2_K 15/15. Median first-divergence step: 69 / 43 / 10 / 6.
2. Margins at divergence (each model's own top-1 minus top-2): Q6/Q4 flips all near-boundary (both-side margins <= 0.13) — classic tie-break jitter. IQ2/Q2 flips often not near-boundary for BF16 (median margin 0.31 / 0.21, max 0.58) while the quant distributions are flatter — broader distribution reshaping, not just jitter.
3. All 45 divergences were top-2 tie-break swaps (each model's choice was the other's rank-2). Never a deeper reranking.
4. Decision quality at identical states: 45/45 both moves safe and exactly tied on every recorded feature. Zero cases of a feature-dominant quant decision. 40/45 occurred with 3 safe moves available — open states, not emergencies.
5. Base rate: BF16's own top-2 moves are feature-equivalent in 47.0% of its decisions (61.9% when both safe). Tie-break states are the norm, and all 45 divergence states were drawn from them.
6. Post-divergence trajectories are descriptive only: the quant trajectory outscored BF16's subsequent trajectory in 11/15 seeds (IQ2_M) and 10/15 (Q2_K).

## Mechanism sketch

Snake frequently presents feature-equivalent move pairs; the BF16 head's tie-breaks are arbitrary but deterministic; quantization reshuffles them; sequential compounding amplifies early divergences into large score gaps. Earlier divergence means more compounding room, matching the observed score ordering.

## External mechanism link (verified 2026-10-08)

Proskurina et al., "When Quantization Affects Confidence of Large Language Models?" (arXiv:2405.00632): GPTQ 4-bit decreases confidence in true labels, and quantization disproportionately affects samples where the full model was already low-confidence. Directly predicts the observed pattern — flips at near-tie decisions, never at confident ones.

## Limitations

- 15 exploratory seeds; IQ2_M/Q2_K added adaptively after early results.
- "Equivalent" is relative to four recorded features; longer-horizon single-state quality not directly observable.
- The BF16-vs-GGUF *level* retains the runtime confound (PyTorch/MPS vs llama.cpp/Metal); the recipe *gradient* does not.
- Post-divergence score comparisons are descriptive, not counterfactual.

## What could change this

- Same-runtime GGUF-BF16 control (`Cloudflare_clef-flash-bf16.gguf` exists in the same bartowski repo; see issue #6).
- Frozen held-out validation.
- Calibration-overlap check (issue #3).

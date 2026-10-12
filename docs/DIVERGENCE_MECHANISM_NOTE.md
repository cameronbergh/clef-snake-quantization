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

## External mechanism link (verified 2026-10-08; scope qualified 2026-10-09 per issue #8)

Proskurina et al., "When Quantization Affects Confidence of Large Language Models?" (arXiv:2405.00632): GPTQ 4-bit decreases confidence in true labels, and quantization disproportionately affects samples where the full model was already low-confidence.

Scope of the analogy: this fits the Q6_K_L/Q4_K_M pattern well — all of their divergences are near-boundary flips on both sides (margins ≤ 0.13). It does not fully explain the IQ2_M/Q2_K pattern, where BF16 is often confident (median margin 0.31 / 0.21, max 0.58) yet the quantized backbone still chooses differently; there the *quantized* models' own distributions are flatter (median margin 0.17 / 0.09), i.e. the flip occurs where the quantized distribution is uncertain. Treat the paper as a relevant hypothesis for the near-boundary component, not an established explanation of all divergences.

Note three distinct properties, which the earlier draft conflated: feature-equivalence (tied on recorded safety/distance/open-space features), top-2 rank swaps, and probability margins. Two moves can tie on every recorded feature while receiving quite different probabilities.

## Limitations

- 15 exploratory seeds; IQ2_M/Q2_K added adaptively after early results.
- "Equivalent" is relative to four recorded features; longer-horizon single-state quality not directly observable.
- The BF16-vs-GGUF *level* retains the runtime confound (PyTorch/MPS vs llama.cpp/Metal); the recipe *gradient* does not.
- Post-divergence score comparisons are descriptive, not counterfactual.

## What could change this

- Same-runtime GGUF-BF16 control (`Cloudflare_clef-flash-bf16.gguf` exists in the same bartowski repo; see issue #6).
- Frozen held-out validation.
- Calibration-overlap check (issue #3).

## Addendum 2026-10-11: IQ2/Q2 confident flips are logit-noise rank swaps

Closes the open hole flagged in the 2026-10-09 scope qualification above: the
Proskurina analogy fits Q6/Q4 near-boundary flips but not the IQ2/Q2 flips where
BF16 was confident (median margin 0.31 / 0.21, max 0.58).

**Method (offline, no new inference).** Ran on the 2026-10-05 five-way decision
records (10,747 records, `data/2026-10-05-fiveway/decisions/*.jsonl.gz`), a
superset of the 45 first-divergence states (so disagreement counts here differ
from the 45-state analysis). On BF16/quant *agree* states, fit a temperature T
minimizing mean KL between the quant's distribution and softmax(BF16 implied
logits / T), grid 0.5–4.0. On *disagree* states, recorded the BF16 rank of the
quant's chosen move, the BF16 top1–top2 implied-logit gap, and both entropies.
Scripts `mech_test.py` / `mech_test2.py` are not yet committed; available on
request.

**Results.**

| pair | common states | disagreements | BF16 margin at disagreements (median / max) | fitted T* (agree n) | mean KL, agree | residual mean KL, disagree |
|---|---|---|---|---|---|---|
| IQ2_M vs BF16 | 537 | 37 | 0.241 / 0.661 | 1.10 (500) | 0.0212 | 0.1258 |
| Q2_K vs BF16 | 589 | 39 | 0.226 / 0.650 | 1.25 (550) | 0.0159 | 0.0647 |
| Q6_K_L vs BF16 | 1521 | 4 | 0.028 / 0.061 | 1.00 (1517) | 0.0002 | 0.0012 |
| Q4_K_M vs BF16 | 990 | 17 | 0.039 / 0.132 | 1.00 (973) | 0.0018 | 0.0066 |

Flip targets by BF16 rank: IQ2_M 36/37 rank-2 (1 rank-3); Q2_K 37/39 rank-2
(2 rank-3); Q6/Q4 all rank-2. Combined IQ2/Q2: 73/76 flips land on BF16's
runner-up. BF16 implied top1–top2 logit gap at IQ2/Q2 flips: median 0.53, range
0.02–1.86. Implied logit spread (median, all common states): BF16 3.319 vs
IQ2_M 3.204; BF16 3.312 vs Q2_K 2.829. Entropy at flips (median): BF16 0.905 vs
IQ2_M 1.007; BF16 0.967 vs Q2_K 1.052 (max ln 4 = 1.386).

**Interpretation.** 2-bit quantization both compresses logit scale (T* 1.10–1.25
with low KL on agree states; Q6/Q4 fit T* = 1.00 with near-zero KL) and adds
per-logit noise. Flattening alone cannot explain the confident flips: at
T = 1.25 a logit gap of 1.86 becomes 1.49, argmax unchanged — yet flips occur at
BF16 margins up to 0.66, so a noise component is required, and the residual KL
on disagree states (4–6x the agree-state KL) is its signature. The flips are
rank swaps to the runner-up, not a better policy: 73/76 pick exactly BF16's
second-ranked move — the adjacent rank, i.e. the smallest gap for noise to
cross. A systematically better policy would not concentrate 96% of its
"improvements" there, and the follow-up experiments found no policy gain
(90-game Kev campaign: no gain; Q2_K −2.3 pts vs same-runtime BF16 on the
866-case classifier benchmark).

**Caveats.**
- The per-logit noise component is inferred, not directly measured.
- One metric in the original script ("flattened-argmax predicts quant choice in
  0/37") is vacuous: temperature scaling preserves argmax, so it is 0 by
  construction on disagree states. The conclusion rests on the compression fit,
  the rank-2 pattern, and the entropy/margin numbers instead.
- "Not a better policy" is a joint conclusion with the feature-equivalence
  finding (all 45 first-divergence states were feature-equivalent top-2
  tie-break swaps) and the null follow-ups, not from this test alone.
- A systematic (non-random) bias favoring runner-up features is not ruled out;
  this test cannot distinguish structured bias from iid noise.
- IQ2_M figures inherit the issue #15 artifact-provenance question (open as of
  2026-10-11): the tested artifact's identity vs the historical Hugging Face
  revision is still being verified.

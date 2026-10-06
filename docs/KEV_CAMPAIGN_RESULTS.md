# Kev 4B: completed prospective Snake campaign

**Neither Q8 nor Q4 met the pre-specified improvement criterion.** All 90 games on 30 fresh paired seeds completed and passed the original local and independent portable audits. This does not establish equivalence, prove that Q4 is worse, or refute the earlier CLEF Q2 observation.

![All paired scores and prospective differences](../analysis/2026-10-06-kev4b-campaign-v1/kev-snake-results.png)

[Versioned data](../data/2026-10-06-kev4b-campaign-v1/) · [All paired rows and uncertainty](../analysis/2026-10-06-kev4b-campaign-v1/uncertainty.json) · [Figure provenance](../analysis/2026-10-06-kev4b-campaign-v1/provenance.json) · [Passive viewer](../watcher/kev-campaign-results.html)

## Results

| Condition | Games | Mean food | Median | Range | Collision / alive cap |
|---|---:|---:|---:|---|---|
| BF16 | 30 | 19.90 | 20.0 | 7–31 | 30 / 0 |
| MLX affine8, group 64 | 30 | 19.90 | 19.5 | 7–33 | 30 / 0 |
| MLX affine4, group 64 | 30 | 17.80 | 17.5 | 2–33 | 30 / 0 |

| Versus BF16 | Paired mean difference | Marginal 95% bootstrap interval | Wins / ties / losses | Two-sided sign-flip p | Holm-adjusted p |
|---|---:|---|---|---:|---:|
| Q8 | 0.00 | −2.60 to +2.70 | 5 / 19 / 6 | 1.00000 | 1.00000 |
| Q4 | −2.10 | −4.50 to +0.13 | 7 / 9 / 14 | 0.09955 | 0.19910 |

The statistical units are 30 paired environment seeds, not 90 independent games or thousands of move calls. There are no missing rows or technical failures. All games collided before the cap. Attempts total 5,728 BF16, 5,746 Q8 and 5,053 Q4; successful-move means are 189.93, 190.53 and 167.43.

## Prospective protocol and model

The [preparation archive](../archives/kev4b_campaign_v1/108fc446/) preserves all 40 frozen files plus `freeze.json`, byte for byte from commit `108fc4464c5012793b82efe8200ab7438d17255e`. This includes historical README, TODO and CI files. Freeze SHA256: `1d9e83c0213d1db7263ff4f21a45c51ccf0a037c9b8213283f817b9df2dea79c`. Current documentation advances separately.

Thirty saved 128-bit food-placement seeds exclude all 51 distinct earlier CLEF/Qwen seeds. All six condition orders occur exactly five times, serially. Each game freshly loads the same merged BF16 checkpoint, derives the selected condition in memory, and receives exactly two fixed warmups. Warmups reuse compatibility fixtures and never move the board; scored seeds are fresh.

Kev uses its supported native typed-choice pointer head, rather than CLEF's joint-schema head or generated move text. The 12×12 board, initial body/direction, four actions, factual safety/distance/reachable-space features, food priorities, unsafe-choice execution and 500-successful-move horizon are preserved. No solver, action mask, sampling, fallback or hidden retry is used.

All three conditions use native MLX. Q8/Q4 apply affine quantization in groups of 64 to 248 eligible linear projections. The FP32 pointer head, temperature `2.406050072164233` and 178 retained backbone tensors remain unchanged. Recipe labels are not uniform whole-model precision or persisted checkpoint sizes; quantized export/reload and HTTP compatibility are not claimed.

| Source | Pinned revision |
|---|---|
| `jaredpalmer/kev-4b` | `6cfce5c2fa4b4bd64026336ab649c5ca78857d52` |
| `Qwen/Qwen3.5-4B-Base` | `1001bb4d826a52d1f399e183466143f4da7b741b` |
| Native Kev code | `5e42a7a03f28134853dd3ff77461457e921e5ec1` |

All 29 payload hashes, package versions and per-load tensor inventories are retained in `native.jsonl.gz`. The [60-decision compatibility preflight](../experiments/kev4b_v1/PREFLIGHT_RESULTS.md) remains separate zero-game evidence.

## Frozen analysis

The [analysis plan](../data/2026-10-06-kev4b-campaign-v1/analysis-plan.json) preceded all outcomes. Primary endpoint: food before collision or the fixed horizon. Comparisons: Q8−BF16 and Q4−BF16 paired means. There are 100,000 shared PCG64 sign draws, seed `2026100602`, using a two-sided absolute-mean statistic and `(1 + extreme count) / 100001`. Holm correction covers exactly two tests. Improvement requires a positive difference and Holm p ≤ 0.05.

Sign-flip interpretation assumes sign exchangeability/symmetry; without it this is not an exact test of only a zero-mean null. Bootstrap intervals use 100,000 common whole-seed row resamples, PCG64 seed `2026100601`, and linear percentile quantiles. Intervals are marginal, not simultaneous. No scores were excluded, imputed or replaced. The versioned analysis records exact NumPy draw operations and all paired scores/end statuses.

## Timing, memory and context

Snake waits for each decision before applying one action; wall time adds no gameplay penalty. There were 16,707 decisions, including 180 warmups, and 33,414 internal backbone passes. Two passes produce one choice. The live viewer refreshed every two seconds and displayed the latest board, so several elapsed moves could appear between pictures.

| Condition | Gameplay native-forward median | Fresh load/merge median | Quantization median |
|---|---:|---:|---:|
| BF16 | 349.88 ms | 2.300 s | — |
| Q8 | 341.81 ms | 2.295 s | 0.154 s |
| Q4 | 335.20 ms | 2.292 s | 0.137 s |

Forward time includes instrumentation but excludes warmups/loading/conversion. Trajectories and state lengths differ, so speed is descriptive. Load/merge and conversion are timed suboperations, not the entire verification/load boundary. Native process-lifetime RSS peaked at 15.95 GiB; MLX active allocation at 7.83 GiB and MLX peak allocation at 14.78 GiB. These overlapping metrics must not be added or interpreted as isolated per-game memory comparisons.

The supervised run lasted 2h 14m 43s, ending `2026-10-06T04:05:22Z` before the `2026-10-06T13:50:40Z` deadline. All decision, time, memory and storage guards stayed within bounds. There were zero failed attempts and no retry/resume; normal game collisions are separate outcomes.

Before execution the tokenizer-only context gate passed 113 synthetic long-body cases: maximum complete row 1,964 tokens and zero truncation, model loads, forwards or games. Its exact report/hash is retained. This bounded synthetic coverage does not prove exhaustive coverage or long-state model-forward behavior.

## Evidence and reproduction

All 203,018 original runner/native/supervisor rows are retained in deterministic gzip with exact numerical values and ordering. Requests/responses, base64 bytes and original SHA256, token IDs, logits, probabilities, inventories and trajectories are unchanged. Only 270 resolver path fields use `$ASSET_ROOT`; three occurrences of the consumed one-use identifier are committed with SHA256, including the public receipt. The portable auditor restores the pinned path alias in memory before using the unchanged frozen reconstruction engine.

Original journals/private receipts remain preserved outside Git, identified by original file hashes. Private approval/claim bytes, authorization conversation/thread history, control files and stdout/stderr are excluded. The public authorization receipt explicitly authorizes no execution. Privacy checks inspect decompressed logs and decoded wire bytes.

The original local audit verifies private one-use approval/claim bytes. The portable standard-library audit rechecks every factual request, transition/food event, unrounded argmax/softmax, exact response match, token structure, source/seed freeze, tensor scope, warmup/load/unload boundary, resource sample and successful worker exit. It cannot independently reverify excluded private authorization history, recompute tokenization without a tokenizer, or prove that hardware produced logged values.

```sh
python -B -m experiments.kev4b_publication_v1.audit data/2026-10-06-kev4b-campaign-v1
```

Recalculate CPU-only analysis into a new directory using `requirements-kev-analysis.txt`:

```sh
python -B -m experiments.kev4b_publication_v1.analysis data/2026-10-06-kev4b-campaign-v1 runs/kev-analysis-reproduction
```

Thirty seeds were a bounded design choice, not a demonstrated power guarantee. This selected Kev checkpoint and Snake protocol do not establish general intelligence improvement, a monotonic precision effect, equivalence or replication across tasks/hardware. Same-runtime held-out CLEF validation and independent reproduction remain open.

# Compact non-game CLEF benchmark results

**Complete:** 49 locked public synthetic tasks, 866 identical cases through all six conditions; **5,196 scored calls + 18 separate compatibility warmups = 5,214 actual forwards**, with zero technical inference failures, retries, answer corrections or dropped cases.

The lowest-bit recipes showed mixed equal-task macro accuracy changes on this suite. The Snake observation does not establish uniform improvement across tasks or recipes.

## Accuracy

| Condition | Correct / 866 | Micro accuracy | Equal-task macro accuracy |
|---|---:|---:|---:|
| official-bf16 | 787 | 90.88% | 90.86% |
| gguf-bf16 | 785 | 90.65% | 90.61% |
| q6-k-l | 784 | 90.53% | 90.47% |
| q4-k-m | 786 | 90.76% | 90.75% |
| iq2-m | 786 | 90.76% | 90.78% |
| q2-k | 766 | 88.45% | 88.32% |

Primary control is **BF16 GGUF on the same llama.cpp/Metal runtime**. Official BF16 uses PyTorch/MPS and is a secondary runtime comparison. All six use the unchanged official BF16 JointSchemaHead; quant recipes also retain their own actual output-embedding rows.

## Frozen paired macro comparisons

Differences are percentage points versus GGUF BF16. The bootstrap resamples complete paired vectors of the 49 tasks, not isolated correlated decision calls: 100,000 NumPy PCG64 resamples with seed 20261007. Intervals are unadjusted two-sided percentile intervals; centered paired-bootstrap p values use a +1 correction; Holm adjusts the four primary quant contrasts. These are descriptive public-suite comparisons, not held-out confirmation.

| Condition | Macro difference | 95% task-bootstrap interval | Holm p |
|---|---:|---:|---:|
| q6-k-l | -0.14 | -0.60 to +0.26 | 1.00000 |
| q4-k-m | +0.13 | -0.61 to +0.98 | 1.00000 |
| iq2-m | +0.16 | -1.13 to +1.46 | 1.00000 |
| q2-k | -2.29 | -4.13 to -0.62 | 0.04180 |

**Interpretation:** Q6, Q4 and IQ2 have small observed macro differences and intervals crossing zero; none establishes an improvement or equivalence. Q2's lower macro accuracy survives the frozen four-comparison adjustment on this descriptive suite. Against GGUF BF16, Q2 corrected 13 cases but spoiled 32 previously correct cases: a net 19 fewer correct answers. The prior Snake gains therefore do not uniformly transfer to these non-game decisions.

## All decision primitives

| Condition | Choice accuracy (362) | Yes/no accuracy (311) | Ordinal accuracy (193) | Ordinal MAE | Within one |
|---|---:|---:|---:|---:|---:|
| official-bf16 | 92.82% | 93.25% | 83.42% | 0.181 | 98.45% |
| gguf-bf16 | 92.82% | 92.60% | 83.42% | 0.181 | 98.45% |
| q6-k-l | 92.82% | 92.60% | 82.90% | 0.187 | 98.45% |
| q4-k-m | 92.54% | 92.93% | 83.94% | 0.176 | 98.45% |
| iq2-m | 92.54% | 93.57% | 82.90% | 0.197 | 98.45% |
| q2-k | 92.54% | 89.07% | 79.79% | 0.228 | 98.45% |

The BF16 runtimes agreed on **864/866 native predictions (99.77%)**. Agreement does not establish bit-identical arithmetic or runtime equivalence. Native-vs-raw rounded-boundary prediction differences are {"gguf-bf16": 0, "iq2-m": 0, "official-bf16": 0, "q2-k": 0, "q4-k-m": 0, "q6-k-l": 0}; no raw probability was used to correct a native answer.

## Exact scoring and preserved probabilities

- Choice: the native returned choice, including its original tie behavior.
- Yes/no: returned four-decimal `noul` probability >= 0.5, as in the upstream adapter.
- Ordinal score: argmax of the returned distribution, with first level on a tie; **not rounding the native expected-score scalar**.
- The wrapper captures unrounded float32 softmax probabilities from the **same** head forward. No second model call, sampling, solver, masking or fallback policy is used.

Every task/primitive, raw-probability Brier score, clipped NLL, fixed ten-bin top-confidence ECE/reliability curve, binary AUC/probability error, ordinal error and paired correctness gain/loss against **both** controls is included in [the complete frozen summary](../analysis/2026-10-07-classifier-benchmark-v2-v1/summary.json). Task/calibration analyses are exploratory secondary outputs.

## Provenance, execution and audit

- Upstream: [jabr/classifier-benchmark@afb83bee](https://github.com/jabr/classifier-benchmark/tree/afb83bee3b74064ae5a5d58c05b352a8d0ef7240), complete locked **v2 only**, CC0. Byte SHA256 `57af4d24ba8d627b12b307c2ffd4ded3c0fe3828c803dc6d70c1449d89f0ac1c`; upstream canonical lock `f9d74c2885657a21690f10e5ca76e4a273b82e851fb61e9f6c00f9dc00d02ceb`.
- Pre-inference commitment `12e80fe1a7079a356fa57670b788563db9b90befa2cb32723d99551a7b956b8f`: 24 immutable preparation/source files, exact case/request/token commitments, asset hashes and tensor inventories, deterministic condition/case order, resource/stopping budget and analysis.
- Exact upstream state/instructions/criteria; common question id `decision`; all 866 inputs are 158–380 tokens, no truncation.
- Fixed blocked condition order: Q4, IQ2, official BF16, Q6, GGUF BF16, Q2. One additional model process at a time; read-only existing assets, offline loading, external Models cache/temp/raw-output only, no downloads/conversions or unrelated-service restart.
- CPU fallback disabled. Official BF16 uses the normal PyTorch implementation because CUDA-only fast-path libraries are absent; the four quants and GGUF BF16 share the pinned native bridge and Metal runtime.
- Independently implemented portable audit reconstructs every request and native answer, verifies token commitments, all 5,196 paired scores, 18 warmups, all 5,214 started/completed forwards, both suite hashes, preparation/output hashes and supervisor resources.
- The [publication-only exporter](../experiments/classifier_benchmark_publication_v1/README.md) narrowly exempts one exact known public synthetic bearer-auth example from an overbroad privacy-marker scan. Original frozen exporter bytes stay preserved. The separate publication auditor also repairs the root-only hash-manifest boundary, retaining every original check and verifying the nested upstream lock. Frozen numerical/statistical analysis is run through that stronger auditor without algorithm changes. No dataset row, request/response byte, model call, scoring rule or statistical calculation changes. Private path configs/worker logs remain external; public data retain log hash commitments. All plain/compressed public payloads pass inspection.

## Verification

All **167 Python tests** pass, including the complete 5,214-record offline fixture, native-answer tamper rejection, nested upstream lock inventory and exact public-fixture privacy checks. Both watcher suites pass; all Python sources parse without model imports/bytecode writes. Qwen/Kev preparation checks, the archived Kev freeze and all three historical published dataset audits pass. The new dataset exhaustively verifies all 43 payload hashes, and the original 24-file freeze remains unchanged. Original CLEF inference counters are unchanged across the campaign.

## Limits

- Public synthetic cases can overlap training/evaluation data; they are **not held-out confirmation**, validated difficulty levels, a full Decision Index ranking or evidence of broad intelligence. Related tasks share families/instructions, so task-bootstrap uncertainty remains conditional/descriptive rather than proof of an independent population sample.
- This is a single checkpoint and machine (Apple M4 Max, 128 GiB unified memory), one execution per case/condition, deterministic blocked order. It does not establish monotonic precision effects, causal regularization, or replication across models/hardware.
- Latency uses identical cases but remains descriptive: model/kernel caches, runtime differences and unrelated services were not globally locked. Process-lifetime peak RSS is **not total GPU/model memory** and is not directly comparable across MPS and Metal allocations.
- The audit establishes internal integrity/reconstruction of recorded evidence; it cannot independently prove that hardware produced logged values. Offline token checks verify retained commitments against recorded requests, without loading a tokenizer or weights.

## Artifacts and offline reproduction

- [Immutable complete public evidence](../data/2026-10-07-classifier-benchmark-v2-v1/)
- [Frozen preparation and protocol](../experiments/classifier_benchmark_v2_v1/README.md)
- [Full analysis JSON and compact table](../analysis/2026-10-07-classifier-benchmark-v2-v1/)
- [Reproduction/storage rules](REPRODUCING.md#compact-non-game-clef-benchmark)

```sh
python3 -B -m experiments.classifier_benchmark_publication_v1.audit data/2026-10-07-classifier-benchmark-v2-v1
python3 -B -m experiments.classifier_benchmark_publication_v1.analysis data/2026-10-07-classifier-benchmark-v2-v1 runs/classifier-analysis-reproduction
```

The public audit needs only the standard library (Python 3.9+); offline analysis additionally needs NumPy. Neither command loads model weights or makes inference calls.

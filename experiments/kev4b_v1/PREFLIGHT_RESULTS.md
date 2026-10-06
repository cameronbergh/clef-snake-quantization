# Kev-4B bounded preflight results — 2026-10-06

The compatibility preflight passed on **12 saved discovery fixtures**, with **60 decision evaluations including 8 warmups, 120 internal backbone passes, and zero games**. Every request retained all four actions and its complete 383–581-token row. No request was truncated and the successful run had no technical failures.

Native BF16 and the Snake wrapper matched exactly on all 12 requests, encodings, probability vectors and choices. Four BF16 repeats also matched exactly, including raw logits. All eight warmups matched their later fixture evaluations. The adapter used an in-process callback; a live HTTP server and end-to-end HTTP transport were not tested. These are fixed-state compatibility results; they establish no gameplay benefit, held-out performance or main-campaign readiness.

| Comparison with BF16 | Fixtures | Choice changes | Largest absolute probability change |
| --- | ---: | ---: | ---: |
| MLX affine 8-bit, group 64 | 12 | 0 | 0.0211357474 |
| MLX affine 4-bit, group 64 | 12 | 0 | 0.1656092405 |

Each quantized condition started from a separate fresh native BF16 merge. The observed inventory contained **248 quantized linear projections and 3,569,090,560 source weights**. All 178 retained backbone tensors, including BF16 embeddings, and the original FP32 pointer head remained identical. The native temperature was **2.406050072164233**; head file SHA256 was `dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c`. Variants were derived in memory. **No quantized export/reload was validated or persisted.**

| Fixture | Tokens | BF16 | Q8 | Q4 |
| --- | ---: | --- | --- | --- |
| bf16-round-01-first | 383 | up | up | up |
| bf16-round-01-middle | 471 | down | down | down |
| bf16-round-01-last | 581 | down | down | down |
| bf16-round-05-first | 384 | right | right | right |
| bf16-round-05-middle | 410 | up | up | up |
| bf16-round-05-last | 465 | up | up | up |
| bf16-round-10-first | 383 | up | up | up |
| bf16-round-10-middle | 482 | up | up | up |
| bf16-round-10-last | 554 | up | up | up |
| bf16-round-15-first | 383 | down | down | down |
| bf16-round-15-middle | 472 | up | up | up |
| bf16-round-15-last | 548 | down | down | down |

The [machine-readable report](preflight-results-2026-10-06.json) includes every fixture's native and wrapped BF16, Q8 and Q4 choice, raw probabilities and logits; all eight warmups and four repeats; all 40 comparisons; and relative evidence filenames with SHA256 hashes. Full raw journals remain retained locally. The [fixture manifest](preflight-cases.json) identifies the twelve states from the prior CLEF discovery data.

Two initial technical failures made zero decision calls: the first source-cleanliness guard refused packaging artifacts; the continuation stopped when native snapshot resolution tried to obtain repository metadata while offline. A reviewed resolver then returned only the exact verified local snapshots, preserving upstream computation and offline mode. Their full initial span, including the intervening gap, was 180.856241916 seconds, rounded up to **181 seconds**. The one-use retry took **73.731192834 seconds** from launch receipt through final supervisor exit: **254.731192834 charged seconds in total**, below the 900-second aggregate limit.

Observed peak RSS was **16,784,687,104 bytes** and MLX peak allocation was **15,867,964,942 bytes**; these overlapping unified-memory measures must not be added. Post-run files occupied approximately **12.436 GB**, below the approved 56 GiB storage ceiling. The report retains the precise observation and timestamp.

Runtime: Python 3.12.13, MLX/Metal 0.32.2, mlx-lm 0.31.3, PyTorch 2.8.0, Transformers 5.17.0, PEFT 0.21.0, NumPy 2.5.3 and Hugging Face Hub 1.32.0. The [pinned provenance](provenance.json) and report retain the upstream code/base/adapter revisions, dependency-lock hash, actual harness/resolver hashes, fixed head identities and execution-plan hashes. No main campaign or further model run is authorized by this report.

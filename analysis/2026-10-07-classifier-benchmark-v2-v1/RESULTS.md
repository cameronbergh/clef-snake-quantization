# Compact non-game CLEF decision benchmark

Complete locked v2: 49 tasks × 866 identical cases × six conditions; 5,196 scored calls plus 18 separate warmups.

| Condition | Micro accuracy | Equal-task macro accuracy | Median ms |
|---|---:|---:|---:|
| q4-k-m | 90.76% | 90.75% | 324.2 |
| iq2-m | 90.76% | 90.78% | 299.5 |
| official-bf16 | 90.88% | 90.86% | 383.2 |
| q6-k-l | 90.53% | 90.47% | 322.2 |
| gguf-bf16 | 90.65% | 90.61% | 291.8 |
| q2-k | 88.45% | 88.32% | 301.3 |

Primary paired macro contrasts (percentage points), versus same-runtime GGUF BF16:

| Condition | Difference | 95% task-bootstrap interval | Holm p |
|---|---:|---:|---:|
| q6-k-l | -0.14 | -0.60 to +0.26 | 1.0000 |
| q4-k-m | +0.13 | -0.61 to +0.98 | 1.0000 |
| iq2-m | +0.16 | -1.13 to +1.46 | 1.0000 |
| q2-k | -2.29 | -4.13 to -0.62 | 0.0418 |

The public synthetic suite is not held-out confirmation, does not establish easy/hard task levels, and supports no broad or monotonic quantization claim. Runtime differences are isolated by the extra BF16 GGUF control, but kernels are not bit-identical and lexical output embeddings follow each actual GGUF. All native answers and raw head probabilities are preserved; any rounded-boundary disagreements are disclosed. Latency is descriptive with unrelated services untouched.

Every task, primitive, calibration metric, reliability bin, paired correctness transition, resource receipt and boundary difference is in `summary.json`.

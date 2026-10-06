# Proposed campaign budget

This budget is prospective. It authorizes no inference and does not promise completion within the ceiling. All observations below come from the completed, unchanged [preflight report](../kev4b_v1/preflight-results-2026-10-06.json) and preserved raw preflight journal, not from campaign gameplay.

| Resource | Proposed bound |
| --- | ---: |
| Paired seeds / conditions / games | 30 / 3 / 90 |
| Fresh loads / warmups per load | 90 / 2 |
| Gameplay decisions | 45,000 maximum |
| Warmup decisions | 180 |
| All decision evaluations | 45,180 maximum |
| Complete tokens per decision | 8,192, no truncation |
| Internal text-backbone invocations | 9 per decision maximum; 406,620 total |
| Whole-campaign wall time | 43,200 seconds (12 hours) |
| Individual decision wall time | 120 seconds |
| Process RSS / MLX active memory | Each ≤48 GiB |
| New campaign evidence | ≤16 GiB |
| Entire external asset root | ≤56 GiB |
| External volume free-space margin | ≥8 GiB |

A terminal collision consumes an attempt. A game colliding after fewer than 500 successful moves therefore needs at most 500 attempts; an alive cap needs exactly 500. Two warmups for each of 90 loads add 180 decisions. Counts include failed evaluations and cannot be reused after a technical stop.

The old preflight's 60 decisions performed 120 internal passes because its complete rows were only 383–581 tokens. Longer rows use multiple 1,024-token prefix chunks. The 9-pass upper bound is conservative for an 8,192-token complete row and one question branch. Counters record actual internal invocations separately; neither figure denotes generated reasoning tokens.

## Time estimates and uncertainty

Ordinary saved-state forward means were 0.325264 seconds for wrapped BF16, 0.315124 for Q8 and 0.312257 for Q4. These include preflight forward instrumentation but not a full gameplay turn's serialization, journal fsync, state-feature construction or repeated storage checking. They are short-state observations, not full-campaign timing measurements.

The three preflight loads through final completion spanned 57.086809 seconds, of which 20.217892 were forward time. Scaling the remaining load/conversion/inventory/unload overhead by 30 gives about **18.43 minutes** for 90 loads. This is a rough scaling of three heterogeneous loads, not a measured 90-load benchmark.

Campaign source verification runs once initially and before each load: **91 verifications** of 9,502,566,604 source bytes, approximately **864.73 GB cumulative reads**, plus metadata and tensor-inventory hashing. This is I/O, not new disk allocation. The previous worker start through input verification was 7.598424 seconds; scaling that observed span gives about **11.52 minutes**. Caching, drive contention, extra per-load checks and future system load can change it. New evidence records verification, load, inventory and conversion timing separately.

| Scenario | Decisions including warmups | Projection using short-state means and the above overhead |
| --- | ---: | ---: |
| Earlier CLEF average call count (10,747 / 75 games), applied only as an illustration | 13,076.4 expected-count arithmetic | About 1.65 hours |
| Every proposed game reaches 500 decisions | 45,180 | About 4.48 hours |
| Every game reaches 500 decisions and decisions average 1 second | 45,180 | About 13.05 hours; exceeds the stop limit |

The CLEF scenario is **not a Kev gameplay forecast**. The fractional count is scenario arithmetic, never an executable call allocation. Longer states have not been timed. Reserve a 12-hour window if the campaign is later approved, but expect an incomplete retained run if the ceiling is reached; do not silently extend it or resume.

## Memory and storage

The successful preflight observed peak RSS of **15.63 GiB** and MLX peak allocation of **14.78 GiB**. These overlap and are not additive; neither establishes long-state peak memory. One model condition is loaded at a time, with unload verification before proceeding. Both worker checks and a separate supervisor enforce limits and preserve measurements.

The external root last occupied about **12.436 GB (11.58 GiB)**, with approximately 226.49 GB free on Models. Recheck the verified mount identity, free space and actual root size immediately before any later execution and throughout it. Existing snapshots are reused; no new checkpoint downloads or persisted quantized weights are needed.

Preflight journal density was approximately 25.6 KB per decision. A linear full-count projection is about **1.08 GiB**, but larger bodies, complete encodings, repeated inventories and extra journal streams increase it. Plan for **one to several GiB of evidence**, bounded by 16 GiB. Existing usage plus that hard evidence allowance is approximately 27.58 GiB; caches and other root growth still count against the independent 56 GiB total-root bound.

## Stops and remaining validation

The supervisor arms a per-decision timer before acknowledging the reserved call. The campaign stops on timeout, failed identity/hash/runtime/head/tensor checks, token overflow, parser/native errors, memory/storage breaches, lost mount, or inconsistent evidence. A failed attempt never becomes a collision score. Every scheduled row remains accounted for; no automatic retry, seed replacement or resume exists.

The long-state tokenizer-only check remains unexecuted. Its 113 deterministic synthetic requests include long bodies and the preserved 145-entry edge state. Passing it would establish only those complete-row counts, not gameplay reachability, exhaustive context coverage, latency, memory or quantization benefit. Actual model execution remains separately approval-gated.

Raw timing provenance: `preflight/2026-10-06-local-resolution-retry/events.jsonl`, SHA256 `d3efd334aebe69b1b76e2673f7a35c8b55a305183ad36cfa362a0de5e550a7b2`, under the external asset root. Public preflight report SHA256: `febffe857923ec6a7f544c5608e0300aaf0514796795e2444982a0c793221253`. No campaign outcomes were used for these estimates.

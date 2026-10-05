# Kev-4B bounded compatibility preflight

**Authorized, not executed.** Acquisition of the pinned sources, isolated dependencies and this bounded preflight are approved. No runtime parity or quantization results have been established. This plan permits zero games and does not freeze or authorize a main campaign.

## Fixed inputs and evaluation budget

Use the twelve requests in [preflight-cases.json](preflight-cases.json), in saved order: first, middle (`floor(n/2)+1`) and final attempts from BF16 discovery rounds **1, 5, 10 and 15**. Each request is bound to its published source record and hash. Preserve the complete factual state, question and option order; change only the serving alias. Do not roll out the states, choose replacements based on outputs or treat these reused discovery states as fresh game evidence.

| Phase | Decision evaluations | Purpose |
|---|---:|---|
| Warmups | 8 | Two each for upstream BF16, wrapped BF16, affine8 and affine4; retain every record |
| Upstream BF16 | 12 | All fixed states through the native MLX path |
| Wrapped BF16 | 12 | Same states and weights through the proposed evidence wrapper |
| Repeatability | 4 | First four saved cases repeated on wrapped BF16 |
| Affine8 | 12 | Same states with proposed 8-bit projections and unchanged FP32 head |
| Affine4 | 12 | Same states with proposed 4-bit projections and unchanged FP32 head |
| **Planned total** | **60** | **Hard maximum: 64; zero scored games** |

The approved counting unit is a complete native decision evaluation. Count every warmup, comparison, repeat and failed evaluation attempt against the total. Record internal prefix/branch backbone subcalls separately; do not describe the 60 decisions as 60 physical neural forward passes. The four unused evaluations are not a retry allowance or permission to add another condition. A changed allocation must be reviewed within the existing hard limits before execution.

Use one inference client, one active request and one loaded condition at a time. Place each condition's warmups immediately before its measured panel, record their fixture identities before the first load, and retain them in the total. No hidden OOM retry, safety override, action sampling or fallback is allowed.

## Time, storage and admission

The **900-second wall-clock limit** begins at the first model load and ends after the final unload. It includes loading, quantization/conversion, warmups, evaluations, evidence capture and unloading within that interval. Acquisition and isolated dependency installation precede that clock. A supervisor must enforce both the deadline and cumulative evaluation limit, including a call still in progress. If time expires before all phases finish, report an incomplete preflight; do not extend it or silently resume.

All model payloads, the isolated environment, caches, conversion temporaries and logs must remain under **`/Volumes/Models/clef-snake-experiments/kev4b-v1`**. Follow the [storage plan](storage-plan.json): approved overall **56 GiB** gate, **32 GiB** initial phase gate, fresh mount/path/free-space checks and explicit per-allocation limits. Internal storage and X9 are not fallbacks. Never modify existing working environments or move/delete existing files to make room.

Before the first evaluation, verify pinned source and runtime identities, actual merged BF16 weights, native FP32 head/calibration and the effective settings in [protocol.json](protocol.json). Load only verified local artifacts in offline mode. Disable prefix caching, date-fact injection, state truncation and hidden retries. Tokenize each complete native row with the actual pinned tokenizer and admit only **1–8,192 tokens**; a synthetic token-count test does not establish real context fit. Preserve exact tokens and option positions.

The proposed quantization must match [QUANTIZATION_PLAN.md](QUANTIZATION_PLAN.md): affine 8-bit or 4-bit, group size 64, eligible linear projections only, each derived directly from the same verified merged BF16 values. Keep the native head and other retained tensors unchanged. Inspect actual tensors rather than trusting a dtype label. Record memory use; unresolved resource limits or a mismatch in the planned scope must stop execution before evaluations.

## Acceptance, logging and stops

Upstream and wrapped BF16 must use identical full tokens, option positions and native head/retained tensors. All twelve choices must agree, with maximum absolute difference in unrounded candidate probabilities at most **1e-6**. Preserve the upstream first-option behavior for a true raw tie. The four repeats check the wrapped baseline's deterministic behavior. Probability or action changes caused by quantization are observations, not reasons to remove a fixture or declare a parity failure.

Save exact ordered requests, raw response bytes, HTTP status when applicable, errors, timing, phase/fixture identifiers and cumulative decision/subcall counters. Independently capture unrounded logits/probabilities, native argmax, token IDs/option positions, model and runtime hashes, tensor inventory, retained-tensor hashes and effective configuration. Rounded API probabilities cannot independently prove raw argmax. Never discard a low-confidence, unsafe or changed choice. If a transport failure has partial bytes, retain them; do not repair or retry the response.

Stop on the first identity, tokenization/truncation, contract, BF16 parity, hidden-retry, evidence-write, storage or concurrency failure, or either hard budget limit. Keep the failing attempt and mark remaining phases not run. No Snake move is applied and no error is a collision or zero-food score. Persisted quantized variants require export/reload validation before later use; in-memory checks alone do not establish that compatibility.

The resulting report must distinguish passed checks, failed checks and unexecuted phases, account for every evaluation and internal subcall, and state whether any derived artifacts were persisted and verified. It can support a later main-campaign design; it cannot establish Snake performance or a quantization benefit. Main seeds, schedule, analysis and an independent new-schema evidence auditor remain unfrozen future work.

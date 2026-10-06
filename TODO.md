# Research TODO

## Smaller quantizations

- [ ] Find one or more CLEF-Flash backbone GGUFs **smaller than the current IQ2_M and Q2_K checkpoints**. Investigate supported lower-bit recipes such as IQ1_S / IQ1_M; their availability for this model is not yet verified.
- [ ] If suitable published files do not exist, make candidate quantizations from a verified high-precision source with pinned quantizer/version/settings. Prefer high-precision input over repeatedly requantizing Q2; record calibration/importance-matrix provenance if used.
- [ ] Verify payload hashes, file size, tensor counts and actual per-tensor types, tokenizer alignment, raw all-token hidden-state extraction and lexical output-row treatment. A recipe label is not a uniform whole-model bit width.
- [ ] Retain the **unchanged official BF16 joint-schema head**, factual state features and head argmax; no safety override, fallback, solver or hidden policy changes. Smaller refers to the backbone checkpoint, not necessarily the entire inference pipeline.
- [ ] Test verified candidates in a separately labeled dataset using shared saved seeds, serial inference and documented order. Report every seed, failure and alive cap, scores, timing and measured memory. Publish negative results as well as gains.

## Validation and mechanism

- [x] Publish the completed and audited Q2_K extension as an additive dataset; preserve the initial 60-game release. Published in `5ee0d86`; hosted offline checks passed. [75-game evidence and audit](docs/FIVE_WAY_RESULTS.md).
- [ ] Validate and publish first-action-divergence analysis on exactly equal recorded states; distinguish observed trajectories from untested counterfactuals.
- [ ] Freeze a fresh held-out validation protocol before seeing its results; keep discovery seeds separate.
- [ ] Add a same-runtime high-precision GGUF control and expand numerical compatibility checks.
- [ ] Seek an independent clean-room reproduction and, if effects persist, test additional initial states/tasks.

## Additional model comparison

- [x] Assess native decision-model candidates from primary sources: [Laya, Kev, Jeff and CLEF-27B](docs/DECISION_MODEL_CANDIDATES.md). Availability is not runtime/parity validation or candidate selection.
- [x] Select Kev-4B for bounded preparation and implement its [offline native-choice adapter](experiments/kev4b_v1/README.md), saved compatibility fixtures, provenance and resource plans. Qwen remains outside the intended model class.
- [x] Obtain approval for pinned acquisition, isolated setup and a [bounded zero-game preflight](experiments/kev4b_v1/PREFLIGHT_PLAN.md): at most 64 decision evaluations, 900 seconds from first load through final unload, and a 56 GiB external Models budget. Approval is not a completed execution.
- [x] Acquire and hash-verify all 29 pinned files on external Models, create the isolated environment and implement storage, identity, deadline, call-count and evidence guards; existing environments were preserved.
- [x] Complete the [60-decision preflight](experiments/kev4b_v1/PREFLIGHT_RESULTS.md): full 383–581-token rows, exact native/wrapped BF16 parity and four repeats, unchanged FP32 head, and 248 affine8/affine4 linear projections. Both initial failures and the explicitly authorized remaining-budget retry are retained; zero games.
- [ ] Complete the frozen tokenizer-only long-state check before any [Kev campaign](experiments/kev4b_campaign_v1/README.md). Model latency/memory for long states remain unmeasured. The campaign uses fresh in-memory variants and needs no HTTP server or quantized export/reload.
- [x] Separately prepare and freeze [30 fresh paired seeds, counterbalanced serial execution and prospective analysis](experiments/kev4b_campaign_v1/README.md), with a native runner and independent auditor. No campaign outcomes informed this preparation.
- [ ] Obtain separate exact-freeze execution approval after the context-validation gate passes; proposed bounds are 90 games, 45,180 decisions and 12 hours. Preserve all errors/partial results, with no automatic resume. No campaign has run.
- [ ] [Issue #1: benchmark another local decision model beyond CLEF](https://github.com/cameronbergh/clef-snake-quantization/issues/1). Preserve board/action rules and factual features, label model-interface differences, use native argmax without safety overrides, and publish a separately audited additive dataset. No new campaign has run.

**Deferred, outside the intended model class:** the [Qwen3 general instruction-model comparator](experiments/qwen3_snake_v1/STATUS.md). Its adapter, 30-seed schedule and analysis were prepared, but that work does not complete candidate selection for issue #1. Preserve the original freeze; do not acquire or run Qwen as an incidental next step.

## Charts, watcher and publication graphics

- [x] Publish the initial per-seed score comparison and paired-difference uncertainty figure in the GitHub README (PNG and SVG).
- [x] Add charts to the **HTML benchmark watcher**, using recorded benchmark data only; viewing charts never triggers inference. [Watcher and setup](watcher/README.md).
- [x] Show score distributions, matched-seed comparisons and paired wins/ties/losses. Label counts and unfinished models; use only shared completed seeds for interim paired comparisons.
- [x] Publish paired mean differences with descriptive uncertainty for every quantized variant. [Five-way analysis](analysis/2026-10-05-fiveway/UNCERTAINTY.md).
- [ ] Add within-game score/move progression from decision traces without repeatedly loading large logs into the watcher.
- [x] Include descriptive gameplay latency charts, explicitly labeling runtime and different-trajectory limits.
- [ ] Add a measured-memory chart if useful; distinguish process-lifetime RSS from per-round measurements and leave missing BF16 memory unavailable.
- [x] Provide descriptive uncertainty and alive-cap markers, preserving all unfavorable seeds. The analysis resamples paired seed units, not correlated moves.
- [x] Export versioned **PNG/SVG** figures, including a mobile layout for sharing and the GitHub README. [Five-way exports](analysis/2026-10-05-fiveway/). No social post has been sent.
- [x] Keep figure captions, dataset/version links, provenance and caveats with exports; preserve the initial four-way graphics unchanged.

See [next-experiment rationale](docs/NEXT_EXPERIMENTS.md) and [agent development rules](AGENTS.md). These are research plans, not claims of completed experiments; adding a TODO does not launch a download or GPU run.

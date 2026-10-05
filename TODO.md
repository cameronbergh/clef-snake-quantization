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

- [ ] [Issue #1: benchmark another local decision model beyond CLEF](https://github.com/cameronbergh/clef-snake-quantization/issues/1). First compare candidate interfaces, runtime support and resource needs. Preserve board/action rules; use each model's supported interface and label interface differences. Prefer a same-runtime high-precision/quantized sweep, fresh saved paired seeds, a frozen analysis plan, serial inference and an additive audited dataset. No new campaign is included in this publication/chart update.

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

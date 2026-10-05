# Research TODO

## Smaller quantizations

- [ ] Find one or more CLEF-Flash backbone GGUFs **smaller than the current IQ2_M and Q2_K checkpoints**. Investigate supported lower-bit recipes such as IQ1_S / IQ1_M; their availability for this model is not yet verified.
- [ ] If suitable published files do not exist, make candidate quantizations from a verified high-precision source with pinned quantizer/version/settings. Prefer high-precision input over repeatedly requantizing Q2; record calibration/importance-matrix provenance if used.
- [ ] Verify payload hashes, file size, tensor counts and actual per-tensor types, tokenizer alignment, raw all-token hidden-state extraction and lexical output-row treatment. A recipe label is not a uniform whole-model bit width.
- [ ] Retain the **unchanged official BF16 joint-schema head**, factual state features and head argmax; no safety override, fallback, solver or hidden policy changes. Smaller refers to the backbone checkpoint, not necessarily the entire inference pipeline.
- [ ] Test verified candidates in a separately labeled dataset using shared saved seeds, serial inference and documented order. Report every seed, failure and alive cap, scores, timing and measured memory. Publish negative results as well as gains.

## Validation and mechanism

- [ ] Finish and publish the complete Q2_K extension as an additive dataset; preserve the initial 60-game release.
- [ ] Analyze first action divergences on exactly equal recorded states; distinguish observed trajectories from untested counterfactuals.
- [ ] Freeze a fresh held-out validation protocol before seeing its results; keep discovery seeds separate.
- [ ] Add a same-runtime high-precision GGUF control and expand numerical compatibility checks.
- [ ] Seek an independent clean-room reproduction and, if effects persist, test additional initial states/tasks.

See [next-experiment rationale](docs/NEXT_EXPERIMENTS.md) and [agent development rules](AGENTS.md). These are research plans, not claims of completed experiments; adding a TODO does not launch a download or GPU run.

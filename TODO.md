# Research TODO

## Priority: determine where lower precision helps, hurts or makes no detectable difference

The open question is **whether more aggressive quantization (lower bit width) helps only this CLEF/Snake setup, other games, simple decisions, or harder non-game tasks**. Do not assume it improves everything or that increasingly smaller recipes produce increasingly better decisions. Current evidence is mixed: CLEF's exploratory IQ2_M/Q2_K gains and Kev's prospective BF16/Q8/Q4 results are separate experiments, not a universal precision law.

- [ ] Review the [latest completed Kev campaign](docs/KEV_CAMPAIGN_RESULTS.md) before choosing the next experiment: 30 paired seeds / 90 games; mean food BF16 **19.90**, Q8 **19.90**, Q4 **17.80**. Neither quantization met the frozen improvement criterion. No detected improvement is not proof of equivalence, and Q4's lower observed mean is not proof of a population-level decline. Kev did not test CLEF's Q2 recipes.
- [ ] Prioritize the existing same-runtime high-precision CLEF control and fresh held-out validation before treating the discovery gain as a precision-only effect; preserve the original data and protocol. More trials on the same seeds do not supply independent confirmation.
- [ ] Design a small **model × precision recipe × task family × difficulty** matrix. Compare precision conditions within each pinned model/checkpoint first; keep model-family replication separate from task transfer. Use the same validated runtime, native decision interface and unchanged head within a model; report actual quantized tensors, calibration and mixed-precision scope rather than treating recipe names as a uniform bit-width ladder.
- [ ] Include **games and non-games**, and **single-step and sequential** decisions. Candidate families: Snake plus a separately versioned maze/navigation or other game; intent classification (BANKING77 / CLINC150+OOS); tool/action selection (Decision Index adaptations of BFCL / API-Bank / When2Call); reasoning (ARC-Easy / ARC-Challenge, ANLI / ContractNLI); and offline workflow decisions (invoice processing, customer service or security-incident routing). Assess public data, licenses, context limits, adapter feasibility and scoring before selection. A multiple-choice adaptation is not automatically the original generative benchmark or a directly comparable leaderboard score.
- [ ] Assess the existing [Decision Index](https://clef-evals.workers-ai-mle.workers.dev/) and [Typesafe workflow evaluations](https://evals.typesafe.ai/) first, using the [CLEF-Flash model card](https://huggingface.co/Cloudflare/clef-flash) as an initial benchmark map, not independent quantization evidence. Prefer a small representative suite over selecting only benchmarks where lower precision happens to win.
- [ ] Define difficulty **before seeing quantized outcomes**: established easy/challenge splits, independently assigned ambiguity, number of options/distractors, required reasoning steps, planning horizon and context length. Where possible vary difficulty within the same task family; cross-benchmark score differences alone cannot identify a difficulty effect. Check ceiling/floor effects and never silently truncate inputs.
- [ ] Freeze datasets/splits, paired seeds or case IDs, prompts/options/features, model order, sample size/stopping rule, primary comparisons and analysis before execution. Keep discovery/calibration examples out of held-out confirmation; document possible benchmark training contamination. State the minimum effect of interest and justify sample size rather than treating 15 or 30 seeds as a power guarantee.
- [ ] Report gains, declines and inconclusive results for **every** condition and task. Use paired seed/case units and task-appropriate metrics (accuracy/macro-F1, calibration/Brier score, workflow exactness, game reward/survival). Keep decision quality separate from latency/memory; use matched-state timing and fixed decision budgets so faster inference does not masquerade as smarter decisions.
- [ ] Analyze whether the precision effect differs by task family or difficulty using pre-specified contrasts/interactions and multiplicity control. Do not infer a difference merely because one subgroup is significant and another is not, and do not average incompatible raw scores into a claim about being better at everything.

## Completed compact non-game decision suite

- [x] Complete all 49 locked classifier-benchmark v2 tasks (866 cases) across official BF16, same-runtime GGUF BF16, Q6_K_L, Q4_K_M, IQ2_M and Q2_K. All 5,196 scored + 18 warmup calls are retained and audited; report all gains/declines, paired macro contrasts, calibration and limits. [Results](docs/CLASSIFIER_BENCHMARK_RESULTS.md) · [Data](data/2026-10-07-classifier-benchmark-v2-v1/). This completes the compact suite, not held-out validation, a difficulty experiment or every future benchmark in the matrix.

## Related work and explanation

- [ ] Conduct a primary-source literature review of **post-training, inference-time quantization improving task accuracy/reward**, alongside negative results and sensitivity studies. Record model/task, quantization scope, runtime controls, calibration, retraining, sample size, uncertainty and artifact availability; distinguish speed/memory gains from decision-quality gains.
- [ ] Review [QeRL: Beyond Efficiency—Quantization-enhanced Reinforcement Learning for LLMs](https://arxiv.org/abs/2510.11696) and its [author project page](https://hanlab.mit.edu/projects/qerl-beyond-efficiency----quantization-enhanced-reinforcement-learning-for-llms) as a related lead. Its reported noise/exploration benefit involves **RL training, LoRA and adaptive stochastic noise**, not simply lowering precision for an unchanged frozen greedy Snake policy. It is not a replication or an explanation established for this study.
- [ ] Use the existing identical-state/first-divergence work to examine action flips, unrounded logit margins, probability calibration, collision choices and numerical sensitivity. Keep later different-trajectory comparisons separate. Probability entropy changes do not establish behavioral exploration in a deterministic argmax policy; uniform positive logit-temperature scaling alone does not change its winner.
- [ ] If pursuing causal mechanisms, design separately frozen matched-state/branch-rollout ablations of quantization error, output-embedding treatment, runtime arithmetic and head/backbone scope. Label perturbation/noise or altered-feature experiments as new protocols; do not retrofit them into published evidence or assert a regularization mechanism from score gains alone.

## Research framing and publication

- [ ] Revisit the title/subtitle and prepare a scoped research summary after reviewing both models. The current **Less Precision, Better Decisions?** question mark is exploratory; a descriptive alternative is **When Does Quantization Change Decision Quality?** Do not rename the project or imply a general benefit without additional evidence.
- [ ] Prepare a reproducible research note and an optional Twitter/X summary covering methods, positive **and negative** results, uncertainty, runtime confounds, data/code links and open questions. An exploratory public write-up need not wait for journal acceptance; assess suitable workshop/preprint/journal routes separately based on novelty and evidence. Preparing a draft does not authorize posting it.

## Smaller quantizations

- [ ] Find one or more CLEF-Flash backbone GGUFs **smaller than the current IQ2_M and Q2_K checkpoints**. Investigate supported lower-bit recipes such as IQ1_S / IQ1_M; their availability for this model is not yet verified.
- [ ] If suitable published files do not exist, make candidate quantizations from a verified high-precision source with pinned quantizer/version/settings. Prefer high-precision input over repeatedly requantizing Q2; record calibration/importance-matrix provenance if used.
- [ ] Verify payload hashes, file size, tensor counts and actual per-tensor types, tokenizer alignment, raw all-token hidden-state extraction and lexical output-row treatment. A recipe label is not a uniform whole-model bit width.
- [ ] Retain the **unchanged official BF16 joint-schema head**, factual state features and head argmax; no safety override, fallback, solver or hidden policy changes. Smaller refers to the backbone checkpoint, not necessarily the entire inference pipeline.
- [ ] Test verified candidates in a separately labeled dataset using shared saved seeds, serial inference and documented order. Report every seed, failure and alive cap, scores, timing and measured memory. Publish negative results as well as gains.

## Validation and mechanism

- [x] Publish the completed and audited Q2_K extension as an additive dataset; preserve the initial 60-game release. Published in `5ee0d86`; hosted offline checks passed. [75-game evidence and audit](docs/FIVE_WAY_RESULTS.md).
- [ ] Validate and publish first-action-divergence analysis on exactly equal recorded states; distinguish observed trajectories from untested counterfactuals.
- [ ] Freeze a fresh held-out **CLEF** validation protocol before observing results. The completed fresh-seed Kev campaign is a separate model comparison, not direct CLEF Q2 replication.
- [ ] Add a same-runtime high-precision GGUF control and expand numerical compatibility checks.
- [ ] Seek an independent clean-room reproduction and, if effects persist, test additional initial states/tasks.

## Additional model comparison

- [x] Assess native decision-model candidates from primary sources: [Laya, Kev, Jeff and CLEF-27B](docs/DECISION_MODEL_CANDIDATES.md). Availability is not runtime/parity validation or candidate selection.
- [x] Select Kev-4B for bounded preparation and implement its [offline native-choice adapter](experiments/kev4b_v1/README.md), saved compatibility fixtures, provenance and resource plans. Qwen remains outside the intended model class.
- [x] Obtain approval for pinned acquisition, isolated setup and a [bounded zero-game preflight](experiments/kev4b_v1/PREFLIGHT_PLAN.md): at most 64 decision evaluations, 900 seconds from first load through final unload, and a 56 GiB external Models budget. Approval is not a completed execution.
- [x] Acquire and hash-verify all 29 pinned files on external Models, create the isolated environment and implement storage, identity, deadline, call-count and evidence guards; existing environments were preserved.
- [x] Complete the [60-decision preflight](experiments/kev4b_v1/PREFLIGHT_RESULTS.md): full 383–581-token rows, exact native/wrapped BF16 parity and four repeats, unchanged FP32 head, and 248 affine8/affine4 linear projections. Both initial failures and the explicitly authorized remaining-budget retry are retained; zero games.
- [x] Complete the tokenizer-only long-state gate: 113 synthetic cases, maximum 1,964 tokens, zero truncation/model calls. [Exact report](data/2026-10-06-kev4b-campaign-v1/context-report.json). Synthetic coverage is not exhaustive long-state forward validation.
- [x] Separately prepare and freeze [30 fresh paired seeds, counterbalanced serial execution and prospective analysis](experiments/kev4b_campaign_v1/README.md), with a native runner and independent auditor. No campaign outcomes informed this preparation.
- [x] Receive separate exact-freeze execution approval and complete all 90 games within the 45,180-decision / 12-hour limits. The one-use approval is consumed; no retry or resume is authorized. [Run evidence and audit](docs/KEV_CAMPAIGN_RESULTS.md).
- [x] [Issue #1: benchmark another local decision model beyond CLEF](https://github.com/cameronbergh/clef-snake-quantization/issues/1): publish the audited Kev BF16/Q8/Q4 dataset, native adapter/setup and frozen prospective report/charts. Neither quantization met the improvement criterion; this does not establish equivalence or refute CLEF Q2.

**Deferred, outside the intended model class:** the [Qwen3 general instruction-model comparator](experiments/qwen3_snake_v1/STATUS.md). Its adapter, 30-seed schedule and analysis were prepared, but that work does not complete candidate selection for issue #1. Preserve the original freeze; do not acquire or run Qwen as an incidental next step.

## Charts, watcher and publication graphics

- [x] Publish separately versioned [Kev PNG/SVG desktop/mobile figures](analysis/2026-10-06-kev4b-campaign-v1/) and [passive results viewer](watcher/kev-campaign-results.html); preserve CLEF figures and datasets unchanged.
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

# Next experiments: separate validation from exploration

The initial 15 seeds are now discovery data. Do not reuse them as an independent confirmation set, discard unfavorable results, or adapt a stopping rule based on the latest scores.

## Completed discovery extension; next control

The [complete five-way snapshot](FIVE_WAY_RESULTS.md) contains all 75 games on the 15 original seeds, including Q2_K, with the original cap and full audits. The initial four-way release is preserved. IQ2_M and Q2_K are different mixed GGUF recipes; both retain the official BF16 decision head.

**Next controlled experiment:** establish a high-precision GGUF backbone on the same llama.cpp/Metal bridge before making a precision-only claim. Start with the fixed identical-request compatibility/timing panel described below, then freeze a held-out game schedule. This is a proposal, not a running job; no model download or additional inference was performed for the five-way publication or watcher charts.

## Offline mechanism exploration: no extra inference needed

For every paired seed, compare decision traces up to their first divergence. Verify that both models saw exactly the same full request at that point. Record action probabilities, confidence margins, immediate collision safety, and whether one action immediately terminates the game. Separate genuinely common-state comparisons from later different-state trajectories. Show every seed, including losses; do not select only dramatic examples.

This can establish where paths separate. It cannot establish what an unexecuted counterfactual would have scored. Such a claim would need controlled branch rollouts.

## Held-out validation protocol (proposed; not run)

- **Primary contrast:** IQ2_M versus Q6_K_L on fresh shared environment seeds, because both use the same llama.cpp/Metal bridge and identical official head. Keep Q2_K and BF16 as secondary contrasts if resources permit.
- **Sample:** propose 100 fresh seeds, generated and saved once before inference. Use a fixed complete schedule; no significance-based early stopping. This is a pragmatic validation target, not a power guarantee.
- **Execution:** rotate or counterbalance model order within seed, serial GPU inference, warm all variants consistently. Save versions, tensor types, config, sources, hardware and timings before the run.
- **Policy:** unchanged greedy head argmax, factual state features, 12×12 initial board, per-event shared food candidate priorities, 500-successful-move cap, no safety masking/fallback.
- **Primary outcome:** mean paired difference in food collected before collision/cap. Report a paired seed-level uncertainty interval, wins/ties/losses, and all raw scores. Report caps separately and do not label surviving games losses.
- **Secondary outcomes:** successful moves, collision type, capped survival, descriptive gameplay latency, and separate matched-state timings. Call-level samples are not independent evidence for game-level skill.
- **Freeze analysis before observing outcomes:** record exact resampling method and comparisons; avoid choosing the best quant post hoc for a confirmatory headline.

## Precision/runtime control

Add an unquantized BF16/F16 GGUF backbone on the same bridge and validate token alignment, all-token hidden-state extraction, output-embedding treatment, and unchanged head. F16 is not BF16; label the exact precision used. Compare that bridge baseline with official PyTorch BF16 on fixed identical requests. No new checkpoint or runtime should be called equivalent until verified.

Also record which tensors are quantized versus BF16/F16, and which output rows are dequantized and cast for head inputs. A quantization label alone does not describe the entire inference pipeline.

## Beyond this task

[Qwen3 Snake v1](../experiments/qwen3_snake_v1/README.md) is now prepared for issue #1: an offline native-generation adapter, 30 fresh paired main seeds plus six separate preflight seeds, a balanced 90-game schedule, prospective analysis and a verified external-only storage plan. It preserves the existing board/features but uses Qwen's own chat/action interface and BF16 tied input/output weights across its planned precision sweep. No Qwen weights were downloaded and no model calls or scored games were run. The serial runner, new-schema auditor, artifact conversion and separately authorized preflight remain gates before a campaign. This does not replace the proposed CLEF runtime control or pool with CLEF discovery seeds.

If the held-out effect persists, repeat with different initial positions, board sizes, and another decision task. Changing any feature/schema/grid turns it into a new protocol; retain the discovery benchmark rather than silently rewriting it. General claims require breadth, not just more repeats on one task.

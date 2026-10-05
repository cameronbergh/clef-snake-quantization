# Methods and reasoning

## Why this experiment?

The motivating browser demo uses deterministic food placement. Resetting the same model with greedy actions reproduced the same path and terminal mistake. Five repetitions therefore tested repeatability, not five different environments. We changed **food placement only**, retaining the existing initial board and decision formulation, to examine whether backbone quantization changed closed-loop outcomes across varied environments.

The original demo also had a recommended-action feature and unsafe-choice override. Those were removed **before this discovery comparison** so programmatic action selection would not be misreported as model skill. Factual features remain explicitly visible in the requests. The benchmark is not raw-vision Snake or an unassisted planning benchmark.

## Why 15 shared seeds?

Fifteen rounds per variant were a practical exploratory expansion chosen during the investigation, **not a power-based sample-size guarantee**. Seeds were generated once before those games and saved. Each distinct seed is one paired statistical unit. More calls within a game establish execution provenance; they do not create thousands of independent skill samples.

Model inclusion was adaptive: BF16/Q6/Q4 ran first, then IQ2_M, then Q2_K was requested after earlier outcomes were observed. These discovery data are not a preregistered confirmation set. A larger fixed, held-out protocol is proposed in [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md), not claimed to have been run.

## Why per-food-event candidate priorities?

For each seed, food event (= score) and Fisher-Yates index, hash the ASCII string `clef-snake-food-v1:{seed}:{event}:{j}` using SHA256. Interpret the full digest as a big-endian integer and take modulo `j+1`, swapping from `j=143` down to `1` in a row-major array of 144 cells. Select the first unoccupied cell in that order. The modulo mapping is the exact saved algorithm; tiny hash-to-index bias is negligible but this is not claimed to be a mathematically perfect unbiased sampler.

This makes priorities independent of a variant's movement path and avoids ordinary rejection sampling advancing a PRNG stream differently when bodies occupy different cells. It does **not** force later food positions to remain identical: different bodies can block different candidates. Full event ranks, occupied cells and selections are logged and audited. A full board retains the original `[0,0]` fallback rather than silently changing the game.

## What is held fixed?

- 12×12 grid, body `[[5,6],[4,6],[3,6],[2,6]]`, direction right.
- Full body/head/food/direction, food deltas, factual collision safety, Manhattan distance and flood-fill reachable-space features.
- Exact four-option question instructions and option order. JSON field order is checked because serialization can affect tokenization.
- Official record encoding and **unchanged 121,762,820-parameter BF16 joint schema head**.
- Argmax action, no sampling, no safety override/fallback/solver. All four moves remain selectable, including terminal ones.
- Move semantics: collision checks the body excluding its vacating tail; failed moves do not change body/direction/score/steps. Eating extends the body and advances the food event.
- 500 **successful** moves maximum; terminal collision counts as an attempt. Caps are alive/censored, not losses.

## What differs between implementations?

Official BF16 is PyTorch/MPS BF16 with the upstream Transformers path. Quantized GGUF backbones share the pinned llama.cpp/Metal bridge and same BF16 official head. The bridge returns causal, final-normalized hidden states for every token, without pooling/unit normalization. Actual GGUF lexical `output.weight` rows are dequantized and cast BF16 for head input.

The GGUFs are mixed quantization recipes, not uniformly 2/4/6-bit models. This changes quantized backbone and output-row arithmetic; the joint head is not quantized. The original BF16 versus quant comparisons also change runtime. Existing five-state BF16-GGUF compatibility evidence is included, but small validation is not equivalent to a full same-runtime precision control.

The primary three variants rotate order within seed, with serial GPU calls. Later IQ2/Q2 phases are not fully counterbalanced, leaving execution-time confounding. Warmup calls are excluded; gameplay timing still includes state-size/trajectory and possible compilation differences. Process lifetime RSS is not round-local and was not recorded for original BF16.

## Outcome and uncertainty

Primary reported outcome: food collected before collision or the cap. Report mean, median, range and all paired scores, plus wins/ties/losses and caps. Capped observed scores are retained; no ultimate uncapped-score/survival claim follows.

The exploratory analysis resamples entire paired seed rows, 100,000 times with NumPy PCG64 seed 20261005, and reports descriptive percentile intervals. Intervals are unadjusted. Symmetry/sign tests and Holm-adjusted included-model comparisons are disclosed for transparency, not advertised as confirmatory or causal. Adaptive model inclusion and post-observation analysis remain limitations even after those adjustments.

Quantization can perturb a greedy action near a decision boundary. Sequential trajectories can amplify small probability changes into large score differences. This is a possible explanation, not established beneficial regularization or improved reasoning. Q6/Q4 near-baseline performance prevents a monotonic quantization-benefit claim.

## Reproducibility boundaries

Original observations used an adapted browser game. That upstream UI has no explicit license; it is not redistributed. The standalone Python driver independently implements documented behavior. Its full request/schema/feature **and field-order parity**, all moves, food events and outcomes are verified against every published browser-run record. This is exhaustive behavioral evidence on the observed trajectories, not a claim of identical source or proof over every possible future board.

Full original benchmark logs and independently captured model-server requests/responses are losslessly compressed in the dataset. Payload and executed-source hashes are preserved; evidence export uses an explicit allowlist. Offline CI verifies data integrity and recorded behavior without downloading models or spending GPU time. Reproduction on a second machine/runtime has not yet been established, and numerical kernel differences can alter actions.

# Less Precision, Better Decisions?

*An exploratory study of CLEF-Flash backbone quantization on Snake.*

## What we can say

On 15 shared random environment seeds, the IQ2_M backbone variant of Cloudflare CLEF-Flash collected an average of **18.53 food**, versus **12.20 for the official BF16 configuration**. IQ2_M scored higher on 11 seeds, tied on two, and scored lower on two. The observed mean difference is **+6.33 food (+51.9%)**. Q6_K_L and Q4_K_M were much closer to BF16: means of 12.33 and 12.73.

The same unmodified official BF16 joint-schema decision head chooses every move. This is **backbone quantization**, not an entirely two-bit decision model. The model receives structured state and factual collision, distance, and flood-fill features, not screenshots. There is no solver-selected action, unsafe-action masking, sampling, or fallback: the head's argmax is executed, even when it loses.

These are exploratory results, not a claim that lower precision universally improves intelligence or that gains are monotonic. The models were added adaptively after looking at results. BF16 uses PyTorch/MPS while quantized backbones use a llama.cpp/Metal bridge, so precision is confounded with runtime. IQ2_M ran after the first three variants. It also outscored Q6_K_L, which uses the same bridge, on these seeds; that reduces but does not eliminate implementation concerns.

## How much evidence is this?

There are **15 paired environment-seed units**, not 60 independent environments or 8,250 independent observations. Thousands of move calls establish provenance, but they do not increase the statistical sample size to thousands.

A descriptive, unadjusted paired-seed bootstrap gives an IQ2_M-minus-BF16 mean-score interval of approximately **+0.93 to +11.27 food**. That is an encouraging signal with considerable uncertainty. It is not confirmatory: an exploratory symmetry test does not meet a 0.05 threshold after adjusting across the initial three quant-versus-BF16 comparisons. See [UNCERTAINTY.md](UNCERTAINTY.md) for methods and all results; do not cherry-pick a p-value.

Each game stops at collision or 500 successful moves. One IQ2_M run reached the cap alive. Scores include that censored run, and we make no claim about its ultimate uncapped score. Seeds pair the food-candidate ordering; later food coordinates can differ if snake bodies occupy different cells.

## Possible explanation — not a demonstrated mechanism

Greedy, sequential decisions can amplify tiny numerical changes. A quantization-induced shift near an action boundary might divert a model away from a losing trajectory, or into one. A large difference in game score need not represent a large improvement in underlying reasoning. This is a hypothesis to test by comparing the first divergent action on identical states and measuring subsequent trajectories. We have not established beneficial regularization or any particular mechanism.

## Where to share

1. **GitHub now:** code, exact seeds, per-round data, full decision evidence, build/run instructions, uncertainty analysis, limitations, and model/runtime hashes.
2. **X/Twitter:** a short exploratory-results thread with the figure and repository link. Draft below; it has not been posted.
3. **Hugging Face community / r/LocalLLaMA:** share the reproducible implementation and invite independent replication. Avoid duplicating posts before the code and data are available.
4. **Versioned archival release / Zenodo:** useful once the package is stable, for an immutable, citable dataset and DOI.
5. **Technical report or workshop paper later:** after held-out validation, runtime controls, and a mechanism analysis. Check prior literature before claiming novelty. arXiv is an archive, not peer review.

## X/Twitter draft — not posted

**1/4**

Less Precision, Better Decisions? In a small CLEF-Flash Snake study, IQ2_M averaged 18.53 food vs BF16's 12.20 (+52%) across 15 shared seeds. It won 11, tied 2, lost 2. Code + data: https://github.com/cameronbergh/clef-snake-quantization

**2/4**

This is backbone quantization, with the SAME official BF16 decision head. Structured Snake state, no vision. Every head argmax is executed—even unsafe moves. No solver, safety override or fallback. Q6/Q4 scores stayed near BF16, so this isn't a monotonic trend.

**3/4**

Exploratory, not proof: only 15 paired seeds; BF16 and GGUF use different runtimes; IQ2 ran later; one IQ2 game hit the 500-move cap alive. Full traces, seeds, methods and uncertainty are public. We want independent replication, not a sweeping quantization claim.

**4/4**

Next: finish Q2_K, test fresh held-out seeds, control runtime, and inspect the first action where models diverge. Can small numerical changes rescue a greedy policy from a losing trajectory? That's a hypothesis—not an established mechanism.

Draft only; update Q2 status before posting and attach the figure to post 1.

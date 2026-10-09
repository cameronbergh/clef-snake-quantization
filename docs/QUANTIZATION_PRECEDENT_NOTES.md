# Quantization-accuracy precedent notes

*Reading notes, 2026-10-08, for the "bigger discovery" question: is there precedent for quantization improving decision quality? Not a systematic review.*

## Papers

Linked, not vendored — arXiv links are the canonical copies; PDFs are not committed to this repo.

### Verified mechanism link

**Proskurina et al., "When Quantization Affects Confidence of Large Language Models?", arXiv:2405.00632 (2024).**
https://arxiv.org/abs/2405.00632

GPTQ 4-bit quantization decreases model confidence in true labels, with quantization disproportionately affecting samples where the full-precision model was already low-confidence. Predicts the Q6/Q4 near-boundary flip pattern in the Snake data; the IQ2/Q2 confident-BF16 flips go beyond it (see DIVERGENCE_MECHANISM_NOTE.md for the qualified scope). A mechanism precedent, not a decision-quality gain.

### Verified finding (from the paper body, 2026-10-09)

**Liu et al., "Evaluating the Generalization Ability of Quantized LLMs: Benchmark, Analysis, and Toolbox", arXiv:2406.12928 (2024).**
https://arxiv.org/abs/2406.12928

Across 26 datasets (plus cross-dataset/cross-subject settings), the authors report that in some cases, quantizing to 4 bits even leads to higher model performance compared to full precision — e.g. GLUE-SST and GLUE-QNLI in zero-shot. The effect is task-specific and non-monotonic: NLI tasks are least sensitive to quantization while scientific QA and commonsense reasoning degrade more; 2-bit mostly degrades. The paper's main focus is calibration-data distribution effects, not beating full precision. Caveats: two 7B model families, point estimates without uncertainty treatment in the reported figures, and "in some cases" is not a systematic gain — consistent with this project's survey finding of no systematic frozen-PTQ improvement. Directly relevant to whether quantization effects transfer across task families (issue #6).

### Red herring — do not cite as a mechanism here

**QeRL (arXiv:2510.11696).** Reports quantization noise aiding RL exploration and final accuracy — but requires an RL training loop with LoRA and scheduled stochastic noise. Does not transfer to a frozen greedy policy. (Already noted in TODO.md.)

## Where this project's own evidence stands

- Formal survey (issue #5): no peer-reviewed paper found showing *systematic* decision-quality gains from frozen post-training quantization.
- Community trawl (issue #7): no informal report survived scrutiny (claims dissolved into distillation-not-PTQ, cross-architecture comparisons, latency confounds, or sub-noise flips).
- So: precedent exists for the *mechanism* (confidence perturbation at low-margin decisions); precedent does *not* exist for systematic *gains*. The transfer question still belongs to the model x precision x task-family matrix (issue #6).

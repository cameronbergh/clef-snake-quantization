# Quantization-accuracy precedent notes

*Reading notes, 2026-10-08, for the "bigger discovery" question: is there precedent for quantization improving decision quality? Not a systematic review.*

## Papers

Linked, not vendored — arXiv links are the canonical copies; PDFs are not committed to this repo.

### Verified mechanism link

**Proskurina et al., "When Quantization Affects Confidence of Large Language Models?", arXiv:2405.00632 (2024).**
https://arxiv.org/abs/2405.00632

GPTQ 4-bit quantization decreases model confidence in true labels, with quantization disproportionately affecting samples where the full-precision model was already low-confidence. Directly predicts the Snake first-divergence pattern (flips concentrate at near-tie decisions, never at confident ones). A mechanism precedent, not a decision-quality gain.

### Lead worth reading (specific claim not yet verified from the paper body)

**Liu et al., "Evaluating the Generalization Ability of Quantized LLMs: Benchmark, Analysis, and Toolbox", arXiv:2406.12928 (2024).**
https://arxiv.org/abs/2406.12928

Benchmark of quantized-LLM generalization across 40+ datasets and multiple quantization algorithms, reporting counter-intuitive task-specific findings (notably around calibration-data distribution effects). Directly relevant to whether quantization effects transfer across task families — the core open question for the task-matrix plan (issue #6). The specific claim that 4-bit sometimes beats full precision on individual tasks needs verification against the paper body before citing.

### Red herring — do not cite as a mechanism here

**QeRL (arXiv:2510.11696).** Reports quantization noise aiding RL exploration and final accuracy — but requires an RL training loop with LoRA and scheduled stochastic noise. Does not transfer to a frozen greedy policy. (Already noted in TODO.md.)

## Where this project's own evidence stands

- Formal survey (issue #5): no peer-reviewed paper found showing *systematic* decision-quality gains from frozen post-training quantization.
- Community trawl (issue #7): no informal report survived scrutiny (claims dissolved into distillation-not-PTQ, cross-architecture comparisons, latency confounds, or sub-noise flips).
- So: precedent exists for the *mechanism* (confidence perturbation at low-margin decisions); precedent does *not* exist for systematic *gains*. The transfer question still belongs to the model x precision x task-family matrix (issue #6).

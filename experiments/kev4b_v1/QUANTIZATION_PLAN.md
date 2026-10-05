# Proposed Kev-4B quantization plan

Status: **proposed_unexecuted**. The user approved pinned acquisition, isolated dependencies and a bounded preflight of at most 64 decision evaluations, 15 minutes and zero games under a 56 GiB external Models budget. This preparation contains no execution runner. No source weights, derived variants, runtime parity, gameplay results or main-campaign freeze have been verified or produced for this plan.

The proposed comparison uses Kev's trained decision interface throughout: one factual state and one four-option `choice` question, followed by its native candidate probabilities and argmax. The backbone, learned adapter and pointer head belong to Kev. CLEF's head and the deferred general instruction-model comparator are not components of this model.

## Pinned sources

| Component | Pin |
| --- | --- |
| Kev serving/model source | [`5e42a7a03f28134853dd3ff77461457e921e5ec1`](https://github.com/jaredpalmer/kev/tree/5e42a7a03f28134853dd3ff77461457e921e5ec1) |
| Kev-4B adapter and native head | [`jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52`](https://huggingface.co/jaredpalmer/kev-4b/tree/6cfce5c2fa4b4bd64026336ab649c5ca78857d52) |
| Base checkpoint | [`Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`](https://huggingface.co/Qwen/Qwen3.5-4B-Base/tree/1001bb4d826a52d1f399e183466143f4da7b741b) |
| Author's locked Mac stack | MLX 0.32.2, mlx-metal 0.32.2, mlx-lm 0.31.3 |

[provenance.json](provenance.json) records source-file hashes, the author's distinct training-code revision, native behavior, requirements and remaining gaps. [artifact-plan.json](artifact-plan.json) contains all 29 source-file metadata rows, including sizes, Git blob identities and advertised LFS SHA256 hashes. These are expected identities from public metadata, not local weight-verification results.

Both complete snapshots total **9,502,566,604 bytes**: 9,342,824,751 for the base and 159,741,853 for Kev. The base snapshot includes vision weights that the text-only MLX loader discards; they still count in this conservative full-snapshot acquisition budget. This total excludes a dedicated environment, derived files, temporary space and evidence. The parent storage plan must reserve those separately on the verified external Models partition. Internal disk and X9 fallback are forbidden.

The upstream code, adapter/head and base declare Apache-2.0 licenses. Retain their accompanying licenses and notices when acquisition is separately authorized.

## Baseline and fixed components

The supported Mac baseline is the official **merged BF16 MLX backbone plus the unchanged FP32 Kev pointer head**. The [native loader](https://github.com/jaredpalmer/kev/blob/5e42a7a03f28134853dd3ff77461457e921e5ec1/kev/checkpoint.py) loads the base's stored BF16 values and merges the LoRA delta in FP32, with one rounding to BF16. The merge scale remains 1.0.

Explicit MLX loading ignores `LoadOptions.dtype` and uses the stored precision. Selecting FP16 therefore does not create an FP16 MLX baseline. Automatic backend selection with an explicit FP32 request switches to PyTorch; it must not be substituted for the same-runtime BF16 baseline.

The [MLX scorer](https://github.com/jaredpalmer/kev/blob/5e42a7a03f28134853dd3ff77461457e921e5ec1/kev/mlx_model.py) reads text-backbone hidden states, converts the selected question/option states to FP32, and applies the original PyTorch `PointerHead`. The head's learned tensors and stored calibration temperature remain fixed in every proposed condition. The model card reports temperature 2.41; this must be checked against the acquired head before any decision evaluation.

The Qwen base has tied token input/output embeddings. Kev bypasses the vocabulary output projection and uses its own pointer head. Token embeddings remain BF16 under this plan. Their role and storage must not be confused with the separate FP32 decision head.

## Proposed conditions

| ID | Projection weights | Other parameters | Stage |
| --- | --- | --- | --- |
| `bf16` | Official merged BF16 values | Native floating-point values; FP32 pointer head | Proposed baseline |
| `mlx-affine8-g64` | MLX affine 8-bit, groups of 64 | Identical retained tensors and pointer head | Proposed primary comparison |
| `mlx-affine4-g64` | MLX affine 4-bit, groups of 64 | Identical retained tensors and pointer head | Proposed primary comparison |
| `mlx-affine2-g64` | MLX affine 2-bit, groups of 64 | Same intended retention rule | Deferred; no enabled comparison |

These are mixed-precision projection-weight experiments. They are not uniformly low-bit whole-model checkpoints, GGUF recipes, activation quantization, or KV-cache quantization. No precision fallback, alternate head, per-variant temperature refit or variant-specific prompt change is implicit.

Every variant must start independently from the same verified merged BF16 values. Never quantize an already quantized condition. The 2-bit condition requires a later explicit protocol decision before outcomes from that condition are observed; it is excluded from the proposed primary comparison family here.

## Proposed implementation boundary

After the official checkpoint loader has merged the LoRA and loaded the head, apply `mlx.nn.quantize` to the native text backbone with `mode="affine"`, `group_size=64`, the selected bit width, and a predicate that accepts only `mlx.nn.Linear` modules. Quantization of input activations remains disabled. Preserve embeddings, normalization parameters, depthwise convolution, recurrent parameters and the native FP32 head.

This route is supported by the generic [MLX 0.32.2 quantization API](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/quantized.py); its [upstream tests](https://github.com/ml-explore/mlx/blob/v0.32.2/python/tests/test_quantized.py) cover 2/4/8-bit affine operations and supported group sizes. Kev itself does not provide this quantization switch or a compatible packed-checkpoint export contract. Source inspection establishes a plausible implementation route, not successful execution.

Static inspection of the [pinned Qwen3.5 implementation](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/qwen3_5.py) predicts **248 eligible linear modules containing 3,569,090,560 weights**: 96 MLP, 32 full-attention and 120 linear-attention projections. Predicted input widths of 2,560, 4,096 and 9,216 divide evenly by 64. These counts and eligibility are unverified against loaded tensors. A mismatch must stop the preparation rather than silently changing the quantization scope.

Do not use MLX's default unrestricted predicate: it also quantizes embeddings. Keeping embeddings unchanged avoids a further compatibility issue in Kev, whose constructor infers pointer-head width from the embedding weight shape; packed embeddings would change that shape.

## Gaps that must be closed before execution

1. **Dedicated environment.** The upstream package requires Python >=3.12,<3.14, PyTorch >=2.6,<2.9, Transformers >=5.17,<6 and PEFT >=0.21. The author's complete lockfile is hashed in provenance. Existing working environments must not be modified or assumed compatible. No environment has been installed or validated for this plan.
2. **Offline loading.** Kev resolves the base and tokenizer by Hugging Face identifier even when given a local adapter directory. A future loader must use verified external cache paths, enforce offline operation and fail if a pinned snapshot is missing. Missing files never authorize an automatic download.
3. **Explicit mixed-precision reporting.** Kev's current `model.dtype` reads only the embedding dtype and would still report BF16 after projection quantization. Record a per-module tensor inventory, quantization mode/bit width/group size, retained tensor hashes, runtime version and effective backend.
4. **Derived artifact format.** Kev's existing full-weight loader validates ordinary floating-point shapes and dtypes; it does not accept packed quantized tensors. Either implement and audit a new versioned reload format or explicitly define in-memory derivation. Neither is implemented here. Original source files must remain immutable.
5. **Native baseline parity.** A separately authorized bounded check must establish that the wrapper's unquantized path matches the pinned upstream BF16 path on identical saved states, including tokenization, option order, logits/probabilities and chosen action. FP32 PyTorch can be an additional diagnostic, clearly identified as another backend, not the main baseline.
6. **Quantized-path validation.** Confirm exact quantized scope, unchanged retained tensors/head, finite outputs, complete four-option probabilities, deterministic repeat behavior and export/reload parity if applicable. Quantization-induced probability or action changes are intended observations; do not choose thresholds or exclude states to make them disappear.
7. **Response audit.** The [native API](https://github.com/jaredpalmer/kev/blob/5e42a7a03f28134853dd3ff77461457e921e5ec1/kev/api.py) computes argmax before rounding serialized probabilities to four decimals. Preserve its selected action and unrounded probabilities/logits. Recomputing argmax from rounded JSON can introduce artificial ties. Keep the original first-option tie behavior and fixed option ordering.
8. **Research authorization.** Acquisition, isolated setup and bounded preflight have task authorization; execution still requires the identity, storage, parity and stop guards. Gameplay is not authorized. Main seeds, schedule, statistical freeze and campaign runner are not provided by this document.

Any later campaign must retain the original factual Snake state, unsafe actions, native decision interface, serial requests, losses, caps and technical failures. It must record runtime and configuration changes without treating them as evidence of a precision-only effect.

# Native decision-model candidates

Primary-source and local inventory review, 2026-10-05. This is a feasibility assessment, not candidate selection, runtime validation or new benchmark evidence. No model weights were downloaded and no inference was run for this review.

## Intended model class

The intended comparison is a model trained to compute typed decisions and option probabilities directly. TypeSafe describes [Jev/System One](https://docs.typesafe.ai/concepts/system-one) this way. A shared `/v1/systemone` API alone is insufficient: a wrapper can also put that schema around generated text. Jev is documented as a hosted service; this review found no official downloadable Jev checkpoint or reproducible low-bit variants. Its exact architecture and parameter count cannot be inferred from the interface. [Official models](https://docs.typesafe.ai/models).

The [Qwen3 preparation is deferred](../experiments/qwen3_snake_v1/STATUS.md): its constrained JSON generation is outside this intended class. The original preparation and freeze remain intact.

## Options

| Candidate | Native mechanism and scale | Precision comparison available | Main limit |
|---|---|---|---|
| **Laya English** | ModernBERT-large plus trained option-scoring decision head, **421M** total | Native llama.cpp **BF16 and Q8_0** published | Default input budget 512 tokens; Q8 also quantizes parts of the decision head |
| **Kev-4B** | Qwen3.5-4B-Base, trained LoRA and pointer decision head | Author's MLX high-precision path; no verified native low-bit sweep in inspected loader | Needs validated quantization while retaining its own head; general MLX quantization support is insufficient evidence |
| **Jeff, firelex** | Qwen3.5-based **0.8B** decision model with trained answer-code readout | v1.3 publishes **Q8_0 and Q4_K_M**; high-precision native weights also exist | v1.3 is adapter-first; author recommends **v1.2** for zero-shot use. Do not mix versions |
| **CLEF-27B** | Qwen3.8-27B plus trained joint-schema decision head | Published BF16/Q6/Q4/IQ2/Q2 backbone GGUF candidates | Within CLEF family, not independent-family replication; requires its own 27B head and bridge validation |
| **Jeff, logan-markewich** | Wrapper around GLiFormer encoder classification | PyTorch and an ONNX INT8 path | A separate project, not the firelex checkpoint; quantization scope/runtime and evolving model size require pinning |

Primary sources: [Laya architecture](https://huggingface.co/convaiinnovations/laya), [native Laya exports](https://huggingface.co/ggml-org/Laya-GGUF), [Kev code](https://github.com/jaredpalmer/kev), [Kev-4B release](https://huggingface.co/jaredpalmer/kev-4b), [firelex Jeff](https://github.com/firelex/jeff), [Jeff GGUF release](https://huggingface.co/mstrasser/jeff-base-gguf), [CLEF-27B](https://huggingface.co/Cloudflare/clef), [CLEF GGUF exports](https://huggingface.co/bartowski/Cloudflare_clef-GGUF), [GLiFormer Jeff](https://github.com/logan-markewich/jeff).

## Laya: published quants differ from the existing local variant

The local installation contains **Laya multilingual, 322M, native MLX FP16**, not the English 421M model in the GGUF pair. The cached safetensors header contains 321,908,998 FP16 tensor elements. Its [published source/export card](https://huggingface.co/aac6fef/laya-multilingual-mlx) identifies mmBERT and a native decision head. Changing multilingual to English changes the checkpoint, not just precision.

At GGUF revision `22265007700297ba9e128297e82540cf28c5d7d4`, BF16 is **844,026,720 bytes** and Q8_0 **449,397,600 bytes**: **1,293,424,320 bytes combined**. The publisher's [conversion log](https://huggingface.co/ggml-org/Laya-GGUF/blob/22265007700297ba9e128297e82540cf28c5d7d4/convert.log) shows Q8 quantization of decision-head transformer matrices and `cls.weight`; some head tensors remain higher precision. This is not a backbone-only sweep with an unchanged BF16 head. The upstream source tensors were FP16 before this BF16 export. Verify the binary tensor inventory after acquisition.

Native llama.cpp support merged October 2 in [PR #29818](https://github.com/ggml-org/llama.cpp/pull/29818); the installed generic June build predates it. Runtime work is therefore required. The English model's documented 512-token input budget must be respected; an encoder positional limit advertised in GGUF metadata does not establish decision-model fidelity at that length. The existing [Laya-MLX Snake demo](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_DEMO.md) includes planner-derived features and a safety layer, so its scores cannot be pooled with unmasked CLEF gameplay.

## Kev and the two Jeff projects

Kev-4B offers an Apple Silicon MLX path. Its pointer head scores supplied options from hidden states; replacing it with ordinary Qwen generation would change the model. Release revision `6cfce5c2fa4b4bd64026336ab649c5ca78857d52` contains a **129,924,032-byte adapter** and **5,249,791-byte head**, excluding the roughly 8 GB base. The [pinned MLX implementation](https://github.com/jaredpalmer/kev/blob/5e42a7a03f28134853dd3ff77461457e921e5ec1/kev/mlx_model.py) merges adapters and retains the native head. A matched low-bit path still needs engineering and parity checks. Other sizes exist, but size changes cannot serve as precision variants.

Firelex Jeff uses a [trained decision readout](https://github.com/firelex/jeff/blob/5f7d0e0e3bd87d60c0df7cd22f760dd6769272fe/src/jeff/model.py). Its [GGUF path](https://jeffhub.ai/docs/llama-cpp) reads trained answer-code logits in one pass; it does not generate and parse move text. The [v1.3 GGUF repository](https://huggingface.co/mstrasser/jeff-base-gguf/tree/a5d0e595a0161864cad99efa9fe9e9a43a797893) publishes **1,082,015,328-byte Q8_0** and **672,329,312-byte Q4_K_M** files, with no F16/BF16 GGUF sibling found. Native v1.3 backbone plus readout is **1,706,550,008 bytes**. A same-runtime high-precision control needs export validation. Keep checkpoint version, adapter and calibration fixed; published v1.3 quants cannot be compared against v1.2 as a precision-only experiment. The author publishes format-specific temperatures, so record any calibration difference explicitly. Snake suitability remains untested.

Logan-Markewich's Jeff instead exposes a GLiFormer classifier behind a Jev-compatible API. This is not an independently trained Jeff release. The current [large checkpoint](https://huggingface.co/knowledgator/gliformer-large-v1/tree/d0a4e53d09cebe6bc963dd9be319d4279084bb2d) reports about **575.6M parameters**, while the wrapper README says 400M; pin the actual model rather than repeating the older size. An MPS high-precision versus CPU ONNX INT8 comparison would also confound backend and precision. Resolve which Jeff is intended before selecting it.

## CLEF-27B and external storage

The official 27B [head configuration](https://huggingface.co/Cloudflare/clef/blob/2f3de3dd85f379784083b0814d997ab627200f0c/joint_head_config.json) requires hidden size **5,120**, versus Flash's **4,096**. Never reuse Flash's head. Official source revision `2f3de3dd85f379784083b0814d997ab627200f0c` totals **54,989,894,057 bytes**, including its **256,125,024-byte** head and vision weights.

At Bartowski revision `e306f00c6c85da175dfb8de952ebb872087426a7`, BF16 GGUF totals **53,808,282,752 bytes**, Q4_K_M **17,203,416,256**, and Q2_K **10,582,559,936**. Those files plus the official source total **136,584,153,001 bytes**. Adding Q6_K and IQ2_M yields **170,489,000,233 bytes**. The publisher's inspected layouts show backbone tensors and no separately published decision head; metadata does not prove native-head execution. Validate the bridge with the official 27B head. [Pinned inventory and layouts](https://huggingface.co/bartowski/Cloudflare_clef-GGUF/tree/e306f00c6c85da175dfb8de952ebb872087426a7).

The dedicated external **Models** partition had **238,862,585,856 bytes free** at inspection. These payload totals fit, but exclude temporary conversions, duplicate caches, logs and memory requirements. Recheck the mounted volume and reserve transient space before any acquisition. All future weights, caches and conversion output must stay on that partition; the detachable X9 and internal disk are not fallback destinations. Existing files were not moved or deleted.

## Decision before implementation

**Laya is the smallest published native quant-pair option; Kev is a larger independently trained decision-model option; CLEF-27B is the closest family extension.** Firelex Jeff is also viable to investigate once its zero-shot/version choice is resolved. None is established as a successful Snake player under this protocol, and none predicts a benefit from quantization.

For the selected checkpoint, first verify complete state/context handling, native argmax, head/quantization scope and same-runtime high-precision parity. Then freeze fresh paired seeds, serial counterbalanced execution and analysis before results. Preserve unsafe actions, failures and alive caps. Compare precisions within each model; cross-model score differences also reflect architecture, size, training and interfaces. No acquisition, new adapter or campaign follows automatically from this assessment.

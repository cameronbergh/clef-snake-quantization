# Reproducing the experiment

## Scope

Recorded discovery games ran in a browser. The public standalone Python game is independently implemented and exhaustively compared against those recordings. Offline verification needs only Python 3.9+. Model-backed replication was prepared for Python 3.12 and the **same Apple Silicon MPS/Metal inference path**; CUDA/Linux support is not claimed or tested.

Recorded package versions are in `requirements-lock.txt`; minimal pinned inference dependencies are in `requirements-inference.txt`. The GGUF Python decoder is installed from the pinned llama.cpp tree, not an unpinned PyPI release. Build tools: Xcode Command Line Tools (C++ compiler/Metal SDK), Git and CMake. Suggested setup:

```sh
xcode-select --install
brew install python@3.12 cmake
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m clef_snake.download --root models --models bf16 q6-k-l q4-k-m iq2-m
python scripts/build_bridge.py --work build
```

Download substantial checkpoints only after checking free disk space. The official checkpoint includes its unchanged tokenizer, custom source, head, and backbone. The GGUFs contain backbone tensors but no joint head. If only running quantized models, use `--head-only` and omit `bf16`. Revisions, exact payload sizes/SHA256, head and source SHA256 are in `clef_snake/catalog.py`; downloads are fully hash-verified. Model files and build output are gitignored.

## Start servers, serial inference

Start each server in its own terminal; do not send game calls until `/health` reports `ready`. Port defaults: BF16 8765, Q6 8767, Q4 8768, IQ2 8769, Q2 8770. **Avoid running an interactive game or another inference client during evaluation.** Disable `PYTORCH_ENABLE_MPS_FALLBACK`; the server refuses an enabled fallback.

```sh
python -m clef_snake.server --model bf16 --official models/official --log runs/bf16-server.jsonl
python -m clef_snake.server --model q6-k-l --official models/official --gguf models/q6-k-l/Cloudflare_clef-flash-Q6_K_L.gguf --bridge build/libclef_bridge.dylib --log runs/q6-server.jsonl
python -m clef_snake.server --model q4-k-m --official models/official --gguf models/q4-k-m/Cloudflare_clef-flash-Q4_K_M.gguf --bridge build/libclef_bridge.dylib --log runs/q4-server.jsonl
python -m clef_snake.server --model iq2-m --official models/official --gguf models/iq2-m/Cloudflare_clef-flash-IQ2_M.gguf --bridge build/libclef_bridge.dylib --log runs/iq2-server.jsonl
```

Loading all servers simultaneously requires substantial unified memory. They are loaded separately, but benchmark calls are strictly serial. Each server checks official head/source hashes and the entire GGUF payload before loading. No model weights are silently rebuilt or changed.

## Replay the discovery schedule and seeds

```sh
python -m clef_snake.benchmark --manifest data/2026-10-05/seed-manifest.json --schedule data/2026-10-05/schedule.json --models bf16 q6-k-l q4-k-m iq2-m --move-cap 500 --out runs/replication
```

Output must be empty. `--manifest` is mandatory: seeds are never regenerated implicitly. The checked-in schedule preserves rotating BF16/Q6/Q4 followed by the later IQ2 phase. A replication can improve order control, but then it is a different schedule and must be labeled. Omit `--schedule` to rotate all selected models per seed. Endpoint overrides are a JSON file, e.g. `{"bf16":"http://127.0.0.1:9000"}`, passed with `--endpoints`.

Each saved request receives a new forward pass. Every full response, inference ID, wall latency, applied move, board snapshot and food-selection event is logged. HTTP/model errors stop the run without fallback. The driver verifies model revision/hash provenance and head argmax; it never uses safe moves to override a decision. No precomputed answers are used for replication.

New Python-driver games write raw JSONL and CSV in the chosen output directory. The public offline audit consumes the released evidence layout (compressed phase logs, round summaries and independent server logs); use `scripts/export_evidence.py --help` for the original archival export, or adapt that layout for new runs. Do not silently mix newly generated games with original discovery data.

## Numerical and timing limits

The bridge sets causal attention, no pooling, and exposes final-normalized hidden states for every token. States and actual GGUF output rows are cast BF16 for the official head. The llama.cpp patch only bypasses the unused language-model vocabulary projection in embedding mode. A validated BF16-GGUF control previously agreed on all five saved test actions; the complete numerical proof is in `data/2026-10-05/provenance/bridge-validation.json`. This is a small compatibility check, not proof of full runtime equivalence.

Replicas can differ across hardware, PyTorch/llama.cpp versions and floating-point kernels. Original warmup calls were excluded, but a first longer unseen input can still trigger compilation effects. Gameplay latency depends on trajectory/token length; do not interpret its median as a perfectly matched precision-only speed comparison. The package is not yet clean-room model-tested end to end on a second machine; offline parity, dependency import and native source compilation are separately verified.

## Recalculate the exploratory statistics and figures

Use a separate analysis environment (initial figure: Python 3.9, NumPy 2.0.2, Matplotlib 3.9.4) and a new output directory so historical artifacts are preserved:

```sh
python3 -m venv .venv-analysis
. .venv-analysis/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/analyze.py data/2026-10-05/results.json runs/analysis
```

The analysis is offline; it makes no model calls. The complete paired seed rows are resampled together, preserving cross-model pairing.

# Agentic development guide

## Project map

- `clef_snake/game.py`: independent Python game, feature/schema encoding and paired food protocol.
- `clef_snake/benchmark.py`: serial real-model driver; saved seeds mandatory; no implicit regeneration.
- `clef_snake/server.py`, `model_bridge.py`: official BF16 or GGUF backbone + unchanged official BF16 joint head.
- `clef_snake/catalog.py`: pinned model, head/source and native revisions/hashes.
- `clef_snake/audit.py`: exhaustive offline evidence/hash/request/trajectory/server audit.
- `native/`: minimal all-token-state C ABI and MIT-context llama.cpp patch.
- `scripts/`: build, allowlisted archival export and exploratory statistical analysis.
- `watcher/`: independently authored passive JSON viewer; never add model calls or inference controls.
- `analysis/`: versioned derived statistics, figures and provenance; retain older snapshots.
- `docs/DECISION_MODEL_CANDIDATES.md`: native typed-decision candidate assessment; no selection, acquisition or inference implied.
- `experiments/kev4b_campaign_v1/`: prospective 30-seed BF16/Q8/Q4 campaign, frozen separately from the spent preflight. Read `README.md`, `RESOURCE_PLAN.md`, `EVIDENCE.md` and `protocol.json`. Default runner/verifier and all tests are offline. The external tokenizer-context report and exact-freeze one-use campaign approval are required before real execution; preparation authorizes no games. Preserve every frozen file and use a new version for outcome-informed amendments.
- `experiments/kev4b_v1/`: native Kev adapter, saved compatibility fixtures and approved bounded acquisition/setup/preflight plans. Read `README.md`, `PREFLIGHT_PLAN.md` and `protocol.json`; offline checks alone establish no runtime result. `PREFLIGHT_RESULTS.md` records the completed 60-decision run and limitations. `preflight.py`, `local_resolver.py` and `retry_budget.py` preserve local-only loading and the spent one-use retry allowance; do not rerun or delete its receipt/claim. Its historical plan does not establish a main-campaign freeze; see the separate campaign namespace for current preparation.
- `experiments/qwen3_snake_v1/`: deferred general instruction-model comparator, outside the intended Jev-style model class. Read `STATUS.md` first. Preserve the offline adapter, external-volume gate and pre-result freeze; no enabled campaign runner. Create a new version for outcome-informed amendments.
- `data/`: versioned public discovery evidence; `docs/`: methods/reproduction/limitations; `tests/`: lightweight protocol checks.

## Read first

Read `README.md`, `docs/METHODS.md`, `docs/REPRODUCING.md`, `NOTICE`, and the relevant code before changing behavior. Existing observations are exploratory; do not turn them into a monotonic, causal or general-intelligence claim. The sample unit is an environment seed, not a move call.

## Safe default commands

```sh
python3 -m unittest discover -s tests -v
node tests/test_watcher.js
python3 -m compileall -q clef_snake experiments scripts tests
python3 -m experiments.qwen3_snake_v1.verify
python3 -m experiments.kev4b_v1.verify
python3 -m experiments.kev4b_campaign_v1.verify
python3 -m clef_snake.audit data/2026-10-05 --verify-hashes
```

These require no model weights, GPU, credentials or network (watcher tests use Node.js). Audit each added complete dataset, too. The parity audit must continue checking every full request, field order, factual feature, transition, food event and independent server match; do not weaken checks to make changed behavior pass.

## Write boundaries and research integrity

- **Published evidence is immutable.** Add new dated/phase dataset directories and new research notes for more trials; do not replace previously published seeds, traces, scores or hash manifests. If a derived calculation needs correction, explicitly document the correction and retain the original evidence. New pipeline versions get new provenance.
- No hidden policy changes: preserve initial state, grid, question/schema/features, official head weights, raw all-token hidden states, output-row treatment, argmax, move/collision semantics and cap unless an experiment explicitly changes one and labels it as a new protocol.
- Never mask unsafe actions, insert a solver/recommended action, silently retry a losing game, sample an action, enable CPU fallback or discard unfavorable seeds. Collision calls are attempts, not successful moves; alive caps are censored.
- Preserve exact model/download/source hashes and serial inference. Record runtime and execution-order changes rather than attributing their effects only to precision.
- **Model downloads, server loading and GPU experiments are opt-in tasks**, never part of CI or an incidental test run. Existing benchmark data can be audited offline. Do not start new experiments just because a development task changes code.
- Kev acquisition, isolated setup and bounded preflight have task authorization under `experiments/kev4b_v1/protocol.json`: at most 64 decision evaluations, 900 seconds from first load through final unload including conversion, zero games and the approved 56 GiB external Models budget. Preserve the planned 60-evaluation allocation, record internal subcalls separately, and implement stop/evidence guards before execution. This authorization does not cover a main campaign, hidden retries or the deferred 2-bit condition.
- Keep checkpoints, build binaries, credentials, tokens, personal machine paths and unrelated workspace history out of Git. Use an explicit allowlist for evidence export and inspect compressed logs as well as plain text.
- On this Mac, future model acquisition, caches, conversion temporaries and output belong on the verified external **Models** partition. Never fall back to the internal disk or assume detachable X9 is the destination. Check the frozen experiment storage plan and active mount before any write; do not move or delete existing files as incidental cleanup.
- Respect third-party licenses. Do not copy the unlicensed original browser UI into this repo. Behavioral parity does not mean the standalone driver is the original executable.
- New analysis must retain seed pairing, state exploratory versus prospective choices, disclose multiple comparisons and uncertainty, and report all seeds/caps.

## Completing a change

Run the relevant offline commands and dataset audits, inspect the diff, and report changes, verification and any numerical/reproducibility limits. Prefer small reviewable commits. Research-result updates should link code, exact dataset, seeds, analysis and provenance together. Remote publication and external communication need task authorization; local code work alone does not authorize posting a social thread.

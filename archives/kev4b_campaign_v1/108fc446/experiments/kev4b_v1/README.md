# Kev-4B: offline preparation and approved bounded preflight

Kev-4B is selected for a bounded native decision-model compatibility check for [issue #1](https://github.com/cameronbergh/clef-snake-quantization/issues/1). **Pinned acquisition, isolated setup and the 60-decision compatibility preflight are complete.** Native/wrapped BF16 probabilities matched exactly on all twelve saved states; affine8 and affine4 retained the same choices on those states. See the [audited preflight results](PREFLIGHT_RESULTS.md), including both preserved initial failures and the explicitly authorized retry. No Kev games or persisted quantized artifacts were produced. The separate [main-campaign preparation](../kev4b_campaign_v1/README.md) now defines fresh seeds and analysis. Campaign execution remains unauthorized, with tokenizer-only long-state validation pending. The historical preflight plans below remain unchanged.

The approved preflight permits at most **64 decision evaluations, 900 seconds from first model load through final unload, and zero scored games**, within the **56 GiB external Models budget**. The plan uses 60 evaluations. Conversion after first load counts toward the time limit. See the [preflight plan](PREFLIGHT_PLAN.md) for the fixed states, counters, evidence and stop conditions.

## Native interface and evidence

The [adapter](adapter.py) preserves the complete ordered `Game.request()` state and four-option question, changing only the serving alias to `kev-latest` for `/v1/systemone`. This alias is not checkpoint identity. Kev uses its trained pointer head to select an option; it does not generate and parse a textual move.

`build_request(game)` and `make_request(game)` create the request without modifying the game. `parse_response(raw_bytes)` validates the native response. `attempt(game, transport)` calls an injected transport exactly once and returns an `Attempt`; no HTTP client, model loader, retry, persistence or game movement is included. `Attempt.to_record()` retains the request hash, exact request/response bytes as base64, response status, choice and any technical error. A future runner must save the full record durably before applying an action. The preflight applies no actions.

All four directions remain available, including unsafe choices. Kev chooses its raw argmax before rounding displayed probabilities to four decimals, so the adapter preserves the returned winner when displayed values tie. Independent runtime evidence must retain the unrounded logits/probabilities, option positions and native choice. Kev's nonzero `output_tokens` counts serialized answers; it is not evidence of generated reasoning tokens. Contract checks alone do not establish model identity, complete tokenization or numerical parity.

## Plans and limits

- [Protocol](protocol.json): native request contract, approved preflight bounds and main-campaign exclusion.
- [Fixed requests](preflight-cases.json): twelve saved CLEF discovery states, with source lines and hashes. These are compatibility fixtures, not fresh validation seeds.
- [Quantization plan](QUANTIZATION_PLAN.md): merged native MLX BF16 baseline, proposed affine 8-bit/4-bit linear projections with groups of 64, and unchanged FP32 pointer head and retained tensors. The 2-bit condition remains deferred.
- [Provenance](provenance.json) and [artifact inventory](artifact-plan.json): pinned code, base, adapter/head, expected file identities and remaining validation gaps. The execution report separately records local acquisition verification.
- [Storage plan](storage-plan.json): destination, isolated environment, cache paths, exact source bytes and estimated derived-file reservations.

The complete source snapshots total **9,502,566,604 bytes** (about 9.502 GB). All weights, environments, caches, temporary conversions and evidence output belong under **`/Volumes/Models/clef-snake-experiments/kev4b-v1`**. Verify the dedicated external Models mount, destination ancestry and current free space before any write; no internal-disk or X9 fallback, moving existing files or incidental cleanup is allowed.

The initial allocation is **29,903,661,260 bytes**, with a **32 GiB** initial free-space gate. The conservative persisted allocation is **54,599,723,212 bytes**, under the approved **56 GiB (60,129,542,144-byte)** overall gate. Estimated derived files are not measured artifacts. The budget includes a deferred 2-bit reservation, which does not authorize that condition or extra evaluations. Keep one verified source copy and recheck space before conversion; stop before exceeding an allocation.

## Offline verification

From the repository root:

```sh
python3 -m unittest discover -s tests -p 'test_kev_*.py' -v
python3 -m experiments.kev4b_v1.verify
```

These checks use saved evidence and synthetic transport responses; they do not load a tokenizer or model, contact HTTP, install dependencies or acquire weights. They validate preparation consistency, not runtime compatibility. The [supervised preflight](preflight.py) has now checked the twelve fixed states. Its [local resolver](local_resolver.py) accepts only exact pinned, previously hash-verified snapshots, changes no native decision math, and restores the upstream resolver after loading. The [one-use retry ledger](retry_budget.py) binds the two initial failures by journal hashes and debits their full elapsed span: 181 seconds charged, at most 719 left for the explicitly approved retry. A permanent external launch receipt covers failures before the ledger claim. This spent claim is not a general-purpose runner or permission for further inference. Export/reload compatibility and gameplay remain untested.

Fresh main seeds, a counterbalanced schedule, prospective analysis and an independent Kev evidence auditor are prepared in the separate campaign namespace. Tokenizer-only context validation and a separate campaign execution decision remain pending. The CLEF evidence, executed preflight and deferred Qwen freeze remain unchanged.

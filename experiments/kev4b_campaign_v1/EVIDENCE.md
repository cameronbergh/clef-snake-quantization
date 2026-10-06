# Campaign evidence and independent audit

This is a prospective evidence contract, not a campaign result or execution approval. The CLI validates preparation by default. Real execution needs a separate, unused approval tied to the final freeze and all frozen limits. No campaign model calls are made by this document, the auditor or its tests.

## Files and durable ordering

| File | Producer | Contents |
| --- | --- | --- |
| `runner.jsonl` | Serial runner | Frozen inputs, scheduled games, reservations, lossless adapter attempts, board transitions, summaries and technical stops |
| `native.jsonl` | Native runtime | Independently verified source/runtime/tensor identities, encoding, forward reservations, internal pass events, raw decisions and serialized responses |
| `supervisor.jsonl` | Separate supervising process | Exact approval/claim, deadline, resource observations and worker exit or failure |
| `approval.json`, `claim.json` | Supervisor | Byte-preserved external approval and consumed one-use claim; private run evidence |
| `context-report.json` | Supervisor copy of the external tokenizer-only report | Exact bytes bound by `context_report_sha256` in approval, claim and runner/supervisor receipts |
| `stdout.txt`, `stderr.txt` | Supervisor | Worker diagnostic output; private run evidence |
| `control.json`, `watchdog-ack.json` | Worker/supervisor | Current watchdog handoff state, not a replacement for the append-only evidence journals |

Every journal has contiguous, one-based `sequence` values, a monotonic timestamp, an event name, and an explicit `synthetic` boolean. Journal writes return only after flush and `fsync`. Records preserve JSON field order; raw request and response bytes are separately retained as canonical base64 with SHA256. The auditor streams the runner and native journals in their common process monotonic order. It rejects malformed JSON, duplicate keys, nonfinite JSON numbers, missing sequence numbers, and torn final lines instead of skipping them.

The runner emits `run_start`, independently cross-checks `inputs_verified`, and follows the frozen 90-game schedule. Each game records:

1. `game_start` with the exact scheduled identity, initial board and food event; runner `load_start`.
2. Native input re-verification and a fresh native load. `checkpoint_metadata`, `local_snapshot_resolution`, `native_inventory`, optional `quantize_start`/`quantized_inventory`, and `load_complete` establish the condition. Runner `load_complete` retains the same native metadata.
3. Exactly two warmups using the first two frozen preflight requests. Warmup actions never change the game.
4. Game decisions, with one durable `attempt` followed by one `transition` for each successful adapter response. A collision consumes a decision attempt but does not increment successful moves. A technical failure never becomes a collision or score.
5. Runner `unload_start`; native rehash of the full loaded inventory and `unloaded`; runner `unload_complete` with matching metadata. Only then may `game_complete` report a collision or an alive 500-move cap.

A successful campaign ends with `run_complete` and supervisor `worker_exit` with code zero. A failure stops execution, retains all preceding evidence, and records the available native/runner/supervisor error and cleanup events. It does not authorize retries, replacement seeds, or resumption.

## Per-decision contract

Every warmup, game attempt and failed evaluation shares this context across journals:

- `call_id`: global contiguous reservation number, including warmups and failures;
- `game_id`: the frozen episode identity;
- `phase`: `warmup` or `game`;
- `game_attempt`: one-based game attempt, otherwise null;
- `warmup_index`: 1 or 2 for a warmup, otherwise null;
- `fixture_id`: the frozen warmup case ID, otherwise null.

The runner first records `call_reserved` with complete request bytes/hash. Native `decision_start` independently reserves the same `call_id` and `decision_id` before encoding. `encoded` retains the complete native textual record, question/option metadata, all token IDs, segmentation, positions, option readouts and decision readout. A single `native_forward_start` precedes all actual text-backbone invocations. Every `backbone_pass_start` has a matching result or error; prefix chunks and the question branch are counted separately.

`raw_decision` records the calibrated logits, unrounded probabilities, first-maximum index and selected action. `native_response` retains the exact serialized response and SHA256. `decision_error` is a terminal failed attempt with any available response bytes. The adapter `attempt` retains the same request/response bytes, status, choice or error, and the successful native response's journal sequence. Only after that attempt is durably written can a game `transition` record its prior/next board and all/new food events.

The audit checks softmax consistency and the **unrounded** native argmax before checking four-decimal API probabilities. Rounded probabilities can tie; selecting another displayed maximum is not permitted. Exact raw ties retain the native first-option rule in the frozen order `up`, `down`, `left`, `right`.

## What the audit establishes

`audit.py` imports only the Python standard library. It does not import the runner, adapter, production game, native helpers, tokenizer, model frameworks or model weights. It independently reconstructs:

- the complete ordered request, factual directional safety reasons, Manhattan distances and breadth-first reachable-space counts;
- SHA256 Fisher–Yates food priorities, occupied-cell selection and every food event;
- neck/body/wall collisions, vacating-tail behavior, food growth, move counts and caps;
- the preserved full-board `[0,0]` fallback, including possible tail-eating into a 145-entry body;
- native textual state/question rendering, encoding structure and full-row/option-position constraints;
- raw argmax, response rounding, lossless native/adapter matching and durable attempt-before-transition order;
- all reservations, forwards, internal passes, warmups, failed attempts, game summaries and serial load/unload boundaries.

The auditor independently rehashes every `freeze.files` entry and checks the 30 distinct fresh seed units, all 90 scheduled games, and all six condition orders exactly five times in their prescribed order. It verifies exclusion from all 51 distinct earlier seeds, including comparator preflight seeds. `analysis-plan.json` and `context_check.py` are mandatory frozen inputs: omitting either or changing its bytes invalidates the audit, just as changing the protocol, seeds or execution code does. Every fresh load must match the accepted native inventory hash. The two packed inventories must match their accepted condition hashes, with exactly 248 quantized linear modules, 3,569,090,560 source projection weights, unchanged 178 retained tensors, and the fixed FP32 head and exact calibration. Final unload evidence must match the loaded inventory. Source, tokenizer/checkpoint file inventory and runtime package identities are checked again for every load.

A real completion also requires the preserved native-tokenizer-only context report. Its exact bytes must match the hash in the external approval and all receipts. The audit independently reconstructs all 113 static stress requests and native textual records, checks their hashes and case order, and verifies the report's frozen utility and source provenance, complete-row accounting, option positions, maxima, absence of truncation and zero forbidden-operation counters. It does not execute the context utility or load a tokenizer. This preparation records the report as not executed; no report is fabricated or included in the freeze. A `fixtures_only` report or unit-test fixture cannot establish native tokenization coverage.

Limits are 45,000 gameplay reservations, 180 warmups and 45,180 total decisions; each game allows at most 500 successful moves and at most 500 attempts. Internal backbone passes are recorded separately. A complete real audit also verifies the preserved approval/claim and the successful supervisor lifecycle. Resource evidence is mandatory:

- Supervisor `resource_sample` records identify `sample_phase` (`initial`, `periodic` or `final`) and `worker_state` (`running` or `exited`). Exactly one initial sample must precede the first native load and contain a positive measured worker RSS plus all three storage measurements (`free_bytes`, `root_bytes`, `evidence_bytes`). Periodic samples also require positive measured RSS. A final sample must follow runner completion and process exit, record all storage measurements, and mark current RSS unavailable (`null`); it must precede the successful `worker_exit`. Missing, late, out-of-order or over-budget samples prevent a pass.
- Native `runtime`, `load_start`, `native_inventory`, `quantized_inventory`, `load_complete`, `raw_decision` and `unloaded` records must each include `rss_peak_bytes`, `mlx_active_bytes`, `mlx_peak_bytes` and `mlx_cache_bytes` for real runs. Values must be nonnegative integers, with positive RSS, peak MLX memory at least current active memory, and observed RSS/current active MLX memory within their frozen ceilings.

These checks establish the presence and consistency of the recorded observations. They do not turn periodic samples into a continuous independent measurement of resource use.

## Running the audit

With the exact frozen checkout and a completed local run directory:

```sh
python3 -B -m experiments.kev4b_campaign_v1.audit PATH_TO_RUN --repo PATH_TO_FROZEN_CHECKOUT
```

The command is read-only and emits JSON to stdout. Exit code zero and `passed: true` require a complete, real, consistent campaign. Incomplete or technically interrupted evidence emits no score dataset or inferential result. Available reservation and completion counts remain visible for diagnosis. Missing, contradictory or malformed evidence cannot be repaired by dropping records.

The offline regression suite uses explicitly synthetic traces:

```sh
python3 -B -m unittest discover -s tests -p test_kev_campaign_audit.py -v
```

The Python API's `allow_synthetic=True` exists only for unit tests; the CLI never enables it. A complete synthetic fixture reports `real: false`, `passed: false`, and `status: synthetic_complete`. These fixtures do not establish native compatibility, timing, model performance or a real-campaign pass.

## Limits and publication boundary

The audit verifies evidence consistency and independently reconstructs game behavior; it cannot prove that hardware produced a logged numerical value. It checks and retains complete token IDs and their structural alignment without loading a tokenizer or independently recomputing tokenization. Likewise, it verifies recorded tensor inventories and frozen payload hashes without loading checkpoint tensors. These boundaries must accompany any claim of audit success.

Raw run journals and approval/claim files may contain local paths, process metadata and diagnostic details. They are private execution evidence. Publication requires a separately reviewed allowlist/export, inspection of compressed as well as plain records, and preservation of provenance. This preparation adds no campaign result dataset and changes no published discovery evidence.

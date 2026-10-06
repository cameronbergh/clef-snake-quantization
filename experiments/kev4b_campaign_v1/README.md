# Kev paired-seed Snake campaign preparation

This is a prospective, separately versioned campaign preparation. **No campaign has run and preparation does not authorize execution.** The saved seed manifest and analysis choices precede every campaign outcome. The final [freeze](freeze.json) binds the reviewed code and inputs; [verification](verify.py) checks them without loading a tokenizer or model.

The completed [compatibility preflight](../kev4b_v1/PREFLIGHT_RESULTS.md) used 60 decision evaluations, including warmups, on twelve reused discovery states. It made 120 internal text-backbone passes and played zero games. Each short request needed one state-prefix pass and one question-branch pass. A decision is one four-option `model.forward(enc)`, not one text-backbone invocation. Those fixtures established native/wrapped BF16 parity and the specified in-memory quantization scope, not gameplay benefit.

## Proposed experiment

- Thirty fresh, saved 128-bit seeds, excluding all 51 distinct seeds in the CLEF and deferred Qwen manifests; 90 games across `bf16`, `mlx-affine8-g64` and `mlx-affine4-g64`.
- Each seed is a paired block. All six condition orders occur five times, in the exact saved [schedule](schedule.json). One condition is loaded at a time, freshly for every game, followed by two fixed warmups that do not move the board.
- All three conditions use the same pinned Kev checkpoint, native MLX implementation, base tokenizer, FP32 pointer head and temperature. Quantization changes 248 eligible linear projections using affine groups of 64; retained tensors and head match the accepted preflight inventories.
- Preserve the original 12×12 board, request features/order, four available actions, deterministic food priorities, unsafe-choice execution and 500-successful-move horizon. The board advances only when a decision has returned and its attempt is durable. Wall time adds no game penalty.
- Preserve the original full-board food fallback, including its possible 145-entry tail-food edge state. This campaign does not silently change the game rules.

The in-process native decision API provides the same serialized response boundary used by the adapter. A new HTTP server adds no necessary experimental control. Quantized weights are freshly derived in memory; this design does not require persisted export/reload. Neither HTTP nor export compatibility is claimed.

## Prospective analysis

The [analysis plan](analysis-plan.json) treats the **30 seeds**, not individual moves, as the independent units. The primary endpoint is food collected before collision or the fixed move horizon. Capped food scores are retained and fully observed at that horizon; uncapped lifetime remains censored.

Compare Q8−BF16 and Q4−BF16 paired mean scores with 100,000 two-sided sign-flip draws and Holm correction across those two comparisons. This relies on sign exchangeability/symmetry. Whole-seed paired bootstrap intervals use 100,000 common row resamples and are marginal, not simultaneous. Random generators, seeds and all reporting choices are saved before outcomes. Thirty seeds are a bounded design choice, not a demonstrated power guarantee.

All 90 scheduled games must finish and pass the independent audit before primary inference. Technical stops retain every row and available record; partial results are descriptive and potentially selected. No zero imputation, replacement seeds, hidden retries or automatic resumption is permitted. Selection of Kev after the CLEF discovery limits interpretation to this model and task.

## Resources and stops

The [resource plan](RESOURCE_PLAN.md) separates estimates from hard limits. The proposal permits at most **45,180 decisions**: 45,000 game attempts plus 180 warmups. Complete native rows must fit 8,192 tokens without truncation. At most eight 1,024-token prefix passes plus one branch pass gives the conservative **406,620 internal-pass ceiling**.

The whole-campaign wall limit is **12 hours**, including verification, imports, loads, conversion, warmups, games and unloads; each decision has a separately armed **120-second watchdog**. Stop on the first technical failure, identity mismatch, limit breach or lost external mount. RSS and MLX active memory each have a 48 GiB limit. Evidence is bounded at 16 GiB, the entire existing external asset root at 56 GiB, and at least 8 GiB volume free space must remain. Memory measures overlap and must not be added.

All assets, caches, temporary files and runtime evidence remain under `/Volumes/Models/clef-snake-experiments/kev4b-v1`. No downloads, internal-disk/X9 fallback or quantized exports are planned.

## Readiness and execution boundary

The serial [runner](runner.py), [native backend](native.py), independent [auditor](audit.py) and [evidence contract](EVIDENCE.md) are implemented and tested offline. Synthetic tests are explicitly labeled and cannot receive a real-campaign pass.

**Still pending before execution:** the CPU-only native tokenizer check on 113 synthetic long-body states. Its command was not executed, so this preparation claims no long-state tokenizer result. [context_check.py](context_check.py) is frozen with source/tokenizer hash guards; default mode only creates synthetic request fixtures. The explicit tokenizer mode opens no checkpoint weights, imports no MLX/checkpoint loader and performs no model forward. Synthetic coverage would still not prove an exhaustive maximum over all reachable states. Long-state model latency and memory remain unmeasured; the future campaign must stop rather than truncate or exceed its budget.

An external, unused approval must identify this exact freeze and all limits. No external execution approval or campaign directory is created by this preparation. The old preflight claim and aggregate allowance are spent provenance, not a reusable campaign budget. A later approval must also satisfy the recorded context-validation gate. Execution starts only through the supervised runner; the default command is read-only validation:

```sh
python3 -B -m experiments.kev4b_campaign_v1.verify
python3 -B -m experiments.kev4b_campaign_v1.runner
python3 -B -m unittest discover -s tests -p 'test_kev_campaign_*.py' -v
```

Future progress checks can read `runner.jsonl`, `native.jsonl` and `supervisor.jsonl` beneath `campaigns/kev4b_campaign_v1/<run-id>` on the external root. No polling schedule or recurring monitoring is created. See the evidence contract for the independent post-run audit. Raw runtime evidence must receive a separate publication allowlist review before export.

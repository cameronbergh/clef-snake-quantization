# Data and integrity

`data/2026-10-05` preserves the initial 60-game discovery snapshot. `data/2026-10-05-fiveway` is the complete 75-game snapshot, including the same earlier games plus Q2_K. Each contains all of its recorded games, not selected highlights. They share 15 environment seeds and must not be concatenated as independent datasets.

- `seed-manifest.json`: exact 15 saved random seeds and food algorithm.
- `schedule.json`: model/seed execution order.
- `rounds.csv` / `rounds.json`: all completed per-round outcomes.
- `results.json`: aggregates, paired score/move differences and limitations.
- `decisions/*.jsonl.gz`: full original browser benchmark logs, losslessly gzipped. Each line includes full request/response, seed/model/attempt, post-action board and cumulative food-event evidence. These are not edited down to selected probabilities.
- `server-logs/*.jsonl.gz`: independently recorded model-server lines filtered only by the published inference IDs; full request/response retained for exact matching.
- `provenance/`: pinned payload/official-head/runtime hashes, GGUF tensor/row checks, bridge validation.
- `evidence-inventory.json`: hashes of original uncompressed logs/results and locally verified executed-source inventory. Original unlicensed UI source is not redistributed. The independently implemented public driver is parity-verified against every original request including schema, factual features and field order.
- `original-audit.json`: original combined source/trajectory/provenance/server audit.
- `portable-audit.json`: public implementation's exhaustive independent parity and trajectory audit.
- `SHA256SUMS`: relative paths, hashes of all evidence files. Run `python3 -m clef_snake.audit data/2026-10-05-fiveway --verify-hashes` (or select the initial dataset).

Hashes provide integrity checks, not independent notarization or a guarantee that an experiment is unbiased. The exact original payloads and checks are made available so other researchers can recalculate and critique the result. No model weights, personal machine paths, private workspace files, credentials or prior git history are published.

One environment seed is the unit of paired analysis. Food candidate permutations are coupled by seed and food event, but taking the first free cell means later actual food can differ when bodies diverge. Score is food eaten before collision/cap. Do not treat per-call records as independent skill samples; do not extrapolate capped scores to uncapped survival.

Derived five-way statistics and PNG/SVG figures live separately in [`analysis/2026-10-05-fiveway`](../analysis/2026-10-05-fiveway), with source/procedure provenance. They do not replace the original four-way uncertainty or figure. The [passive watcher](../watcher/README.md) reads existing data only.

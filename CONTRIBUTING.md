# Contributing and independent replication

Contributions from humans and coding agents are welcome. Start with [AGENTS.md](AGENTS.md) and [methods/reasoning](docs/METHODS.md).

## Development checks

No model download is needed for the default development workflow:

```sh
python3 -m unittest discover -s tests -v
python3 -m compileall -q clef_snake scripts tests
python3 -m clef_snake.audit data/2026-10-05 --verify-hashes
```

CI runs these deterministic offline checks only. Analysis/figures additionally need NumPy and Matplotlib; inference needs the separately documented pinned environment. A compiler/real-model test is not required for a documentation-only change, and model-backed testing must be requested explicitly.

## Reporting a replication

Include model/runtime revisions, actual tensor types, hardware/OS/package versions, head/source hashes, full saved seed manifest, execution order, unchanged or changed protocol details, per-round scores/caps, and complete request/response provenance. Record failures instead of silently substituting results. Reusing discovery seeds is a replication, **not an independent held-out confirmation set**.

Put new evidence in a distinct dataset directory. Do not replace the original data or regenerate its seeds. Include an integrity manifest and offline audit; report deviations from the original driver/runtime clearly. Independently obtained game trajectories may differ numerically even with the same weights and seed.

## Pull requests

Explain the hypothesis or bug, changed behavior, meaningful verification, and remaining limitations. Keep policy changes separate from implementation fixes. Do not weaken parity checks or edit historical logs to fit a changed implementation. Cite upstream code and preserve licensing notices. Include AI assistance disclosure when relevant.

Do not attach checkpoints, access tokens, private logs, personal filesystem paths or unrelated git history. An independently verified new outcome is welcome even if it contradicts the initial observation.

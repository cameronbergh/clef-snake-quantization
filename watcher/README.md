# Passive benchmark watcher

`benchmark-watch.html` is a standalone, dependency-free data viewer. It contains no game controls, inference requests, remote libraries, or model loading. Its board is independently drawn from saved coordinates; no original third-party game UI is included.

From the repository root:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/watcher/benchmark-watch.html`. The page first reads `./benchmark-live.json`. If that file returns 404, it loads `../data/2026-10-05-fiveway/results.json`. Browser restrictions usually prevent JSON fetches when opening the HTML directly as a `file:` URL.

For the existing live watcher, deploy only this HTML beside the existing runtime JSON files. The viewer reads `benchmark-live.json`, optional `benchmark-amendment.json`, and optional `benchmark-persisted-aggregates.json` from that same directory. It does not write or modify those files. No data-URL query parameter or arbitrary network destination is supported, and redirects are rejected.

## Reading the charts

- Per-seed scores and wins/ties/losses use the common completed seed subset across **all five** configurations. Pairing uses the actual saved seed, never the displayed round number alone. Before all five models complete a seed, that seed does not enter these charts.
- The distribution and summary use each model's available completed games. Scores, means, medians and caps come from those rows. An outlined diamond marks a move-cap survivor; its observed score is censored.
- Latency uses the saved **server latency** mean, median and p95 across individual calls. It is not request-wall time, an average of round medians, or a matched-input benchmark. BF16 and the quantized backbones use different runtimes, and gameplay trajectories differ. A model's saved timing is hidden if its aggregate game count disagrees with its accepted rows.
- The legacy persisted IQ2 summary is used only when that model has neither live rows nor a live aggregate. It is labeled as a legacy summary, never creates seed rows, and never increases top-level game, seed, or cap counts. It does not enter paired charts. Current live evidence takes precedence.
- Exact repeated model/seed records count once. Conflicting records are excluded instead of choosing a preferred result. Missing or malformed rows produce a visible warning; they are not imputed. These checks support visualization and do not replace the repository's exhaustive evidence audit.
- The completed-game selector shows saved final boards and the last structured head decision when available. “Current / last live board” follows the live snapshot. Missing boards or decision fields remain unavailable rather than being simulated.

The fixed research caveat refers to this 75-game, 15-seed discovery snapshot. There are 15 independent environment units, not 75 independent games or thousands of independent move calls. Models were added after earlier results, and the observations establish neither a causal precision effect nor a monotonic relationship.

## Refresh and verification

An active benchmark refreshes every two seconds. Complete or published data and stopped benchmarks pause automatic polling; **Refresh** checks again. Hidden tabs pause polling, and failed requests back off to at most one request cycle per minute while preserving the last successful view. Loading a corrupt live file surfaces an error instead of silently replacing it with older published results.

The offline data checks require Node.js and use only built-in modules:

```sh
node tests/test_watcher.js
```

They evaluate the exact DOM-free normalization script embedded in the page against the immutable published fixture, malformed and partial rows, seed pairing, duplicates/conflicts, stale aggregates and the legacy IQ2 fallback. No model or GPU is involved.

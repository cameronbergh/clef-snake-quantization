# CLEF Maze protocol

Second sequential decision environment for CLEF quantization comparisons, beyond
Snake. Implementation spec: issue #10. This document covers rules, interfaces,
generation, recording, and offline validation. **No weights, downloads,
inference runs, or quantization-improvement claims are part of this protocol.**

## Rules

- Rectangular grid of cells (`width` x `height`, minimum 2x2) with interior wall
  segments, one start cell and one goal cell. Coordinates: x increases right,
  y increases down (same convention as Snake).
- Four cardinal actions: up, down, left, right. No diagonal moves.
- Fully observable: the complete wall layout, start, goal, and current position
  are visible in every request.
- No opponent, no real-time penalty.
- **Invalid move** (into a wall segment or outside the grid): position unchanged,
  logged, and consumes one attempt.
- **Goal move**: moving onto the goal ends the run in success. The goal move
  counts as an attempt.
- **Attempt cap**: exhausting the cap without reaching the goal ends the run as
  a capped failure. A run that reaches the goal on its final allowed attempt
  counts as success (goal takes precedence over the cap).
- Acting on a terminal maze raises an error; terminal runs are never extended.

## Maze generation (version `maze-v1`)

Deterministic from explicit seeds and parameters via a SHA256 counter-mode
keyed stream (`clef-maze-gen-v1:{seed}:{counter}`); no platform-dependent RNG.

1. Randomized iterative depth-first carve over the cell grid produces a perfect
   maze (spanning tree: every cell reachable, many dead ends).
2. A `loop_fraction` of the remaining interior walls is removed (keyed shuffle),
   adding loops and branching. `0.0` keeps the perfect maze.
3. Start cell: keyed uniform draw.
4. Goal cell: `target_distance="far"` picks a farthest-from-start cell (ties by
   keyed order); an integer target picks the reachable cell whose shortest-path
   distance is closest to it (ties by keyed order).
5. Reachability of every cell is asserted at generation time.

Difficulty controls (all pre-outcome, recorded in the manifest):

- `width`, `height`: grid dimensions.
- `loop_fraction`: fraction of extra walls removed; more loops, more route choice.
- `target_distance`: `"far"` or an integer shortest-path distance start-to-goal.

Exact layouts are saved in manifests (`clef_maze/fixtures/sample-manifest.json`
is a 5-maze example). Comparisons must use saved layouts, never regenerate
seeds ad hoc. Manifests are immutable once written; the generator refuses to
overwrite.

## Request / action format

`Maze.request()` returns a deterministic structured request with a fixed schema
across precision conditions:

```json
{"model": "clef-flash",
 "state": {
   "grid": {"width": 7, "height": 7,
            "coordinates": "x increases right, y increases down",
            "walls": [[x1, y1, x2, y2], "..."],
            "start": [sx, sy], "goal": [gx, gy]},
   "position": [x, y],
   "attempts_used": 3, "attempt_cap": 120, "invalid_moves": 1,
   "move_analysis": {"up": {"valid": true, "reason": "open", "next_position": [x, y]},
                     "...": {"valid": false, "reason": "wall|bounds", "next_position": [x, y]}},
   "rules": ["..."]},
 "questions": {"move": {"type": "choice", "instructions": "...",
                        "criteria": {"up": "Move north (y - 1)", "..."}}}}
```

Deliberately **not** included: recommended moves, optimal distances, or any
solver guidance. `move_analysis` reports only factual per-action validity
(open / wall / bounds), mirroring Snake's factual `move_analysis`.

## CLEF four-choice adapter

`clef_maze/adapter.py` builds the request above and verifies recorded model
responses with the same contract as the Snake driver: provenance must match the
official repo/revision with no fallback or safety override, probabilities must
cover exactly the four actions, and the recorded choice must equal the native
argmax. The adapter changes only the question text and state schema; the
official decision head and argmax behavior are preserved.

Unsupported in this implementation task (documented, not silently missing):
model acquisition/loading, the inference server, and any live campaign runner.
All offline checks run with this package only.

## Recording

The offline runner (`python3 -m clef_maze.runner`) writes, per run directory:

- `maze-manifest.json`: byte copy of the input manifest (exact layouts used).
- `config.json`: policies, policy seed, driver version. Labeled
  "scripted policies only; not model results".
- `decisions.jsonl`: one row per attempt:
  `{sequence, maze_id, policy, attempt, action, request, after}`.
- `rounds.csv`: per episode `{sequence, maze_id, policy, reached, attempts,
  invalid_moves, shortest_path_length, path_inefficiency, end_reason}`.

Paired conditions use exactly the same saved maze/start/goal/attempt cap; each
condition gets a fresh `Maze` instance whose state evolves independently
(isolation is covered by tests).

## Metrics

- Primary: fraction of episodes reaching the goal within the cap.
- Secondary: attempts, invalid moves, path inefficiency
  (`attempts / shortest_path_length`, reported for successful episodes).
- No score rewards fast failures: capped failures are failures regardless of
  attempts used; inefficiency is success-conditioned.

## Offline validation

- `python3 -m clef_maze.generate --seeds ... --out manifest.json [...]`
- `python3 -m clef_maze.runner --manifest manifest.json --out run-dir --policies greedy optimal`
- `python3 -m clef_maze.replay run-dir` (exit 0 = exact replay; non-zero names
  the first mismatch). The auditor checks, in order: (1) manifest integrity —
  sha256 of `maze-manifest.json` matches `config.json`; (2) manifest fidelity —
  every layout regenerates exactly from its recorded seed/parameters via the
  versioned generator, walls are canonical, the goal is reachable, and the
  recorded shortest-path length matches recomputation; (3) episode replay —
  every recorded request equals the recomputed request, every action applies
  cleanly, every after-state matches, with no missing/duplicate/extra rows;
  (4) summary reconciliation — `rounds.csv` reached/attempts/invalid_moves,
  end_reason, and path_inefficiency all match the replay, episode sets agree
  across files, and policies match the configured set.
- `python3 -m clef_maze.demo --manifest manifest.json --maze-id maze-001 --policy greedy`
  (add `--frames` for every step, `--manual` for WASD play). Demos are labeled
  as scripted/manual, never as model results.
- Unit tests: `python3 -m unittest discover -s tests -v` (see
  `tests/test_maze_*.py`).

Scripted policies (`clef_maze/policies.py`: random, greedy, wall-follower,
optimal) are test fixtures only — never model fallbacks, never recommendations
to a model, never mixed into model results. The optimal policy exists to
validate the generator's `shortest_path_length` and the replay auditor.

## Follow-up protocol outline (references #6; NOT frozen)

This section is an outline only. Sample size, manifests, primary contrasts,
effect thresholds, multiplicity handling, model order, and stopping rules must
be frozen in a separate protocol document before any separately authorized
model campaign. Nothing here authorizes downloads, inference, or GPU execution.

- **Environment set**: saved maze manifests at two or three difficulty settings
  (e.g. 7x7 low-loop, 9x9 mid-loop, 11x11 with integer target distances),
  disjoint from any seeds used during implementation validation.
- **Design**: paired within-maze across precision conditions (BF16 GGUF
  same-runtime control plus quants), each condition an independent trajectory
  from the identical saved start state. Counterbalanced execution order,
  recorded per episode.
- **Primary contrast**: goal-reaching fraction within cap, BF16 vs each quant,
  paired by maze. Pre-registered direction-agnostic (two-sided) tests; the
  question is whether quantization changes sequential decision quality, not
  whether it "improves" it.
- **Secondary**: invalid-move rate, path inefficiency on successful episodes,
  first-divergence analysis mirroring the Snake method (identical-state
  disagreements, feature-equivalence vs probability margins).
- **Multiplicity**: number of contrasts fixed at freeze; report all
  mazes/difficulties run, including capped failures.
- **Guardrails**: reuse the Snake integrity rules — no solver input to the
  model, no action masking, no seed substitution, provenance + argmax
  verification on every response, independent replay audit of the recorded run.

## Limitations

- The adapter's `choice()` path is specified and unit-tested against fixtures,
  but no live model campaign has been run through it.
- Difficulty controls are structural (size, loops, distance); they do not
  measure cognitive difficulty for a model.
- The ASCII demo is a visualization aid, not a metric.

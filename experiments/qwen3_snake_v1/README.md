# Qwen3 Snake v1: prepared, not run

Preparation for [issue #1](https://github.com/cameronbergh/clef-snake-quantization/issues/1). This directory contains a tested offline adapter and a prospective protocol. **There are no Qwen Snake results, model downloads, model conversions or inference calls from this preparation.** Existing CLEF evidence is unchanged.

We selected [Qwen3-4B-Instruct-2507](CANDIDATES.md), using its native non-thinking chat interface. The comparison is BF16 versus Q4_K_M and Q2_K backbone recipes, with the model's tied input/output weights retained in BF16 throughout. All variants use one pinned standard llama.cpp runtime. This is a different action interface from CLEF's decision head; it is not a direct model-skill ranking against CLEF.

## What is frozen

- [Protocol](protocol.json): original factual game state and four actions; fixed native chat template, JSON action schema, greedy decoding, 32-token response cap and explicit failure rules.
- [Seeds](seed-manifest.json) and [schedule](schedule.json): 30 fresh paired main seeds, six separate preflight seeds, 90 planned games; each of the six model orders occurs five times. Seeds are disjoint from published CLEF seeds.
- [Analysis plan](ANALYSIS_PLAN.md): paired capped-food differences, fixed resampling/test procedures and Holm adjustment for two comparisons. All failures, losses and alive caps remain visible.
- [Artifact plan](artifact-plan.json): official source revision, exact source file sizes/hashes, native template hash, installed runtime/library fingerprints and converter revision. Generated GGUF hashes do not exist yet.
- [Storage plan](storage-plan.json): external destination, exact source-byte accounting, separate reservations for files that do not exist yet, and a conservative free-space gate.
- [Freeze manifest](freeze.json): SHA256 bindings for these plans, the adapter, verification code, tests and shared game implementation. The publishing Git commit establishes this preparation before results. Future amendments require a new version; do not rewrite this freeze after outcomes.

## Offline use

From the repository root:

```sh
python3 -m unittest discover -s tests -v
python3 -m experiments.qwen3_snake_v1.verify
# macOS only; read-only, creates no directories and starts no model:
python3 -m experiments.qwen3_snake_v1.storage
```

`adapter.build_request(game)` and `serialize_request(game)` produce the request for `/v1/chat/completions`. `parse_response(raw)` validates a complete raw response. `decide(game, transport)` makes exactly one call to an injected transport and returns the exact request, losslessly retained response, action/usage or explicit error. It does not apply a move, retry, call HTTP, start a server or load weights. The eventual runner must persist that record before applying a valid action once. Invalid or truncated responses stop execution; they are not zero scores or collisions.

## External storage is mandatory

The dedicated mounted partition is **`/Volumes/Models`**, an external APFS volume. It is separate from `Crucial X9`; X9 is not this experiment's fallback. The planned root is `/Volumes/Models/clef-snake-experiments/qwen3-v1`. All model weights, HF caches, temporary files, conversions and evidence output must remain there. The storage module refuses a missing/redirected mount, an internal disk, a different volume, insufficient free space, or a cache path outside that volume. It never creates a mount directory or moves/deletes anything.

At preparation the Models volume had **238,862,585,856 bytes** available. The complete pinned source snapshot is **8,060,917,568 bytes**, including **8,044,982,000 weight bytes**. No verified matching cache payload is credited. The total reservation is **39,467,865,920 bytes**, including a 6 GiB untouched margin; the launch gate is **40 GiB (42,949,672,960 bytes)**. That leaves **199,394,719,936 bytes** after the detailed reservations at the recorded observation. Recheck immediately before any future write. These are exact arithmetic totals, not measurements of uncreated GGUF files or logs.

Use a single source cache on that volume, not both a cache and a second copied snapshot. [The storage plan](storage-plan.json) specifies `HF_HOME`, `HF_HUB_CACHE`, `XDG_CACHE_HOME` and `TMPDIR`, including download/conversion transients. The internal disk remains excluded even if it temporarily has free space.

## Remaining work and smallest next step

The smallest acquisition step, after explicit authorization, is to rerun the external-volume gate and fetch only the pinned official source snapshot into the external cache, verifying every byte count and payload hash. No quantization or game is implied by that permission. Confirm dependency availability without installing missing packages automatically.

Then verify BF16 conversion, tensor inventory, tied I/O equality, native template and runtime compatibility before producing the two custom quantizations. Unsupported BF16 or a required configuration change stops this version; no silent F16 fallback is allowed. A separately authorized bounded preflight consists of 18 initial-state requests on the six saved preflight seeds, without scored rollouts.

A serial HTTP runner and an independent auditor for the new response schema still need implementation and testing before the 90-game campaign. They must enforce the frozen order, record independent server evidence, monitor storage, keep failures and prevent concurrent clients. No campaign is enabled by this preparation or by passing its offline checks.

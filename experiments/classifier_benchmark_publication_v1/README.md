# Compact benchmark publication adapter v1

A publication-only repair, identified during execution before any result analysis: the pinned public CC0 suite includes an intentional PII-detection case with a synthetic bearer-auth header ending in `...example`. The original frozen exporter used an overly broad token-marker rejection. That original source remains immutable in the 24-file preparation freeze.

This exporter preserves every original request/response/source byte. Its privacy inspection exempts **only the exact known public synthetic header**, after verifying the immutable upstream suite byte digest and canonical lock and confirming one source case contains it. Other Bearer markers, private user/home paths and private token prefixes are still rejected. Plain and compressed payloads are inspected. Public provenance records the changed export source and where this known public example occurs. No inference, answer, scoring, order, stopping budget or frozen analysis is changed; no dataset case is dropped or redacted.

The allowed export remains a fixed preparation allowlist, all six full compressed decision logs, complete status/resource receipts and portable commitments. Private local asset-path config and raw worker stdout/stderr stay excluded; their hashes are retained. Publication authorizes no further model calls.

```sh
python3 -B -m experiments.classifier_benchmark_publication_v1.export EXTERNAL_RUN NEW_PUBLIC_DATA_DIRECTORY
```

## Nested hash inventory audit repair

The first completed public export exposed a second publication-only defect: the frozen auditor excluded every filename `hashes.json`, including the vendored upstream `cases/hashes.json`, although the exporter correctly includes that lock in the complete portable manifest. The original frozen source and failed export/log are preserved.

The publication auditor retains every original exhaustive check and changes only the exclusion boundary to the **root** manifest. It therefore verifies an additional nested lock file, rather than omitting evidence or weakening verification. `analysis.py` routes the unchanged frozen numerical/statistical analysis through this corrected auditor; no score, sample, bootstrap, metric or statistical algorithm changes. All wrappers and source hashes are included in public provenance.

Use the corrected offline entrypoints:

```sh
python3 -B -m experiments.classifier_benchmark_publication_v1.audit PUBLIC_DATA_DIRECTORY
python3 -B -m experiments.classifier_benchmark_publication_v1.analysis PUBLIC_DATA_DIRECTORY NEW_ANALYSIS_DIRECTORY
```

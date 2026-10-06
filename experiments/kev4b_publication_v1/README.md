# Completed Kev campaign publication, version 1

This offline pipeline publishes the completed `kev4b_campaign_v1`. It authorizes no acquisition, model loading, inference, retry or resume. [Results](../../docs/KEV_CAMPAIGN_RESULTS.md), [data](../../data/2026-10-06-kev4b-campaign-v1/) and [figures](../../analysis/2026-10-06-kev4b-campaign-v1/) report that neither primary comparison met the improvement criterion.

All 40 frozen preparation files plus `freeze.json` remain byte-identical in [the preparation archive](../../archives/kev4b_campaign_v1/108fc446/). This includes historical root documentation/CI. The portable auditor imports the unchanged independent frozen engine and verifies its source hash against the archive. Validate preparation with the explicit archive:

```sh
python -B -c "from experiments.kev4b_campaign_v1.verify import verify; print(verify('archives/kev4b_campaign_v1/108fc446'))"
python -B -m experiments.kev4b_publication_v1.audit data/2026-10-06-kev4b-campaign-v1
```

`export.py` requires a new output directory and a complete passing original audit. It reads six allowlisted files, retains every journal row and exact wire payload, aliases pinned asset paths, commits the consumed one-use identity, excludes private receipts/history, emits deterministic gzip and checks the portable reconstruction/hash manifest. Originals are rehashed afterward to verify they were unchanged. Public receipts are non-executable. Redacted public logs are not byte-identical archives of private original files; original hashes identify those preserved originals.

`analysis.py` audits public data before applying the frozen seed-level analysis. It needs NumPy/Matplotlib only for CPU statistics/figures, refuses an existing output directory and records every paired row, multiplicity/interval choices and source/input/output hashes. Neither auditor nor analysis imports model frameworks.

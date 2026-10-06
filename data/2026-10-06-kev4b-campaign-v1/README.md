# Kev 4B prospective Snake campaign

90 complete games, 30 fresh paired seeds, BF16 / affine8 / affine4; 180 warmups. All three conditions use the same native MLX runtime and unchanged FP32 pointer head. No improvement criterion was met. See ../../docs/KEV_CAMPAIGN_RESULTS.md and ../../analysis/2026-10-06-kev4b-campaign-v1/.

Every runner/native/supervisor row is retained in deterministic gzip. Requests, responses, logits, token IDs, inventories and trajectories retain exact values and order. Only resolver asset paths are aliased and one-use IDs committed with SHA256. Original journals are preserved outside Git; their hashes are in provenance.json. Private control/approval/history files are excluded. authorization-receipt.json is not an execution approval.

Audit without weights, tokenizer or third-party packages:

```sh
python -B -m experiments.kev4b_publication_v1.audit data/2026-10-06-kev4b-campaign-v1
```

The original local audit verified private approval bytes. The portable audit rechecks all gameplay, numerical, tensor, source, context and resource evidence, but cannot recheck private authorization history or prove that hardware produced the logged values.

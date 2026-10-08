# Compact non-game CLEF decision benchmark v1

Explicitly authorized by Cameron on 2026-10-07 to start and complete the compact suite. This namespace is separate from all historical Snake/Kev data and consumed approvals. Imports and audits are offline; running `supervisor` is the deliberate opt-in model execution action. It never contacts inference endpoints of the original servers.

Pinned upstream: [jabr/classifier-benchmark at afb83bee](https://github.com/jabr/classifier-benchmark/tree/afb83bee3b74064ae5a5d58c05b352a8d0ef7240). Full locked **v2 only**: 49 tasks, 866 public synthetic cases, CC0. The retained vendor files are attribution/protocol references, not imported as the local runner. Their instructions/criteria/state remain exact. The derived `vendor/cases/v2.json` mirror enables standard-library Python 3.9 audits; its canonical digest is independently checked against the upstream lock, while the original TOML byte digest is also verified. The upstream canonical digest and file-byte digest are both verified.

Six conditions retain the unchanged official JointSchemaHead: official PyTorch/MPS BF16, same-runtime GGUF BF16, Q6_K_L, Q4_K_M, IQ2_M, Q2_K. Primary contrasts use GGUF BF16. Each has three compatibility warmups followed by all 866 cases, serially. Complete budget: 5,196 scored + 18 warmup = 5,214 forwards. Read `protocol.json` for stopping, storage, interpretation and the prospective analysis. Fixed order is SHA256-derived before inference; this is serial blocked order, not full counterbalancing.

`freeze.py` verifies/hashes existing read-only assets, generates exact requests and full token commitments without inference, inventories tensor types and dependencies, and freezes all preparation source. It never downloads, converts or substitutes model files. Execution uses the existing quant environment and bridge, with all new cache/temp/log outputs on the verified external Models volume; Python bytecode and MPS CPU fallback are disabled. The supervisor protects memory, disk, output size and time. Fatal errors stop with partial evidence, without retries. `handoff.json` on the external root is durable progress/recovery evidence.

The worker instruments only the wrapper's forward method to capture unrounded float32 head-softmax probabilities from the same pass used by the official `systemone` response. No second pass or head-weight mutation occurs. Native choice, native rounded noul threshold and returned score distribution argmax match upstream scoring. Score expected-value rounding is not a permitted substitute. Raw-vs-rounded prediction boundary differences are disclosed. Each exact input is token-hash checked at inference.

Offline portable evidence audit:

```sh
python3.12 -B -m experiments.classifier_benchmark_v2_v1.audit data/2026-10-07-classifier-benchmark-v2-v1
```

Analysis (NumPy required, no model/GPU calls):

```sh
python3.12 -B -m experiments.classifier_benchmark_v2_v1.analysis data/2026-10-07-classifier-benchmark-v2-v1 runs/classifier-analysis-reproduction
```

`export.py` copies only a fixed preparation/source allowlist, full compressed response logs, status/resource receipts and portable commitments. Private local path config and raw worker stdout/stderr remain excluded; public provenance retains their hashes. All plain/compressed payloads are inspected before delivery. No weights, credentials, personal histories or earlier server logs are exported.

Public synthetic data are not held-out confirmation; no validated difficulty split or broad intelligence claim is supported. Descriptive task bootstrap preserves paired task vectors. All tasks and probability calibration are exploratory secondary analyses. Latency is descriptive because unrelated services are untouched.

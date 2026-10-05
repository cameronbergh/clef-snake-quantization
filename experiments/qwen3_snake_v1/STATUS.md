# Deferred: general instruction-model comparator

Status correction, 2026-10-05. This note supersedes any suggestion in the preserved preparation documents that Qwen3-4B-Instruct-2507 satisfies the intended Jev-style decision-model comparison in issue #1.

Qwen3-4B-Instruct is a general instruction model. This adapter constrains its generated JSON and parses an action. Native decision models instead compute typed answers and option probabilities through their decision architecture. Sharing an output schema does not make these the same model class.

The preparation committed in `7b92994813fda55d5f131945c699c7251dbaa976` remains unchanged: adapter, saved seeds/schedule, prospective analysis, artifact/storage plan and all 17 freeze-listed files. This additive status note does not modify that historical freeze or claim a new experiment. The offline verifier confirms the preserved preparation, not eligibility for issue #1.

**No model acquisition, conversion, preflight, inference or scored campaign has been executed.** Qwen is deferred as an optional general instruction-model comparator. Do not proceed with its acquisition or runner as an incidental next step.

See the [native decision-model candidate assessment](../../docs/DECISION_MODEL_CANDIDATES.md) and [current research TODO](../../TODO.md). Selection and compatibility validation come before a separately authorized new campaign.

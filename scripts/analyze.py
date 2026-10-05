"""Recalculate exploratory seed-level statistics and publication figures.

Usage: python analyze.py path/to/combined/results.json [output_directory]
No model calls. All resampling units are complete, paired seed rows.
"""
import itertools
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ["bf16", "q6-k-l", "q4-k-m", "iq2-m", "q2-k"]
LABELS = {"bf16": "BF16", "q6-k-l": "Q6_K_L", "q4-k-m": "Q4_K_M",
          "iq2-m": "IQ2_M", "q2-k": "Q2_K"}
COLORS = ["#4b5563", "#2563eb", "#06a096", "#a855f7", "#e88d24"]


def interval(samples):
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


def sign_flip_p(differences):
    """Exact two-sided symmetry test; zeros don't change enumeration."""
    d = np.asarray(differences, dtype=int)
    d = d[d != 0]
    if not len(d):
        return 1.0
    observed = abs(int(d.sum()))
    extreme = sum(abs(sum(s * int(v) for s, v in zip(signs, d))) >= observed
                  for signs in itertools.product((-1, 1), repeat=len(d)))
    return extreme / (2 ** len(d))


def sign_test_p(differences):
    d = np.asarray(differences)
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n)


def main(source, out):
    raw = json.loads(source.read_text())
    paired = raw["paired_seed_differences"]
    models = [m for m in MODELS if m in paired[0]["scores"]]
    seeds = raw["seed_manifest"]["seeds"]
    assert len(paired) == 15 == len(set(seeds))
    assert [p["seed"] for p in paired] == seeds
    assert raw["total_rounds"] == len(models) * len(seeds)
    scores = np.array([[p["scores"][m] for m in models] for p in paired], dtype=float)
    rng = np.random.default_rng(20261005)
    draws = rng.integers(0, len(seeds), size=(100_000, len(seeds)))
    bootstrap_means = scores[draws].mean(axis=1)
    means = scores.mean(axis=0)
    statistics = {}
    for index, model in enumerate(models):
        rounds = [r for r in raw["rounds"] if r["model"] == model]
        assert len(rounds) == 15
        assert sorted(r["score"] for r in rounds) == sorted(scores[:, index])
        summary = {"n_seeds": 15, "mean": float(means[index]),
                   "median": float(np.median(scores[:, index])),
                   "range": [float(scores[:, index].min()), float(scores[:, index].max())],
                   "sample_sd": float(scores[:, index].std(ddof=1)),
                   "bootstrap_95_ci_mean": interval(bootstrap_means[:, index]),
                   "capped_alive": sum(r["end_reason"] == "move_cap" for r in rounds)}
        if index:
            d = scores[:, index] - scores[:, 0]
            summary["versus_bf16"] = {
                "differences": d.astype(int).tolist(), "mean_difference": float(d.mean()),
                "relative_mean_change_percent": float((means[index] / means[0] - 1) * 100),
                "bootstrap_95_ci_mean_difference": interval(bootstrap_means[:, index] - bootstrap_means[:, 0]),
                "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
                "sign_test_two_sided_p_uncorrected": sign_test_p(d),
                "sign_flip_two_sided_p_uncorrected": sign_flip_p(d),
            }
        statistics[model] = summary
    # Family is all included quant variants versus BF16, not a selected best one.
    ordered = sorted(models[1:], key=lambda m: statistics[m]["versus_bf16"]["sign_flip_two_sided_p_uncorrected"])
    previous = 0.0
    for rank, model in enumerate(ordered):
        p = statistics[model]["versus_bf16"]["sign_flip_two_sided_p_uncorrected"]
        previous = max(previous, min(1.0, p * (len(ordered) - rank)))
        statistics[model]["versus_bf16"]["sign_flip_holm_adjusted_p"] = previous
    iq_index, q6_index = models.index("iq2-m"), models.index("q6-k-l")
    iq_q6 = scores[:, iq_index] - scores[:, q6_index]
    notes = [
        "Exploratory, post-observation analyses. Models were added adaptively; intervals are unadjusted, descriptive percentile-bootstrap intervals, not confirmatory evidence.",
        "Independent sample size is 15 paired environment seeds, not 60/75 games or thousands of correlated move decisions.",
        "100,000 nonparametric paired-seed bootstrap resamples; NumPy PCG64 default_rng seed 20261005; 2.5th/97.5th percentile intervals. Resample entire seed rows, retaining cross-model pairing.",
        "Sign-flip p-values assume symmetry/exchangeability of paired differences under the null. Model labels were not randomized treatments; these are not causal randomization tests. Holm correction covers included quant-versus-BF16 tests only, not the whole adaptive analysis process.",
        "Score is food collected by collision or the 500-successful-move cap. Capped runs remain in score summaries; no uncensored survival or ultimate-score claim is made.",
        "BF16 versus GGUF variants changes both precision and runtime. IQ2 and Q2 ran later. Paired seed/event priorities may produce different food coordinates once bodies diverge.",
        "Selected same-runtime IQ2-versus-Q6 comparison is also exploratory, with unadjusted interval.",
    ]
    result = {"source_name": source.name, "total_executions": raw["total_rounds"],
              "independent_paired_seed_units": 15, "resampling_replicates": 100000,
              "resampling_seed": 20261005, "models": statistics,
              "exploratory_iq2_vs_q6": {"mean_difference": float(iq_q6.mean()),
                  "bootstrap_95_ci_mean_difference": interval(bootstrap_means[:, iq_index] - bootstrap_means[:, q6_index]),
                  "wins": int((iq_q6 > 0).sum()), "ties": int((iq_q6 == 0).sum()), "losses": int((iq_q6 < 0).sum())},
              "methods_and_limitations": notes}
    out.mkdir(parents=True, exist_ok=True)
    (out / "uncertainty.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Exploratory seed-level uncertainty", "", f"{raw['total_rounds']} executions, **15 paired seed units**. Score is food collected before collision or the 500-move cap.", "",
             "| Model | Mean score | Difference vs BF16 | Descriptive paired-bootstrap 95% interval for difference | Wins / ties / losses |",
             "|---|---:|---:|---|---|"]
    for m in models:
        s = statistics[m]
        if m == "bf16":
            lines.append(f"| {LABELS[m]} | {s['mean']:.2f} | — | — | — |")
        else:
            d = s["versus_bf16"]; lo, hi = d["bootstrap_95_ci_mean_difference"]
            lines.append(f"| {LABELS[m]} | {s['mean']:.2f} | {d['mean_difference']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {d['wins']} / {d['ties']} / {d['losses']} |")
    lines += ["", "## Interpretation", "", "These data are a promising task-specific observation, not evidence that quantization universally or monotonically improves models. The IQ2 gain is larger than the Q6/Q4 differences on these seeds. Fresh, prospectively specified held-out trials are needed before a confirmatory claim.", "", "## Methods and caveats", "", *["- " + n for n in notes], "", "## Exploratory tests", "", "P-values are supplied for transparency, not as a headline. The symmetry assumption, adaptive model selection, small sample, and runtime/order confounds limit their interpretation.", "",
             "| Variant vs BF16 | Two-sided sign test, uncorrected | Two-sided sign-flip, uncorrected | Holm-adjusted sign-flip (included variants) |", "|---|---:|---:|---:|"]
    for m in models[1:]:
        d = statistics[m]["versus_bf16"]
        lines.append(f"| {LABELS[m]} | {d['sign_test_two_sided_p_uncorrected']:.4f} | {d['sign_flip_two_sided_p_uncorrected']:.4f} | {d['sign_flip_holm_adjusted_p']:.4f} |")
    (out / "UNCERTAINTY.md").write_text("\n".join(lines) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.2, 1]}, layout="constrained")
    x = np.arange(len(models))
    for row in scores:
        axes[0].plot(x, row, color="#a1a1aa", alpha=.25, lw=.75, zorder=1)
    for j, m in enumerate(models):
        axes[0].scatter(np.repeat(j, len(seeds)), scores[:, j], color=COLORS[j], alpha=.65, s=20, zorder=2)
        axes[0].scatter(j, means[j], color=COLORS[j], marker="D", s=90, edgecolor="white", linewidth=1.3, zorder=3)
        axes[0].annotate(f"{means[j]:.2f}", (j, means[j]), xytext=(8, 4), textcoords="offset points", color=COLORS[j], weight="bold")
    axes[0].set_xticks(x, [LABELS[m] for m in models], rotation=20)
    axes[0].set_ylabel("Food collected (500-move-cap score)")
    axes[0].set_title("15 shared seeds · dots = games, diamonds = means", loc="left", fontsize=11)
    axes[0].grid(axis="y", alpha=.15)
    for j, m in enumerate(models[1:]):
        d = statistics[m]["versus_bf16"]; lo, hi = d["bootstrap_95_ci_mean_difference"]; mean = d["mean_difference"]
        axes[1].errorbar(mean, j, xerr=[[mean-lo], [hi-mean]], fmt="o", color=COLORS[j+1], capsize=5, markersize=7, lw=2)
    axes[1].axvline(0, color="#71717a", linestyle="--", lw=1)
    axes[1].set_yticks(np.arange(len(models)-1), [LABELS[m] for m in models[1:]])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean food difference versus BF16")
    axes[1].set_title("Paired-seed differences · descriptive 95% bootstrap CI", loc="left", fontsize=11)
    axes[1].grid(axis="x", alpha=.15)
    fig.suptitle("CLEF-Flash Snake: exploratory quantization results", fontsize=16, weight="bold")
    fig.text(.5, -.035, "n=15 seeds; intervals unadjusted; BF16/runtime and later IQ2/Q2 order confounds; not a monotonic or causal claim", ha="center", fontsize=9, color="#52525b")
    for ext in ("png", "svg"):
        fig.savefig(out / ("quantization-snake-results." + ext), dpi=220, bbox_inches="tight")
    print(json.dumps({"total_executions": raw["total_rounds"], "models": statistics,
                      "exploratory_iq2_vs_q6": result["exploratory_iq2_vs_q6"]}, indent=2))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(__file__).parent)

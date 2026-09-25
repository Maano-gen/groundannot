#!/usr/bin/env python3
"""
stats.py — aggregate N-run GroundAnnot benchmark with bootstrap CIs,
learning curves, and leave-one-out sensitivity analysis.

Reads all llm_baseline_*-run*.json files, groups by model, and reports:
  - mean ± SD of unsupported / allowed_not_enriched / valid_enriched
  - bootstrap 95% CI on valid rate (1000 resamples)
  - learning curve: estimate of valid rate vs number of runs
  - leave-one-disease-out sensitivity
  - leave-one-model-out sensitivity
  - per-term stability: fraction of runs each GO ID appeared in
  - Jaccard similarity between pairs of runs

Writes to stats/:
  summary.csv
  bootstrap_ci.csv
  per_disease_per_run.csv
  per_term_stability_<model>.csv
  leave_one_disease_out.csv
  leave_one_model_out.csv
  valid_rate_by_model.png
  learning_curve.png
  heatmap_<model>.png
  per_term_stability.png
  run_overlap.png
"""

import json
import os
import re
import glob
import random
from collections import defaultdict
from statistics import mean, stdev
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN_PATTERN = re.compile(r"^llm_baseline_(.+)\.json$")
RUN_SUFFIX  = re.compile(r"-run\d+$")
GO_RE       = re.compile(r"GO:\d{7}")

STATS_DIR = Path("stats")
STATS_DIR.mkdir(exist_ok=True)

BOOTSTRAP_N = 1000
random.seed(42)


def load_runs():
    models = defaultdict(list)
    for path in sorted(
        glob.glob("runs/llm_baseline_*-run*.json")
        + glob.glob("llm_baseline_*-run*.json")):
        m = RUN_PATTERN.match(os.path.basename(path))
        if not m:
            continue
        tag = m.group(1)
        model_key = RUN_SUFFIX.sub("", tag)
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
        models[model_key].append((tag, rows))
    return models


def m_sd(x):
    return (mean(x), stdev(x) if len(x) > 1 else 0.0)


def valid_rate_of_run(rows):
    total = sum(r["llm_ids"] for r in rows)
    valid = sum(r["valid_enriched"] for r in rows)
    return valid / max(total, 1)


def bootstrap_ci(values, n=BOOTSTRAP_N, alpha=0.05):
    if len(values) < 2:
        return (values[0], values[0]) if values else (0.0, 0.0)
    means = []
    for _ in range(n):
        sample = [random.choice(values) for _ in values]
        means.append(mean(sample))
    means.sort()
    lo = means[int(alpha / 2 * n)]
    hi = means[int((1 - alpha / 2) * n)]
    return lo, hi


def summarise(models):
    rows = ["model,n_runs,llm_ids_mean,llm_ids_sd,unsupported_mean,unsupported_sd,"
            "ane_mean,ane_sd,valid_mean,valid_sd,valid_rate_mean,valid_rate_sd"]
    boot = ["model,n_runs,valid_rate_mean,ci_low,ci_high"]

    for model, runs in sorted(models.items()):
        llm_ids = [sum(r["llm_ids"] for r in rows_) for _, rows_ in runs]
        unsup   = [sum(r["unsupported"] for r in rows_) for _, rows_ in runs]
        ane     = [sum(r["allowed_not_enriched"] for r in rows_) for _, rows_ in runs]
        valid   = [sum(r["valid_enriched"] for r in rows_) for _, rows_ in runs]
        rates   = [valid_rate_of_run(rows_) for _, rows_ in runs]

        lm, ls = m_sd(llm_ids)
        um, us = m_sd(unsup)
        am, asd = m_sd(ane)
        vm, vs = m_sd(valid)
        rm, rs = m_sd(rates)

        rows.append(
            f"{model},{len(runs)},{lm:.2f},{ls:.2f},{um:.2f},{us:.2f},"
            f"{am:.2f},{asd:.2f},{vm:.2f},{vs:.2f},{rm:.4f},{rs:.4f}"
        )

        lo, hi = bootstrap_ci(rates)
        boot.append(f"{model},{len(runs)},{rm:.4f},{lo:.4f},{hi:.4f}")

    (STATS_DIR / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (STATS_DIR / "bootstrap_ci.csv").write_text("\n".join(boot) + "\n", encoding="utf-8")


def per_disease_table(models):
    lines = ["model,disease,run,llm_ids,unsupported,ane,valid_enriched"]
    for model, runs in sorted(models.items()):
        for tag, rows in runs:
            for r in rows:
                lines.append(
                    f"{model},{r['disease']},{tag},{r['llm_ids']},"
                    f"{r['unsupported']},{r['allowed_not_enriched']},"
                    f"{r['valid_enriched']}"
                )
    (STATS_DIR / "per_disease_per_run.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def per_term_stability(models):
    for model, runs in sorted(models.items()):
        n_runs = len(runs)
        per_disease = defaultdict(lambda: defaultdict(int))
        for _, rows in runs:
            for r in rows:
                ids = set(GO_RE.findall(r.get("llm_text", "")))
                for tid in ids:
                    per_disease[r["disease"]][tid] += 1

        out = ["disease,term_id,appeared_in_runs,n_runs,stability"]
        for disease, counts in sorted(per_disease.items()):
            for tid, c in sorted(counts.items(), key=lambda x: -x[1]):
                out.append(f"{disease},{tid},{c},{n_runs},{c / n_runs:.2f}")
        (STATS_DIR / f"per_term_stability_{model}.csv").write_text(
            "\n".join(out) + "\n", encoding="utf-8"
        )


def leave_one_disease_out(models):
    lines = ["model,excluded_disease,n_runs,valid_rate"]
    for model, runs in sorted(models.items()):
        # collect per-disease valid rate per run
        all_diseases = [r["disease"] for r in runs[0][1]]
        for excluded in all_diseases:
            rates = []
            for _, rows in runs:
                tot = 0
                val = 0
                for r in rows:
                    if r["disease"] == excluded:
                        continue
                    tot += r["llm_ids"]
                    val += r["valid_enriched"]
                rates.append(val / max(tot, 1))
            lines.append(f"{model},{excluded},{len(runs)},{mean(rates):.4f}")
    (STATS_DIR / "leave_one_disease_out.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def leave_one_model_out(models):
    lines = ["excluded_model,n_models_kept,valid_rate_of_remaining"]
    all_rates = {}
    for model, runs in models.items():
        all_rates[model] = [valid_rate_of_run(rows) for _, rows in runs]

    for excluded in sorted(models.keys()):
        kept = [r for m, rs in all_rates.items() if m != excluded for r in rs]
        lines.append(f"{excluded},{len(models) - 1},{mean(kept):.4f}")
    (STATS_DIR / "leave_one_model_out.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def plot_valid_rate(models):
    ordered = sorted(models.items())
    means = []
    sds = []
    names = []
    for model, runs in ordered:
        rates = [valid_rate_of_run(rows) for _, rows in runs]
        m, s = m_sd(rates)
        names.append(model)
        means.append(m)
        sds.append(s)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=sds, capsize=6, color="#4a90d9", edgecolor="black")
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(i, m + s + 0.02, f"{m:.3f}\n±{s:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Valid enriched rate (mean ± SD)")
    ax.set_title("Valid enriched rate across runs per model")
    ax.set_ylim(0, max(means) + 0.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(STATS_DIR / "valid_rate_by_model.png", dpi=150)
    plt.close(fig)


def plot_learning_curve(models):
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, runs in sorted(models.items()):
        rates = [valid_rate_of_run(rows) for _, rows in runs]
        running = [mean(rates[: i + 1]) for i in range(len(rates))]
        ax.plot(range(1, len(running) + 1), running, marker="o",
                markersize=3, label=model)
    ax.set_xlabel("Number of runs")
    ax.set_ylabel("Running mean of valid enriched rate")
    ax.set_title("Learning curve — estimate vs number of runs")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(STATS_DIR / "learning_curve.png", dpi=150)
    plt.close(fig)


def plot_per_disease_heatmap(models):
    for model, runs in sorted(models.items()):
        diseases = [r["disease"] for r in runs[0][1]]
        matrix = np.zeros((len(diseases), len(runs)))
        for j, (_, rows) in enumerate(runs):
            for i, r in enumerate(rows):
                matrix[i, j] = r["valid_enriched"]

        fig, ax = plt.subplots(
            figsize=(0.25 * len(runs) + 3, 0.7 * len(diseases) + 2)
        )
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([f"r{i+1}" for i in range(len(runs))],
                           rotation=90, fontsize=6)
        ax.set_yticks(range(len(diseases)))
        ax.set_yticklabels(diseases)
        ax.set_title(f"{model}: valid_enriched per disease × run")
        fig.colorbar(im, ax=ax, label="valid_enriched")
        fig.tight_layout()
        fig.savefig(STATS_DIR / f"heatmap_{model}.png", dpi=150)
        plt.close(fig)


def plot_term_stability_hist(models):
    fig, axes = plt.subplots(1, len(models),
                             figsize=(5 * len(models), 4), squeeze=False)
    for ax, (model, runs) in zip(axes[0], sorted(models.items())):
        n_runs = len(runs)
        per_disease = defaultdict(lambda: defaultdict(int))
        for _, rows in runs:
            for r in rows:
                ids = set(GO_RE.findall(r.get("llm_text", "")))
                for tid in ids:
                    per_disease[r["disease"]][tid] += 1

        stability = [c / n_runs
                     for counts in per_disease.values()
                     for c in counts.values()]
        ax.hist(stability, bins=10, range=(0, 1),
                color="#4a90d9", edgecolor="black")
        ax.set_title(model)
        ax.set_xlabel("Term stability (fraction of runs)")
        ax.set_ylabel("Unique GO IDs")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(STATS_DIR / "per_term_stability.png", dpi=150)
    plt.close(fig)


def plot_run_overlap(models):
    fig, axes = plt.subplots(1, len(models),
                             figsize=(5 * len(models), 5), squeeze=False)
    for ax, (model, runs) in zip(axes[0], sorted(models.items())):
        n = len(runs)
        sets = []
        for _, rows in runs:
            s = set()
            for r in rows:
                s |= set(GO_RE.findall(r.get("llm_text", "")))
            sets.append(s)
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                a, b = sets[i], sets[j]
                mat[i, j] = len(a & b) / max(len(a | b), 1)
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{model}\nJaccard overlap")
        fig.colorbar(im, ax=ax, label="Jaccard")
    fig.tight_layout()
    fig.savefig(STATS_DIR / "run_overlap.png", dpi=150)
    plt.close(fig)


def main():
    models = load_runs()
    if not models:
        print("No llm_baseline_*-run*.json files found. Run ./run_many.sh first.")
        return

    print(f"Found {sum(len(v) for v in models.values())} runs "
          f"across {len(models)} models:")
    for m, runs in models.items():
        print(f"  {m}: {len(runs)} runs")

    print("\nSummarising...")
    summarise(models)
    per_disease_table(models)
    per_term_stability(models)

    print("Sensitivity analysis...")
    leave_one_disease_out(models)
    leave_one_model_out(models)

    print("Plotting...")
    plot_valid_rate(models)
    plot_learning_curve(models)
    plot_per_disease_heatmap(models)
    plot_term_stability_hist(models)
    plot_run_overlap(models)

    print(f"\nWrote outputs to {STATS_DIR.resolve()}/")
    for p in sorted(STATS_DIR.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()

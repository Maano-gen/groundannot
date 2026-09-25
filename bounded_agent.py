#!/usr/bin/env python3
"""
bounded_agent.py — Compute the bounded-agent result from existing runs.

The bounded agent runs the same LLM but intersects its picks with the
backend enriched set (enriched_*.json). Any pick not in the enriched set
is discarded. This script computes, for every existing run:
  - how many of the LLM's picks survive the filter
  - how many are dropped
  - how many diseases end up with zero surviving picks

No new LLM calls. Reads existing llm_baseline_*-run*.json.

Outputs:
  stats/bounded_per_run.csv     one row per (model, disease, run)
  stats/bounded_summary.csv     per model: mean coverage, zero-pick rate
  stats/bounded_coverage.png    bar chart of mean kept picks per model
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GO_RE = re.compile(r"GO:\d{7}")
STATS_DIR = Path("stats")
STATS_DIR.mkdir(exist_ok=True)


def find_data_dir(explicit=None):
    if explicit:
        return Path(explicit).resolve()
    env = os.environ.get("GROUNDANNOT_DATA")
    if env:
        return Path(env).resolve()
    for start in (Path(__file__), Path.cwd()):
        p = start.resolve()
        for parent in [p] + list(p.parents):
            snap = parent / "data" / "snapshot_20260924"
            if snap.is_dir():
                return snap.resolve()
    return None


def load_ids(path):
    """Load GO IDs from allowed/enriched JSON (same logic as benchmark_llm)."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    ids = set()

    def _extract(x):
        if isinstance(x, str) and x.startswith("GO:"):
            ids.add(x)
        elif isinstance(x, dict):
            tid = x.get("id") or x.get("term_id") or x.get("native")
            if tid and str(tid).startswith("GO:"):
                ids.add(str(tid))

    if isinstance(data, dict):
        if isinstance(data.get("terms"), list):
            for x in data["terms"]:
                _extract(x)
        for v in data.values():
            if isinstance(v, list):
                for x in v:
                    _extract(x)
    elif isinstance(data, list):
        for x in data:
            _extract(x)
    return ids


def normalise_model(tag):
    return re.sub(r"-run\d+$", "", tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    data_dir = find_data_dir(args.data_dir)
    if not data_dir:
        raise SystemExit("Could not locate data directory. Set GROUNDANNOT_DATA.")
    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.is_dir():
        raise SystemExit(f"Runs directory not found: {runs_dir}")

    print(f"[INFO] data_dir = {data_dir}")
    print(f"[INFO] runs_dir = {runs_dir}")

    # Load enriched sets per disease
    enriched = {}
    for path in sorted(data_dir.glob("enriched_*.json")):
        if "fdr" in path.stem:
            continue
        disease = path.stem.replace("enriched_", "")
        enriched[disease] = load_ids(path)
    print(f"[INFO] loaded enriched sets for {len(enriched)} diseases")

    run_files = sorted(runs_dir.glob("llm_baseline_*-run*.json"))
    if not run_files:
        raise SystemExit(f"No run files matched in {runs_dir}")
    print(f"[INFO] {len(run_files)} run files")

    per_run = ["model,disease,run,llm_n,kept,dropped,coverage,zero_kept"]
    accum = defaultdict(lambda: defaultdict(list))
    zero_kept = defaultdict(lambda: defaultdict(int))
    total_runs = defaultdict(lambda: defaultdict(int))

    for rf in run_files:
        tag = rf.stem.replace("llm_baseline_", "")
        model = normalise_model(tag)
        try:
            with open(rf, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"  skip {rf.name}: {e}")
            continue

        for r in data:
            disease = r["disease"]
            if disease not in enriched:
                continue
            picks = r.get("llm_ids_list") or GO_RE.findall(r.get("llm_text", ""))
            picks = set(picks)
            llm_n = len(picks)
            kept = picks & enriched[disease]
            dropped = picks - enriched[disease]
            coverage = len(kept) / llm_n if llm_n else 0.0
            zk = 1 if len(kept) == 0 else 0

            per_run.append(
                f"{model},{disease},{tag},{llm_n},{len(kept)},{len(dropped)},"
                f"{coverage:.3f},{zk}"
            )
            accum[model]["kept"].append(len(kept))
            accum[model]["llm_n"].append(llm_n)
            accum[model]["coverage"].append(coverage)
            total_runs[model][disease] += 1
            if zk:
                zero_kept[model][disease] += 1

    (STATS_DIR / "bounded_per_run.csv").write_text(
        "\n".join(per_run) + "\n", encoding="utf-8"
    )
    print(f"[INFO] wrote bounded_per_run.csv ({len(per_run) - 1} rows)")

    def msd(xs):
        return (mean(xs), stdev(xs) if len(xs) > 1 else 0.0)

    summary = ["model,n_runs,mean_llm_n,mean_kept,mean_dropped,mean_coverage,"
               "zero_kept_runs,zero_kept_pct"]
    for model in sorted(accum.keys()):
        kept = accum[model]["kept"]
        llm_n = accum[model]["llm_n"]
        cov = accum[model]["coverage"]
        n = len(kept)
        mk = mean(kept); mn = mean(llm_n)
        md = mn - mk
        mc = mean(cov)
        zkr = sum(zero_kept[model].values())
        zkp = zkr / n if n else 0.0
        summary.append(
            f"{model},{n},{mn:.2f},{mk:.2f},{md:.2f},{mc:.3f},{zkr},{zkp:.3f}"
        )

    (STATS_DIR / "bounded_summary.csv").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print("[INFO] wrote bounded_summary.csv")
    print()
    for line in summary:
        print("  " + line)

    # Plot mean kept picks per model
    models = [l.split(",")[0] for l in summary[1:]]
    means = [float(l.split(",")[3]) for l in summary[1:]]
    if models:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(models))
        ax.bar(x, means, color="#4a90d9", edgecolor="black")
        for i, m in enumerate(means):
            ax.text(i, m + 0.15, f"{m:.2f}", ha="center", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel("Mean kept picks per run (out of up to 10)")
        ax.set_title("Bounded-agent coverage: how many LLM picks survive the enriched filter")
        ax.set_ylim(0, 10.5)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(STATS_DIR / "bounded_coverage.png", dpi=150)
        plt.close(fig)
        print("[INFO] wrote bounded_coverage.png")


if __name__ == "__main__":
    main()

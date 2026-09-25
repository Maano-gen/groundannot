#!/usr/bin/env python3
"""
fdr_baseline.py — Compare LLM term picks against a statistical FDR baseline.

For each disease:
  1. Read enriched_fdr_*.json (contains FDR values, already sorted ascending).
  2. Take the top 10 terms by lowest FDR from each backend — the "FDR baseline".
  3. Compare against every LLM run's picks for that disease.

LLM runs are read from <repo>/runs/ if it exists, otherwise from the data dir.

Outputs to stats/:
  fdr_top10_per_disease.csv   the reference top-10 lists (one per backend)
  fdr_comparison.csv          one row per (disease, model, run)
  fdr_summary.csv             aggregated per (disease, model): mean ± SD
  fdr_overlap.png             bar chart of overlap by model
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
            if any(parent.glob("enriched_fdr_*.json")):
                return parent.resolve()
    return None


def top10_by_fdr(fdr_file):
    """Return {backend: [(id, label, fdr), ...top10]} from an enriched_fdr file."""
    with open(fdr_file, encoding="utf-8") as fh:
        data = json.load(fh)
    out = {}
    for backend, rows in data.items():
        top = []
        for r in rows:
            if not r.get("id"):
                continue
            top.append((r["id"], r.get("label", ""), r.get("fdr")))
            if len(top) >= 10:
                break
        out[backend] = top
    return out


def load_run(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser(description="FDR baseline vs LLM picks.")
    ap.add_argument("--data-dir", default=None,
                    help="Directory with enriched_fdr_*.json")
    ap.add_argument("--runs-dir", default=None,
                    help="Directory with llm_baseline_*.json (default: <repo>/runs)")
    ap.add_argument("--llm-glob", default="llm_baseline_*.json",
                    help="Glob for LLM run files inside runs-dir")
    args = ap.parse_args()

    data_dir = find_data_dir(args.data_dir)
    if not data_dir:
        raise SystemExit("Could not locate data directory. Set GROUNDANNOT_DATA.")
    print(f"[INFO] data_dir = {data_dir}")

    # Where are the LLM run files?
    if args.runs_dir:
        runs_dir = Path(args.runs_dir).resolve()
    else:
        # try <repo>/runs, then a sibling of data_dir, then data_dir itself
        candidates = [
            Path(__file__).resolve().parent / "runs",
            data_dir.parent.parent / "runs",
            data_dir.parent / "runs",
            data_dir,
        ]
        runs_dir = None
        for c in candidates:
            if c.is_dir() and any(c.glob(args.llm_glob)):
                runs_dir = c.resolve()
                break
        if runs_dir is None:
            runs_dir = data_dir
    print(f"[INFO] runs_dir = {runs_dir}")

    # ---------------------------------------------------------------
    # 1. Build FDR top-10 reference per disease
    # ---------------------------------------------------------------
    fdr_files = sorted(data_dir.glob("enriched_fdr_*.json"))
    if not fdr_files:
        raise SystemExit(f"No enriched_fdr_*.json found in {data_dir}")

    top10_by_disease = {}
    ref_lines = ["disease,backend,rank,term_id,label,fdr"]
    for path in fdr_files:
        disease = path.stem.replace("enriched_fdr_", "")
        top10 = top10_by_fdr(path)
        top10_by_disease[disease] = top10
        for backend, rows in top10.items():
            for rank, (tid, label, fdr) in enumerate(rows, start=1):
                label_escaped = (label or "").replace(",", ";")
                ref_lines.append(
                    f"{disease},{backend},{rank},{tid},{label_escaped},{fdr}"
                )
    (STATS_DIR / "fdr_top10_per_disease.csv").write_text(
        "\n".join(ref_lines) + "\n", encoding="utf-8"
    )
    print(f"[INFO] wrote fdr_top10_per_disease.csv "
          f"({len(fdr_files)} diseases)")

    # ---------------------------------------------------------------
    # 2. For each LLM run, compare against FDR top-10 per disease
    # ---------------------------------------------------------------
    run_files = sorted(runs_dir.glob(args.llm_glob))
    if not run_files:
        raise SystemExit(
            f"No LLM run files matched {args.llm_glob} in {runs_dir}"
        )
    print(f"[INFO] {len(run_files)} LLM run files")

    per_run_rows = ["disease,model,run,llm_n,overlap_panther,overlap_enrichr,"
                    "overlap_gprofiler,overlap_union,union_size,"
                    "jaccard_union,recall_union"]

    accum = defaultdict(lambda: defaultdict(list))

    for rf in run_files:
        tag = rf.stem.replace("llm_baseline_", "")
        model = re.sub(r"-run\d+$", "", tag)
        try:
            rows = load_run(rf)
        except Exception as e:
            print(f"  skip {rf.name}: {e}")
            continue

        for r in rows:
            disease = r["disease"]
            if disease not in top10_by_disease:
                continue

            ids_from_list = r.get("llm_ids_list")
            if ids_from_list:
                llm_ids = set(ids_from_list)
            else:
                llm_ids = set(GO_RE.findall(r.get("llm_text", "")))
            llm_n = len(llm_ids)

            tp = top10_by_disease[disease]
            set_p = {t[0] for t in tp.get("PANTHER", [])}
            set_e = {t[0] for t in tp.get("Enrichr", [])}
            set_g = {t[0] for t in tp.get("g:Profiler", [])}
            union = set_p | set_e | set_g

            ov_p = len(llm_ids & set_p)
            ov_e = len(llm_ids & set_e)
            ov_g = len(llm_ids & set_g)
            ov_u = len(llm_ids & union)
            j_u = jaccard(llm_ids, union)
            recall_u = ov_u / len(union) if union else 0.0

            per_run_rows.append(
                f"{disease},{model},{tag},{llm_n},"
                f"{ov_p},{ov_e},{ov_g},{ov_u},{len(union)},"
                f"{j_u:.3f},{recall_u:.3f}"
            )
            accum[(model, disease)]["ov_u"].append(ov_u)
            accum[(model, disease)]["j_u"].append(j_u)
            accum[(model, disease)]["recall_u"].append(recall_u)
            accum[(model, disease)]["ov_p"].append(ov_p)
            accum[(model, disease)]["ov_e"].append(ov_e)
            accum[(model, disease)]["ov_g"].append(ov_g)

    (STATS_DIR / "fdr_comparison.csv").write_text(
        "\n".join(per_run_rows) + "\n", encoding="utf-8"
    )
    print(f"[INFO] wrote fdr_comparison.csv "
          f"({len(per_run_rows) - 1} rows)")

    # ---------------------------------------------------------------
    # 3. Summary: mean ± SD per (model, disease)
    # ---------------------------------------------------------------
    def msd(xs):
        return (mean(xs), stdev(xs) if len(xs) > 1 else 0.0)

    summary_lines = ["model,disease,n_runs,overlap_union_mean,overlap_union_sd,"
                     "jaccard_mean,jaccard_sd,recall_mean,recall_sd,"
                     "overlap_panther_mean,overlap_enrichr_mean,"
                     "overlap_gprofiler_mean"]
    for (model, disease), d in sorted(accum.items()):
        n = len(d["ov_u"])
        om, os_ = msd(d["ov_u"])
        jm, js = msd(d["j_u"])
        rm, rs = msd(d["recall_u"])
        pm = mean(d["ov_p"])
        em = mean(d["ov_e"])
        gm = mean(d["ov_g"])
        summary_lines.append(
            f"{model},{disease},{n},{om:.3f},{os_:.3f},{jm:.3f},{js:.3f},"
            f"{rm:.3f},{rs:.3f},{pm:.3f},{em:.3f},{gm:.3f}"
        )
    (STATS_DIR / "fdr_summary.csv").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    print("[INFO] wrote fdr_summary.csv")

    # ---------------------------------------------------------------
    # 4. Plot: mean overlap_union by model (across all diseases)
    # ---------------------------------------------------------------
    by_model = defaultdict(list)
    for (model, disease), d in accum.items():
        by_model[model].extend(d["ov_u"])
    if by_model:
        models = sorted(by_model.keys())
        means = [mean(by_model[m]) for m in models]
        sds = [stdev(by_model[m]) if len(by_model[m]) > 1 else 0.0
               for m in models]

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(models))
        ax.bar(x, means, yerr=sds, capsize=6,
               color="#4a90d9", edgecolor="black")
        for i, (m, s) in enumerate(zip(means, sds)):
            ax.text(i, m + s + 0.2, f"{m:.1f}\n±{s:.1f}",
                    ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel("Mean overlap with FDR top-10 union (out of ≤30)")
        ax.set_title("LLM picks vs statistical FDR baseline")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(STATS_DIR / "fdr_overlap.png", dpi=150)
        plt.close(fig)
        print("[INFO] wrote fdr_overlap.png")


if __name__ == "__main__":
    main()

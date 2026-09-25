#!/usr/bin/env python3
"""
hallmark_truth.py — Score LLM picks against the hallmark's defining pathway.

For each hallmark (e.g. DNA_REPAIR), check whether the LLM's picks include
at least one term whose label matches a keyword for that hallmark.

Hallmarks are split into two categories:
  direct    — the hallmark name literally names the biology
              (DNA_REPAIR, APOPTOSIS, HYPOXIA, ...)
  indirect  — the hallmark names a regulator or programme, not the biology
              (MYC_TARGETS_V1, E2F_TARGETS, KRAS_SIGNALING_UP, ...)

The direct subset is the clean ground-truth benchmark. The indirect subset
is reported separately because a correct answer there often does not
contain the hallmark's own name.

Inputs:
  data/snapshot_hallmarks/enriched_fdr_HALLMARK*.json   (id -> label)
  data/snapshot_hallmarks/llm_baseline_*-hallmark.json  (LLM picks)

Outputs:
  stats/hallmark_truth.csv           one row per (hallmark, model)
  stats/hallmark_truth_summary.csv   per model, direct / indirect / all
  stats/hallmark_truth.png           bar chart of hit rate by model
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("data/snapshot_hallmarks")
STATS_DIR = Path("stats")
STATS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# Keyword matchers per hallmark
# ---------------------------------------------------------------
KEYWORDS = {
    "ADIPOGENESIS": ["adipogen", "adipocyte", "fat cell"],
    "ALLOGRAFT_REJECTION": ["allograft", "graft rejection", "transplant rejection"],
    "ANDROGEN_RESPONSE": ["androgen"],
    "ANGIOGENESIS": ["angiogen", "blood vessel", "vasculature"],
    "APICAL_JUNCTION": ["apical junction", "tight junction", "adherens junction", "cell junction"],
    "APICAL_SURFACE": ["apical"],
    "APOPTOSIS": ["apoptot"],
    "BILE_ACID_METABOLISM": ["bile acid", "bile salt"],
    "CHOLESTEROL_HOMEOSTASIS": ["cholesterol"],
    "COAGULATION": ["coagulation", "blood clot", "hemost", "haemost", "fibrin"],
    "COMPLEMENT": ["complement"],
    "DNA_REPAIR": ["dna repair", "dna damage", "double-strand break", "mismatch repair"],
    "E2F_TARGETS": ["e2f"],
    "EPITHELIAL_MESENCHYMAL_TRANSITION": ["epithelial mesenchymal", "mesenchymal transition",
                                          "epithelial-mesenchymal"],
    "ESTROGEN_RESPONSE_EARLY": ["estrogen", "oestrogen"],
    "ESTROGEN_RESPONSE_LATE": ["estrogen", "oestrogen"],
    "FATTY_ACID_METABOLISM": ["fatty acid"],
    "G2_M_CHECKPOINT": ["g2", "g2/m", "cell cycle checkpoint", "mitotic checkpoint",
                        "spindle checkpoint"],
    "GLYCOLYSIS": ["glycoly"],
    "HEDGEHOG_SIGNALING": ["hedgehog"],
    "HEME_METABOLISM": ["heme", "haem", "porphyrin"],
    "HYPOXIA": ["hypox"],
    "IL_2_STAT5_SIGNALING": ["stat5", "interleukin-2", "il-2"],
    "IL_6_JAK_STAT3_SIGNALING": ["stat3", "interleukin-6", "il-6", "jak"],
    "INFLAMMATORY_RESPONSE": ["inflammat"],
    "INTERFERON_ALPHA_RESPONSE": ["interferon alpha", "interferon-alpha", "type i interferon",
                                   "interferon type i"],
    "INTERFERON_GAMMA_RESPONSE": ["interferon gamma", "interferon-gamma", "type ii interferon",
                                   "interferon type ii"],
    "KRAS_SIGNALING_DN": ["kras", "k-ras", "ras signal"],
    "KRAS_SIGNALING_UP": ["kras", "k-ras", "ras signal"],
    "MITOTIC_SPINDLE": ["mitotic spindle", "spindle"],
    "MTORC1_SIGNALING": ["mtorc1", "mtor"],
    "MYC_TARGETS_V1": ["myc"],
    "MYC_TARGETS_V2": ["myc"],
    "MYOGENESIS": ["myogen", "muscle", "myotube", "myoblast"],
    "NOTCH_SIGNALING": ["notch"],
    "OXIDATIVE_PHOSPHORYLATION": ["oxidative phosphorylation", "respiratory electron transport",
                                   "electron transport chain", "respiratory chain"],
    "P53_PATHWAY": ["p53", "tp53"],
    "PANCREAS_BETA_CELLS": ["beta cell", "pancrea", "insulin secretion"],
    "PPEROXISOME": ["peroxisom"],
    "PI3K_AKT_MTOR_SIGNALING": ["pi3k", "akt", "mtor"],
    "PROTEIN_SECRETION": ["protein secretion", "secretion", "secretory"],
    "REACTIVE_OXYGEN_SPECIES_PATHWAY": ["reactive oxygen", "reactive oxygen species",
                                          "oxidative stress", "superoxide"],
    "SPERMATOGENESIS": ["spermatogen", "sperm", "male gamete"],
    "TGF_BETA_SIGNALING": ["tgf"],
    "TNF_ALPHA_SIGNALING_VIA_NF_KB": ["nf-kb", "nfkb", "tnf", "nf-kappa"],
    "UNFOLDED_PROTEIN_RESPONSE": ["unfolded protein", "er stress", "endoplasmic reticulum stress",
                                    "upr"],
    "UV_RESPONSE_DN": ["uv", "ultraviolet", "dna photolyase"],
    "UV_RESPONSE_UP": ["uv", "ultraviolet", "dna photolyase"],
    "WNT_BETA_CATENIN_SIGNALING": ["wnt", "beta-catenin", "beta catenin"],
    "XENOBIOTIC_METABOLISM": ["xenobiotic", "drug metabol", "cytochrome p450", "cyp"],
}

# ---------------------------------------------------------------
# Direct vs indirect classification
# Direct: the hallmark name literally names the biology that should
#         appear as a term for that gene set.
# Indirect: the hallmark names a regulator, programme, or direction, and
#           the correct terms are usually the downstream biology.
# ---------------------------------------------------------------
DIRECT = {
    "ADIPOGENESIS",
    "ALLOGRAFT_REJECTION",
    "ANDROGEN_RESPONSE",
    "ANGIOGENESIS",
    "APICAL_JUNCTION",
    "APICAL_SURFACE",
    "APOPTOSIS",
    "BILE_ACID_METABOLISM",
    "CHOLESTEROL_HOMEOSTASIS",
    "COAGULATION",
    "COMPLEMENT",
    "DNA_REPAIR",
    "EPITHELIAL_MESENCHYMAL_TRANSITION",
    "ESTROGEN_RESPONSE_EARLY",
    "ESTROGEN_RESPONSE_LATE",
    "FATTY_ACID_METABOLISM",
    "GLYCOLYSIS",
    "HEDGEHOG_SIGNALING",
    "HEME_METABOLISM",
    "HYPOXIA",
    "IL_2_STAT5_SIGNALING",
    "IL_6_JAK_STAT3_SIGNALING",
    "INFLAMMATORY_RESPONSE",
    "INTERFERON_ALPHA_RESPONSE",
    "INTERFERON_GAMMA_RESPONSE",
    "MITOTIC_SPINDLE",
    "MTORC1_SIGNALING",
    "MYOGENESIS",
    "NOTCH_SIGNALING",
    "OXIDATIVE_PHOSPHORYLATION",
    "P53_PATHWAY",
    "PANCREAS_BETA_CELLS",
    "PPEROXISOME",
    "PI3K_AKT_MTOR_SIGNALING",
    "PROTEIN_SECRETION",
    "REACTIVE_OXYGEN_SPECIES_PATHWAY",
    "SPERMATOGENESIS",
    "TGF_BETA_SIGNALING",
    "TNF_ALPHA_SIGNALING_VIA_NF_KB",
    "UNFOLDED_PROTEIN_RESPONSE",
    "UV_RESPONSE_DN",
    "UV_RESPONSE_UP",
    "WNT_BETA_CATENIN_SIGNALING",
    "XENOBIOTIC_METABOLISM",
}
# Everything else in KEYWORDS is treated as indirect
# (E2F_TARGETS, G2_M_CHECKPOINT, KRAS_SIGNALING_DN/UP, MYC_TARGETS_V1/V2)


def category_of(hallmark):
    return "direct" if hallmark in DIRECT else "indirect"


def build_id_to_label():
    """From all enriched_fdr_*.json, build {hallmark: {id: label}}."""
    mapping = {}
    for path in sorted(DATA_DIR.glob("enriched_fdr_*.json")):
        name = path.stem.replace("enriched_fdr_", "")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        id2label = {}
        for backend, rows in data.items():
            for r in rows:
                tid = r.get("id")
                if tid:
                    lab = (r.get("label") or "").strip()
                    if lab:
                        id2label[tid] = lab
        mapping[name] = id2label
    return mapping


def label_matches(label, keywords):
    if not label:
        return False
    low = label.lower()
    return any(kw in low for kw in keywords)


def normalise_model(tag):
    """Turn run tags into clean model names."""
    m = tag.replace("-hallmark", "")
    if m in ("test", "test-local", "test-hallmark"):
        return "qwen2.5-7b"
    return m


def main():
    id_to_label = build_id_to_label()
    print(f"[INFO] loaded label maps for {len(id_to_label)} hallmarks")

    run_files = sorted(DATA_DIR.glob("llm_baseline_*-hallmark.json"))
    if not run_files:
        raise SystemExit("No llm_baseline_*-hallmark.json files found")

    rows = ["hallmark,category,model,run,llm_n,matched_picks,hit"]
    # summary keyed by (model, category)
    summary = defaultdict(lambda: {"hits": 0, "total": 0})

    for rf in run_files:
        tag = rf.stem.replace("llm_baseline_", "")
        model = normalise_model(tag)
        with open(rf, encoding="utf-8") as fh:
            data = json.load(fh)

        for r in data:
            hallmark = r["disease"]
            if hallmark not in KEYWORDS:
                continue
            keywords = KEYWORDS[hallmark]
            cat = category_of(hallmark)

            ids = r.get("llm_ids_list") or re.findall(r"GO:\d{7}", r.get("llm_text", ""))
            ids = list(ids)
            llm_n = len(ids)

            id2label = id_to_label.get(hallmark, {})
            matched = 0
            for tid in ids:
                lab = id2label.get(tid, "")
                if label_matches(lab, keywords):
                    matched += 1

            hit = 1 if matched > 0 else 0
            rows.append(f"{hallmark},{cat},{model},{tag},{llm_n},{matched},{hit}")

            summary[(model, cat)]["hits"] += hit
            summary[(model, cat)]["total"] += 1
            summary[(model, "all")]["hits"] += hit
            summary[(model, "all")]["total"] += 1

    (STATS_DIR / "hallmark_truth.csv").write_text("\n".join(rows) + "\n",
                                                   encoding="utf-8")
    print(f"[INFO] wrote hallmark_truth.csv ({len(rows) - 1} rows)")

    # Summary CSV
    lines = ["model,category,n_hallmarks,n_hits,hit_rate"]
    for (model, cat), d in sorted(summary.items()):
        n = d["total"]
        hits = d["hits"]
        rate = hits / n if n else 0.0
        lines.append(f"{model},{cat},{n},{hits},{rate:.3f}")
    (STATS_DIR / "hallmark_truth_summary.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("[INFO] wrote hallmark_truth_summary.csv")
    print()
    print("Summary:")
    for line in lines:
        print("  " + line)

    # Plot: grouped bar per model, direct / indirect / all
    models = sorted({m for (m, _) in summary.keys()})
    cats = ["direct", "indirect", "all"]
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    for i, cat in enumerate(cats):
        vals = []
        for m in models:
            d = summary.get((m, cat), {"hits": 0, "total": 0})
            vals.append(d["hits"] / d["total"] if d["total"] else 0.0)
        ax.bar(x + (i - 1) * width, vals, width, label=cat)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Fraction of hallmarks with a matching term")
    ax.set_title("LLM ground-truth recovery on 50 MSigDB Hallmarks")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Category")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(STATS_DIR / "hallmark_truth.png", dpi=150)
    plt.close(fig)
    print("[INFO] wrote hallmark_truth.png")


if __name__ == "__main__":
    main()

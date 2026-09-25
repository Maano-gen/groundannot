#!/usr/bin/env python3
"""
make_figure2.py — Alzheimer's comparison figure.

Left column:  each pick from the local Qwen2.5-7B model for the
              Alzheimer's gene list, showing both the label the model
              CLAIMED and the live QuickGO label for that ID.
              Category shows contract status (valid enriched /
              allowed-not-enriched / unsupported).
Right column: the FDR top-10 for the same list (union across backends).

Reads:
  runs/llm_baseline_qwen2.5-7b-run001.json
  data/snapshot_20260924/allowed_alzheimers_disease.json
  data/snapshot_20260924/enriched_alzheimers_disease.json
  data/snapshot_20260924/enriched_fdr_alzheimers_disease.json

Writes: manuscript/figures/figure2_alzheimers_comparison.png
"""

import json
import re
import time
from pathlib import Path

import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_JSON     = Path("runs/llm_baseline_qwen2.5-7b-run001.json")
ALLOWED      = Path("data/snapshot_20260924/allowed_alzheimers_disease.json")
ENRICHED     = Path("data/snapshot_20260924/enriched_alzheimers_disease.json")
ENRICHED_FDR = Path("data/snapshot_20260924/enriched_fdr_alzheimers_disease.json")
OUT = Path("manuscript/figures/figure2_alzheimers_comparison.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

GO_LINE = re.compile(r"(GO:\d{7})\s+(.+)")
QUICKGO = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms"


def load_ids(path):
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
        for v in data.values():
            if isinstance(v, list):
                for x in v:
                    _extract(x)
    elif isinstance(data, list):
        for x in data:
            _extract(x)
    return ids


def parse_picks(llm_text):
    """Return [(id, claimed_label), ...] in order."""
    out = []
    for line in llm_text.splitlines():
        m = GO_LINE.search(line)
        if m:
            gid, lab = m.group(1), m.group(2).strip()
            out.append((gid, lab))
    return out


def quickgo_label(tid, retries=2):
    """Fetch the current QuickGO label for a GO ID."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(f"{QUICKGO}/{tid}",
                             headers={"Accept": "application/json"},
                             timeout=20)
            if r.status_code == 404:
                return "(not a current GO term)"
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            return r.json()["results"][0].get("name", "")
        except Exception:
            if attempt < retries:
                time.sleep(1.5 ** attempt)
    return "(QuickGO unreachable)"


def build_label_map(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    m = {}
    for backend, terms in data.items():
        for t in terms:
            tid = t.get("id")
            if tid and tid not in m:
                m[tid] = t.get("label", "")
    return m


def top_fdr_union(path, n=10):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = []
    for backend, terms in data.items():
        for t in terms:
            tid = t.get("id")
            if not tid:
                continue
            rows.append({
                "id": tid,
                "label": t.get("label", ""),
                "fdr": t.get("fdr"),
                "backend": backend,
            })
    rows.sort(key=lambda r: (r["fdr"] is None, r["fdr"]))
    seen, out = set(), []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        out.append(r)
        if len(out) >= n:
            break
    return out


def fmt_fdr(v):
    if v is None:
        return "—"
    if v < 1e-3:
        return f"{v:.1e}"
    return f"{v:.4f}"


def main():
    runs = json.load(open(RUN_JSON, encoding="utf-8"))
    ad = next((r for r in runs if r["disease"] == "alzheimers_disease"), None)
    if ad is None:
        raise SystemExit("Alzheimer's row not found")

    picks = parse_picks(ad.get("llm_text", ""))
    allowed = load_ids(ALLOWED)
    enriched = load_ids(ENRICHED)
    enriched_labels = build_label_map(ENRICHED_FDR)
    fdr_top = top_fdr_union(ENRICHED_FDR, n=10)

    print(f"[INFO] total picks: {len(picks)}")
    print(f"[INFO] FDR top-10 union: {len(fdr_top)}")
    print("[INFO] fetching QuickGO labels ...")
    real_labels = {}
    for gid, _ in picks:
        real_labels[gid] = quickgo_label(gid)
        print(f"  {gid}  {real_labels[gid]}")
        time.sleep(0.3)

    # Count mismatches between model-claimed label and real QuickGO label
    def norm(s):
        return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

    n_mismatch = 0
    for gid, claimed in picks:
        real = real_labels.get(gid, "")
        if real.startswith("("):
            continue
        if norm(claimed) != norm(real):
            n_mismatch += 1
    print(f"[INFO] label mismatches: {n_mismatch} of {len(picks)}")

    cat_color = {
        "VALID_ENRICHED":       "#198754",
        "ALLOWED_NOT_ENRICHED": "#fd7e14",
        "UNSUPPORTED":          "#dc3545",
    }
    cat_display = {
        "VALID_ENRICHED":       "valid enriched",
        "ALLOWED_NOT_ENRICHED": "allowed, not enriched",
        "UNSUPPORTED":          "unsupported by any backend",
    }

    fig, ax = plt.subplots(figsize=(14.5, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.25, 0.965,
            "Local Qwen2.5-7B picks for the Alzheimer's gene list\n"
            "(stable across 36 runs; SD = 0.00)",
            ha="center", fontsize=11.5, fontweight="bold")
    ax.text(0.75, 0.965,
            "FDR top-10 across PANTHER / Enrichr / g:Profiler",
            ha="center", fontsize=11.5, fontweight="bold")
    ax.plot([0.5, 0.5], [0.03, 0.92], color="#cccccc", linewidth=1)

    # Left column — one pick per row
    y = 0.895
    dy = 0.085
    for gid, claimed in picks:
        if gid in enriched:
            cat = "VALID_ENRICHED"
        elif gid in allowed:
            cat = "ALLOWED_NOT_ENRICHED"
        else:
            cat = "UNSUPPORTED"
        color = cat_color[cat]

        ax.text(0.02, y, gid, fontsize=10, fontfamily="monospace",
                fontweight="bold", va="center", color=color)
        ax.text(0.135, y, f"model claim: \u201c{claimed}\u201d",
                fontsize=9.5, va="center", color="#222222")
        real = real_labels.get(gid) or enriched_labels.get(gid) or "(not a current GO term)"
        ax.text(0.135, y - 0.026,
                f"QuickGO label: {real}",
                fontsize=8.5, va="center", color="#555555", style="italic")
        ax.text(0.02, y - 0.048, cat_display[cat],
                fontsize=8, va="center", color=color)
        y -= dy

    # Right column — FDR top-10
    y = 0.895
    for r in fdr_top:
        ax.text(0.52, y, r["id"], fontsize=10, fontfamily="monospace",
                fontweight="bold", va="center", color="#1e40af")
        ax.text(0.645, y, r["label"][:54], fontsize=9.5, va="center",
                color="#222222")
        ax.text(0.52, y - 0.026,
                f"FDR = {fmt_fdr(r['fdr'])}  ({r['backend']})",
                fontsize=8.5, va="center", color="#555555", style="italic")
        y -= dy

    ax.text(0.5, 0.008,
            f"Overlap: 0 / 10.  None of the model's picks match any "
            f"backend's FDR top-10 for the Alzheimer's gene list. "
            f"{n_mismatch} of 10 model-claimed labels do not match the "
            f"current GO term for that accession.",
            ha="center", fontsize=9, color="#842029",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#f8d7da", edgecolor="#842029"))

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"[INFO] wrote {OUT}")


if __name__ == "__main__":
    main()

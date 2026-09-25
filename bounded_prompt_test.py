#!/usr/bin/env python3
"""
bounded_prompt_test.py — Run the LLM with the enriched list IN the prompt.

This is the actual bounded-agent condition. The model is given:
  - the gene list
  - a numbered shortlist of enriched GO terms (top N by FDR)
and asked to pick 10 from that list.

If the model picks only from the provided list, the bounded contract works.
If it picks terms outside the list, it ignores the vocabulary and still
generates from memory.

Reuses LLMClient from benchmark_llm.py.

Usage:
  python3 bounded_prompt_test.py --tag local-bounded
  python3 bounded_prompt_test.py --provider groq \\
      --model-id openai/gpt-oss-120b --tag groq-gpt-oss-120b-bounded

Outputs:
  stats/bounded_prompt_<tag>.json
  stats/bounded_prompt_<tag>.log
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from benchmark_llm import LLMClient, load_dotenv, find_data_dir, API_BASE_URLS

GO_RE = re.compile(r"GO:\d{7}")
TOP_N = 50   # how many enriched terms to show the model


def log(path, msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


SYSTEM = "You are a bioinformatician. You must answer only using the provided list."

USER_TEMPLATE = """You are given a gene list and a shortlist of Gene Ontology
biological-process terms that are statistically enriched in that gene list.

Gene list: {genes}

Enriched GO terms (pick ONLY from this list, use the exact GO ID):
{shortlist}

Select exactly 10 of the terms above that best describe the biology of the
gene list. Output one per line, no preamble, no numbering:
GO:XXXXXXX term name
"""


def load_shortlist(fdr_path, n=TOP_N):
    """Return list of (id, label) from enriched_fdr_*.json, top n by FDR."""
    with open(fdr_path, encoding="utf-8") as fh:
        data = json.load(fh)
    # data: {backend: [{id, label, fdr, ...}, ...]} sorted by fdr ascending
    seen = {}
    order = []
    for backend, rows in data.items():
        for r in rows:
            tid = r.get("id")
            if not tid or tid in seen:
                continue
            seen[tid] = r.get("label") or ""
            order.append(tid)
    return [(tid, seen[tid]) for tid in order[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="local",
                    choices=["local"] + list(API_BASE_URLS.keys()))
    ap.add_argument("--model", default=None)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    load_dotenv(Path(__file__).parent, Path.cwd())

    data_dir = find_data_dir(explicit=args.data_dir)
    if not data_dir:
        raise SystemExit("No data directory found.")
    genes_root = Path(__file__).resolve().parent / "examples" / "lists"
    if not genes_root.is_dir():
        raise SystemExit(f"Gene list dir not found: {genes_root}")

    stats_dir = Path("stats")
    stats_dir.mkdir(exist_ok=True)
    log_path = stats_dir / f"bounded_prompt_{args.tag}.log"
    out_path = stats_dir / f"bounded_prompt_{args.tag}.json"

    log(log_path, f"data_dir = {data_dir}")
    log(log_path, f"genes_root = {genes_root}")
    log(log_path, f"tag = {args.tag}")

    # Resolve model
    if args.provider == "local":
        model_id = args.model or os.environ.get("GROUNDANNOT_LLM")
        if not model_id:
            raise SystemExit("No local model. Pass --model or set GROUNDANNOT_LLM.")
    else:
        model_id = args.model_id
        if not model_id:
            raise SystemExit(f"--provider {args.provider} requires --model-id.")

    log(log_path, f"provider = {args.provider}")
    log(log_path, f"model = {model_id}")

    client = LLMClient(args.provider, model_id)
    log(log_path, "client ready")

    results = []

    for fdr_path in sorted(data_dir.glob("enriched_fdr_*.json")):
        disease = fdr_path.stem.replace("enriched_fdr_", "")
        genes_file = genes_root / f"{disease}.txt"
        if not genes_file.exists():
            log(log_path, f"  skip {disease}: no {genes_file}")
            continue

        with open(genes_file, encoding="utf-8") as fh:
            genes = [g.strip() for g in fh if g.strip()]
        genes_str = ", ".join(genes)

        shortlist = load_shortlist(fdr_path, TOP_N)
        shortlist_ids = {tid for tid, _ in shortlist}
        shortlist_str = "\n".join(f"{tid} {lab}" for tid, lab in shortlist)

        prompt = USER_TEMPLATE.format(genes=genes_str, shortlist=shortlist_str)

        log(log_path, f"querying {disease} ({len(genes)} genes, "
                      f"shortlist={len(shortlist)})")

        text = client.chat(SYSTEM, prompt)
        if text is None:
            log(log_path, f"  skip {disease}: no response")
            continue

        picked = set(GO_RE.findall(text))
        in_list = picked & shortlist_ids
        out_list = picked - shortlist_ids

        log(log_path, f"  picked {len(picked)} | in_list {len(in_list)} "
                      f"| out_of_list {len(out_list)}")
        for tid in sorted(out_list):
            log(log_path, f"    OUT_OF_LIST {tid}")

        results.append({
            "disease": disease,
            "n_genes": len(genes),
            "shortlist_size": len(shortlist),
            "picked_n": len(picked),
            "in_list": len(in_list),
            "out_of_list": len(out_list),
            "in_list_ids": sorted(in_list),
            "out_of_list_ids": sorted(out_list),
            "llm_text": text,
        })

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    total_picked = sum(r["picked_n"] for r in results)
    total_in = sum(r["in_list"] for r in results)
    total_out = sum(r["out_of_list"] for r in results)
    rate = total_in / total_picked if total_picked else 0.0

    log(log_path, "=" * 50)
    log(log_path, f"TOTAL picked across {len(results)} diseases: {total_picked}")
    log(log_path, f"  in_list:      {total_in}")
    log(log_path, f"  out_of_list:  {total_out}")
    log(log_path, f"  in_list rate: {rate:.3f}")
    log(log_path, f"wrote {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
fetch_hallmarks.py — download MSigDB Hallmark gene sets from Enrichr
and write gene-list .txt files compatible with the GroundAnnot pipeline.

Enrichr hosts the same 50 hallmark sets as MSigDB.
Source URL:
  https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020

Outputs:
  examples/lists_hallmark/<SANITISED_NAME>.txt  (one file per set)
  data/hallmark_gmt/MSigDB_Hallmark_2020.gmt    (cached raw download)
"""

import re
import sys
from pathlib import Path

import requests

URL = ("https://maayanlab.cloud/Enrichr/geneSetLibrary"
       "?mode=text&libraryName=MSigDB_Hallmark_2020")

OUT_DIR = Path("examples/lists_hallmark")
CACHE_DIR = Path("data/hallmark_gmt")
CACHE_FILE = CACHE_DIR / "MSigDB_Hallmark_2020.gmt"


def download():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists() and CACHE_FILE.stat().st_size > 10_000:
        print(f"[INFO] using cached file {CACHE_FILE}")
        return CACHE_FILE.read_text(encoding="utf-8")

    print(f"[INFO] downloading {URL}")
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    text = r.text
    CACHE_FILE.write_text(text, encoding="utf-8")
    print(f"[INFO] cached {CACHE_FILE} ({len(text):,} chars)")
    return text


def sanitise(name):
    """Turn 'TNF-alpha Signaling via NF-kB' into
    'TNF_ALPHA_SIGNALING_VIA_NF_KB'."""
    s = name.upper()
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def parse(text):
    """Return list of (name, [genes])."""
    sets = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        name = parts[0].strip()
        genes = [g.strip() for g in parts[1:] if g.strip()]
        if name and genes:
            sets.append((name, genes))
    return sets


def main():
    text = download()
    sets = parse(text)
    print(f"[INFO] parsed {len(sets)} gene sets")

    if not sets:
        sys.exit("[FAIL] no gene sets parsed — check the raw file")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, genes in sets:
        clean = sanitise(name)
        if not clean:
            continue
        # uppercase symbols, dedupe preserving order
        seen = set()
        unique = []
        for g in genes:
            g = g.upper()
            if g not in seen:
                seen.add(g)
                unique.append(g)
        out = OUT_DIR / f"{clean}.txt"
        out.write_text("\n".join(unique) + "\n", encoding="utf-8")
        print(f"  {clean:55s} {len(unique):4d} genes")
        written += 1

    print()
    print(f"[INFO] wrote {written} gene lists to {OUT_DIR}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
benchmark_llm.py — GroundAnnot LLM baseline (single-file, portable).

Runs an LLM on each disease gene list with NO backend access, extracts GO IDs
from the free-text response, and compares them against:

  allowed_*.json   → Contract A (every ID returned by any backend)
  enriched_*.json  → Contract B (only significant terms, FDR/g:SCS < 0.05)

Three-way split per disease:

  unsupported           ID not in any backend payload     (Contract A violation)
  allowed_not_enriched  ID in allowed but not in enriched (Contract B only)
  valid_enriched        ID in enriched set                (legitimate hit)

Contract A violations are verified live against QuickGO.

Portable behaviour:
  - .env is loaded from the repo root (or CWD) if present.
  - allowed_*.json and enriched_*.json are found by walking up from this file
    or from $GROUNDANNOT_DATA.
  - Gene lists are read from <repo>/examples/lists/ regardless of CWD.
  - Run from any directory; outputs are written to the current directory.

Usage (local GGUF):
  python3 benchmark_llm.py
  python3 benchmark_llm.py --tag qwen2.5-7b

Usage (API):
  export GROQ_API_KEY=gsk_...
  python3 benchmark_llm.py --provider groq --model-id qwen/qwen3.8-27b --tag groq-qwen3.8-27b

Environment variables:
  GROUNDANNOT_LLM        default GGUF path
  GROUNDANNOT_DATA       data directory containing allowed_*.json
  GROUNDANNOT_PROVIDER   default API provider
  GROUNDANNOT_MODEL_ID   default API model ID
  GROQ_API_KEY / OPENROUTER_API_KEY / TOGETHER_API_KEY / SCALEWAY_API_KEY / NVIDIA_API_KEY
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

QUICKGO = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms"

API_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "groq":       "https://api.groq.com/openai/v1",
    "together":   "https://api.together.xyz/v1",
    "scaleway":   "https://api.scaleway.ai/v1",
    "nvidia":     "https://integrate.api.nvidia.com/v1",
}
API_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "groq":       "GROQ_API_KEY",
    "together":   "TOGETHER_API_KEY",
    "scaleway":   "SCALEWAY_API_KEY",
    "nvidia":     "NVIDIA_API_KEY",
}

LOG_PATH = None  # set in main()


# ---------------------------------------------------------------------------
# .env loader — parsed from repo root or CWD; existing env always wins
# ---------------------------------------------------------------------------
def load_dotenv(*search_from):
    candidates = []
    for start in search_from:
        if start is None:
            continue
        p = Path(start).resolve()
        for parent in [p] + list(p.parents):
            candidates.append(parent / ".env")
    candidates.append(Path.cwd() / ".env")

    for path in candidates:
        if path.exists():
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(
                    k.strip(), v.strip().strip('"').strip("'")
                )
            return path
    return None


# ---------------------------------------------------------------------------
# data directory discovery — walk up from this file, then CWD
# ---------------------------------------------------------------------------
def find_data_dir(explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("GROUNDANNOT_DATA")
    if env:
        candidates.append(Path(env))

    for start in (Path(__file__), Path.cwd()):
        p = start.resolve()
        for parent in [p] + list(p.parents):
            candidates.append(parent)

    for c in candidates:
        try:
            if c.is_dir() and any(c.glob("allowed_*.json")):
                return c.resolve()
        except OSError:
            continue
    return None


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ---------------------------------------------------------------------------
# QuickGO
# ---------------------------------------------------------------------------
def quickgo(tid, retries=2):
    url = f"{QUICKGO}/{tid}"
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers={"Accept": "application/json"}, timeout=20)
            if r.status_code == 404:
                return None
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            res = r.json()["results"][0]
            return {
                "name": res.get("name", ""),
                "aspect": res.get("aspect", ""),
                "obsolete": bool(res.get("isObsolete", False)),
            }
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 ** attempt)
    log(f"    QuickGO transport failure for {tid}: {last_err}")
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------
def load_ids(path):
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


# ---------------------------------------------------------------------------
# unified LLM client
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, provider, model_id):
        self.provider = provider
        self.model_id = model_id

        if provider == "local":
            from llama_cpp import Llama
            path = Path(model_id).expanduser()
            if not path.exists():
                raise SystemExit(f"local model not found: {path}")
            self.client = Llama(
                model_path=str(path), n_ctx=2048, verbose=False, seed=42
            )
            self.mode = "local"
            return

        if provider not in API_BASE_URLS:
            raise SystemExit(f"unknown provider: {provider}")

        key_env = API_KEY_ENV[provider]
        api_key = os.environ.get(key_env)
        if not api_key:
            raise SystemExit(
                f"{key_env} is not set. Put it in .env in the repo root, or export it."
            )

        from openai import OpenAI
        self.client = OpenAI(base_url=API_BASE_URLS[provider], api_key=api_key)
        self.mode = "api"

    def chat(self, system_prompt, user_prompt,
             max_tokens=2048, temperature=0.0, retries=3):
        last_err = None
        for attempt in range(retries):
            try:
                if self.mode == "local":
                    out = self.client.create_chat_completion(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return out["choices"][0]["message"]["content"]

                extra = {}
                if self.provider == "groq" and "gpt-oss" in self.model_id.lower():
                    extra["reasoning_effort"] = "low"

                # Groq's DeepSeek R1 models do not accept a system role.
                # Fold the system prompt into the user message instead.
                if "deepseek-r1" in self.model_id.lower():
                    messages = [
                        {"role": "user",
                         "content": f"{system_prompt}\n\n{user_prompt}"},
                    ]
                else:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]

                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **extra,
                )
                return resp.choices[0].message.content
            except Exception as e:
                last_err = e
                log(f"    API attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        log(f"  gave up: {last_err}")
        return None


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------
SYSTEM = "You are a bioinformatician. Answer concisely and only in the requested format."
USER = """List exactly 10 Gene Ontology biological process terms that are enriched in this gene list.

Gene list: {genes}

Output format: one per line, no preamble, no numbering:
GO:XXXXXXX term name
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GroundAnnot LLM baseline.")
    p.add_argument("--provider", default=None,
                   choices=["local"] + list(API_BASE_URLS.keys()),
                   help="LLM provider. If omitted, inferred from env/flags.")
    p.add_argument("--model", default=None,
                   help="Path to GGUF (used with --provider local).")
    p.add_argument("--model-id", default=None,
                   help="Model ID for API providers.")
    p.add_argument("--data-dir", default=None,
                   help="Directory containing allowed_*.json.")
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: llm_baseline_<tag>.json).")
    p.add_argument("--tag", default="",
                   help="Tag for output/log filenames.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    global LOG_PATH
    args = parse_args()

    # 1. Load .env from repo root or CWD, if present
    env_file = load_dotenv(Path(__file__).parent, Path.cwd())
    if env_file:
        print(f"[INFO] loaded env from {env_file}", flush=True)

    # 2. Resolve data directory before anything else
    data_dir = find_data_dir(explicit=args.data_dir)
    if not data_dir:
        raise SystemExit(
            "No data directory found (looked for allowed_*.json). "
            "Pass --data-dir or set GROUNDANNOT_DATA."
        )

    # Remember where the gene lists live, before we chdir away
    genes_root = Path(__file__).resolve().parent / "examples" / "lists"
    if not genes_root.is_dir():
        raise SystemExit(
            f"Gene-list directory not found: {genes_root}\n"
            f"Expected examples/lists/*.txt next to benchmark_llm.py."
        )

    os.chdir(data_dir)

    # 3. Resolve provider and model
    provider = args.provider or os.environ.get("GROUNDANNOT_PROVIDER")
    model_id = args.model_id or os.environ.get("GROUNDANNOT_MODEL_ID")

    if provider is None:
        if args.model or os.environ.get("GROUNDANNOT_LLM"):
            provider = "local"
        else:
            raise SystemExit(
                "No LLM configured. Examples:\n"
                "  python3 benchmark_llm.py --model /path/to/model.gguf\n"
                "  python3 benchmark_llm.py --provider groq "
                "--model-id qwen/qwen3.8-27b"
            )

    if provider == "local":
        model_id = model_id or args.model or os.environ.get("GROUNDANNOT_LLM")
        if not model_id:
            raise SystemExit(
                "No local GGUF model. Pass --model, set GROUNDANNOT_LLM, "
                "or place a .gguf under ~/llm-models/."
            )
        model_id = str(Path(model_id).expanduser())
    else:
        if not model_id:
            raise SystemExit(
                f"--provider {provider} requires --model-id "
                f"(or GROUNDANNOT_MODEL_ID)."
            )

    # 4. Output paths — written to CWD
    tag = args.tag or (
        Path(model_id).stem if provider == "local"
        else model_id.replace("/", "_").replace(":", "_")
    )
    out_path = args.out or f"llm_baseline_{tag}.json"
    LOG_PATH = f"llm_baseline_{tag}.log"

    log(f"data_dir   = {data_dir}")
    log(f"genes_root = {genes_root}")
    log(f"provider   = {provider}")
    log(f"model      = {model_id}")
    log(f"tag        = {tag}")

    client = LLMClient(provider, model_id)
    log("client ready")

    results = []

    for f in sorted(glob.glob("allowed_*.json")):
        disease = f.replace("allowed_", "").replace(".json", "")
        genes_file = genes_root / f"{disease}.txt"

        if not genes_file.exists():
            log(f"  skip {disease}: no {genes_file}")
            continue

        with open(genes_file, encoding="utf-8") as fh:
            genes = [g.strip() for g in fh if g.strip()]
        genes_str = ", ".join(genes)

        allowed_ids = load_ids(f)
        enriched_ids = load_ids(f.replace("allowed_", "enriched_"))

        log(f"querying {disease} ({len(genes)} genes) "
            f"| allowed={len(allowed_ids)} enriched={len(enriched_ids)}")

        text = client.chat(SYSTEM, USER.format(genes=genes_str))
        if text is None:
            log(f"  skipping {disease}: no response")
            continue

        llm_ids = set(re.findall(r"GO:\d{7}", text))
        unsupported = llm_ids - allowed_ids
        allowed_not_enriched = (llm_ids & allowed_ids) - enriched_ids
        valid_enriched = llm_ids & enriched_ids
        not_enriched = unsupported | allowed_not_enriched

        categorised = []
        for tid in sorted(unsupported):
            info = quickgo(tid)
            if info == "UNKNOWN":
                cat, name = "UNKNOWN", "—"
            elif info is None:
                cat, name = "NOT_IN_GO", "—"
            elif info["obsolete"]:
                cat, name = "OBSOLETE", info["name"]
            elif info["aspect"] != "biological_process":
                cat, name = "WRONG_BRANCH", info["name"]
            else:
                cat, name = "WRONG_BIOLOGY", info["name"]
            categorised.append({"id": tid, "name": name, "category": cat})

        counts = {}
        for c in categorised:
            counts[c["category"]] = counts.get(c["category"], 0) + 1

        log(f"  LLM {len(llm_ids)} IDs | "
            f"unsupported {len(unsupported)} | "
            f"allowed_not_enriched {len(allowed_not_enriched)} | "
            f"valid_enriched {len(valid_enriched)}")
        for c in categorised:
            log(f"    {c['category']:16s} {c['id']}  {c['name']}")

        results.append({
            "disease": disease,
            "n_genes": len(genes),
            "provider": provider,
            "model_id": model_id,
            "llm_ids": len(llm_ids),
            "allowed_ids": len(allowed_ids),
            "enriched_ids": len(enriched_ids),
            "unsupported": len(unsupported),
            "allowed_not_enriched": len(allowed_not_enriched),
            "not_enriched": len(not_enriched),
            "valid_enriched": len(valid_enriched),
            "unsupported_categorised": categorised,
            "category_counts": counts,
            "llm_text": text,
            "llm_ids_list": sorted(llm_ids),
        })

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    total_llm = sum(r["llm_ids"] for r in results)
    total_unsup = sum(r["unsupported"] for r in results)
    total_ane = sum(r["allowed_not_enriched"] for r in results)
    total_not_enr = sum(r["not_enriched"] for r in results)
    total_valid = sum(r["valid_enriched"] for r in results)
    totals = {}
    for r in results:
        for k, v in r["category_counts"].items():
            totals[k] = totals.get(k, 0) + v

    log("=" * 50)
    log(f"provider={provider} model={model_id}")
    log(f"TOTAL: {total_llm} LLM IDs across {len(results)} disease lists")
    log(f"  Contract A unsupported (not in allowed):     {total_unsup}")
    log(f"  Contract B allowed but not enriched:         {total_ane}")
    log(f"  Contract A+B violated (not in enriched):     {total_not_enr}")
    log(f"  Valid enriched hits:                         {total_valid}")
    log(f"  QuickGO breakdown: {totals}")
    log(f"wrote {out_path}")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as e:
        sys.exit(f"[FAIL] network error: {e}")

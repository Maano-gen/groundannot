# GroundAnnot

A closed-vocabulary contract for grounding LLM gene-set annotation in live enrichment backends.

**Maano Malima**^1, **Setshaba Taukobong**^2

^1 South African Medical Research Council (SAMRC) Genomics Platform, Cape Town, South Africa
^2 South African Medical Research Council (SAMRC) Biomedical Research and Innovation Platform (BRIP), Cape Town, South Africa

Corresponding author: Maano Malima — maano.malima@mrc.ac.za

---

## What this is

`GroundAnnot` is a small Python client for three live gene-list enrichment backends — PANTHER, Enrichr, and g:Profiler — plus MyGene.info for per-gene context. For any input gene list it writes two bounded vocabularies to disk:

- `allowed_term_ids.json` — every GO term ID any backend returned, with its backend label
- `enriched_term_ids.json` — the subset each backend flags as significant at FDR < 0.05 (or the backend's equivalent default)

A downstream LLM agent may **select** from these vocabularies, not generate terms from memory. This is the "closed-vocabulary contract". It removes three classes of error before the model speaks:

1. IDs that no backend returned
2. Enrichment claims that no backend flagged as significant
3. IDs paired with labels that do not match the backend's own label for that accession

The tool also writes a portable HTML report that separates the enrichment table (agent may cite) from per-gene context (agent may read only).

## Why it matters

An unconstrained LLM asked to name enriched GO terms for a gene list will produce real GO IDs that are not in any backend's enriched set, and — in a result reported in the accompanying manuscript — can pair every one of its picks with a label that does not correspond to that accession. See the paper for the full benchmark.

## Install

Requires Python 3.9+.

    python3 -m pip install --user -e .

Optional dependencies for the LLM benchmark:

    python3 -m pip install --user ".[benchmark]"      # llama-cpp-python + openai

## Quickstart

Run the tool on one of the bundled gene lists:

    python3 -m groundannot.cli \
        --genes examples/lists/alzheimers_disease.txt \
        --organism human \
        --label "Alzheimer's disease" \
        --output report_alzheimers_disease.html

Outputs three files in the current directory:

- `report_alzheimers_disease.html` — portable HTML report
- `allowed_term_ids.json` — Contract A (every ID returned, with labels)
- `enriched_term_ids.json` — Contract B (only significant terms)
- `enriched_with_fdr.json` — significant terms with FDR values, sorted ascending

## Repository layout

    groundannot/                The package
      backends.py               PANTHER, Enrichr, g:Profiler, MyGene clients
      cli.py                    Command-line entry point
      report.py                 HTML report generator
      species.py                3-tier organism resolver

    examples/lists/             Six curated disease gene lists
    examples/lists_hallmark/    50 MSigDB Hallmark gene sets (from Enrichr)

    data/snapshot_20260924/     Dated backend snapshot: 6 disease lists
    data/snapshot_20260925/     Second dated snapshot, same 6 disease lists
    data/snapshot_hallmarks/    Dated backend snapshot: 50 hallmark lists
    data/hallmark_gmt/          Source GMT file for the hallmark sets

    runs/                       215 disease-list LLM benchmark runs (JSON + log)
    runs_snapshot_20260925/     3 single-run cross-checks on the second snapshot

    stats/                      All analysis CSVs and plots

    benchmark_llm.py            LLM benchmark harness (local GGUF or OpenAI-compatible API)
    bounded_agent.py            Post-hoc bounded-agent coverage
    bounded_prompt_test.py      Bounded-prompt compliance test
    fdr_baseline.py             LLM picks vs FDR top-10 union
    hallmark_truth.py           Hallmark theme-recovery scoring
    fetch_hallmarks.py          Download MSigDB Hallmark gene sets from Enrichr
    make_figure2.py             Generate Figure 2 (Alzheimer's comparison)
    make_figure3.py             Generate Figure 3 (summary panels)
    stats.py                    Bootstrap CIs, learning curves, leave-one-out

## Reproducing the analysis

All benchmark runs and backend snapshots are committed, so the analysis scripts read from disk without any network access.

    python3 stats.py               # bootstrap CIs, learning curve, stability
    python3 fdr_baseline.py        # LLM picks vs statistical FDR baseline
    python3 hallmark_truth.py      # hallmark theme-recovery scoring
    python3 bounded_agent.py       # coverage of unconstrained picks

The LLM benchmark itself can be rerun for any OpenAI-compatible provider:

    python3 benchmark_llm.py \
        --provider groq \
        --model-id openai/gpt-oss-120b \
        --tag my-run

Local GGUF models are supported via `llama-cpp-python`:

    python3 benchmark_llm.py \
        --model ~/llm-models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
        --tag my-local

API keys are read from `.env` in the repo root (copy `.env.example` to `.env` and fill in).

## Data included

| File | Content |
|---|---|
| `data/snapshot_20260924/` | 6 disease lists × {allowed, enriched, enriched_fdr, report.html} |
| `data/snapshot_20260925/` | Same 6 lists, second dated pull |
| `data/snapshot_hallmarks/` | 50 Hallmark lists × {allowed, enriched, enriched_fdr, report.html} |
| `runs/` | 215 benchmark runs: 36 local Qwen2.5-7B, 92 Qwen3.8-27B, 87 GPT-OSS-120B |
| `runs_snapshot_20260925/` | 3 runs on the second snapshot (one per model) |
| `stats/` | Bootstrap CIs, FDR comparison, hallmark recovery, bounded coverage, all plots |

Every run JSON contains the raw model output, the extracted IDs, the three-way contract classification, and the QuickGO category for every unsupported ID.

## Key results (from the manuscript)

- Mean valid-enriched rates on six disease lists: **15.25%** (Qwen2.5-7B local), **20.14%** (Qwen3.8-27B), **35.05%** (GPT-OSS-120B). Bootstrap 95% CIs ≤ 0.7 pp wide.
- **Zero fabricated GO accessions** across all 215 runs.
- **10/10 label mismatches** for the local 7B model on the Alzheimer's list: every pick carried a label that does not match the live QuickGO label for the same accession.
- Bounded-prompt compliance: **60/60 for each model** (180/180 overall) when the enriched shortlist is placed in the prompt.

## Citation

If you use this code or data, please cite the accompanying manuscript:

> Malima M, Taukobong S. *GroundAnnot: a closed-vocabulary contract for grounding LLM gene-set annotation in live enrichment backends.* (Manuscript in preparation; DOI to be added on preprint.)

## License

MIT. See `LICENSE`.

## Contact

Maano Malima — maano.malima@mrc.ac.za

import argparse, json
from . import backends, report


def main():
    ap = argparse.ArgumentParser(description="GroundAnnot")
    ap.add_argument("--genes", required=True)
    ap.add_argument("--organism", default="human")
    ap.add_argument("--output", default="groundannot_report.html")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    with open(args.genes) as f:
        genes = [g.strip() for g in f if g.strip()]
    print(f"[INFO] {len(genes)} genes loaded")

    enrichments = []
    for name, fn in [("PANTHER", backends.panther),
                     ("Enrichr", backends.enrichr),
                     ("g:Profiler", backends.gprofiler)]:
        try:
            print(f"[INFO] querying {name}...")
            result = fn(genes, organism=args.organism)
            enrichments.append({"source": name, "terms": result["terms"]})
        except Exception as e:
            print(f"[WARN] {name} failed: {e}")

    print("[INFO] querying MyGene.info...")
    mg = backends.mygene(genes, organism=args.organism)

    # Contract A: every ID returned by any backend
    allowed = {b["source"]: [t["id"] for t in b["terms"]] for b in enrichments}

    # Contract B: only IDs flagged significant (FDR or p_value < 0.05)
    enriched = {b["source"]: [t["id"] for t in b["terms"] if t.get("significant")]
                for b in enrichments}

    # Enriched terms with their statistics (for FDR baseline comparison)
    enriched_with_fdr = {}
    for b in enrichments:
        rows = []
        for t in b["terms"]:
            if not t.get("significant"):
                continue
            rows.append({
                "id": t.get("id"),
                "label": t.get("label"),
                "fdr": t.get("fdr"),
                "p_value": t.get("p_value"),
                "fold": t.get("fold"),
                "source": t.get("source") or b["source"],
            })
        # sort by FDR ascending so the top of the list is the most significant
        rows.sort(key=lambda r: (r["fdr"] is None, r["fdr"]))
        enriched_with_fdr[b["source"]] = rows

    with open("allowed_term_ids.json", "w") as f:
        json.dump(allowed, f, indent=2)
    with open("enriched_term_ids.json", "w") as f:
        json.dump(enriched, f, indent=2)
    with open("enriched_with_fdr.json", "w") as f:
        json.dump(enriched_with_fdr, f, indent=2)

    report.write_report(args.output, genes, args.organism,
                        enrichments, mg["cards"], label=args.label)
    print(f"[INFO] wrote {args.output}, allowed_term_ids.json, "
          f"enriched_term_ids.json, enriched_with_fdr.json")


if __name__ == "__main__":
    main()

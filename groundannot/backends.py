import requests, re, time
from . import species


def _sp(organism, backend):
    r = species.resolve(organism)
    if backend == "panther":
        return r["taxon_id"]
    if backend == "gprofiler":
        return r["gprofiler"]
    if backend == "mygene":
        return r["mygene"]
    raise ValueError(backend)


def _as_list(x):
    """Normalise dict/None/scalar to a list."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _dedupe(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _go_terms(entries):
    """Extract GO term labels from a list of dicts or strings."""
    out = []
    for e in _as_list(entries):
        if isinstance(e, dict) and e.get("term"):
            out.append(e["term"])
        elif isinstance(e, str):
            out.append(e)
    return _dedupe(out)


def _interpro_terms(entries):
    out = []
    for e in _as_list(entries):
        if isinstance(e, dict) and e.get("desc"):
            out.append(e["desc"])
        elif isinstance(e, str):
            out.append(e)
    return _dedupe(out)


def _flatten_pathways(pw):
    if not isinstance(pw, dict):
        return []
    out = []
    for src, entries in pw.items():
        for e in _as_list(entries):
            if isinstance(e, dict) and e.get("name"):
                out.append(f"{e['name']} ({src})")
    return _dedupe(out)


def panther(genes, organism="human", dataset="GO:0008150"):
    taxon = _sp(organism, "panther")
    base = "https://pantherdb.org/services/oai/pantherdb"
    gs = ",".join(genes)
    r = requests.post(f"{base}/geneinfo",
        data={"geneInputList": gs, "organism": taxon}, timeout=60)
    r2 = requests.post(f"{base}/enrich/overrep",
        data={"geneInputList": gs, "organism": taxon,
              "annotDataSet": dataset,
              "enrichmentTestType": "FISHER",
              "correction": "FDR"}, timeout=60)
    terms = r2.json()["results"]["result"]
    out = []
    for t in terms:
        term = t.get("term", {})
        tid = term.get("id")
        if not tid:
            continue
        fdr = t.get("fdr")
        out.append({"id": tid, "label": term.get("label"),
                    "p_value": t.get("pValue"), "fdr": fdr,
                    "fold": t.get("fold_enrichment"), "source": "PANTHER",
                    "significant": (fdr is not None and fdr < 0.05)})
    return {"source": "panther", "terms": out,
            "raw": {"geneinfo": r.json(), "enrich": r2.json()}}


def enrichr(genes, organism="human", library="GO_Biological_Process_2025"):
    if organism.lower() != "human":
        raise ValueError("Enrichr only supports human via this backend")
    base = "https://maayanlab.cloud/Enrichr"
    resp = requests.post(f"{base}/addList",
        files={"list": (None, "\n".join(genes)),
               "description": (None, "groundannot")}, timeout=30)
    uid = resp.json()["userListId"]
    time.sleep(1)
    r = requests.get(f"{base}/enrich",
        params={"userListId": uid, "backgroundType": library}, timeout=30)
    terms = r.json().get(library, [])
    out = []
    for t in terms:
        raw_label = t[1]
        m = re.search(r"\(GO:\d+\)", raw_label)
        tid = m.group(0).strip("()") if m else raw_label
        # strip parenthetical GO id from the label
        label = re.sub(r"\s*\(GO:\d+\)\s*$", "", raw_label).strip()
        adj = t[6]
        out.append({"id": tid, "label": label,
                    "p_value": t[2], "fdr": adj, "source": "Enrichr",
                    "significant": (adj is not None and adj < 0.05)})
    return {"source": "enrichr", "terms": out, "raw": r.json()}


def gprofiler(genes, organism="human",
              sources=("GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC")):
    org = _sp(organism, "gprofiler")
    r = requests.post("https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
        json={"organism": org, "query": genes,
              "sources": list(sources), "user_threshold": 0.05,
              "all_results": False, "no_iea": True}, timeout=60)
    out = []
    for t in r.json().get("result", []):
        p = t.get("p_value")
        # g:Profiler's p_value is already g:SCS-corrected;
        # report it in both columns for uniformity
        out.append({"id": t.get("native"), "label": t.get("name"),
                    "p_value": p, "fdr": p,
                    "source": "g:Profiler/" + t.get("source", ""),
                    "significant": (p is not None and p < 0.05)})
    return {"source": "gprofiler", "terms": out, "raw": r.json()}


def mygene(genes, organism="human"):
    species_code = _sp(organism, "mygene")
    fields = ("symbol,name,entrezgene,summary,"
              "go.BP,go.MF,go.CC,"
              "pathway.kegg,pathway.reactome,pathway.wikipathways,pathway.panther,"
              "interpro")
    r = requests.post("https://mygene.info/v3/query",
        data={"q": ",".join(genes), "scopes": "symbol",
              "species": species_code, "fields": fields}, timeout=90)
    cards = []
    for g in r.json():
        if not g.get("symbol"):
            continue
        go = g.get("go", {}) or {}
        cards.append({
            "symbol": g.get("symbol"),
            "name": g.get("name", ""),
            "summary": (g.get("summary") or "").replace("\n", " "),
            "go_bp": _go_terms(go.get("BP")),
            "go_mf": _go_terms(go.get("MF")),
            "go_cc": _go_terms(go.get("CC")),
            "pathways": _flatten_pathways(g.get("pathway")),
            "interpro": _interpro_terms(g.get("interpro")),
        })
    return {"source": "mygene", "cards": cards, "raw": r.json()}

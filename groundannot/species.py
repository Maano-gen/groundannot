import requests

CACHE = {}

def _panther_catalogue():
    if "panther" not in CACHE:
        r = requests.get(
            "https://pantherdb.org/services/oai/pantherdb/supportedgenomes",
            timeout=30)
        CACHE["panther"] = r.json()["search"]["output"]["genomes"]["genome"]
    return CACHE["panther"]

def _gprofiler_catalogue():
    if "gprofiler" not in CACHE:
        r = requests.get(
            "https://biit.cs.ut.ee/gprofiler/api/util/organisms_list",
            timeout=30)
        CACHE["gprofiler"] = r.json()
    return CACHE["gprofiler"]

def resolve(query):
    q = str(query).strip().lower()

    LOCAL = {
        "human":     ("9606",  "hsapiens",      "human"),
        "mouse":     ("10090", "mmusculus",     "mouse"),
        "rat":       ("10116", "rnorvegicus",   "rat"),
        "fly":       ("7227",  "dmelanogaster", "fruitfly"),
        "worm":      ("6239",  "celegans",      "worm"),
        "yeast":     ("559292","scerevisiae",   "yeast"),
        "zebrafish": ("7955",  "drerio",        "zebrafish"),
    }
    if q in LOCAL:
        taxon, gp, mg = LOCAL[q]
        return {"taxon_id": taxon, "gprofiler": gp, "mygene": mg, "name": q}

    taxon_id = None
    name = q
    for g in _panther_catalogue():
        if (q == str(g["taxon_id"]) or q == g["name"].lower()
                or q == g["long_name"].lower()
                or q in g["long_name"].lower()):
            taxon_id = str(g["taxon_id"])
            name = g["long_name"]
            break

    if taxon_id is None:
        if q.isdigit():
            taxon_id = q
        else:
            raise ValueError(f"Species '{query}' not found in PANTHER catalogue.")

    gp_code = None
    for o in _gprofiler_catalogue():
        if o["taxonomy_id"] == taxon_id:
            gp_code = o["id"]
            break
    if gp_code is None:
        raise ValueError(f"Species '{query}' (taxon {taxon_id}) not in g:Profiler.")

    return {"taxon_id": taxon_id, "gprofiler": gp_code,
            "mygene": taxon_id, "name": name}

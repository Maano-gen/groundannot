from jinja2 import Template
from datetime import datetime
import json

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>GroundAnnot report</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 2em auto; }
h1 { border-bottom: 2px solid #333; }
h2 { margin-top: 2em; color: #444; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }
th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; }
th { background: #f0f0f0; }
.gene-card { border: 1px solid #ddd; padding: 1em; margin: 1em 0; border-radius: 6px; }
.gene-card h3 { margin-top: 0; }
.small { color: #666; font-size: 0.85em; }
.section-tag { display: inline-block; background: #eee; padding: 2px 8px;
               border-radius: 4px; font-size: 0.8em; }
.closed { background: #e8f4ea; }
.context { background: #f4f0e8; }
</style></head><body>

<h1>GroundAnnot report — {{ label }}</h1>
<p class="small">Run: {{ run_date }} &nbsp;|&nbsp; Organism: {{ organism }}
&nbsp;|&nbsp; Input genes: {{ genes|length }}</p>
<p class="small">Genes: {{ genes|join(', ') }}</p>

<h2><span class="section-tag closed">Closed vocabulary</span> Enrichment</h2>
<p>The language model may cite term IDs only from the tables below.</p>

{% for block in enrichments %}
<h3>{{ block.source }}</h3>
<table>
<tr><th>Term ID</th><th>Label</th><th>p-value</th><th>FDR</th></tr>
{% for t in block.terms[:25] %}
<tr><td>{{ t.id }}</td><td>{{ t.label }}</td>
<td>{{ '%.2e'|format(t.p_value) if t.p_value else '—' }}</td>
<td>{{ '%.2e'|format(t.fdr) if t.fdr else '—' }}</td></tr>
{% endfor %}
</table>
{% endfor %}

<h2><span class="section-tag context">Context only</span> Per-gene annotations</h2>
<p>The language model may read these for narrative but must not cite them as enrichment evidence.</p>

{% for c in cards %}
<div class="gene-card">
<h3>{{ c.symbol }} — {{ c.name }}</h3>
<p><b>Summary:</b> {{ c.summary or '—' }}</p>
<p><b>GO:BP:</b> {{ c.go_bp[:6]|join('; ') or '—' }}</p>
<p><b>GO:MF:</b> {{ c.go_mf[:6]|join('; ') or '—' }}</p>
<p><b>GO:CC:</b> {{ c.go_cc[:6]|join('; ') or '—' }}</p>
<p><b>Pathways:</b> {{ c.pathways[:8]|join('; ') or '—' }}</p>
<p><b>InterPro:</b> {{ c.interpro[:6]|join('; ') or '—' }}</p>
</div>
{% endfor %}

<h2>Provenance</h2>
<ul>
<li>PANTHER: pantherdb.org, REST (geneinfo + enrich/overrep)</li>
<li>Enrichr: maayanlab.cloud/Enrichr, GO_Biological_Process_2025</li>
<li>g:Profiler: biit.cs.ut.ee/gprofiler, GO/KEGG/REAC</li>
<li>MyGene.info: mygene.info/v3/query</li>
</ul>
</body></html>"""

def write_report(path, genes, organism, enrichments, cards, label=None):
    html = Template(HTML).render(
        run_date=datetime.now().isoformat(timespec="seconds"),
        organism=organism, genes=genes,
        enrichments=enrichments, cards=cards,
        label=label or path.replace(".html", ""))
    with open(path, "w") as f:
        f.write(html)

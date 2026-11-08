"""Pull scoping tables from QLever into ../qlever/*.csv (cached; skip if present)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from qlever import run  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "scoping" / "qlever"
WP = 'FILTER(STRENDS(STR(?site), ".wikipedia.org/") && ?site != <https://abstract.wikipedia.org/>)'

QUERIES = {
    "n_humans": "SELECT (COUNT(?h) AS ?n) WHERE { ?h wdt:P31 wd:Q5 }",
    "sites": """SELECT ?site ?g WHERE { VALUES ?site { <https://en.wikipedia.org/> <https://abstract.wikipedia.org/>
        <https://en.wikiquote.org/> <https://simple.wikipedia.org/> } ?site wikibase:wikiGroup ?g }""",
    # one row per human with >=1 Wikipedia article: how many editions (group on the IRI, not a BIND string --
    # sorting 30M computed strings is what timed out)
    "wp_counts": """SELECT ?h (COUNT(?a) AS ?nwp) WHERE {
        ?h wdt:P31 wd:Q5 . ?a schema:about ?h ; schema:isPartOf ?site . ?site wikibase:wikiGroup "wikipedia" .
        FILTER(?site != <https://abstract.wikipedia.org/>) } GROUP BY ?h""",
    "en_titles": """SELECT ?h ?title WHERE {
        ?h wdt:P31 wd:Q5 . ?a schema:about ?h ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?title }""",
    # every birth statement (rank picked in pandas; QLever mishandles rank FILTERs)
    "births": """SELECT ?h ?y ?prec ?rank WHERE {
        ?h wdt:P31 wd:Q5 ; p:P569 ?st . ?st psv:P569 ?v ; wikibase:rank ?rank .
        ?v wikibase:timeValue ?t ; wikibase:timePrecision ?prec . BIND(YEAR(?t) AS ?y) }""",
    "deaths": """SELECT ?h (MIN(YEAR(?d)) AS ?dy) WHERE { ?h wdt:P31 wd:Q5 ; wdt:P570 ?d } GROUP BY ?h""",
    "floruit": """SELECT ?h (MIN(YEAR(?f)) AS ?fy) WHERE { ?h wdt:P31 wd:Q5 ; wdt:P1317 ?f } GROUP BY ?h""",
}

for name, query in QUERIES.items():
    path = OUT / f"{name}.csv"
    if path.exists():
        print(f"{name}: cached")
        continue
    print(f"{name}: querying...", flush=True)
    try:
        body = run(query, timeout=1800)
    except RuntimeError as e:
        print(f"{name}: FAILED {e}", flush=True)
        time.sleep(30)
        continue
    tmp = path.with_suffix(".csv.tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.rename(path)
    print(f"{name}: {body.count(chr(10)) - 1:,} rows", flush=True)
    time.sleep(30)  # be gentle with the public instance

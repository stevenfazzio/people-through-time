"""Stage 00: the roster -- every Wikidata human (P31=Q5) with >=1 language-Wikipedia article.

One QLever query, GROUP BY on the item IRI (it ran in 17 s on 2026-11-07 for
13.8M humans; grouping on a BIND-ed string instead timed out while sorting).
Birth dates are NOT fetched here: the all-humans P569 query exceeds QLever's
30 s cap, and only people above the views floor need them (a later stage).

The raw QLever CSV is cached at RAW_CSV and reused when present, because the
public instance rate-limits hard (nginx 429s for ~50 min at a time). The first
cache was seeded on 2026-11-08 from scoping's wp_counts.csv -- the identical
query, run 2026-11-07 23:03 -- after QLever was 429ing again. Delete the cache
to refresh the roster.

Output: data/roster.parquet [qid, n_wikipedias]
"""

import io

import pandas as pd

import config
from util import qlever_csv, write_parquet_atomic

RAW_CSV = config.DATA_DIR / "raw" / "roster_qlever.csv"

QUERY = """SELECT ?h (COUNT(?a) AS ?nwp) WHERE {
  ?h wdt:P31 wd:Q5 . ?a schema:about ?h ; schema:isPartOf ?site . ?site wikibase:wikiGroup "wikipedia" .
  FILTER(?site != <https://abstract.wikipedia.org/>) } GROUP BY ?h"""

ENTITY_PREFIX = "http://www.wikidata.org/entity/"


def main() -> None:
    if RAW_CSV.exists():
        print(f"Using cached QLever response {RAW_CSV}")
        body = RAW_CSV.read_text(encoding="utf-8")
    else:
        body = qlever_csv(QUERY)
        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        tmp = RAW_CSV.with_suffix(".csv.tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(RAW_CSV)
    df = pd.read_csv(io.StringIO(body))
    assert list(df.columns) == ["h", "nwp"], f"unexpected columns {list(df.columns)}"
    assert df["h"].str.startswith(ENTITY_PREFIX + "Q").all(), "non-item IRI in roster"
    df = pd.DataFrame({"qid": df["h"].str.slice(len(ENTITY_PREFIX)), "n_wikipedias": df["nwp"].astype("int32")})
    assert not df["qid"].duplicated().any()
    # 4,848,996 on 2026-11-07. A result far below that means a truncated
    # response or a changed predicate, not a real shrinkage.
    assert len(df) > 4_000_000, f"roster suspiciously small: {len(df):,}"
    print(f"Roster: {len(df):,} humans with >=1 Wikipedia article")
    print(df["n_wikipedias"].describe().to_string())
    write_parquet_atomic(df, config.ROSTER_PARQUET)
    print(f"Wrote {config.ROSTER_PARQUET}")


if __name__ == "__main__":
    main()

"""Stage 04: Wikidata facts for everyone above the views floor -- dates, gender, label, description.

Birth year is the one input the fame score (median_views x age) still lacks.
WDQS, one VALUES query per config.WDQS_CHUNK QIDs (3-5 s each on 2026-11-08),
serial: WDQS gives a client ~60 s of query time per minute, and QLever's 30 s
cap plus its 429 lockouts (2026-11-07) ruled out one big query there.

Fetch: each chunk's rows go to config.FACTS_RAW_DIR/chunk_NNNNN.parquet as
returned (statement level, every rank), plus one "_requested" row per QID
asked for, so a rerun fetches only QIDs no chunk has asked for yet, whatever
the input order. Build: one value per property, junk filtered where the output
row is built, never in the cache.

Picking a date (P569 birth, P570 death, P1636 baptism, P1317 floruit, P2031
work-period start): drop deprecated, keep preferred if any else normal, then
the highest precision, then the lower median year. WDQS writes BCE years in
astronomical numbering (-99 = 100 BC: Julius Caesar), so age = 2026 - year
needs no year-zero fix.

Century / millennium precision (checked 2026-11-08, scoping/century_precision.py):
a ROUND stated year (ending 00 or 01) is a boundary of the century it names --
CE 1600 is "16th century", its last year (Malyuta Skuratov, died 1573, "born
1600"); BCE -900 is "9th century BC", its earliest -- so it moves to that
century's midpoint (1550, -850). A non-round year is the editors' estimate and
is kept (Pythagoras: preferred -582 at century precision, his year-precision
values -569..-581). Among coarse births with a precise death, deaths before
birth fall from 18.2% to 4.0% (CE) and lifespans over 122 from 14.2% to 1.2%
(BCE); moving every coarse value to its midpoint does slightly worse.
Decades (precision 8) are kept as stated: at most ~5 years off.

birth_year_est: birth_year, else the first of baptism / floruit / work start /
death minus its config.BIRTH_FALLBACK_OFFSETS offset. A birth or death after
config.FAME_REFERENCE_YEAR counts as missing. Null (source "conflict") when the
birth and death disagree -- death first, or a lifespan over MAX_LIFESPAN, both
at year precision or finer -- never clamped.

Output: data/people_facts.parquet, one row per person above the floor, in
people_views order: [qid, label, label_lang, description, gender, birth_year,
birth_precision, n_birth_values, death_year, death_precision, baptism_year,
floruit_year, work_start_year, birth_year_est, birth_source]. Every *_year is
after the round-century move; birth_year_est is the one stage 05 scores.
"""

import time

import numpy as np
import pandas as pd

import config
from util import WdqsTimeout, log, make_session, wdqs_select, write_parquet_atomic

DATE_PROPS = {"P569": "birth", "P570": "death", "P1636": "baptism", "P1317": "floruit", "P2031": "work_start"}
RAW_COLUMNS = ["qid", "kind", "rank", "t", "prec", "val", "lang", "vlabel"]
REQUESTED = "_requested"
ENTITY = "http://www.wikidata.org/entity/"
ONTOLOGY = "http://wikiba.se/ontology#"
YEAR_PRECISION = 9  # Wikidata time precision: 6 millennium, 7 century, 8 decade, 9 year, 10 month, 11 day

QUERY = """SELECT ?item ?kind ?rank ?t ?prec ?val ?lang ?vlabel WHERE {
  VALUES ?item { {values} }
  { VALUES (?p ?psv ?kind) { {date_props} }
    ?item ?p ?st . ?st ?psv ?v ; wikibase:rank ?rank . ?v wikibase:timeValue ?t ; wikibase:timePrecision ?prec . }
  UNION { ?item wdt:P21 ?val . BIND("P21" AS ?kind) OPTIONAL { ?val rdfs:label ?vlabel FILTER(LANG(?vlabel) = "en") } }
  UNION { ?item rdfs:label ?val . FILTER(LANG(?val) IN ("en", "mul")) BIND("label" AS ?kind) BIND(LANG(?val) AS ?lang) }
  UNION { ?item schema:description ?val . FILTER(LANG(?val) = "en") BIND("desc" AS ?kind) }
}"""


def build_query(qids: list[str]) -> str:
    date_props = " ".join(f'(p:{p} psv:{p} "{p}")' for p in DATE_PROPS)
    return QUERY.replace("{values}", " ".join(f"wd:{q}" for q in qids)).replace("{date_props}", date_props)


def rows_frame(rows: list[dict[str, str]], qids: list[str]) -> pd.DataFrame:
    """WDQS rows as stored in the cache (prefixes stripped, nothing else), plus the _requested rows."""
    df = pd.DataFrame(rows, columns=["item", *RAW_COLUMNS[1:]]).rename(columns={"item": "qid"}).astype("string")
    df["qid"] = df["qid"].str.removeprefix(ENTITY)
    df["rank"] = df["rank"].str.removeprefix(ONTOLOGY)
    is_p21 = df["kind"] == "P21"
    df.loc[is_p21, "val"] = df.loc[is_p21, "val"].str.removeprefix(ENTITY)
    requested = pd.DataFrame({"qid": qids, "kind": REQUESTED})
    return pd.concat([df, requested], ignore_index=True)[RAW_COLUMNS].astype("string")


def fetch_chunk(session, qids: list[str]) -> pd.DataFrame:
    """One VALUES query; on a WDQS timeout, halve the chunk and recurse."""
    try:
        rows = wdqs_select(session, build_query(qids))
    except WdqsTimeout:
        if len(qids) == 1:
            raise
        log(f"WDQS timeout on {len(qids)} QIDs; splitting")
        half = len(qids) // 2
        return pd.concat([fetch_chunk(session, qids[:half]), fetch_chunk(session, qids[half:])], ignore_index=True)
    return rows_frame(rows, qids)


def chunk_files() -> list:
    return sorted(config.FACTS_RAW_DIR.glob("chunk_*.parquet"))


def requested_qids() -> set[str]:
    done: set[str] = set()
    for path in chunk_files():
        raw = pd.read_parquet(path, columns=["qid", "kind"])
        done.update(raw.loc[raw["kind"] == REQUESTED, "qid"])
    return done


def fetch(qids: list[str]) -> None:
    """Fetch every QID no cached chunk has requested yet, in input order, one chunk file per query."""
    done = requested_qids()
    todo = [q for q in qids if q not in done]
    files = chunk_files()
    next_n = int(files[-1].stem.split("_")[1]) + 1 if files else 0
    n_chunks = -(-len(todo) // config.WDQS_CHUNK)
    log(f"Facts: {len(qids) - len(todo):,} of {len(qids):,} QIDs cached; {len(todo):,} to fetch in {n_chunks} chunks")
    session = make_session()
    t0 = time.time()
    for i in range(n_chunks):
        chunk = todo[i * config.WDQS_CHUNK : (i + 1) * config.WDQS_CHUNK]
        write_parquet_atomic(fetch_chunk(session, chunk), config.FACTS_RAW_DIR / f"chunk_{next_n + i:05d}.parquet")
        if (i + 1) % 25 == 0 or i + 1 == n_chunks:
            rate = (time.time() - t0) / (i + 1)
            log(f"  {i + 1}/{n_chunks} chunks, {rate:.1f} s each, ~{rate * (n_chunks - i - 1) / 60:.0f} min left")
        time.sleep(config.WDQS_SLEEP)


def load_raw(qids: list[str]) -> pd.DataFrame:
    """Cached rows for qids, each QID from the latest chunk that requested it (never doubled)."""
    wanted = set(qids)
    parts = []
    for path in chunk_files():
        raw = pd.read_parquet(path)
        parts.append(raw[raw["qid"].isin(wanted)].assign(chunk=int(path.stem.split("_")[1])))
    raw = pd.concat(parts, ignore_index=True)
    req = raw[raw["kind"] == REQUESTED]
    missing = wanted - set(req["qid"])
    if missing:
        raise RuntimeError(f"{len(missing):,} QIDs never fetched, e.g. {sorted(missing)[:5]}")
    latest = raw["qid"].map(req.groupby("qid")["chunk"].max())
    return raw[(raw["chunk"] == latest) & (raw["kind"] != REQUESTED)].drop(columns="chunk").reset_index(drop=True)


def pick_dates(raw: pd.DataFrame) -> pd.DataFrame:
    """Per (qid, kind): non-deprecated, best rank, highest precision, lower median year.

    Returns long [qid, kind, year, prec, n_values]; kind is the DATE_PROPS name.
    n_values = distinct non-deprecated years stated for that property.
    """
    d = raw[raw["kind"].isin(list(DATE_PROPS)) & (raw["rank"] != "DeprecatedRank")]
    d = pd.DataFrame({
        "qid": d["qid"].astype(str),
        "kind": d["kind"].map(DATE_PROPS),
        "year": pd.to_numeric(d["t"].str.extract(r"^([+-]?\d+)-", expand=False), errors="coerce"),
        "prec": pd.to_numeric(d["prec"]).astype(int),
        "pref": d["rank"] == "PreferredRank",
    }).dropna(subset=["year"])  # fmt: skip
    d["year"] = d["year"].astype(int)
    keys = ["qid", "kind"]
    n_values = d.groupby(keys)["year"].nunique().rename("n_values")
    d = d[d["pref"] == d.groupby(keys)["pref"].transform("max")]
    d = d[d["prec"] == d.groupby(keys)["prec"].transform("max")]
    d = d.sort_values([*keys, "year"])
    pos = d.groupby(keys).cumcount()
    size = d.groupby(keys)["year"].transform("size")
    picked = d[pos == (size - 1) // 2].set_index(keys)[["year", "prec"]]
    return picked.join(n_values).reset_index()


def coarse_midpoint(year: pd.Series, prec: pd.Series) -> pd.Series:
    """Round century- or coarser-precision years -> the midpoint of the century/millennium they name.

    Ordinal k = ceil(|year| / unit): CE 1600 and 1501 are both the 16th century
    (midpoint 1550); BCE -900 and -801 the 9th century BC (midpoint -850).
    Non-round years and anything at decade precision or finer are returned as is.
    """
    y = year.to_numpy(dtype=float)
    p = prec.to_numpy(dtype=float)
    coarse = p <= 7
    unit = 10.0 ** (YEAR_PRECISION - np.where(coarse, p, YEAR_PRECISION))
    round_ = coarse & np.isin(np.abs(y) % unit, [0, 1])
    k = np.maximum(np.ceil(np.abs(y) / unit), 1)  # year 0 (= 1 BC) is in the 1st century BC
    mid = np.where(y > 0, unit * k - unit / 2, -unit * k + unit / 2)
    return pd.Series(np.where(round_, mid, y), index=year.index).astype(int)


def text_facts(raw: pd.DataFrame) -> pd.DataFrame:
    """label (en over mul), label_lang, description (en), gender (en labels of every truthy P21 value)."""
    lab = raw[raw["kind"] == "label"].assign(rank_lang=lambda x: (x["lang"] != "en").astype(int))
    lab = lab.sort_values(["qid", "rank_lang"]).drop_duplicates("qid").set_index("qid")
    desc = raw[raw["kind"] == "desc"].drop_duplicates("qid").set_index("qid")["val"]
    g = raw[raw["kind"] == "P21"]
    g = g.assign(name=g["vlabel"].fillna(g["val"])).drop_duplicates(["qid", "name"])
    gender = g.sort_values("name").groupby("qid")["name"].agg("; ".join)
    return pd.DataFrame({
        "label": lab["val"], "label_lang": lab["lang"], "description": desc, "gender": gender,
    }).rename_axis("qid")  # fmt: skip


def build(qids: list[str], raw: pd.DataFrame) -> pd.DataFrame:
    dates = pick_dates(raw)
    dates["year"] = coarse_midpoint(dates["year"], dates["prec"])
    wide = dates.pivot(index="qid", columns="kind", values=["year", "prec", "n_values"])
    out = pd.DataFrame({"qid": qids}).set_index("qid")
    out = out.join(text_facts(raw))

    def col(field: str, kind: str) -> pd.Series:
        if (field, kind) not in wide.columns:
            return pd.Series(pd.NA, index=out.index, dtype="Int32")
        return wide[(field, kind)].reindex(out.index).astype("Int32")

    out["birth_year"], out["birth_precision"] = col("year", "birth"), col("prec", "birth").astype("Int8")
    out["n_birth_values"] = col("n_values", "birth").fillna(0).astype("int16")
    out["death_year"], out["death_precision"] = col("year", "death"), col("prec", "death").astype("Int8")
    for kind in ("baptism", "floruit", "work_start"):
        out[f"{kind}_year"] = col("year", kind)

    # Junk: a birth or death in the future counts as missing.
    future = config.FAME_REFERENCE_YEAR
    for kind in ("birth", "death"):
        bad = out[f"{kind}_year"] > future
        out.loc[bad.fillna(False), [f"{kind}_year", f"{kind}_precision"]] = pd.NA
    life = out["death_year"] - out["birth_year"]
    precise = (out["birth_precision"] >= YEAR_PRECISION) & (out["death_precision"] >= YEAR_PRECISION)
    conflict = (precise & ((life < 0) | (life > config.MAX_LIFESPAN))).fillna(False)

    est = out["birth_year"].copy()
    source = pd.Series(np.where(est.notna(), "P569", None), index=out.index, dtype="string")
    for kind, offset in config.BIRTH_FALLBACK_OFFSETS.items():
        fill = est.isna() & out[f"{kind}_year"].notna() & ~conflict
        est[fill] = out.loc[fill, f"{kind}_year"] - offset
        source[fill] = kind
    est[conflict], source[conflict] = pd.NA, "conflict"
    out["birth_year_est"], out["birth_source"] = est.astype("Int32"), source
    return out.reset_index()


def fallback_gaps(out: pd.DataFrame) -> pd.Series:
    """Median gap (other date - P569) at year precision, the basis of BIRTH_FALLBACK_OFFSETS."""
    precise_birth = out["birth_precision"] >= YEAR_PRECISION
    gaps = {}
    for kind in config.BIRTH_FALLBACK_OFFSETS:
        both = precise_birth & out[f"{kind}_year"].notna() & (out["birth_source"] == "P569")
        if kind == "death":
            both &= out["death_precision"] >= YEAR_PRECISION
        gaps[kind] = (out.loc[both, f"{kind}_year"] - out.loc[both, "birth_year"]).median()
    return pd.Series(gaps)


def main() -> None:
    views = pd.read_parquet(config.PEOPLE_VIEWS_PARQUET, columns=["qid", "median_views"])
    qids = views.loc[views["median_views"] >= config.MIN_MEDIAN_VIEWS, "qid"].tolist()
    fetch(qids)
    out = build(qids, load_raw(qids))

    n = len(out)
    log(f"{n:,} people; birth_source:\n{out['birth_source'].value_counts(dropna=False).to_string()}")
    log(f"No birth year at all: {out['birth_year_est'].isna().sum():,} ({out['birth_year_est'].isna().mean():.1%})")
    log(f"birth precision (P569):\n{out['birth_precision'].value_counts(dropna=False).sort_index().to_string()}")
    gaps = fallback_gaps(out)
    for kind, offset in config.BIRTH_FALLBACK_OFFSETS.items():
        log(f"  {kind}: median gap to P569 {gaps[kind]:.0f} y (config offset {offset})")
    log(f"label {out['label'].notna().mean():.1%} (mul only {(out['label_lang'] == 'mul').mean():.1%}), "
        f"description {out['description'].notna().mean():.1%}, gender {out['gender'].notna().mean():.1%}")  # fmt: skip
    log(f"gender:\n{out['gender'].value_counts().head(8).to_string()}")
    write_parquet_atomic(out, config.PEOPLE_FACTS_PARQUET)
    log(f"Wrote {config.PEOPLE_FACTS_PARQUET}")


if __name__ == "__main__":
    main()

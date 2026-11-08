"""enwiki biographies + birth/death years from category dumps. Cached steps -> ../out/*.parquet.

Categories used (namespace 14):
  N births / N BC births / AD N births      -> year
  1950s births / 1950s BC births            -> decade
  5th-century births / 5th-century BC births -> century   (same shapes for deaths, millennia too)
  Living people, Possibly living people
  Year of birth missing/unknown/uncertain... -> birth_missing (and Year of death ... likewise)
"""

import gzip
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data" / "scoping"
DUMPS, OUT = ROOT / "dumps", ROOT / "out"  # dumps: enwiki-latest-{linktarget,categorylinks,page}.sql.gz
ESC = re.compile(r"\\(.)")

YEAR = re.compile(r"(?:AD_)?(\d{1,4})(_BC)?_(births|deaths)")
DECADE = re.compile(r"(\d{1,3})0s(_BC)?_(births|deaths)")
CENT = re.compile(r"(\d{1,2})(?:st|nd|rd|th)-(century|millennium)(_BC)?_(births|deaths)")


def classify(t: str):
    """-> (field, lo, hi, precision) | None. BC years are negative."""
    if m := YEAR.fullmatch(t):
        y = int(m[1])
        y = -y if m[2] else y
        return (m[3][:-1], y, y, "year")
    if m := DECADE.fullmatch(t):
        d = int(m[1]) * 10
        lo, hi = (-(d + 9), -d) if m[2] else (d, d + 9)
        return (m[3][:-1], lo, hi, "decade")
    if m := CENT.fullmatch(t):
        n, span = int(m[1]), (100 if m[2] == "century" else 1000)
        lo, hi = ((n - 1) * span + 1, n * span)
        if m[3]:
            lo, hi = -hi, -lo
        return (m[4][:-1], lo, hi, m[2])
    if t == "Living_people":
        return ("living", None, None, "flag")
    if t == "Possibly_living_people":
        return ("possibly_living", None, None, "flag")
    if t.startswith(("Year_of_birth_", "Date_of_birth_")):
        return ("birth_missing", None, None, t)
    if t.startswith(("Year_of_death_", "Date_of_death_")):
        return ("death_missing", None, None, t)
    return None


def step_cats() -> pd.DataFrame:
    path = OUT / "cats.parquet"
    if path.exists():
        return pd.read_parquet(path)
    row = re.compile(r"\((\d+),(-?\d+),'((?:[^'\\]|\\.)*)'\)")
    rows, seen_births = [], []
    with gzip.open(DUMPS / "enwiki-latest-linktarget.sql.gz", "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith(("INSERT INTO", "(")):
                continue
            for m in row.finditer(line):
                if m[2] != "14":
                    continue
                t = ESC.sub(r"\1", m[3])
                c = classify(t)
                if c:
                    rows.append((int(m[1]), t, *c))
                elif t.endswith(("_births", "_deaths")):
                    seen_births.append(t)
    df = pd.DataFrame(rows, columns=["lt_id", "cat", "field", "lo", "hi", "prec"])
    assert len(df), "no categories matched -- linktarget format changed?"
    print(df.groupby(["field", "prec"]).size().to_string())
    unmatched = pd.Series(seen_births)
    print(f"\n{len(unmatched):,} other *_births/*_deaths categories NOT used, e.g.:")
    print(unmatched.sample(min(25, len(unmatched)), random_state=0).to_string(index=False))
    df.to_parquet(path)
    return df


def step_catlinks(cats: pd.DataFrame) -> pd.DataFrame:
    path = OUT / "catlinks.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # One row per line: (cl_from,'sortkey','ts','prefix','type',collation,target_id),
    # The sortkey is binary and may hold commas, so slice from both ends instead of regexing the middle.
    # (macOS grep -F with ~10k patterns crawled at ~20 MB/min; plain slicing is far faster.)
    want = set(cats["lt_id"].tolist())
    proc = subprocess.Popen(["gzip", "-dc", str(DUMPS / "enwiki-latest-categorylinks.sql.gz")], stdout=subprocess.PIPE)
    out, n_rows = [], 0
    for raw in proc.stdout:
        if raw[:1] != b"(":
            continue
        n_rows += 1
        parts = raw.rstrip(b"\n;,")[:-1].rsplit(b",", 3)  # [..., b"'page'", b"1", b"22896130"]
        if int(parts[3]) in want and parts[1] == b"'page'":
            out.append((int(raw[1 : raw.index(b",")]), int(parts[3])))
        if n_rows % 50_000_000 == 0:
            print(f"  categorylinks: {n_rows:,} rows, {len(out):,} kept", flush=True)
    proc.wait()
    assert n_rows, "no categorylinks rows -- format changed?"
    df = pd.DataFrame(out, columns=["page_id", "lt_id"])
    assert len(df), "no categorylinks rows parsed -- format changed?"
    print(f"categorylinks: {len(df):,} (page, category) rows, {df.page_id.nunique():,} pages")
    df.to_parquet(path)
    return df


def step_pages(ids: set[int]) -> pd.DataFrame:
    path = OUT / "pages.parquet"
    if path.exists():
        return pd.read_parquet(path)
    row = re.compile(r"\((\d+),(-?\d+),'((?:[^'\\]|\\.)*)',([01]),")
    out, n_rows = [], 0
    proc = subprocess.Popen(["gzip", "-dc", str(DUMPS / "enwiki-latest-page.sql.gz")], stdout=subprocess.PIPE)
    for raw in proc.stdout:
        if raw[:1] != b"(":
            continue
        n_rows += 1
        i = raw.index(b",")
        # cheap prefilter (namespace 0, wanted id) before the regex
        if raw[i + 1 : i + 3] != b"0," or int(raw[1:i]) not in ids:
            continue
        m = row.match(raw.decode("utf-8", errors="replace"))
        if m and m[4] == "0":
            out.append((int(m[1]), ESC.sub(r"\1", m[3]).replace("_", " ")))
    proc.wait()
    assert n_rows, "no page rows -- format changed?"
    df = pd.DataFrame(out, columns=["page_id", "title"])
    print(f"page: {len(df):,} of {len(ids):,} category members are ns0 non-redirect articles")
    df.to_parquet(path)
    return df


PREC_ORDER = {"year": 0, "decade": 1, "century": 2, "millennium": 3}


def main() -> None:
    cats = step_cats()
    cl = step_catlinks(cats).merge(cats, on="lt_id")
    pages = step_pages(set(cl["page_id"]))
    cl = cl[cl.page_id.isin(pages.page_id)]

    def best(field: str) -> pd.DataFrame:
        d = cl[cl.field == field].copy()
        d["p"] = d["prec"].map(PREC_ORDER)
        d = d.sort_values(["page_id", "p", "lo"])
        n_conf = d[d.p == 0].groupby("page_id")["lo"].nunique()
        d = d.drop_duplicates("page_id")
        d = d.set_index("page_id")[["lo", "hi", "prec"]].add_prefix(f"{field}_")
        d[f"{field}_conflict"] = n_conf.reindex(d.index).fillna(0) > 1
        return d

    bio = pages.set_index("page_id")
    bio = bio.join(best("birth")).join(best("death"))
    for flag in ["living", "possibly_living", "birth_missing", "death_missing"]:
        bio[flag] = bio.index.isin(cl.loc[cl.field == flag, "page_id"])
    bio.reset_index().to_parquet(OUT / "bios.parquet")
    print(f"biographies: {len(bio):,}")


if __name__ == "__main__":
    sys.exit(main())

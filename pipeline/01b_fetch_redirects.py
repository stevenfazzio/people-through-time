"""Stage 01b: enwiki redirect map (source title -> target title).

Pageview dumps record views under the title AS REQUESTED that month, so
(1) renamed articles split their history across old and new titles ("Vijay
(actor)" -> "C. Joseph Vijay" zeroed out a top-500 celebrity in
living-people-map's naive join) and (2) alias redirects ("The Rock") never
reach the target. Stage 02 folds these sources into its lookup.

enwiki only -- other languages' redirects are not resolved (a documented
undercount, inherited from living-people-map).

Sources (streamed, deleted after unless KEEP_RAW_DUMPS):
  enwiki-latest-page.sql.gz     (~2.4 GB) -> redirect page_id -> ns0 source title
  enwiki-latest-redirect.sql.gz (~160 MB) -> rd_from page_id -> ns0 target title

Output: data/redirects.parquet [source_title, target_title] (underscored, ns0->ns0)
"""

import re
from pathlib import Path

import pandas as pd

import config
from util import download, require_rows, sql_row_lines, write_parquet_atomic

# page: (page_id,page_namespace,'page_title',page_is_redirect,...)
PAGE_ROW_RE = re.compile(rb"\((\d+),(-?\d+),'((?:[^'\\]|\\.)*)',([01]),")
# redirect: (rd_from,rd_namespace,'rd_title','rd_interwiki','rd_fragment')
REDIRECT_ROW_RE = re.compile(rb"\((\d+),(-?\d+),'((?:[^'\\]|\\.)*)'")
SQL_ESCAPE_RE = re.compile(r"\\(.)")


def unescape(raw: bytes) -> str:
    return SQL_ESCAPE_RE.sub(r"\1", raw.decode("utf-8", errors="replace"))


def parse_redirect_pages(path: Path) -> dict[int, str]:
    """ns0 redirect page_id -> its (source) title. Prefilters on the namespace before the regex."""
    titles: dict[int, str] = {}
    n_rows = 0
    for line in sql_row_lines(path):
        n_rows += 1
        c1 = line.index(b",")
        if line[c1 + 1 : c1 + 3] != b"0,":  # namespace 0 only
            continue
        m = PAGE_ROW_RE.match(line)
        if m and m.group(4) == b"1":
            titles[int(m.group(1))] = unescape(m.group(3))
    require_rows(path, n_rows)
    print(f"page.sql: {n_rows:,} rows, {len(titles):,} ns0 redirect pages", flush=True)
    return titles


def parse_redirect_targets(path: Path) -> dict[int, str]:
    """rd_from page_id -> ns0 target title."""
    targets: dict[int, str] = {}
    n_rows = 0
    for line in sql_row_lines(path):
        n_rows += 1
        m = REDIRECT_ROW_RE.match(line)
        if m and m.group(2) == b"0":
            targets[int(m.group(1))] = unescape(m.group(3))
    require_rows(path, n_rows)
    print(f"redirect.sql: {n_rows:,} rows, {len(targets):,} ns0-target redirects", flush=True)
    return targets


def main() -> None:
    paths = {
        name: download(config.ENWIKI_DUMP_URL.format(name=name), config.DUMPS_DIR / name)
        for name in ["enwiki-latest-page.sql.gz", "enwiki-latest-redirect.sql.gz"]
    }
    sources = parse_redirect_pages(paths["enwiki-latest-page.sql.gz"])
    targets = parse_redirect_targets(paths["enwiki-latest-redirect.sql.gz"])

    rows = [(src, targets[pid]) for pid, src in sources.items() if pid in targets and targets[pid] != src]
    df = pd.DataFrame(rows, columns=["source_title", "target_title"]).drop_duplicates(subset="source_title")
    print(f"Redirect map: {len(df):,} source->target pairs")
    assert len(df) > 5_000_000, "far fewer enwiki redirects than expected (~11M) -- format change?"
    write_parquet_atomic(df, config.REDIRECTS_PARQUET)
    print(f"Wrote {config.REDIRECTS_PARQUET}")

    if not config.KEEP_RAW_DUMPS:
        for p in paths.values():
            p.unlink()


if __name__ == "__main__":
    main()

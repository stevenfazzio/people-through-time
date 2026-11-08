"""Stage 01: per-language Wikipedia titles for every roster human.

Streams the wb_items_per_site SQL dump (~1.9 GB gz, every Wikidata sitelink)
and keeps rows whose item is in the roster and whose site is a Wikipedia
language edition ({lang}wiki, excluding Commons/Meta/etc.). This is the
(site, title) -> QID lookup stage 02 credits pageviews through.

Rows look like (ips_row_id,ips_item_id,'ips_site_id','ips_site_page'), one per
line. The item id is sliced out before any regex, so the ~90% of rows for
non-roster items cost one int() and a set lookup.

Titles are stored underscored (the pageview dumps' form); the dump stores
spaces.

Output: data/sitelinks.parquet [qid, site, title]  (~14.7M rows)
"""

import re
from pathlib import Path

import pandas as pd

import config
from util import download, require_rows, sql_row_lines, write_parquet_atomic

ROW_RE = re.compile(rb"\((\d+),(\d+),'((?:[^'\\]|\\.)*)','((?:[^'\\]|\\.)*)'\)")
SQL_ESCAPE_RE = re.compile(r"\\(.)")
SITE_RE = re.compile(r"^[a-z][a-z0-9_]*wiki$")
PROGRESS_EVERY = 20_000_000  # rows
NON_LANGUAGE_WIKIS = {
    "commonswiki", "specieswiki", "wikidatawiki", "metawiki", "mediawikiwiki", "sourceswiki",
    "foundationwiki", "outreachwiki", "incubatorwiki", "wikifunctionswiki", "wikimaniawiki",
    "loginwiki", "votewiki", "testwiki", "test2wiki", "testwikidatawiki", "abstractwiki",
}  # fmt: skip


def parse_sitelinks(dump_path: Path, roster_ids: set[int]) -> list[tuple[str, str, str]]:
    """(qid, site, underscored title) for each roster item's language-Wikipedia sitelinks."""
    rows = []
    n_rows = 0
    for line in sql_row_lines(dump_path):
        n_rows += 1
        if n_rows % PROGRESS_EVERY == 0:
            print(f"{n_rows:,} sitelinks scanned, {len(rows):,} kept", flush=True)
        c1 = line.index(b",")
        c2 = line.index(b",", c1 + 1)
        if int(line[c1 + 1 : c2]) not in roster_ids:
            continue
        m = ROW_RE.match(line)
        if m is None:
            raise ValueError(f"unparseable roster sitelink row: {line[:200]!r}")
        site = m.group(3).decode()
        if not SITE_RE.match(site) or site in NON_LANGUAGE_WIKIS:
            continue
        title = SQL_ESCAPE_RE.sub(r"\1", m.group(4).decode("utf-8", errors="replace")).replace(" ", "_")
        rows.append((f"Q{int(m.group(2))}", site, title))
    require_rows(dump_path, n_rows)
    print(f"{n_rows:,} sitelinks scanned in total", flush=True)
    return rows


def main() -> None:
    roster = pd.read_parquet(config.ROSTER_PARQUET, columns=["qid", "n_wikipedias"])
    roster_ids = {int(q[1:]) for q in roster["qid"]}
    print(f"Roster: {len(roster_ids):,} QIDs")

    dump_path = download(config.WB_ITEMS_PER_SITE_URL, config.DUMPS_DIR / Path(config.WB_ITEMS_PER_SITE_URL).name)
    df = pd.DataFrame(parse_sitelinks(dump_path, roster_ids), columns=["qid", "site", "title"])
    assert not df.duplicated(["qid", "site"]).any(), "an item has two sitelinks on one site"

    per_person = df.groupby("qid").size()
    print(f"Sitelinks: {len(df):,} rows, {df['site'].nunique()} sites, {per_person.size:,} people")
    print(f"People with an enwiki sitelink: {(df['site'] == 'enwiki').sum():,}")
    # The QLever count (stage 00) and this dump are different snapshots; they
    # should agree closely, not exactly.
    missing = len(roster_ids) - per_person.size
    print(f"Roster people with no Wikipedia sitelink in the dump: {missing:,} ({missing / len(roster_ids):.2%})")
    assert missing / len(roster_ids) < 0.02, "dump and roster disagree by >2% -- wrong site filter?"
    write_parquet_atomic(df, config.SITELINKS_PARQUET)
    print(f"Wrote {config.SITELINKS_PARQUET}")

    if not config.KEEP_RAW_DUMPS:
        dump_path.unlink()


if __name__ == "__main__":
    main()

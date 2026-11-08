"""Stage 02: monthly pageviews per (person, language, access method) for every roster human.

Per month: download the ~5.7 GB pageview_complete -user.bz2 (human-classified
traffic), filter to Wikipedia rows with bzip2|grep, credit each row through
the (site, title) -> QID lookup built from stage-01 sitelinks plus stage-01b
enwiki redirect sources, aggregate, write, delete the bz2. The per-month
parquet is the resume unit.

Differences from living-people-map's stage 01e:
  * the roster is every human with a Wikipedia article, not living people;
  * the access method (desktop / mobile-web / mobile-app) is KEPT, for the bot
    check (steady desktop-heavy traffic), instead of being summed away;
  * matches accumulate in compact int64 arrays and are aggregated with numpy
    at the end -- a dict of ~25M tuple keys per month would cost several GB
    per worker.

Dump row: "<wiki> <title> <page_id> <access> <daily_total> <hourly>". The
page_id column is unreliable (never join on it); titles containing double
quotes are CSV-quoted with backslash escapes.

Output: data/pageviews/pageviews_YYYYMM.parquet [qid_int, lang, access, views]
"""

import subprocess
import time
from array import array
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import config
from util import download, parse_dump_title, write_parquet_atomic

# pageview dump wiki codes use hyphens; Wikidata site ids use underscores,
# plus a few legacy ids that don't follow the rule.
SPECIAL_SITE_IDS = {"be-tarask": "be_x_oldwiki", "gsw": "alswiki", "lzh": "zh_classicalwiki", "nan": "zh_min_nanwiki"}
ACCESS_INDEX = {a.encode(): i for i, a in enumerate(config.ACCESS_METHODS)}
LANG_BITS, ACCESS_BITS = 12, 2  # key = ((qid << 12) | lang_idx) << 2 | access_idx; qid < 2**49 fits int64


def site_for_wiki_code(lang: str) -> str:
    return SPECIAL_SITE_IDS.get(lang, lang.replace("-", "_") + "wiki")


def build_lookup() -> dict[bytes, int]:
    """(site + ' ' + underscored_title) -> numeric QID: all roster sitelinks + enwiki redirect sources."""
    sitelinks = pd.read_parquet(config.SITELINKS_PARQUET)
    lookup: dict[bytes, int] = {}
    for qid, site, title in zip(sitelinks["qid"], sitelinks["site"], sitelinks["title"]):
        lookup[f"{site} {title}".encode()] = int(qid[1:])
    print(f"Lookup: {len(lookup):,} sitelinks", flush=True)

    redirects = pd.read_parquet(config.REDIRECTS_PARQUET)
    en_title_qid = {key[7:]: qid for key, qid in lookup.items() if key.startswith(b"enwiki ")}
    n_added = 0
    for source, target in zip(redirects["source_title"], redirects["target_title"]):
        qid = en_title_qid.get(target.encode())
        if qid is not None and lookup.setdefault(b"enwiki " + source.encode(), qid) == qid:
            n_added += 1
    print(f"Lookup: +{n_added:,} enwiki redirect titles -> {len(lookup):,} total", flush=True)
    return lookup


def aggregate_lines(lines, lookup: dict[bytes, int], langs: list[str], month: str = "") -> tuple[pd.DataFrame, int]:
    """Credit dump lines through the lookup. Returns ([qid_int, lang, access, views], n_rows_seen).

    `langs` is the shared lang-index table (appended to as new languages appear).
    """
    keys, views = array("q"), array("q")
    site_cache: dict[bytes, tuple[bytes, int]] = {}  # wiki code -> (site prefix bytes, lang index)
    n_rows = 0
    for line in lines:
        parts = line.rstrip(b"\n").split(b" ")
        if len(parts) < 5:
            continue
        n_rows += 1
        if month and n_rows % 25_000_000 == 0:
            print(f"{month}: {n_rows / 1e6:.0f}M wikipedia rows, {len(keys) / 1e6:.1f}M matched", flush=True)
        wiki = parts[0]
        cached = site_cache.get(wiki)
        if cached is None:
            lang = wiki[: -len(b".wikipedia")].decode()
            if lang not in langs:
                langs.append(lang)
            cached = (site_for_wiki_code(lang).encode() + b" ", langs.index(lang))
            site_cache[wiki] = cached
        site_prefix, lang_idx = cached
        title = parts[1]
        if title[:1] == b'"':
            title = parse_dump_title(title.decode("utf-8", errors="replace")).encode()
        qid = lookup.get(site_prefix + title)
        if qid is None:
            continue
        acc = ACCESS_INDEX.get(parts[3])
        if acc is None:
            raise ValueError(f"unknown access method {parts[3]!r} in {line[:200]!r}")
        try:
            v = int(parts[4])
        except ValueError:
            continue
        keys.append(((qid << LANG_BITS | lang_idx) << ACCESS_BITS) | acc)
        views.append(v)

    k = np.frombuffer(keys, dtype=np.int64)
    uniq, inv = np.unique(k, return_inverse=True)
    totals = np.bincount(inv, weights=np.frombuffer(views, dtype=np.int64)).astype(np.int64)
    lang_codes = np.array(langs, dtype=object)
    df = pd.DataFrame(
        {
            "qid_int": uniq >> (LANG_BITS + ACCESS_BITS),
            "lang": pd.Categorical(lang_codes[(uniq >> ACCESS_BITS) & ((1 << LANG_BITS) - 1)]),
            "access": pd.Categorical.from_codes(uniq & ((1 << ACCESS_BITS) - 1), categories=config.ACCESS_METHODS),
            "views": totals,
        }
    )
    return df, n_rows


def filter_month(bz2_path, month: str, lookup: dict[bytes, int]) -> pd.DataFrame:
    bunzip = subprocess.Popen(["bzip2", "-dc", str(bz2_path)], stdout=subprocess.PIPE)
    grep = subprocess.Popen(["grep", "-E", r"^[a-z-]+\.wikipedia "], stdin=bunzip.stdout, stdout=subprocess.PIPE)
    assert bunzip.stdout is not None and grep.stdout is not None
    bunzip.stdout.close()  # let bzip2 get SIGPIPE if grep dies
    df, n_rows = aggregate_lines(grep.stdout, lookup, langs=[], month=month)
    grep.wait()
    bunzip.wait()
    if bunzip.returncode != 0:
        raise RuntimeError(f"{month}: bzip2 exited {bunzip.returncode}")
    if grep.returncode != 0:  # 1 = zero matches, which is also fatal here
        raise RuntimeError(f"{month}: grep exited {grep.returncode}")
    print(f"{month}: {n_rows:,} wikipedia rows -> {len(df):,} (person, lang, access) rows", flush=True)
    return df


def make_process_month(lookup: dict[bytes, int]):
    def process_month(month: str) -> str:
        out_path = config.pageview_month_parquet(month)
        if out_path.exists():
            return f"{month}: parquet exists, skipping"
        year, mm = month.split("-")
        url = config.PAGEVIEW_URL_TEMPLATE.format(year=year, month=mm)
        bz2_path = download(url, config.DUMPS_DIR / f"pageviews-{year}{mm}-user.bz2")
        df = filter_month(bz2_path, month, lookup)
        if df.empty or df["qid_int"].nunique() < 1_000_000:
            raise RuntimeError(f"{month}: only {df['qid_int'].nunique():,} people matched -- lookup or format broken")
        write_parquet_atomic(df, out_path)
        if not config.KEEP_RAW_DUMPS:
            bz2_path.unlink()
        return (
            f"{month}: {len(df):,} rows, {df['qid_int'].nunique():,} people, {df['views'].sum():,} views "
            f"({df.groupby('access', observed=True)['views'].sum().div(df['views'].sum()).round(3).to_dict()})"
        )

    return process_month


def main() -> None:
    t0 = time.time()
    lookup = build_lookup()
    with ThreadPoolExecutor(max_workers=config.PAGEVIEW_WORKERS) as pool:
        for result in pool.map(make_process_month(lookup), config.PAGEVIEW_MONTHS):
            print(result, flush=True)
    print(f"All {len(config.PAGEVIEW_MONTHS)} months done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()

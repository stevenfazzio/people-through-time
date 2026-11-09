"""Stage 05: fame score, the map roster, and its slider bins.

Score (decided 2026-11-08): median_views x age, age = 2026 - birth year
(config.fame_score), for everyone at or above config.MIN_MEDIAN_VIEWS. The
roster is the top config.MAP_N by score. People with no birth year can't be
scored (stage 04: 38,873, of whom only 11-58 would clear the bar at a plausible
age). Ties break on median views, then QID, so the roster is deterministic.

Birth year = stage 04's birth_year_est, except for the hand-curated rows of
config.BIRTH_YEAR_OVERRIDES_CSV (birth_source "override"): historical people
whose Wikidata date comes from a traditional chronology (Alvar saints at
-4202), and plain Wikidata errors (Tommy Hilfiger "born 29"). An empty year
there means no usable birth year, so the person can't be scored. An
override whose wikidata_year no longer matches stage 04 still applies, with a
warning, because Wikidata may have been corrected since.

Slider: config.bin_edges() over the roster's birth years gives N_BINS
equal-count bins with whole-year edges, so a person's bin depends only on their
birth year. Window w = bins w .. w + WINDOW_BINS - 1, so a person in bin b
appears in windows max(0, b - WINDOW_BINS + 1) .. min(b, n_windows - 1).

Output: data/map_people.parquet, one row per roster person in rank order:
  rank, qid, label, description, gender, birth_year_est, birth_source,
  birth_year (Wikidata P569 after stage 04's century fix), birth_precision,
  death_year, median_views, score, bin, n_wikipedias, top_lang, top_lang_share,
  enwiki_title, top_title (the article on the most-viewed edition, the lead
  source when there is no enwiki article). Titles are underscored, as in
  sitelinks.
"""

import re

import numpy as np
import pandas as pd

import config
from util import log, site_for_wiki_code, write_parquet_atomic

OVERRIDE_COLUMNS = ["qid", "label", "wikidata_year", "birth_year", "source"]
ERAS = [(-np.inf, 0, "BCE"), (0, 1500, "1-1499"), (1500, 1800, "1500-1799"), (1800, 1900, "1800-1899"),
        (1900, 1950, "1900-1949"), (1950, np.inf, "1950+")]  # fmt: skip
OUT_COLUMNS = [
    "rank", "qid", "label", "description", "gender", "birth_year_est", "birth_source", "birth_year",
    "birth_precision", "death_year", "median_views", "score", "bin", "n_wikipedias", "top_lang",
    "top_lang_share", "enwiki_title", "top_title",
]  # fmt: skip


def load_overrides(path) -> pd.DataFrame:
    """The curated CSV, validated: QID format, one row per QID, a source, and a year no later than the reference
    year -- or none, meaning no usable birth year (the person can't be scored)."""
    ov = pd.read_csv(path, comment="#", dtype={"qid": str, "label": str, "source": str})
    if list(ov.columns) != OVERRIDE_COLUMNS:
        raise ValueError(f"{path}: columns {list(ov.columns)}, expected {OVERRIDE_COLUMNS}")
    bad = ov.loc[~ov["qid"].fillna("").map(lambda q: bool(re.fullmatch(r"Q\d+", q))), "qid"].tolist()
    if bad:
        raise ValueError(f"{path}: malformed QIDs {bad}")
    dup = ov.loc[ov["qid"].duplicated(), "qid"].tolist()
    if dup:
        raise ValueError(f"{path}: QIDs listed twice {dup}")
    if (ov["birth_year"].dropna() > config.FAME_REFERENCE_YEAR).any():
        raise ValueError(f"{path}: birth_year later than {config.FAME_REFERENCE_YEAR}")
    if ov["source"].fillna("").str.strip().eq("").any():
        raise ValueError(f"{path}: every row needs a source")
    ov["birth_year"] = ov["birth_year"].astype("Int32")
    ov["wikidata_year"] = ov["wikidata_year"].astype("Int32")
    return ov


def apply_overrides(people: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """birth_year_est / birth_source from the overrides; Wikidata's birth_year is left as it was."""
    out = people.copy()
    pos = pd.Series(np.arange(len(out)), index=out["qid"])
    present = overrides["qid"].isin(pos.index)
    for _, row in overrides[~present].iterrows():
        log(f"override {row['qid']} ({row['label']}) not above the views floor; skipped")
    ov = overrides[present]
    rows = pos[ov["qid"]].to_numpy()
    current = out["birth_year"].iloc[rows].astype("Int32").to_numpy()
    for (_, row), now in zip(ov.iterrows(), current):
        was = row["wikidata_year"]
        if pd.isna(was) != pd.isna(now) or (not pd.isna(now) and was != now):
            log(f"WARNING: override {row['qid']} ({row['label']}) was written against Wikidata year {was}, "
                f"stage 04 now has {now}; applying it anyway -- check whether it is stale")  # fmt: skip
    col_src = out.columns.get_loc("birth_source")
    est = out["birth_year_est"].copy()
    est.iloc[rows] = ov["birth_year"].array  # NA: no usable birth year
    out["birth_year_est"] = est.astype("Int32")
    out.iloc[rows, col_src] = "override"
    return out


def select_roster(people: pd.DataFrame, n: int) -> pd.DataFrame:
    """Top n by fame score among dated people at or above the floor, with 1-based rank."""
    ok = (people["median_views"] >= config.MIN_MEDIAN_VIEWS) & people["birth_year_est"].notna()
    df = people[ok].copy()
    df["score"] = config.fame_score(df["median_views"].to_numpy(float), df["birth_year_est"].to_numpy(float))
    df = df.sort_values(["score", "median_views", "qid"], ascending=[False, False, True], kind="mergesort")
    df = df.head(n).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1, dtype=np.int32))
    return df


def birth_bins(years, edges: np.ndarray) -> np.ndarray:
    """Bin index per birth year: bin i = [edges[i], edges[i + 1])."""
    return (np.searchsorted(edges, np.asarray(years, dtype=float), side="right") - 1).astype(np.int8)


def window_sizes(bins: np.ndarray, n_bins: int) -> np.ndarray:
    counts = np.bincount(bins, minlength=n_bins)
    return np.array([counts[w : w + config.WINDOW_BINS].sum() for w in range(n_bins - config.WINDOW_BINS + 1)])


def article_titles(roster: pd.DataFrame, sitelinks: pd.DataFrame) -> pd.DataFrame:
    """[enwiki_title, top_title] aligned with roster rows; top_title is on the top_lang edition."""
    sl = sitelinks.set_index(["qid", "site"])["title"]
    en_keys = pd.MultiIndex.from_arrays([roster["qid"], np.full(len(roster), "enwiki")])
    top_sites = roster["top_lang"].map(lambda lang: site_for_wiki_code(lang) if isinstance(lang, str) else "")
    top_keys = pd.MultiIndex.from_arrays([roster["qid"], top_sites])
    return pd.DataFrame(
        {"enwiki_title": sl.reindex(en_keys).to_numpy(), "top_title": sl.reindex(top_keys).to_numpy()},
        index=roster.index,
    )


def era(years: pd.Series) -> pd.Series:
    edges = [e[0] for e in ERAS] + [np.inf]
    return pd.cut(years.astype(float), edges, right=False, labels=[e[2] for e in ERAS])


def log_override_effects(before: pd.DataFrame, people: pd.DataFrame, roster: pd.DataFrame, ov: pd.DataFrame) -> None:
    bar = roster["score"].min()
    p = people.set_index("qid")
    for _, row in ov[ov["qid"].isin(p.index)].iterrows():
        views = p.at[row["qid"], "median_views"]
        old = config.fame_score(views, float(row["wikidata_year"])) if pd.notna(row["wikidata_year"]) else np.nan
        new = config.fame_score(views, float(row["birth_year"])) if pd.notna(row["birth_year"]) else np.nan
        year = row["birth_year"] if pd.notna(row["birth_year"]) else "unknown"
        log(f"  {row['label']}: {row['wikidata_year']} -> {year}, {views:,.0f} views/mo, "
            f"score {old:,.0f} -> {new:,.0f} (bar {bar:,.0f}); in roster "
            f"{row['qid'] in set(before['qid'])} -> {row['qid'] in set(roster['qid'])}")  # fmt: skip


def main() -> None:
    views = pd.read_parquet(
        config.PEOPLE_VIEWS_PARQUET, columns=["qid", "n_wikipedias", "median_views", "top_lang", "top_lang_share"]
    )
    facts = pd.read_parquet(config.PEOPLE_FACTS_PARQUET)
    people = facts.merge(views, on="qid", how="left", validate="1:1")
    if people["median_views"].isna().any():
        raise SystemExit("people_facts has QIDs missing from people_views -- rerun stage 04")

    overrides = load_overrides(config.BIRTH_YEAR_OVERRIDES_CSV)
    before = select_roster(people, config.MAP_N)
    people = apply_overrides(people, overrides)
    roster = select_roster(people, config.MAP_N)
    log(f"{len(overrides)} birth-year overrides:")
    log_override_effects(before, people, roster, overrides)
    log(f"Roster changes from the overrides: {len(set(before['qid']) - set(roster['qid']))} out, "
        f"{len(set(roster['qid']) - set(before['qid']))} in")  # fmt: skip

    dated = people["birth_year_est"].notna().sum()
    bar = roster["score"].min()
    log(f"{len(people):,} people above the floor, {dated:,} with a birth year; roster {len(roster):,}; "
        f"lowest score admitted {bar:,.0f} views/mo x years")  # fmt: skip

    edges = config.bin_edges(roster["birth_year_est"])
    if len(edges) - 1 != config.N_BINS:
        raise SystemExit(f"bin_edges gave {len(edges) - 1} bins, not {config.N_BINS}: whole-year snapping merged some")
    roster["bin"] = birth_bins(roster["birth_year_est"], edges)

    sitelinks = pd.read_parquet(config.SITELINKS_PARQUET, filters=[("qid", "in", roster["qid"].tolist())])
    roster = roster.join(article_titles(roster, sitelinks))
    if roster["top_title"].isna().any():
        log(f"WARNING: {roster['top_title'].isna().sum()} roster people have no sitelink on their top-language edition")

    roster["era"] = era(roster["birth_year_est"])
    by_era = roster.groupby("era", observed=False).agg(
        n=("qid", "size"), min_views=("median_views", "min"), no_enwiki=("enwiki_title", lambda s: s.isna().sum())
    )
    by_era["share"] = (by_era["n"] / len(roster)).map("{:.1%}".format)
    log(f"By era:\n{by_era.to_string()}")
    log(f"birth_source:\n{roster['birth_source'].value_counts().to_string()}")
    bins = roster.groupby("bin").agg(n=("qid", "size"), lo=("birth_year_est", "min"), hi=("birth_year_est", "max"))
    log(f"Bins:\n{bins.T.to_string()}")
    sizes = window_sizes(roster["bin"].to_numpy(), config.N_BINS)
    log(f"{len(sizes)} windows of {sizes.min():,}-{sizes.max():,} people, {sizes.sum():,} AlignedUMAP points")
    log(f"No enwiki article (lead to translate): {roster['enwiki_title'].isna().sum():,}; top languages "
        f"{roster.loc[roster['enwiki_title'].isna(), 'top_lang'].value_counts().head(6).to_dict()}")  # fmt: skip
    top = roster[["rank", "label", "birth_year_est", "median_views", "score"]].head(15)
    log(f"Top 15:\n{top.to_string(index=False)}")

    out = roster[OUT_COLUMNS]
    write_parquet_atomic(out, config.MAP_PEOPLE_PARQUET)
    log(f"Wrote {config.MAP_PEOPLE_PARQUET}")


if __name__ == "__main__":
    main()

"""Bot-check features on the real all-language views (one-off, 2026-11-08).

Question: which pages draw suspicious desktop-heavy "user" traffic, and how much
of the views x age roster would a desktop-share filter touch, era by era?

Desktop share differs by language (a German reader is likelier on a desktop
than an Indonesian one), so raw desktop share mostly measures language mix.
The feature here is EXCESS desktop share: actual minus what the person's
language mix predicts from each language's roster-wide baseline.

Birth years are a proxy until the birth-date stage exists: scoping's enwiki
birth categories (data/scoping/out/bios.parquet, midpoint of lo..hi), joined via
the enwiki sitelink. People without an enwiki article or a birth category are
missing from the era tables.

Reads data/people_views.parquet, data/sitelinks.parquet, data/pageviews/*.
Output: scoping/results/bot_features.txt (stdout).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

BIOS = config.DATA_DIR / "scoping" / "out" / "bios.parquet"
ERAS = [(-np.inf, 0, "BCE"), (0, 1500, "1-1499"), (1500, 1800, "1500-1799"), (1800, 1900, "1800-1899"),
        (1900, 1950, "1900-1949"), (1950, np.inf, "1950+")]  # fmt: skip
CONTROLS = ["Ansel_Adams", "Cleopatra", "Elizabeth_II", "Jesus", "Plato", "Genghis_Khan", "Taylor_Swift"]
pd.set_option("display.width", 250)


def site_for(lang: str) -> str:
    special = {"be-tarask": "be_x_oldwiki", "gsw": "alswiki", "lzh": "zh_classicalwiki", "nan": "zh_min_nanwiki"}
    return special.get(lang, lang.replace("-", "_") + "wiki")


def lang_access_cells(keep: np.ndarray) -> tuple[pd.Series, pd.DataFrame]:
    """Roster-wide desktop share per language, and 12-month (qid_int, lang, access) views for `keep`."""
    base, cells = [], []
    for m in config.PAGEVIEW_MONTHS:
        df = pd.read_parquet(config.pageview_month_parquet(m))
        base.append(df.groupby(["lang", "access"], observed=True)["views"].sum())
        df = df[np.isin(df["qid_int"].to_numpy(), keep)]
        cells.append(df.groupby(["qid_int", "lang", "access"], observed=True)["views"].sum())
        print(f"  read {m}", flush=True)
    base = pd.concat(base).groupby(level=[0, 1]).sum().unstack(fill_value=0)
    lang_desktop = base["desktop"] / base.sum(axis=1)
    cells = pd.concat(cells).groupby(level=[0, 1, 2]).sum().rename("views").reset_index()
    return lang_desktop.rename("lang_desktop_share"), cells


def excess_desktop(cells: pd.DataFrame, lang_desktop: pd.Series) -> pd.DataFrame:
    per_lang = cells.groupby(["qid_int", "lang"], observed=True)["views"].sum().rename("v").reset_index()
    per_lang["expected"] = per_lang["v"] * per_lang["lang"].astype(str).map(lang_desktop).to_numpy()
    expected = per_lang.groupby("qid_int")["expected"].sum() / per_lang.groupby("qid_int")["v"].sum()
    top = cells.sort_values("views", ascending=False).drop_duplicates("qid_int").set_index("qid_int")
    total = cells.groupby("qid_int")["views"].sum()
    out = pd.DataFrame({"expected_desktop": expected})
    out["top_cell"] = top["lang"].astype(str) + "/" + top["access"].astype(str)
    out["top_cell_share"] = top["views"] / total
    return out


def main() -> None:
    pv = pd.read_parquet(config.PEOPLE_VIEWS_PARQUET)
    pv["qid_int"] = pv["qid"].str.slice(1).astype(np.int64)
    months = [c for c in pv.columns if c.startswith("views_")]
    above = pv[pv["median_views"] >= config.MIN_MEDIAN_VIEWS].copy()
    print(f"{len(above):,} people at or above {config.MIN_MEDIAN_VIEWS} median views/mo\n")

    lang_desktop, cells = lang_access_cells(above["qid_int"].to_numpy())
    above = above.merge(excess_desktop(cells, lang_desktop), left_on="qid_int", right_index=True, how="left")
    above["excess_desktop"] = above["desktop_share"] - above["expected_desktop"]
    m = above[months].to_numpy(dtype=float)
    above["cv"] = m.std(axis=1) / np.maximum(m.mean(axis=1), 1)  # month-to-month variation; bots can be flat

    sl = pd.read_parquet(config.SITELINKS_PARQUET)
    en = sl[sl["site"] == "enwiki"].set_index("qid")["title"]
    above["title"] = above["qid"].map(en)
    sl_any = sl.set_index(["qid", "site"])["title"]
    no_en = above["title"].isna()
    keys = list(zip(above.loc[no_en, "qid"], above.loc[no_en, "top_lang"].map(site_for)))
    above.loc[no_en, "title"] = [f"{sl_any.get(k, '?')} [{k[1]}]" for k in keys]

    bios = pd.read_parquet(BIOS, columns=["title", "birth_lo", "birth_hi"])
    bios["title"] = bios["title"].str.replace(" ", "_")
    bios = bios.dropna(subset=["birth_lo"]).drop_duplicates("title").set_index("title")
    above["birth"] = above["title"].map((bios["birth_lo"] + bios["birth_hi"]) / 2)

    print("Roster-wide desktop share by language (top 15 languages by views):")
    big = cells.groupby("lang", observed=True)["views"].sum().sort_values(ascending=False).head(15).index
    print(lang_desktop.reindex(big.astype(str)).round(3).to_string(), "\n")

    w = above["total_views"]
    for col in ["desktop_share", "excess_desktop", "app_share", "cv"]:
        q = above[col].quantile([0.01, 0.1, 0.5, 0.9, 0.99, 0.999]).round(3).tolist()
        wmean = np.average(above[col], weights=w)
        print(f"{col:>15} quantiles 1/10/50/90/99/99.9%: {q}; views-weighted mean {wmean:.3f}")
    for t in [0.2, 0.3, 0.4, 0.5]:
        print(f"  excess desktop > {t}: {(above['excess_desktop'] > t).sum():,} people")
    print()

    cols = ["title", "birth", "median_views", "desktop_share", "expected_desktop", "excess_desktop", "top_cell",
            "top_cell_share", "cv", "n_langs"]  # fmt: skip
    print("Controls:")
    print(above[above["title"].isin(CONTROLS)][cols].round(3).to_string(index=False), "\n")

    print("Top 40 by median views with excess desktop share > 0.3:")
    print(above[above["excess_desktop"] > 0.3].nlargest(40, "median_views")[cols].round(3).to_string(index=False))
    print()

    dated = above.dropna(subset=["birth"]).copy()
    dated["score"] = config.fame_score(dated["median_views"], dated["birth"])
    roster = dated.nlargest(config.MAP_N, "score")
    print(f"Proxy roster: top {config.MAP_N:,} by views x age among {len(dated):,} with an enwiki birth year")
    print(f"{'era':>10} {'n':>6} {'>0.2':>6} {'>0.3':>6} {'>0.4':>6}  median excess")
    for lo, hi, name in ERAS:
        r = roster[(roster["birth"] >= lo) & (roster["birth"] < hi)]
        x = r["excess_desktop"]
        counts = " ".join(f"{(x > t).sum():>6}" for t in (0.2, 0.3, 0.4))
        print(f"{name:>10} {len(r):>6} {counts}  {x.median():.3f}")
    print()
    for lo, hi, name in ERAS:
        r = roster[(roster["birth"] >= lo) & (roster["birth"] < hi) & (roster["excess_desktop"] > 0.2)]
        print(f"{name}: top 12 by score with excess desktop > 0.2")
        print(r.nlargest(12, "score")[cols].round(3).to_string(index=False), "\n")


if __name__ == "__main__":
    main()

"""Per-edition desktop background vs the early-era fame bar (one-off, 2026-11-08).

desktop_floor.py found every enwiki page -- living people's too -- carries
~10-15 desktop views/mo even when almost nobody reads it on mobile, while the
early-era desktop excess above that is proportional and dips in summer (real
readers). The background is per PAGE, so all-language views multiply it by the
number of editions: 50 stub editions x ~10/mo is ~500/mo, the size of the
early-era fame bar.

1. Estimate the background per language: median desktop views/mo among a
   person's editions with almost no mobile readers (< 2/mo).
2. Subtract F desktop views/mo from every (person, edition), clipped at 0,
   and see how the views x age proxy roster changes, era by era.

Birth-year proxy: enwiki birth category midpoint (scoping bios), else Wikidata
death year - 50 (scoping deaths.csv), so non-enwiki early people are covered.
Reads data/scoping/cells_above_floor.parquet (from bot_access_ratio.py).
Output: scoping/results/edition_floor.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

CELLS = config.DATA_DIR / "scoping" / "cells_above_floor.parquet"
BIOS = config.DATA_DIR / "scoping" / "out" / "bios.parquet"
DEATHS = config.DATA_DIR / "scoping" / "qlever" / "deaths.csv"
FLOORS = [5, 10, 15, 20]
ERAS = [(-np.inf, 0, "BCE"), (0, 1500, "1-1499"), (1500, 1800, "1500-1799"), (1800, 1900, "1800-1899"),
        (1900, 1950, "1900-1949"), (1950, np.inf, "1950+")]  # fmt: skip
pd.set_option("display.width", 250)


def people_table(views_col: str = "median_views") -> pd.DataFrame:
    """Floor survivors by `views_col` (returned as median_views) with proxy birth years and enwiki titles."""
    pv = pd.read_parquet(config.PEOPLE_VIEWS_PARQUET, columns=["qid", views_col, "n_wikipedias", "n_langs"])
    pv = pv.rename(columns={views_col: "median_views"})
    pv = pv[pv["median_views"] >= config.MIN_MEDIAN_VIEWS].copy()
    pv["qid_int"] = pv["qid"].str.slice(1).astype(np.int64)
    pv = pv.set_index("qid_int")

    sl = pd.read_parquet(config.SITELINKS_PARQUET, filters=[("site", "==", "enwiki")])
    en = sl.assign(qid_int=sl["qid"].str.slice(1).astype(np.int64)).set_index("qid_int")["title"]
    pv["title"] = en.reindex(pv.index)
    bios = pd.read_parquet(BIOS, columns=["title", "birth_lo", "birth_hi"]).dropna(subset=["birth_lo"])
    bios = bios.assign(title=bios["title"].str.replace(" ", "_")).drop_duplicates("title").set_index("title")
    birth = pv["title"].map((bios["birth_lo"] + bios["birth_hi"]) / 2)
    d = pd.read_csv(DEATHS)
    d["qid_int"] = d["h"].str.rsplit("Q", n=1).str[1].astype(np.int64)
    death = d.groupby("qid_int")["dy"].min()
    pv["birth"] = birth.fillna(death.reindex(pv.index) - 50)
    pv["birth_src"] = np.where(birth.notna(), "enwiki", np.where(pv["birth"].notna(), "death-50", "none"))
    pv["title"] = pv["title"].fillna(pv["qid"] + " (no enwiki)")
    return pv


def main() -> None:
    cells = pd.read_parquet(CELLS)
    cells["side"] = np.where(cells["access"] == "desktop", "D", "M")
    pl = cells.pivot_table(index=["qid_int", "lang"], columns="side", values="views", aggfunc="sum", fill_value=0)
    pl = (pl / 12).reset_index()  # views per month

    quiet = pl[pl["M"] < 2]
    print(f"Desktop background: (person, edition) cells with < 2 mobile views/mo: {len(quiet):,}")
    print(f"  all editions: median desktop {quiet['D'].median():.1f}/mo, p25 {quiet['D'].quantile(0.25):.1f}, "
          f"p75 {quiet['D'].quantile(0.75):.1f}")  # fmt: skip
    top = quiet.groupby("lang")["D"].agg(["size", "median"]).sort_values("size", ascending=False).head(12)
    print("  by edition (most cells):", ", ".join(f"{k} {v:.1f}" for k, v in top["median"].items()), "\n")

    people = people_table()
    print("Birth proxy source:", people["birth_src"].value_counts().to_dict(), "\n")
    total = pl.groupby("qid_int")[["D", "M"]].sum().sum(axis=1)
    small = pl[(pl["D"] + pl["M"]) < 30].groupby("qid_int")[["D", "M"]].sum().sum(axis=1)
    people["tail_share"] = (small.reindex(people.index).fillna(0) / total.reindex(people.index)).fillna(0)
    people["n_editions"] = pl.groupby("qid_int").size().reindex(people.index)

    dated = people.dropna(subset=["birth"]).copy()
    dated["score"] = config.fame_score(dated["median_views"], dated["birth"])
    before = dated.nlargest(config.MAP_N, "score")
    print(f"Proxy roster: top {config.MAP_N:,} of {len(dated):,} dated floor survivors")
    print(f"{'era':>10} {'n':>6} {'median views':>12} {'editions':>8} {'tail share':>10}")
    for lo, hi, name in ERAS:
        e = before[(before["birth"] >= lo) & (before["birth"] < hi)]
        print(f"{name:>10} {len(e):>6} {e['median_views'].median():>12,.0f} {e['n_editions'].median():>8.0f} "
              f"{e['tail_share'].median():>10.2f}")  # fmt: skip
    print()

    for f in FLOORS:
        kept = (np.maximum(pl["D"] - f, 0) + pl["M"]).groupby(pl["qid_int"]).sum()
        keep = (kept / total).reindex(dated.index).fillna(1)
        dated["adj_median"] = dated["median_views"] * keep
        dated["adj_score"] = config.fame_score(dated["adj_median"], dated["birth"])
        after = dated[dated["adj_median"] >= config.MIN_MEDIAN_VIEWS].nlargest(config.MAP_N, "adj_score")
        out, inn = before.index.difference(after.index), after.index.difference(before.index)
        print(f"=== subtract {f} desktop views/mo per edition: proxy roster churn {len(out):,} of {config.MAP_N:,}; "
              f"median keep in roster {keep.reindex(before.index).median():.2f}")  # fmt: skip
        for lo, hi, name in ERAS:
            b = before[(before["birth"] >= lo) & (before["birth"] < hi)]
            a = after[(after["birth"] >= lo) & (after["birth"] < hi)]
            k = keep.reindex(b.index)
            print(f"   {name:>10}: {len(b):>6} -> {len(a):>6}; keep median {k.median():.2f}, p10 {k.quantile(0.1):.2f}")
        cols = ["title", "birth", "birth_src", "median_views", "n_editions", "tail_share"]
        ex = dated.loc[out].assign(keep=keep.reindex(out)).nlargest(15, "score")
        print("   biggest exits:")
        print(ex[cols + ["keep"]].round(2).to_string(index=False))
        en_ = dated.loc[inn].assign(keep=keep.reindex(inn)).nlargest(8, "adj_score")
        print("   biggest entries:")
        print(en_[cols + ["keep"]].round(2).to_string(index=False), "\n")


if __name__ == "__main__":
    main()

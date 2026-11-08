"""Bot check, take 2: per-language desktop:mobile ratio vs that language's baseline (one-off, 2026-11-08).

bot_features.py showed raw desktop share is mostly audience, not bots: academic
subjects (Aristotle, Descartes, Knuth) run 45-70% desktop on every language,
while the suspected crawler page, Ansel Adams, is 92% en/mobile-web. A crawler
counted as "user" inflates one (language, access) cell; real readers move every
cell of a language together. So per (person, language), over 12 months:

    r = log((D + 1) / (M + 1)) - log(baseline D/M of that language)

D = desktop, M = mobile-web + mobile-app. Real audience skew shifts r by up to
~1-2 (academic +, celebrity -). Clipping r to a band [-lo, +hi] by shrinking
the over-represented side gives "capped" views; the cut is the share of a
person's views the band removes. This sweeps bands and shows who gets cut.

Birth-year proxy and roster preview as in bot_features.py.
Reads data/people_views.parquet, data/sitelinks.parquet, data/pageviews/*.
Caches the 12-month (person, lang, access) cells for floor survivors in
data/scoping/cells_above_floor.parquet. Output: scoping/results/bot_access_ratio.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

BIOS = config.DATA_DIR / "scoping" / "out" / "bios.parquet"
CELLS = config.DATA_DIR / "scoping" / "cells_above_floor.parquet"
BASE = config.DATA_DIR / "scoping" / "lang_access_baseline.parquet"
ERAS = [(-np.inf, 0, "BCE"), (0, 1500, "1-1499"), (1500, 1800, "1500-1799"), (1800, 1900, "1800-1899"),
        (1900, 1950, "1900-1949"), (1950, np.inf, "1950+")]  # fmt: skip
BANDS = [(np.log(4), np.log(4)), (np.log(8), np.log(4)), (np.log(8), np.log(8)), (np.log(16), np.log(8))]
CONTROLS = ["Ansel_Adams", "Cleopatra", "Michelangelo", "Bruce_Lee", "Aristotle", "Guido_van_Rossum",
            "Silvina_Ocampo", "Craig_Small", "Elizabeth_II", "Taylor_Swift", "Cristiano_Ronaldo"]  # fmt: skip
pd.set_option("display.width", 250)


def load_cells(keep: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    if CELLS.exists() and BASE.exists():
        return pd.read_parquet(CELLS), pd.read_parquet(BASE)
    base, cells = [], []
    for m in config.PAGEVIEW_MONTHS:
        df = pd.read_parquet(config.pageview_month_parquet(m))
        df["lang"] = df["lang"].astype(str)
        df["access"] = df["access"].astype(str)
        base.append(df.groupby(["lang", "access"])["views"].sum())
        df = df[np.isin(df["qid_int"].to_numpy(), keep)]
        cells.append(df.groupby(["qid_int", "lang", "access"])["views"].sum())
        print(f"  read {m}", flush=True)
    base = pd.concat(base).groupby(level=[0, 1]).sum().rename("views").reset_index()
    cells = pd.concat(cells).groupby(level=[0, 1, 2]).sum().rename("views").reset_index()
    CELLS.parent.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(CELLS)
    base.to_parquet(BASE)
    return cells, base


def per_lang(cells: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """(qid_int, lang) -> D, M, r."""
    cells = cells.assign(side=np.where(cells["access"] == "desktop", "D", "M"))
    pl = cells.pivot_table(index=["qid_int", "lang"], columns="side", values="views", aggfunc="sum", fill_value=0)
    b = base.assign(side=np.where(base["access"] == "desktop", "D", "M")).groupby(["lang", "side"])["views"].sum()
    b = b.unstack()
    log_base = np.log(b["D"] / b["M"]).rename("log_base")
    pl = pl.reset_index().merge(log_base, left_on="lang", right_index=True)
    pl["r"] = np.log((pl["D"] + 1) / (pl["M"] + 1)) - pl["log_base"]
    return pl


def capped_total(pl: pd.DataFrame, hi: float, lo: float) -> pd.Series:
    """Per person: views kept after clipping each language's r to [-lo, +hi]."""
    ratio_hi = np.exp(pl["log_base"] + hi)  # max allowed D/M
    ratio_lo = np.exp(pl["log_base"] - lo)  # min allowed D/M
    d = np.minimum(pl["D"], pl["M"] * ratio_hi)
    m = np.minimum(pl["M"], pl["D"] / ratio_lo)
    return (d + m).groupby(pl["qid_int"]).sum()


def names_and_births(people: pd.DataFrame) -> pd.DataFrame:
    sl = pd.read_parquet(config.SITELINKS_PARQUET)
    en = sl[sl["site"] == "enwiki"].set_index("qid")["title"]
    people["title"] = people["qid"].map(en)
    no_en = people["title"].isna()
    first = sl[sl["qid"].isin(people.loc[no_en, "qid"])].drop_duplicates("qid").set_index("qid")
    people.loc[no_en, "title"] = people.loc[no_en, "qid"].map(first["title"] + " [" + first["site"] + "]")
    bios = pd.read_parquet(BIOS, columns=["title", "birth_lo", "birth_hi"]).dropna(subset=["birth_lo"])
    bios["title"] = bios["title"].str.replace(" ", "_")
    bios = bios.drop_duplicates("title").set_index("title")
    people["birth"] = people["title"].map((bios["birth_lo"] + bios["birth_hi"]) / 2)
    return people


def main() -> None:
    pv = pd.read_parquet(config.PEOPLE_VIEWS_PARQUET, columns=["qid", "median_views", "total_views", "n_langs"])
    people = pv[pv["median_views"] >= config.MIN_MEDIAN_VIEWS].copy()
    people["qid_int"] = people["qid"].str.slice(1).astype(np.int64)
    cells, base = load_cells(people["qid_int"].to_numpy())
    pl = per_lang(cells, base)
    people = names_and_births(people.set_index("qid_int")).copy()

    big = pl[pl["D"] + pl["M"] >= 12_000]  # >= ~1,000 views/mo in that language
    w = big["D"] + big["M"]
    qs = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    print(f"r over {len(big):,} (person, language) cells with >= 12k views/yr")
    print("  quantiles  ", " ".join(f"{q:>6}" for q in qs))
    print("  by cell    ", " ".join(f"{v:>6.2f}" for v in big["r"].quantile(qs)))
    order = np.argsort(big["r"].to_numpy())
    cw = np.cumsum(w.to_numpy()[order]) / w.sum()
    by_views = [big["r"].to_numpy()[order][np.searchsorted(cw, q)] for q in qs]
    print("  by views   ", " ".join(f"{v:>6.2f}" for v in by_views), "\n")

    show = ["title", "lang", "D", "M", "r"]
    big = big.merge(people[["title"]], left_on="qid_int", right_index=True)
    print("Most mobile-skewed cells with >= 1M views/yr:")
    print(big[big["D"] + big["M"] >= 1_000_000].nsmallest(20, "r")[show].round(2).to_string(index=False), "\n")
    print("Most desktop-skewed cells with >= 300k views/yr:")
    print(big[big["D"] + big["M"] >= 300_000].nlargest(20, "r")[show].round(2).to_string(index=False), "\n")

    actual = pl.groupby("qid_int")[["D", "M"]].sum().sum(axis=1)
    for hi, lo in BANDS:
        tag = f"band [-log {np.exp(lo):.0f}, +log {np.exp(hi):.0f}]"
        keep = (capped_total(pl, hi, lo) / actual).reindex(people.index).fillna(1)
        people["keep"] = keep
        people["capped_median"] = people["median_views"] * keep
        cut = 1 - keep
        dated = people.dropna(subset=["birth"]).copy()
        dated["score"] = config.fame_score(dated["median_views"], dated["birth"])
        dated["score_capped"] = config.fame_score(dated["capped_median"], dated["birth"])
        before = set(dated.nlargest(config.MAP_N, "score").index)
        floor_ok = dated[dated["capped_median"] >= config.MIN_MEDIAN_VIEWS]
        after = set(floor_ok.nlargest(config.MAP_N, "score_capped").index)
        print(f"=== {tag}: {(cut > 0.2).sum():,} people lose >20%, {(cut > 0.5).sum():,} lose >50%; "
              f"views removed {1 - (people['capped_median'].sum() / people['median_views'].sum()):.2%}; "
              f"proxy roster churn {len(before - after):,} of {config.MAP_N:,}")  # fmt: skip
        r = dated.loc[list(before)]
        line = []
        for lo_y, hi_y, name in ERAS:
            e = r[(r["birth"] >= lo_y) & (r["birth"] < hi_y)]
            line.append(f"{name} {len(e)}: out {len(set(e.index) - after)}")
        print("   ", "; ".join(line))
        ctl = people[people["title"].isin(CONTROLS)][["title", "median_views", "keep"]]
        print("    controls:", ", ".join(f"{t} {k:.2f}" for t, k in zip(ctl["title"], ctl["keep"])))
        lost = dated.loc[list(before - after)].nlargest(25, "score")
        print("    biggest roster exits (by original score):")
        print(lost[["title", "birth", "median_views", "keep", "n_langs"]].round(2).to_string(index=False), "\n")


if __name__ == "__main__":
    main()

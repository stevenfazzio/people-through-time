"""Stage 03: per-person view metrics from the 12 monthly pageview files, with the access-ratio bot filter.

Per person (a month with no rows = 0 views, as in living-people-map):
  views_YYYYMM     monthly total over all languages and access methods, after the bot filter
  median_views     median of the 12 monthly totals -- THE fame metric
  median_views_raw the same before the bot filter
  clip_keep        share of the person's 12-month views the bot filter kept
  total_views, max_views, n_months (months with >=1 view), median_present   (after the filter)
  desktop_share, app_share   share of RAW views by access method
  top_lang, top_lang_share, en_share, n_langs   (after the filter; floor survivors only)

Bot filter (decided 2026-11-08; config.MAX_DESKTOP_SKEW / MAX_MOBILE_SKEW): each
(person, language)'s desktop:mobile log-ratio in its median month, taken
relative to that language's ratio over every roster human, is clipped, and the
over-represented side's views are scaled down by the same fraction in every
month. Steady single-edition crawlers get cut; one-month bursts are left to the
median of monthly totals, which already ignores them. It runs only
for people whose raw median clears config.MIN_MEDIAN_VIEWS -- it can only lower
views, so nobody below the floor could rise above it -- and their columns stay
raw (clip_keep = 1).

Output: data/people_views.parquet (one row per roster human)
"""

import numpy as np
import pandas as pd

import config
from util import write_parquet_atomic

CELL_INDEX = ["qid_int", "lang", "desktop"]


def month_col(month: str) -> str:
    return f"views_{month.replace('-', '')}"


def person_index(qids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Sorted numeric QIDs and the permutation back to roster order."""
    q = qids.str.slice(1).astype(np.int64).to_numpy()
    order = np.argsort(q)
    return q[order], order


def locate(sorted_q: np.ndarray, qid_int: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(sorted_q, qid_int)
    pos = np.minimum(pos, len(sorted_q) - 1)
    if not np.all(sorted_q[pos] == qid_int):
        raise ValueError("pageview rows for QIDs not in the roster -- stale stage-02 output?")
    return pos


def row_keep(df: pd.DataFrame, cell_keep: pd.Series | None) -> np.ndarray:
    """Per pageview row: the fraction of its views the bot filter keeps (1 unless its cell is in cell_keep)."""
    keep = np.ones(len(df))
    if cell_keep is None or cell_keep.empty:
        return keep
    hit = np.isin(df["qid_int"].to_numpy(), cell_keep.index.get_level_values("qid_int").unique())
    sub = df[hit]
    idx = pd.MultiIndex.from_arrays(
        [sub["qid_int"].to_numpy(), sub["lang"].astype(str).to_numpy(), (sub["access"] == "desktop").to_numpy()],
        names=CELL_INDEX,
    )
    keep[hit] = cell_keep.reindex(idx).fillna(1.0).to_numpy()
    return keep


def accumulate(
    roster: pd.DataFrame, months: dict[str, pd.DataFrame], cell_keep: pd.Series | None = None
) -> pd.DataFrame:
    """Pure core: roster [qid, ...] + {month: [qid_int, lang, access, views]} -> metrics frame.

    cell_keep: (qid_int, lang, desktop) -> fraction of that cell's views to keep in every month (the bot
    filter); None = raw views. Access shares always use raw views.
    """
    sorted_q, order = person_index(roster["qid"])
    n, k = len(sorted_q), len(config.ACCESS_METHODS)
    monthly = np.zeros((n, len(months)))
    by_access = np.zeros(n * k)
    for j, df in enumerate(months.values()):
        pos = locate(sorted_q, df["qid_int"].to_numpy())
        views = df["views"].to_numpy()
        monthly[:, j] = np.bincount(pos, weights=views * row_keep(df, cell_keep), minlength=n)
        acc = pd.Categorical(df["access"], categories=config.ACCESS_METHODS).codes.astype(np.int64)
        if (acc < 0).any():
            raise ValueError(f"unknown access methods: {set(df['access'][acc < 0])}")
        by_access += np.bincount(pos * k + acc, weights=views, minlength=n * k)

    # back to roster order
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    monthly = np.rint(monthly[inv]).astype(np.int64)
    by_access = by_access.reshape(n, k)[inv]

    out = roster.copy()
    for j, month in enumerate(months):
        out[month_col(month)] = monthly[:, j]
    out["median_views"] = np.median(monthly, axis=1)
    out["total_views"] = monthly.sum(axis=1)
    out["max_views"] = monthly.max(axis=1)
    out["n_months"] = (monthly > 0).sum(axis=1).astype(np.int8)
    out["median_present"] = np.ma.median(np.ma.masked_equal(monthly, 0), axis=1).filled(0)
    total = np.maximum(by_access.sum(axis=1), 1)
    out["desktop_share"] = (by_access[:, config.ACCESS_METHODS.index("desktop")] / total).astype(np.float32)
    out["app_share"] = (by_access[:, config.ACCESS_METHODS.index("mobile-app")] / total).astype(np.float32)
    return out


def access_cells(months: dict[str, pd.DataFrame], keep_qid_int: np.ndarray) -> tuple[pd.Series, pd.DataFrame]:
    """log(desktop / mobile) per language over every roster human, and per (person, language) for keep_qid_int:
    [qid_int, lang, D, M, D_med, M_med] -- 12-month totals and median months.

    D = desktop views, M = mobile-web + mobile-app views; a month without rows counts as 0.
    """
    base, parts = [], []
    for j, df in enumerate(months.values()):
        desktop = pd.Series((df["access"] == "desktop").to_numpy(), index=df.index, name="desktop")
        base.append(df.groupby([df["lang"], desktop], observed=True)["views"].sum())
        hit = np.isin(df["qid_int"].to_numpy(), keep_qid_int)
        sub = df[hit]
        keys = [sub["qid_int"], sub["lang"].astype(str), desktop[hit]]
        parts.append(sub.groupby(keys, observed=True)["views"].sum().rename(j))

    base = pd.concat(base).groupby(level=[0, 1]).sum().unstack("desktop", fill_value=0)
    base.index = base.index.astype(str)
    with np.errstate(divide="ignore"):
        log_base = np.log(base.get(True, 0) / base.get(False, 0)).rename("log_base")

    monthly = pd.concat(parts, axis=1).fillna(0)  # rows (qid_int, lang, desktop), one column per month
    stats = pd.DataFrame({"sum": monthly.sum(axis=1), "med": monthly.median(axis=1)}).unstack("desktop", fill_value=0)

    def col(stat: str, is_desktop: bool):
        return stats[(stat, is_desktop)] if (stat, is_desktop) in stats.columns else 0

    cells = pd.DataFrame(
        {"D": col("sum", True), "M": col("sum", False), "D_med": col("med", True), "M_med": col("med", False)},
        index=stats.index,
    ).reset_index()
    return log_base, cells


def clip_cells(cells: pd.DataFrame, log_base: pd.Series) -> pd.DataFrame:
    """Add keep_D / keep_M: the fraction of each (person, language)'s desktop / mobile views the bot filter keeps.

    r = log((D_med + 1) / (M_med + 1)) - log_base[lang], on the cell's MEDIAN month, is clipped to
    [-log MAX_MOBILE_SKEW, +log MAX_DESKTOP_SKEW] by shrinking the over-represented side; the same fraction then
    applies to every month. Median months, not 12-month sums: a one-month crawler burst is already ignored by the
    median of monthly totals, and in a sum it would scale down the clean months too (Gungunum lost 12% of his
    median that way). Languages without a finite baseline are left alone.
    """
    ratio = np.exp(cells["lang"].map(log_base).to_numpy(dtype=float))  # the language's D/M
    d, m = cells["D_med"].to_numpy(dtype=float), cells["M_med"].to_numpy(dtype=float)
    ok = np.isfinite(ratio) & (ratio > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_max = (m + 1) * ratio * config.MAX_DESKTOP_SKEW - 1
        m_max = (d + 1) / ratio * config.MAX_MOBILE_SKEW - 1
    out = cells.copy()
    out["keep_D"] = np.where(ok & (d > d_max) & (d > 0), np.maximum(d_max, 0) / np.maximum(d, 1), 1.0)
    out["keep_M"] = np.where(ok & (m > m_max) & (m > 0), np.maximum(m_max, 0) / np.maximum(m, 1), 1.0)
    return out


def cell_keep_series(clipped: pd.DataFrame) -> pd.Series:
    """(qid_int, lang, desktop) -> keep fraction, for the clipped cells only (the input accumulate() takes)."""
    parts = []
    for side, is_desktop in (("D", True), ("M", False)):
        c = clipped[clipped[f"keep_{side}"] < 1]
        parts.append(pd.DataFrame({"qid_int": c["qid_int"], "lang": c["lang"], "desktop": is_desktop,
                                   "keep": c[f"keep_{side}"]}))  # fmt: skip
    return pd.concat(parts).set_index(CELL_INDEX)["keep"]


def language_metrics(per_lang: pd.DataFrame) -> pd.DataFrame:
    """[qid_int, lang, views] (12-month views) -> top_lang / top_lang_share / en_share / n_langs per person."""
    per_lang = per_lang[per_lang["views"] > 0].copy()
    tot = per_lang.groupby("qid_int")["views"].transform("sum")
    per_lang["share"] = per_lang["views"] / tot
    top = per_lang.sort_values(["qid_int", "views"], ascending=[True, False]).drop_duplicates("qid_int")
    en = per_lang[per_lang["lang"] == "en"].set_index("qid_int")["share"]
    res = pd.DataFrame(
        {
            "top_lang": top.set_index("qid_int")["lang"].astype(str),
            "top_lang_share": top.set_index("qid_int")["share"].astype(np.float32),
        }
    )
    res["en_share"] = en.reindex(res.index).fillna(0).astype(np.float32)
    res["n_langs"] = per_lang.groupby("qid_int").size().astype(np.int16)
    return res.rename_axis("qid_int").reset_index()


def enwiki_titles() -> pd.Series:
    """qid -> enwiki title, for readable log lines."""
    sl = pd.read_parquet(config.SITELINKS_PARQUET, filters=[("site", "==", "enwiki")])
    return sl.set_index("qid")["title"]


def main() -> None:
    roster = pd.read_parquet(config.ROSTER_PARQUET)
    missing = [m for m in config.PAGEVIEW_MONTHS if not config.pageview_month_parquet(m).exists()]
    if missing:
        raise SystemExit(f"stage 02 incomplete; missing {missing}")
    months = {m: pd.read_parquet(config.pageview_month_parquet(m)) for m in config.PAGEVIEW_MONTHS}
    raw = accumulate(roster, months)
    above = raw["median_views"] >= config.MIN_MEDIAN_VIEWS
    print(f"{int(above.sum()):,} of {len(raw):,} people at or above {config.MIN_MEDIAN_VIEWS} raw median views/mo")

    log_base, cells = access_cells(months, raw.loc[above, "qid"].str.slice(1).astype(np.int64).to_numpy())
    cells = clip_cells(cells, log_base)
    cut_d = (cells["D"] * (1 - cells["keep_D"])).sum()
    cut_m = (cells["M"] * (1 - cells["keep_M"])).sum()
    share = (cut_d + cut_m) / cells[["D", "M"]].sum().sum()
    print(f"Bot filter over {len(cells):,} (person, language) cells: {(cells['keep_D'] < 1).sum():,} desktop-side "
          f"and {(cells['keep_M'] < 1).sum():,} mobile-side clips remove {cut_d:,.0f} desktop + "
          f"{cut_m:,.0f} mobile views ({share:.3%})")  # fmt: skip

    out = accumulate(roster, months, cell_keep_series(cells))
    out.insert(out.columns.get_loc("median_views") + 1, "median_views_raw", raw["median_views"])
    keep = np.where(raw["total_views"] > 0, out["total_views"] / np.maximum(raw["total_views"], 1), 1.0)
    out.insert(out.columns.get_loc("median_views_raw") + 1, "clip_keep", keep.astype(np.float32))
    now_above = out["median_views"] >= config.MIN_MEDIAN_VIEWS
    print(f"{int(now_above.sum()):,} at or above the floor after the filter ({int((above & ~now_above).sum()):,} fell "
          f"below); {int((out['clip_keep'] < 0.8).sum()):,} people lost >20% of their views")  # fmt: skip
    hit = out[out["clip_keep"] < 0.8].nlargest(15, "median_views_raw")
    hit = hit.assign(title=hit["qid"].map(enwiki_titles()))
    print("Most-viewed people losing >20%:")
    print(hit[["qid", "title", "median_views_raw", "median_views", "clip_keep"]].to_string(index=False))

    kept = cells.assign(views=cells["D"] * cells["keep_D"] + cells["M"] * cells["keep_M"])
    out["qid_int"] = out["qid"].str.slice(1).astype(np.int64)
    out = out.merge(language_metrics(kept[["qid_int", "lang", "views"]]), on="qid_int", how="left")
    out = out.drop(columns="qid_int")

    out = out.sort_values(["median_views", "total_views"], ascending=False, kind="mergesort").reset_index(drop=True)
    print(out[["qid", "median_views", "desktop_share", "top_lang", "n_langs"]].head(15).to_string(index=False))
    write_parquet_atomic(out, config.PEOPLE_VIEWS_PARQUET)
    print(f"Wrote {config.PEOPLE_VIEWS_PARQUET}")


if __name__ == "__main__":
    main()

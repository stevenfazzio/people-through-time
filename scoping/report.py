"""Rank enwiki biographies by median monthly enwiki views; report era distribution and birth-year coverage."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Reads living-people-map's redirect map + enwiki monthly pageview parquets (2025-07..2026-06) in place, read-only.
sys.path.insert(0, str(Path.home() / "repos" / "living-people-map" / "pipeline"))
import config  # noqa: E402  (living-people-map's config, NOT this repo's)

OUT = Path(__file__).resolve().parent.parent / "data" / "scoping" / "out"
pd.set_option("display.width", 220, "display.max_columns", 30)


def pct(a, b):
    return f"{a:,} ({100 * a / b:.1f}%)"


def rank() -> pd.DataFrame:
    path = OUT / "bios_ranked.parquet"
    if path.exists():
        return pd.read_parquet(path)
    bio = pd.read_parquet(OUT / "bios.parquet")
    redirects = pd.read_parquet(config.REDIRECTS_PARQUET)  # 2026-07 snapshot; titles renamed since then fall through
    src = pa.array(redirects["source_title"])
    tgt = pc.replace_substring(pa.array(redirects["target_title"]), pattern="_", replacement=" ")
    keyset = pa.array(bio["title"].unique())
    cols = []
    for month in config.PAGEVIEW_MONTHS:
        t = pq.read_table(config.pageview_month_parquet(month), columns=["title", "views"])
        titles = t["title"].combine_chunks()
        ri = pc.index_in(titles, value_set=src)
        keys = pc.if_else(pc.is_valid(ri), pc.take(tgt, ri), pc.replace_substring(titles, pattern="_", replacement=" "))
        m = pc.is_in(keys, value_set=keyset)
        per = pd.DataFrame({"k": keys.filter(m).to_pandas(), "v": t["views"].combine_chunks().filter(m).to_pandas()})
        c = f"v{month}"
        bio[c] = bio["title"].map(per.groupby("k")["v"].sum()).fillna(0).astype("int64")
        cols.append(c)
        print(f"  {month}: {int((bio[c] > 0).sum()):,} bios with views", file=sys.stderr, flush=True)
    mat = bio[cols].to_numpy()
    bio["median_views"] = np.median(mat, axis=1)
    bio["total_views"] = mat.sum(axis=1)
    bio = bio.drop(columns=cols).sort_values(["median_views", "total_views"], ascending=False, kind="mergesort")
    bio = bio.reset_index(drop=True)
    bio["rank"] = np.arange(1, len(bio) + 1)
    bio.to_parquet(path)
    return bio


bio = rank()
N = len(bio)
bio["birth_mid"] = (bio["birth_lo"] + bio["birth_hi"]) / 2
has_b = bio["birth_lo"].notna()
dead = bio["death_lo"].notna()

print("=== enwiki biographies (pages in a births/deaths/Living people/Year-of-birth category) ===")
print(f"total:                              {N:,}")
print(f"  Living people:                    {pct(int(bio.living.sum()), N)}   (roster crawl in July: 1,149,897)")
print(f"  has a deaths category:            {pct(int(dead.sum()), N)}")
print(f"  neither (status unknown):         {pct(int((~bio.living & ~dead).sum()), N)}")

print("\n=== Birth-year coverage ===")
print(f"has a births category:              {pct(int(has_b.sum()), N)}")
print(f"  no births cat, 'Year of birth ...': {pct(int((~has_b & bio.birth_missing).sum()), N)}")
print(f"  no births cat, has death:         {pct(int((~has_b & dead).sum()), N)}")
print(f"  no births info at all:            {pct(int((~has_b & ~bio.birth_missing).sum()), N)}")
print("precision of best births category:")
print(bio.loc[has_b, "birth_prec"].value_counts().to_string())
print(f"conflicting year categories:        {int(bio.birth_conflict.fillna(False).sum()):,}")
print("plausibility flags:")
print(f"  birth after death:                {int((bio.death_hi < bio.birth_lo).sum()):,}")
print(f"  lifespan > 122:                   {int((bio.death_lo - bio.birth_hi > 122).sum()):,}")
print(f"  Living people born < 1900:        {int((bio.living & (bio.birth_hi < 1900)).sum()):,}")

ERAS = [-100_000, 0, 500, 1000, 1500, 1700, 1800, 1900, 1950, 1980, 3000]
LBL = ["BCE", "1-499", "500-999", "1000-1499", "1500-1699", "1700s", "1800s", "1900-49", "1950-79", "1980+"]
bio["era"] = pd.cut(bio["birth_mid"], ERAS, right=False, labels=LBL)


def era_counts(df):
    s = df["era"].value_counts(sort=False)
    s["(no birth yr)"] = int(df["birth_mid"].isna().sum())
    return s


print("\n=== Birth era: all enwiki biographies vs top-N by median monthly enwiki views ===")
tab = pd.DataFrame(
    {"all bios": era_counts(bio)}
    | {f"top {n // 1000}k": era_counts(bio.head(n)) for n in [10_000, 25_000, 100_000, 250_000]}
)
print(tab.to_string())
print("\n% of column:")
print((100 * tab / tab.sum()).round(1).to_string())
for n in [10_000, 25_000, 100_000, 250_000]:
    top = bio.head(n)
    print(
        f"top {n // 1000:>3}k: cutoff {top.median_views.min():>7,.0f} views/mo"
        f" | living {pct(int(top.living.sum()), n)}"
        f" | born before 1900: {pct(int((top.birth_mid < 1900).sum()), n)}"
    )

print("\n=== Era depth: biographies per era at or above a median-views bar ===")
depth = pd.DataFrame(
    {
        f">={t:,}/mo": bio[bio.median_views >= t]["era"].value_counts(sort=False)
        for t in [10_000, 3_000, 1_000, 300, 100]
    }
)
print(depth.to_string())

print("\n=== Equal-count windows: birth-year edges of 20 bins ===")
for n in [25_000, 100_000]:
    top = bio.head(n)
    edges = np.quantile(top["birth_mid"].dropna(), np.linspace(0, 1, 21))
    print(f"top {n // 1000}k: " + "  ".join(f"{e:.0f}" for e in edges))

print("\n=== Sanity: top 3 per era ===")
for e in LBL:
    sub = bio[bio.era == e].head(3)
    print(f"{e:>10}: " + " | ".join(f"{t} ({m:,.0f})" for t, m in zip(sub.title, sub.median_views)))
for n in [25_000, 100_000]:
    r = bio.iloc[n - 3 : n]
    print(f"around rank {n:,}: " + " | ".join(f"{t} ({m:,.0f})" for t, m in zip(r.title, r.median_views)))

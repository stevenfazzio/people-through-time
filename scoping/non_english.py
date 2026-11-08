"""How many humans lack an English article -- sizing the translate-fallback (from fetch.py's QLever pulls)."""

from pathlib import Path

import pandas as pd

QL = Path(__file__).resolve().parent.parent / "data" / "scoping" / "qlever"


def load(name: str, **kw) -> pd.DataFrame:
    df = pd.read_csv(QL / f"{name}.csv", **kw)
    df["q"] = df.pop("h").str.rsplit("/", n=1).str[1]
    return df


n_humans = int(pd.read_csv(QL / "n_humans.csv")["n"].iloc[0])
wp = load("wp_counts")
en = load("en_titles", dtype=str)
d = load("deaths")
p = wp.merge(en[["q"]].assign(has_en=True), on="q", how="left").merge(d, on="q", how="left")
p["has_en"] = p["has_en"].notna()
N = len(p)

print(f"Wikidata humans (P31=Q5): {n_humans:,}")
print(f"  with >=1 Wikipedia article: {N:,}")
print(f"    with English: {p.has_en.sum():,} ({p.has_en.mean():.1%})   without: {(~p.has_en).sum():,}")
print("\nshare WITHOUT an English article, by number of Wikipedia editions:")
for lo, hi in [(1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 999)]:
    s = p[p.nwp.between(lo, hi)]
    print(f"  {lo:>2}-{hi:<3} editions: {len(s):>9,} people, {(~s.has_en).mean():6.1%} lack English")
p = p.sort_values("nwp", ascending=False)
for n in [10_000, 31_000, 100_000]:
    print(f"top {n:>7,} by #editions: {(~p.head(n).has_en).mean():.1%} lack English")

print("\namong people with >=5 editions, by death year (birth dates timed out on QLever):")
five = p[p.nwp >= 5].copy()
five["era"] = pd.cut(five.dy, [-5000, 500, 1000, 1500, 1700, 1800, 1900, 1950, 2100], right=False)
t = five.groupby("era", observed=True).agg(n=("q", "size"), lack_english=("has_en", lambda s: f"{(~s).mean():.1%}"))
print(t.to_string())

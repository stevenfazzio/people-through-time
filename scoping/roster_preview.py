"""What the CURRENT config's roster rule yields, on the enwiki-biography proxy.

Proxy, not the real roster: enwiki biographies (category-based, report.py)
with median monthly ENGLISH views. The real roster ranks Wikidata humans by
all-language views, so expect more early, non-Western people.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "scoping" / "out"
ERAS = [-np.inf, 0, 500, 1000, 1500, 1700, 1800, 1900, 1950, 1980, np.inf]
LBL = ["BCE", "1-499", "500-999", "1000-1499", "1500-1699", "1700s", "1800s", "1900-49", "1950-79", "1980+"]

b = pd.read_parquet(OUT / "bios_ranked.parquet")
b["y"] = (b.birth_lo + b.birth_hi) / 2
b = b[b.y.notna() & (b.y <= config.FAME_REFERENCE_YEAR)].copy()
b["score"] = config.fame_score(b.median_views, b.y)
b = b.sort_values("score", ascending=False)

no_floor = b.head(config.MAP_N)
top = b[b.median_views >= config.MIN_MEDIAN_VIEWS].head(config.MAP_N).copy()
print(f"MAP_N={config.MAP_N:,}  AGE_OFFSET={config.AGE_OFFSET}  floor={config.MIN_MEDIAN_VIEWS}/mo")
print(
    f"floor effect: {(no_floor.median_views < config.MIN_MEDIAN_VIEWS).sum():,} of the unfloored top {config.MAP_N:,}"
    f" sit below the floor (lowest {no_floor.median_views.min():,.0f}/mo)"
)

top["era"] = pd.cut(top.y, ERAS, right=False, labels=LBL)
era = top.groupby("era", observed=False).agg(people=("title", "size"), lowest_views=("median_views", "min"))
era["share"] = (era.people / len(top)).map("{:.1%}".format)
print("\nby birth era (lowest = the implied fame bar):")
print(era.to_string())
print(f"living {top.living.mean():.1%}   born<1900 {(top.y < 1900).mean():.1%}   born<1500 {(top.y < 1500).mean():.1%}")

edges = config.bin_edges(top.y)
top["bin"] = pd.cut(top.y, edges, labels=False, right=False)
per = top.groupby("bin").size().to_numpy()
win = np.convolve(per, np.ones(config.WINDOW_BINS, int), "valid")
lo = top.groupby("bin").y.min().to_numpy()
hi = top.groupby("bin").y.max().to_numpy()
print(f"\n{len(per)} equal-count bins (birth-year span of each):")
print("  ".join(f"{a:.0f}..{z:.0f}" for a, z in zip(lo, hi)))
print(f"people per bin min/max = {per.min():,}/{per.max():,}")
print(
    f"{len(win)} windows of {config.WINDOW_BINS} bins: people/window min/median/max = "
    f"{win.min():,}/{int(np.median(win)):,}/{win.max():,};  AlignedUMAP points = {win.sum():,}"
)
print(f"bins wholly before 1900: {(hi < 1900).sum()} of {len(per)} ({(hi < 1900).mean():.0%} of the slider)")

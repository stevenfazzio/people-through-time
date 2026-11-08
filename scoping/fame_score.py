"""Compare roster rules on the enwiki proxy: raw median views vs. views x age vs. views x (age - 20).

Top 31k, 32 equal-count windows, no floor -- the 2026-11-08 comparison behind
the views-x-age decision (see CLAUDE.md).
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "scoping" / "out"
ERAS = [-np.inf, 0, 500, 1000, 1500, 1700, 1800, 1900, 1950, 1980, np.inf]
LBL = ["BCE", "1-499", "500-999", "1000-1499", "1500-1699", "1700s", "1800s", "1900-49", "1950-79", "1980+"]
N, W = 31_000, 32

b = pd.read_parquet(OUT / "bios_ranked.parquet")
b["y"] = (b.birth_lo + b.birth_hi) / 2
b = b[b.y.notna() & (b.y < 2022)].copy()
b["age"] = 2026 - b.y
b["era"] = pd.cut(b.y, ERAS, right=False, labels=LBL)


def show(score: pd.Series, name: str) -> pd.DataFrame:
    top = b.assign(s=score).sort_values("s", ascending=False).head(N)
    e = top.era.value_counts(sort=False)
    edges = np.quantile(top.y, np.linspace(0, 1, W + 1))
    bars = top.groupby("era", observed=True).median_views.min()
    print(f"\n=== {name}  (top {N:,}, {W} equal-count windows) ===")
    print("share by era:  " + "  ".join(f"{k} {v / N:.1%}" for k, v in e.items()))
    pre1900, pre1500 = (top.y < 1900).mean(), (top.y < 1500).mean()
    print(f"living {top.living.mean():.1%}   born<1900 {pre1900:.1%}   born<1500 {pre1500:.1%}")
    print("window edges: " + " ".join(f"{x:.0f}" for x in edges))
    print("lowest median views/mo admitted, by era: " + "  ".join(f"{k} {v:,.0f}" for k, v in bars.items()))
    return top


show(b.median_views, "median_views (absolute)")
c = show(b.median_views * b.age, "median_views x age")
show(b.median_views * (b.age - 20).clip(lower=1), "median_views x max(age-20, 1)  [fame starts ~20]")

print("\nmedian_views x age -- spot checks:")
inn = c[c.y < 1500].tail(3)
print(
    "  last-in pre-1500:",
    " | ".join(f"{t} ({v:,.0f}/mo, b.{y:.0f})" for t, v, y in zip(inn.title, inn.median_views, inn.y)),
)
out = b[~b.title.isin(c.title) & (b.y > 1995)].sort_values("median_views", ascending=False).head(5)
print(
    "  most-viewed EXCLUDED post-1995 births:",
    " | ".join(f"{t} ({v:,.0f}/mo)" for t, v in zip(out.title, out.median_views)),
)

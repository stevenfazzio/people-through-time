"""Bot-filter granularity: which months should the access-ratio clip look at? (one-off, 2026-11-08)

The settled clip (config.MAX_DESKTOP_SKEW / MAX_MOBILE_SKEW) shrinks the
over-represented side of a (person, language) whose desktop:mobile ratio is too
far from its language's. Three ways to measure that ratio:
  A  12-month sums, one fraction applied to every month (the scoping version)
  B  the cell's median month, one fraction applied to every month
  C  each month on its own
A one-month crawler burst is already ignored by the median of monthly totals;
A then scales the clean months down too. C is noisy on small cells.
Stage 03 uses B.

Reads data/people_views.parquet (median_views_raw), data/pageviews/*, and the
scoping caches. Output: scoping/results/clip_granularity.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "scoping"))
from edition_floor import people_table  # noqa: E402

import config  # noqa: E402

HI, LO = config.MAX_DESKTOP_SKEW, config.MAX_MOBILE_SKEW
pv = pd.read_parquet(config.PEOPLE_VIEWS_PARQUET, columns=["qid", "median_views_raw"])
surv = pv[pv["median_views_raw"] >= config.MIN_MEDIAN_VIEWS].copy()
surv["qid_int"] = surv["qid"].str.slice(1).astype(np.int64)
keep_q = np.sort(surv["qid_int"].to_numpy())

base = pd.read_parquet(config.DATA_DIR / "scoping" / "lang_access_baseline.parquet")
base["desktop"] = base["access"] == "desktop"
b = base.groupby(["lang", "desktop"])["views"].sum().unstack()
ratio = (b[True] / b[False]).rename("ratio")

# dense per-cell monthly arrays for floor survivors
parts = []
for j, m in enumerate(config.PAGEVIEW_MONTHS):
    df = pd.read_parquet(config.pageview_month_parquet(m))
    df = df[np.isin(df["qid_int"].to_numpy(), keep_q)]
    g = (
        df.groupby(
            [df["qid_int"], df["lang"].astype(str), (df["access"] == "desktop").rename("desktop")], observed=True
        )["views"]
        .sum()
        .rename(j)
    )
    parts.append(g)
wide = pd.concat(parts, axis=1).fillna(0)
wide = wide.unstack("desktop", fill_value=0)  # columns (month, desktop)
D = np.stack([wide[(j, True)].to_numpy() if (j, True) in wide.columns else np.zeros(len(wide)) for j in range(12)], 1)
M = np.stack([wide[(j, False)].to_numpy() if (j, False) in wide.columns else np.zeros(len(wide)) for j in range(12)], 1)
idx = wide.index.to_frame(index=False)
r = idx["lang"].map(ratio).to_numpy(dtype=float)[:, None]
ok = np.isfinite(r) & (r > 0)
print(f"cells {len(idx):,}")


def keep_frac(d, m):
    d_max = (m + 1) * r * HI - 1
    m_max = (d + 1) / r * LO - 1
    kd = np.where(ok & (d > d_max) & (d > 0), np.maximum(d_max, 0) / np.maximum(d, 1), 1.0)
    km = np.where(ok & (m > m_max) & (m > 0), np.maximum(m_max, 0) / np.maximum(m, 1), 1.0)
    return kd, km


variants = {}
kd, km = keep_frac(D.sum(1, keepdims=True), M.sum(1, keepdims=True))
variants["A 12-month sums"] = D * kd + M * km
kd, km = keep_frac(np.median(D, 1, keepdims=True), np.median(M, 1, keepdims=True))
variants["B median-month ratio"] = D * kd + M * km
kd, km = keep_frac(D, M)
variants["C per-month"] = D * kd + M * km
variants["raw"] = D + M

qi = idx["qid_int"].to_numpy()
people = people_table("median_views_raw")  # raw floor survivors
meds = {}
for name, v in variants.items():
    monthly = pd.DataFrame(v).groupby(qi).sum()
    meds[name] = pd.Series(np.median(monthly.to_numpy(), 1), index=monthly.index)
res = pd.DataFrame(meds).reindex(people.index)
dated = people.join(res).dropna(subset=["birth", "raw"])
dated = dated[dated["raw"] >= config.MIN_MEDIAN_VIEWS]
rosters = {}
for name in variants:
    sc = config.fame_score(dated[name], dated["birth"])
    rosters[name] = set(sc[dated[name] >= config.MIN_MEDIAN_VIEWS].nlargest(config.MAP_N).index)
for name in variants:
    if name == "raw":
        continue
    drop = 1 - dated[name] / dated["raw"]
    print(
        f"{name:>22}: median drop >5% for {(drop > 0.05).sum():,} people, >20% for {(drop > 0.2).sum():,}; "
        f"fell below floor {(dated[name] < config.MIN_MEDIAN_VIEWS).sum():,}; "
        f"roster churn vs raw {len(rosters['raw'] - rosters[name])}"
    )
print(
    "A vs B roster diff",
    len(rosters["A 12-month sums"] ^ rosters["B median-month ratio"]),
    "| A vs C",
    len(rosters["A 12-month sums"] ^ rosters["C per-month"]),
    "| B vs C",
    len(rosters["B median-month ratio"] ^ rosters["C per-month"]),
)
show = [
    "Gungunum",
    "Tenskwatawa",
    "Harriet_Martineau",
    "Fiann_Paul",
    "Craig_Small",
    "Charles_Stuart_(abolitionist)",
    "Cristina_Rodríguez_(journalist)",
    "Michelangelo",
    "Kyösti_Kallio",
    "Nek",
    "Ansel_Adams",
    "Aristotle",
    "Ikunum",
    "Ammuna",
    "Neferu_II",
]
cols = ["title", "birth", "raw", "A 12-month sums", "B median-month ratio", "C per-month"]
t = dated[dated["title"].isin(show)][cols]
print(t.round(0).to_string(index=False))
for name in ["A 12-month sums", "B median-month ratio", "C per-month"]:
    others = [o for o in ["A 12-month sums", "B median-month ratio", "C per-month"] if o != name]
    drop = 1 - dated[name] / dated["raw"]
    only = dated[(drop > 0.1) & (dated[others].min(axis=1) > dated[name] * 1.08)]
    print(f"\nDropped >10% by {name} but >8% less by the others ({len(only):,}); top by raw x age:")
    only = only.assign(score=config.fame_score(only["raw"], only["birth"])).nlargest(12, "score")
    print(only[cols].round(0).to_string(index=False))

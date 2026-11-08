"""Is there a steady background of desktop "user" views on enwiki pages? (one-off, 2026-11-08)

Obscure ancient pages draw about as many desktop as mobile views on enwiki
(Ashur-rabi I: ~125 desktop, ~125 mobile a month), where enwiki as a whole
reads 25% desktop. Either their readers are desktop-bound (students, scholars)
or a crawler counted as "user" adds a steady trickle of desktop hits to every
page. The 12-month median can't remove a steady trickle, and at the early-era
fame bar (a few hundred views/mo) a trickle is most of the signal.

Three tests on enwiki, binned by a page's mobile views:
  1. Additive floor: if desktop levels off at some D0 as mobile -> 0, for
     modern living people too (no scholarly audience), the floor is a crawler.
  2. Seasonality: student and scholar reading dips in July-August and late
     December; a crawler doesn't. Summer index = (Jul + Aug) / (Mar + Apr).
  3. Common factor: month-to-month changes shared across many pages' desktop
     series (a crawler's schedule) but not their mobile series.

Eras: early = Wikidata death year < 1500 (scoping's deaths.csv, all humans,
not just enwiki-dated ones); living = enwiki "Living people" (scoping bios).
Output: scoping/results/desktop_floor.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

DEATHS = config.DATA_DIR / "scoping" / "qlever" / "deaths.csv"
BIOS = config.DATA_DIR / "scoping" / "out" / "bios.parquet"
EN = config.DATA_DIR / "scoping" / "en_monthly.parquet"
MONTHS = [m.replace("-", "") for m in config.PAGEVIEW_MONTHS]
SUMMER, SPRING = ["202607", "202608"], ["202603", "202604"]
BINS = [0, 10, 25, 50, 100, 250, 1_000, 10_000, np.inf]  # en mobile views per month
pd.set_option("display.width", 250)


def load_en() -> pd.DataFrame:
    """[qid_int, D_YYYYMM..., M_YYYYMM...] for every roster person with enwiki views (cached)."""
    if EN.exists():
        return pd.read_parquet(EN)
    parts = []
    for m, tag in zip(config.PAGEVIEW_MONTHS, MONTHS):
        df = pd.read_parquet(config.pageview_month_parquet(m))
        df = df[df["lang"] == "en"]
        side = np.where(df["access"] == "desktop", f"D_{tag}", f"M_{tag}")
        parts.append(df.assign(side=side).groupby(["qid_int", "side"])["views"].sum())
        print(f"  read {m}", flush=True)
    out = pd.concat(parts).unstack(fill_value=0)
    out = out[[f"D_{t}" for t in MONTHS] + [f"M_{t}" for t in MONTHS]].reset_index()
    out.to_parquet(EN)
    return out


def eras(qid_int: pd.Series) -> pd.Series:
    d = pd.read_csv(DEATHS)
    d["qid_int"] = d["h"].str.rsplit("Q", n=1).str[1].astype(np.int64)
    early = set(d.loc[d["dy"] < 1500, "qid_int"]) - set(d.loc[d["dy"] >= 1500, "qid_int"])
    sl = pd.read_parquet(config.SITELINKS_PARQUET, filters=[("site", "==", "enwiki")])
    bios = pd.read_parquet(BIOS, columns=["title", "living"])
    living_titles = set(bios.loc[bios["living"], "title"].str.replace(" ", "_"))
    living = set(sl.loc[sl["title"].isin(living_titles), "qid"].str.slice(1).astype(np.int64))
    out = pd.Series("other", index=qid_int.index)
    out[qid_int.isin(early)] = "early (died <1500)"
    out[qid_int.isin(living)] = "living"
    return out


def floor_table(en: pd.DataFrame) -> None:
    dcols, mcols = [f"D_{t}" for t in MONTHS], [f"M_{t}" for t in MONTHS]
    en = en.assign(D=en[dcols].sum(axis=1) / 12, M=en[mcols].sum(axis=1) / 12)
    en["bin"] = pd.cut(en["M"], BINS, right=False)
    print("Desktop vs mobile on enwiki, by mobile views/mo (medians per person; seasonality from summed views)")
    print("enwiki overall desktop:mobile = 0.33 (25% desktop)\n")
    for era, g in en.groupby("era"):
        rows = []
        for b, h in g.groupby("bin", observed=True):
            rows.append({
                "mobile/mo": str(b), "n": len(h), "median M": h["M"].median(), "median D": h["D"].median(),
                "median D/M": (h["D"] / h["M"].clip(lower=0.5)).median(),
                "D summer idx": h[[f"D_{t}" for t in SUMMER]].sum().sum() / h[[f"D_{t}" for t in SPRING]].sum().sum(),
                "M summer idx": h[[f"M_{t}" for t in SUMMER]].sum().sum() / h[[f"M_{t}" for t in SPRING]].sum().sum(),
            })  # fmt: skip
        print(f"--- {era} ({len(g):,} people)")
        print(pd.DataFrame(rows).round(2).to_string(index=False), "\n")


def common_factor(en: pd.DataFrame) -> None:
    """Share of month-to-month variance in log views carried by the first principal component."""
    print("Common factor in month-to-month changes (pages with 50-500 desktop AND mobile views/mo)")
    for era, g in en.groupby("era"):
        res = []
        for side in "DM":
            x = g[[f"{side}_{t}" for t in MONTHS]].to_numpy(dtype=float)
            res.append(x)
        d, m = res
        ok = (d.mean(1) >= 50) & (d.mean(1) <= 500) & (m.mean(1) >= 50) & (m.mean(1) <= 500)
        if ok.sum() < 200:
            continue
        line = [f"{era:>20} n={ok.sum():,}"]
        for side, x in zip("DM", (d[ok], m[ok])):
            z = np.log(x + 1)
            z = z - z.mean(axis=1, keepdims=True)
            _, s, vt = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
            share = s[0] ** 2 / (s**2).sum()
            avg = np.exp(z.mean(axis=0))  # the shared monthly profile (geometric, relative to each page's mean)
            line.append(f"{side}: PC1 {share:.2f}, profile " + " ".join(f"{v:.2f}" for v in avg))
        print("\n  ".join(line))
    print("  (profile months:", " ".join(t[2:] for t in MONTHS) + ")\n")


def main() -> None:
    en = load_en()
    en["era"] = eras(en["qid_int"])
    print(en["era"].value_counts().to_string(), "\n")
    floor_table(en)
    common_factor(en)


if __name__ == "__main__":
    main()

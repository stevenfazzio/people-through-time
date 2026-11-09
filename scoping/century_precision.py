"""What a century- or millennium-precision Wikidata year means (one-off, 2026-11-08).

Stage 04's first build kept coarse-precision years as stated, after a handful
of famous cases (Pythagoras -582, Laozi -579 at century precision) looked like
editors' estimates. The roster preview then showed CE births stored as round
century years that cannot be birth years: Malyuta Skuratov died 1573, "born
1600"; Kamehameha I "born 1800". Test: among coarse births whose death is known
to the year, how often does the death come first, and how long is the life,
under three readings of the stated year?

  C  as stated
  A  every coarse year -> the midpoint of the century/millennium it names
  B  only ROUND years (ending 00 or 01) -> midpoint; others kept

Reads the stated (pre-correction) years straight from stage 04's raw chunks.
Output: scoping/results/century_precision.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from conftest import load_stage  # noqa: E402

facts = load_stage("04_fetch_facts.py")
config = facts.config


def midpoint(y: np.ndarray, prec: np.ndarray) -> np.ndarray:
    unit = 10.0 ** (9 - prec)
    k = np.maximum(np.ceil(np.abs(y) / unit), 1)
    return np.where(y > 0, unit * k - unit / 2, -unit * k + unit / 2)


def main() -> None:
    qids = pd.read_parquet(config.PEOPLE_FACTS_PARQUET, columns=["qid"])["qid"].tolist()
    picked = facts.pick_dates(facts.load_raw(qids))  # stated years, before coarse_midpoint
    wide = picked.pivot(index="qid", columns="kind", values=["year", "prec"])
    df = pd.DataFrame({
        "y": wide[("year", "birth")], "prec": wide[("prec", "birth")],
        "d": wide[("year", "death")], "dprec": wide[("prec", "death")],
    }).dropna(subset=["y"])  # fmt: skip
    precise = df[(df["prec"] >= 9) & (df["dprec"] >= 9)]
    life = precise["d"] - precise["y"]
    for lo, hi, name in [(-1e9, 0, "BCE"), (0, 1500, "1-1499"), (1500, 1e9, "1500+")]:
        sel = (precise["y"] > lo) & (precise["y"] <= hi)
        print(
            f"Baseline, year-precision birth and death, {name}: median life {life[sel].median():.0f} (n={sel.sum():,})"
        )

    print("\nBy precision and roundness, stated years (life = death - stated birth; deaths at year precision):")
    rows = []
    for prec, unit in [(7, 100), (6, 1000), (8, 10)]:
        p = df[df["prec"] == prec]
        for side, s in [("CE", p["y"] > 0), ("BCE", p["y"] <= 0)]:
            q = p[s]
            rnd = (q["y"].abs() % unit).isin([0, 1])
            for kind, m in [("round", rnd), ("not round", ~rnd)]:
                r = q[m & (q["dprec"] >= 9)]
                lf = r["d"] - r["y"]
                rows.append({"prec": prec, "side": side, "kind": kind, "n": int(m.sum()), "n_with_death": len(r),
                             "median_life": lf.median(), "death_before_birth": f"{(lf < 0).mean():.1%}"})  # fmt: skip
    print(pd.DataFrame(rows).to_string(index=False))
    p7 = df[df["prec"] == 7]
    print(
        "\nCentury-precision years, last two digits:",
        (p7["y"].abs() % 100).astype(int).value_counts().head(6).to_dict(),
    )

    y, prec = df["y"].to_numpy(float), df["prec"].to_numpy(float)
    coarse = prec <= 7
    unit = 10.0 ** (9 - np.where(coarse, prec, 9))
    rnd = coarse & np.isin(np.abs(y) % unit, [0, 1])
    rules = {
        "C as stated": y,
        "A midpoint, every coarse": np.where(coarse, midpoint(y, prec), y),
        "B midpoint, round only": np.where(rnd, midpoint(y, prec), y),
    }
    d, has_death = df["d"].to_numpy(float), (df["dprec"] >= 9).to_numpy()
    print("\nCoarse births (precision <= 7) with a year-precision death:")
    for name, yy in rules.items():
        out = []
        for side, s in [("CE", y > 0), ("BCE", y <= 0)]:
            m = coarse & has_death & s
            lf = d[m] - yy[m]
            out.append(f"{side}: median life {np.median(lf):.0f}, death<birth {np.mean(lf < 0):.1%}, "
                       f"life>122 {np.mean(lf > 122):.1%} (n={m.sum():,})")  # fmt: skip
        print(f"  {name:26} | " + " | ".join(out))


if __name__ == "__main__":
    main()

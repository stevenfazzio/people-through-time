"""Who the birth-year gaps and fallbacks would cost the roster (one-off, 2026-11-08).

Stage 04 gave 94.0% of the 792k people above the views floor a Wikidata P569,
1.0% a fallback (death - 71, work start - 20, floruit - 46, baptism) and left
4.9% with none. Questions before stage 05 scores them:

1. Roster preview with real birth years: top MAP_N by median_views x age, by era,
   by birth source and precision, vs the enwiki-proxy expectations in CLAUDE.md.
2. The undated: how many would make the roster at a plausible age, and who.
3. Conflicts (death before birth, lifespan > 122) and coarse-precision moderns.
4. The oldest people admitted: a sanity check on myth and junk dates.

Output: scoping/results/birth_coverage.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

ERAS = [(-np.inf, 0, "BCE"), (0, 1500, "1-1499"), (1500, 1800, "1500-1799"), (1800, 1900, "1800-1899"),
        (1900, 1950, "1900-1949"), (1950, np.inf, "1950+")]  # fmt: skip
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)


def era(years: pd.Series) -> pd.Series:
    edges = [e[0] for e in ERAS] + [np.inf]
    return pd.cut(years.astype(float), edges, right=False, labels=[e[2] for e in ERAS])


def main() -> None:
    views = pd.read_parquet(config.PEOPLE_VIEWS_PARQUET, columns=["qid", "median_views", "top_lang"])
    facts = pd.read_parquet(config.PEOPLE_FACTS_PARQUET)
    df = facts.merge(views, on="qid", how="left")
    assert df["median_views"].ge(config.MIN_MEDIAN_VIEWS).all()
    df["score"] = config.fame_score(df["median_views"], df["birth_year_est"].astype(float))
    df["era"] = era(df["birth_year_est"])
    dated = df.dropna(subset=["score"]).sort_values("score", ascending=False)
    roster = dated.head(config.MAP_N)
    bar = roster["score"].min()
    cols = ["qid", "label", "description", "median_views", "birth_year_est", "birth_source"]

    print(f"{len(df):,} people above the floor; {len(dated):,} with a birth year; MAP_N = {config.MAP_N:,}")
    print(f"Score of the last person admitted: {bar:,.0f} (views/mo x years)\n")

    print("== 1. Roster preview by era (share of roster; lowest views admitted) ==")
    by_era = roster.groupby("era", observed=False).agg(n=("qid", "size"), min_views=("median_views", "min"))
    by_era["share"] = (by_era["n"] / len(roster)).map("{:.1%}".format)
    print(by_era.to_string())
    living = (
        roster["era"].isin(["1900-1949", "1950+"])
        & facts.set_index("qid").loc[roster["qid"], "death_year"].isna().to_numpy()
    )
    born = roster["birth_year_est"]
    print(f"before 1500: {(born < 1500).mean():.1%}; before 1900: {(born < 1900).mean():.1%}; "
          f"born 1900+ with no death date: {living.mean():.1%}")  # fmt: skip
    print("\nBirth source in the roster:")
    print(roster["birth_source"].value_counts().to_string())
    print("\nP569 precision in the roster (6 millennium, 7 century, 8 decade, 9 year, 11 day):")
    print(roster["birth_precision"].value_counts(dropna=False).sort_index().to_string())
    coarse_modern = roster[(roster["birth_precision"] < 9) & (roster["birth_year_est"] >= 1500)]
    print(f"\nCoarse-precision (< year) births from 1500 on in the roster: {len(coarse_modern)}")
    print(coarse_modern[[*cols, "birth_precision"]].head(15).to_string(index=False))
    print("\nFallback-dated people in the roster, by views:")
    print(roster[roster["birth_source"] != "P569"][cols].head(20).to_string(index=False))

    print("\n== 2. The undated ==")
    undated = df[df["score"].isna()].sort_values("median_views", ascending=False)
    print(f"{len(undated):,} undated ({(undated['birth_source'] == 'conflict').sum()} conflicts)")
    for age in (25, 40, 60):
        n_in = (undated["median_views"] * age >= bar).sum()
        print(f"  would clear the bar at age {age}: {n_in:,}")
    print(f"  with a description mentioning 'born': {undated['description'].str.contains('born', na=False).sum():,}")
    print("\nMost-viewed undated people:")
    print(undated[[*cols[:4], "gender", "top_lang"]].head(30).to_string(index=False))
    desc_words = (
        undated.head(2000)["description"]
        .str.lower()
        .str.extract(
            r"(actor|actress|singer|footballer|player|rapper|youtuber|model|wrestler|politician|streamer|dj|musician)",
            expand=False,
        )
    )
    print("\nOccupation words in the 2,000 most-viewed undated descriptions:")
    print(desc_words.value_counts(dropna=False).head(15).to_string())

    print("\n== 3. Conflicts, by views ==")
    conflict = df[df["birth_source"] == "conflict"].sort_values("median_views", ascending=False)
    print(
        conflict[["qid", "label", "description", "median_views", "birth_year", "death_year"]]
        .head(15)
        .to_string(index=False)
    )

    print("\n== 4. Oldest 25 people in the roster ==")
    print(roster.sort_values("birth_year_est")[[*cols, "birth_precision"]].head(25).to_string(index=False))


if __name__ == "__main__":
    main()

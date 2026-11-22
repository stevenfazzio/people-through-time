"""Nearest-neighbour outliers: who is farthest from everyone else in the map's embedding? (one-off, 2026-11-18)

For curiosity and the write-up. Space: the vectors stage 08 lays out (config.MAP_EMBEDDING, the role-instructed
Qwen3 vectors with gender projected out by stage 07b), cosine distance on unit rows.

  global         each person's nearest other person among all 31k
  within window  the same inside each slider window (the people a reader sees together); a person is in up to 5
                 windows and is ranked by the *smallest* of their per-window distances, so "lonely" means lonely
                 in every window they appear in
  2-d            where stage 08's layout puts the lonely: are they grey (outside every coarse region) or inside
                 a labelled region, how far is their 2-d nearest neighbour (percentile within the window), and
                 who is it -- UMAP has to put everyone next to someone
  suspect years  early-born people whose nearest people are all modern: the within-window ranking's top two were
                 a German actress (Wikidata P569 = 20, "20th century") and a manga artist (26) in windows 0-3

Output: scoping/results/nn_outliers.txt (stdout)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

TOP = 25  # people listed per ranking
DETAIL = 10  # of those, people whose every window is shown in 2-d
CHUNK = 2048
LEAD_CHARS = 220
SUSPECT_BORN_BEFORE = 1500  # a person this early whose nearest people are modern may have a junk birth year
SUSPECT_K = 10
SUSPECT_NEIGHBOURS_FROM = 1800
ERAS = [(-1e9, 0, "BCE"), (0, 1000, "1-999"), (1000, 1500, "1000-1499"), (1500, 1800, "1500-1799"),
        (1800, 1900, "1800-1899"), (1900, 1950, "1900-1949"), (1950, 1e9, "1950+")]  # fmt: skip


def nearest(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row index and cosine distance of each unit row's nearest other row."""
    idx = np.empty(len(v), np.int64)
    dist = np.empty(len(v), np.float32)
    for s in range(0, len(v), CHUNK):
        block = v[s : s + CHUNK] @ v.T
        r = np.arange(len(block))
        block[r, s + r] = -np.inf
        best = block.argmax(1)
        idx[s : s + CHUNK], dist[s : s + CHUNK] = best, 1 - block[r, best]
    return idx, dist


def load() -> tuple[pd.DataFrame, np.ndarray]:
    people = pd.read_parquet(config.MAP_PEOPLE_PARQUET)
    z = np.load(config.embeddings_npz(config.MAP_EMBEDDING))
    if not (z["qid"] == people["qid"].to_numpy(dtype=str)).all():
        raise SystemExit("embeddings are not row-aligned with map_people")
    v = z["vectors"].astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    cols = ["qid", "lead", "lead_source", "n_chars", "enwiki_description"]
    people = people.merge(pd.read_parquet(config.LEADS_PARQUET, columns=cols), on="qid", how="left", validate="1:1")
    people["desc"] = people["enwiki_description"].fillna(people["description"]).fillna("")
    return people, v


def window_slices(people: pd.DataFrame, v: np.ndarray) -> pd.DataFrame:
    """One row per (window, person): high-d and 2-d nearest neighbour inside the window, and the person's labels."""
    row = pd.Series(np.arange(len(people)), index=people["qid"])
    lay = pd.read_parquet(config.layout_parquet()).merge(
        pd.read_parquet(config.tracks_parquet()), on=["window", "qid"], validate="1:1"
    )
    lay["e"] = row.loc[lay["qid"]].to_numpy()
    parts = []
    for w, g in lay.groupby("window"):
        e = g["e"].to_numpy()
        i, d = nearest(v[e])
        xy = g[["x", "y"]].to_numpy()
        d2, i2 = cKDTree(xy).query(xy, k=2)
        parts.append(pd.DataFrame({
            "window": w, "e": e, "nn": e[i], "dist": d, "nn2": e[i2[:, 1]],
            "dist_pct": pd.Series(d).rank(pct=True).to_numpy(),
            "dist2_pct": pd.Series(d2[:, 1]).rank(pct=True).to_numpy(),
            "track_0": g["track_0"].to_numpy(), "track_1": g["track_1"].to_numpy(), "track_2": g["track_2"].to_numpy(),
        }))  # fmt: skip
    return pd.concat(parts, ignore_index=True)


def show_person(people: pd.DataFrame, e: int, dist: float, nn: int, where: str = "") -> None:
    p, q = people.iloc[e], people.iloc[nn]
    print(f"{dist:.3f}  #{p['rank']:<6} {p['label']}  (born {p['birth_year_est']}{where}) -- {p['desc']}")
    print(f"        nearest: {q['label']} -- {q['desc']}")
    lead = " ".join(str(p["lead"]).split())
    more = "..." if len(lead) > LEAD_CHARS else ""
    print(f"        lead [{p['lead_source']}, {p['n_chars']} ch]: {lead[:LEAD_CHARS]}{more}")


def main() -> None:
    people, v = load()
    names = pd.read_parquet(config.track_names_parquet())
    coarse_name = dict(zip(names.loc[names["layer"] == 2, "track"], names.loc[names["layer"] == 2, "name"]))

    # --- global
    nn, dist = nearest(v)
    people["nn"], people["dist"] = nn, dist
    q = np.quantile(dist, [0.01, 0.1, 0.5, 0.9, 0.99, 0.999])
    print("GLOBAL: each person's nearest other person among all", f"{len(people):,}")
    print(f"cosine distance: p1 {q[0]:.3f}  p10 {q[1]:.3f}  median {q[2]:.3f}  p90 {q[3]:.3f}  p99 {q[4]:.3f}  "
          f"p99.9 {q[5]:.3f}  max {dist.max():.3f}")  # fmt: skip
    print(f"spearman with lead length: {people['dist'].corr(people['n_chars'], method='spearman'):+.3f}")
    print("median by lead source:", ", ".join(f"{k} {g.median():.3f} (n={len(g)})"
                                              for k, g in people.groupby("lead_source")["dist"]))  # fmt: skip
    bar = np.quantile(dist, 0.99)
    print("\nby era: people, median distance, share in the global top 1%, loneliest")
    for lo, hi, label in ERAS:
        g = people[(people["birth_year_est"] >= lo) & (people["birth_year_est"] < hi)]
        top = g.loc[g["dist"].idxmax()]
        print(f"  {label:>10}  {len(g):6,}  {g['dist'].median():.3f}  {(g['dist'] >= bar).mean():5.1%}  "
              f"{top['label']} ({top['dist']:.3f})")  # fmt: skip
    print(f"\ntop {TOP}")
    global_top = people.sort_values("dist", ascending=False).head(TOP)
    for e, r in global_top.iterrows():
        show_person(people, e, r["dist"], r["nn"])

    # --- within window
    sl = window_slices(people, v)
    best = sl.loc[sl.groupby("e")["dist"].idxmin()].sort_values("dist", ascending=False)
    q = np.quantile(best["dist"], [0.5, 0.99])
    shared = best.head(TOP)["e"].isin(global_top.index).sum()
    print(f"\n\nWITHIN WINDOW: nearest other person in the same window, each person ranked by their smallest "
          f"distance over the windows they're in\nmedian {q[0]:.3f}, p99 {q[1]:.3f}, max {best['dist'].max():.3f}; "
          f"{shared} of the top {TOP} are also in the global top {TOP}")  # fmt: skip
    print("\nby era: median of each person's smallest within-window distance, share in this ranking's top 1%")
    best = best.join(people["birth_year_est"], on="e")
    bar = np.quantile(best["dist"], 0.99)
    for lo, hi, label in ERAS:
        g = best[(best["birth_year_est"] >= lo) & (best["birth_year_est"] < hi)]
        print(f"  {label:>10}  {g['dist'].median():.3f}  {(g['dist'] >= bar).mean():5.1%}")
    print(f"\ntop {TOP} (window shown: the one where they're least alone)")
    window_top = best.head(TOP)
    for r in window_top.itertuples():
        show_person(people, r.e, r.dist, r.nn, f", window {r.window}")

    # --- 2-d
    print("\n\n2-D: where the layout puts people, by decile of their within-window high-d distance (all slices)")
    print("decile  high-d dist  grey (no coarse region)  no fine label  2-d NN distance pct (median)")
    sl["decile"] = pd.qcut(sl["dist"], 10, labels=False)
    for dec, g in sl.groupby("decile"):
        print(f"  {dec + 1:>4}  {g['dist'].min():.3f}-{g['dist'].max():.3f}  {(g['track_2'] < 0).mean():19.1%}  "
              f"{(g['track_0'] < 0).mean():13.1%}  {g['dist2_pct'].median():.2f}")  # fmt: skip
    rho = sl["dist"].corr(sl["dist2_pct"], method="spearman")
    print(f"spearman(high-d, 2-d NN distance) within windows: {rho:+.3f}")

    for title, top in (("global", global_top.index), ("within-window", window_top["e"])):
        print(f"\n{title} top {DETAIL}, every window: high-d pct | 2-d NN pct | coarse region | 2-d nearest")
        for e in list(top)[:DETAIL]:
            print(f"  {people.iloc[e]['label']}")
            for r in sl[sl["e"] == e].sort_values("window").itertuples():
                region = coarse_name.get(r.track_2, "?") if r.track_2 >= 0 else "grey"
                fine = "" if r.track_0 >= 0 else " (no fine label)"
                print(f"    w{r.window:<2}  {r.dist_pct:.3f}  {r.dist2_pct:.2f}  {region}{fine}  | "
                      f"{people.iloc[r.nn2]['label']}")  # fmt: skip

    # --- suspect birth years: the within-window ranking's top two were moderns with junk Wikidata years
    early = np.flatnonzero(people["birth_year_est"].to_numpy(dtype=float) < SUSPECT_BORN_BEFORE)
    by = people["birth_year_est"].to_numpy(dtype=float)
    sims = v[early] @ v.T
    sims[np.arange(len(early)), early] = -np.inf
    near = np.argpartition(-sims, SUSPECT_K, axis=1)[:, :SUSPECT_K]
    med = np.median(by[near], axis=1)
    print(f"\n\nSUSPECT BIRTH YEARS: born before {SUSPECT_BORN_BEFORE}, but the median birth year of their "
          f"{SUSPECT_K} nearest people (global) is {SUSPECT_NEIGHBOURS_FROM} or later")  # fmt: skip
    for i in np.argsort(people["rank"].to_numpy()[early]):
        if med[i] < SUSPECT_NEIGHBOURS_FROM:
            continue
        p = people.iloc[early[i]]
        lead = " ".join(str(p["lead"]).split())
        print(f"  #{p['rank']:<6} {p['label']}  born {p['birth_year_est']} ({p['birth_source']}, precision "
              f"{p['birth_precision']}), neighbours' median {med[i]:.0f}, bin {p['bin']} -- {lead[:150]}")  # fmt: skip


if __name__ == "__main__":
    main()

"""Benchmark AlignedUMAP at roughly the real job's size, on living-people-map's 25k embeddings.

Slices living people into this project's slider structure (config.N_BINS
equal-count birth-year bins, windows of config.WINDOW_BINS consecutive bins),
then fits one AlignedUMAP across all windows. Reports:
  * runtime (the open question in CLAUDE.md -- the real job is ~137k points;
    this is ~25k x 5 = ~120k);
  * smoothness: how far a person moves between neighbouring windows, in units
    of the layout's RMS radius, vs. independent per-window UMAPs aligned
    after the fact with Procrustes (the no-AlignedUMAP baseline);
  * fidelity: share of each person's 15 nearest 2-d neighbours that are also
    among their 15 nearest in the 1024-d embedding, for both.

Living people only span ~1909-2021, so this tests mechanics and cost, not how
history will look. Positions are saved for the frontend prototype.

Reads living-people-map/data (read-only). Writes data/bench/aligned_living.npz.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402
from util import save_npz_atomic  # noqa: E402

LPM = Path.home() / "repos" / "living-people-map" / "data"
OUT = config.DATA_DIR / "bench" / "aligned_living.npz"
UMAP_KW = dict(n_neighbors=15, min_dist=0.05, metric="cosine", random_state=42)  # family defaults
K = 15


def load() -> pd.DataFrame:
    z = np.load(LPM / "embeddings_global_lead.npz")
    people = pd.DataFrame({"pageid": z["pageid"], "row": np.arange(len(z["pageid"]))})
    wd = pd.read_parquet(LPM / "people_wikidata_global.parquet", columns=["pageid", "qid", "birth_year"])
    names = pd.read_parquet(LPM / "people_top25k_global.parquet", columns=["qid", "title"])
    people = people.merge(wd, on="pageid").merge(names, on="qid").dropna(subset=["birth_year"])
    people["emb"] = list(z["emb"][people["row"].to_numpy()])
    return people.reset_index(drop=True)


def windows(people: pd.DataFrame) -> list[np.ndarray]:
    edges = config.bin_edges(people["birth_year"])
    bins = pd.cut(people["birth_year"], edges, labels=False, right=False).to_numpy()
    n_windows = len(edges) - 1 - config.WINDOW_BINS + 1
    return [np.flatnonzero((bins >= w) & (bins < w + config.WINDOW_BINS)) for w in range(n_windows)]


def displacement(layouts: list[np.ndarray], members: list[np.ndarray]) -> np.ndarray:
    """Per-person movement between consecutive windows, in units of the later layout's RMS radius."""
    out = []
    for (a, ma), (b, mb) in zip(zip(layouts, members), zip(layouts[1:], members[1:])):
        shared, ia, ib = np.intersect1d(ma, mb, return_indices=True)
        scale = np.sqrt(((b - b.mean(0)) ** 2).sum(1).mean())
        out.append(np.linalg.norm(a[ia] - b[ib], axis=1) / scale)
    return np.concatenate(out)


def procrustes_chain(layouts: list[np.ndarray], members: list[np.ndarray]) -> list[np.ndarray]:
    """Rotate/reflect/scale/shift each independent layout onto its predecessor via shared people."""
    out = [layouts[0] - layouts[0].mean(0)]
    for prev_m, cur, cur_m in zip(members, layouts[1:], members[1:]):
        _, ip, ic = np.intersect1d(prev_m, cur_m, return_indices=True)
        p, c = out[-1][ip], cur[ic]
        pc, cc = p - p.mean(0), c - c.mean(0)
        r, s = orthogonal_procrustes(cc, pc)
        out.append((cur - c.mean(0)) @ r * (s / (cc**2).sum()) + p.mean(0))
    return out


def knn_fidelity(emb: np.ndarray, layout: np.ndarray) -> float:
    hi = NearestNeighbors(n_neighbors=K + 1, metric="cosine").fit(emb).kneighbors(return_distance=False)[:, 1:]
    lo = NearestNeighbors(n_neighbors=K + 1).fit(layout).kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([len(set(h) & set(lo_)) / K for h, lo_ in zip(hi, lo)]))


def main() -> None:
    import umap  # slow import; keep it out of module import time

    people = load()
    emb = np.stack(people["emb"].to_numpy())
    members = windows(people)
    sizes = [len(m) for m in members]
    print(f"{len(people):,} people, {len(members)} windows of {min(sizes):,}-{max(sizes):,}; {sum(sizes):,} points")
    relations = [
        {int(i): int(j) for i, j in zip(*np.intersect1d(a, b, return_indices=True)[1:])}
        for a, b in zip(members, members[1:])
    ]

    t0 = time.time()
    aligned = umap.AlignedUMAP(**UMAP_KW).fit([emb[m] for m in members], relations=relations)
    t_aligned = time.time() - t0
    layouts_a = [np.asarray(e) for e in aligned.embeddings_]
    print(f"AlignedUMAP: {t_aligned / 60:.1f} min for {sum(sizes):,} points", flush=True)

    t0 = time.time()
    layouts_i = [umap.UMAP(**UMAP_KW).fit_transform(emb[m]) for m in members]
    t_indep = time.time() - t0
    layouts_i = procrustes_chain(layouts_i, members)
    print(f"independent UMAPs: {t_indep / 60:.1f} min", flush=True)

    for name, layouts in [("AlignedUMAP", layouts_a), ("independent + Procrustes", layouts_i)]:
        d = displacement(layouts, members)
        fid = np.mean([knn_fidelity(emb[m], lay) for m, lay in zip(members[::4], layouts[::4])])
        print(
            f"{name:>25}: move between neighbouring windows (x RMS radius) median {np.median(d):.3f}, "
            f"p90 {np.quantile(d, 0.9):.3f}, p99 {np.quantile(d, 0.99):.3f};  15-NN fidelity {fid:.3f}"
        )

    save_npz_atomic(
        OUT,
        qid=people["qid"].to_numpy().astype(str),
        title=people["title"].to_numpy().astype(str),
        birth_year=people["birth_year"].to_numpy(),
        member_offsets=np.cumsum([0] + sizes),
        members=np.concatenate(members),
        positions=np.concatenate(layouts_a).astype(np.float32),
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

"""Stage 08: one AlignedUMAP layout across every slider window.

Window w = the roster people in bins w .. w + WINDOW_BINS - 1 (config), in
rank order. Neighbouring windows share ~80% of their people; AlignedUMAP is
told who they are (`relations`), so a person's position changes smoothly from
window to window. One fit over all windows, on config.MAP_EMBEDDING's vectors
(4096-d Qwen3 with gender projected out by stage 07b, cosine), with
config.UMAP_KW and the alignment knobs.

Output: data/layout_<MAP_EMBEDDING>.parquet, one row per (window, person):
  window (int16), qid, x, y (float32, as AlignedUMAP returns them; the
  frontend rescales). Rows in window order, then rank.
Logs: motion between neighbouring windows in units of the layout's RMS radius
(benchmark median 0.049), 15-NN fidelity (2-d vs embedding, benchmark 0.288),
and the 2-d same-gender neighbour share by era (for the erasure question).
"""

import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import config
from util import log, write_parquet_atomic

K = 15


def window_members(bins: np.ndarray, n_windows: int = config.N_WINDOWS) -> list[np.ndarray]:
    """Row indices (into the rank-ordered roster) of each window's people."""
    return [np.flatnonzero((bins >= w) & (bins < w + config.WINDOW_BINS)) for w in range(n_windows)]


def relations(members: list[np.ndarray]) -> list[dict[int, int]]:
    """For each pair of neighbouring windows: local index in window w -> local index in window w + 1."""
    out = []
    for a, b in zip(members, members[1:]):
        _, ia, ib = np.intersect1d(a, b, assume_unique=True, return_indices=True)
        out.append({int(i): int(j) for i, j in zip(ia, ib)})
    return out


def displacement(layouts: list[np.ndarray], members: list[np.ndarray]) -> np.ndarray:
    """Per shared person, movement between consecutive windows in units of the later layout's RMS radius."""
    out = []
    for (a, ma), (b, mb) in zip(zip(layouts, members), zip(layouts[1:], members[1:])):
        _, ia, ib = np.intersect1d(ma, mb, assume_unique=True, return_indices=True)
        scale = np.sqrt(((b - b.mean(0)) ** 2).sum(1).mean())
        out.append(np.linalg.norm(a[ia] - b[ib], axis=1) / scale)
    return np.concatenate(out)


def knn_2d(layout: np.ndarray, k: int = K) -> np.ndarray:
    return NearestNeighbors(n_neighbors=k + 1).fit(layout).kneighbors(return_distance=False)[:, 1:]


def knn_fidelity(emb: np.ndarray, layout: np.ndarray, k: int = K) -> float:
    """Share of each person's k nearest 2-d neighbours that are among their k nearest in the embedding."""
    hi = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb).kneighbors(return_distance=False)[:, 1:]
    lo = knn_2d(layout, k)
    return float(np.mean([len(set(h) & set(lo_)) / k for h, lo_ in zip(hi, lo)]))


def same_share(values: np.ndarray, nn: np.ndarray) -> tuple[float, float]:
    """Mean share of a labelled person's labelled neighbours with the same value; and its chance rate."""
    known = pd.notna(values)
    vals = np.where(known, values, "").astype(str)
    counts = pd.Series(vals[known]).value_counts()
    same, chance = [], []
    for i in np.flatnonzero(known):
        nb = nn[i][known[nn[i]]]
        if len(nb):
            same.append(np.mean(vals[nb] == vals[i]))
            chance.append((counts[vals[i]] - 1) / (known.sum() - 1))
    return float(np.mean(same)), float(np.mean(chance))


def binary_gender(gender: pd.Series) -> np.ndarray:
    """ "male" / "female" as plain strings, anything else None (not pd.NA, which can't be compared)."""
    g = gender.astype(object)
    return g.where(g.isin(["male", "female"]), None).to_numpy(dtype=object)


def main() -> None:
    import umap  # slow import (numba)

    people = pd.read_parquet(config.MAP_PEOPLE_PARQUET, columns=["qid", "bin", "gender", "birth_year_est"])
    z = np.load(config.embeddings_npz(config.MAP_EMBEDDING))
    if not (z["qid"] == people["qid"].to_numpy(dtype=str)).all():
        raise SystemExit("embeddings are not row-aligned with map_people -- rerun stage 07")
    emb = z["vectors"].astype(np.float32)
    members = window_members(people["bin"].to_numpy())
    sizes = [len(m) for m in members]
    log(f"{config.MAP_EMBEDDING}: {emb.shape[1]}-d vectors; {len(members)} windows of {min(sizes):,}-{max(sizes):,}, "
        f"{sum(sizes):,} points; {config.UMAP_KW}, alignment {config.ALIGNMENT_REGULARISATION} over "
        f"{config.ALIGNMENT_WINDOW_SIZE} windows")  # fmt: skip

    t0 = time.monotonic()
    aligned = umap.AlignedUMAP(
        **config.UMAP_KW,
        alignment_regularisation=config.ALIGNMENT_REGULARISATION,
        alignment_window_size=config.ALIGNMENT_WINDOW_SIZE,
        verbose=True,
    ).fit([emb[m] for m in members], relations=relations(members))
    layouts = [np.asarray(e, dtype=np.float32) for e in aligned.embeddings_]
    log(f"AlignedUMAP: {(time.monotonic() - t0) / 60:.1f} min")
    if not all(np.isfinite(lay).all() for lay in layouts):
        raise SystemExit("non-finite coordinates in the layout")

    out = pd.DataFrame(
        {
            "window": np.repeat(np.arange(len(members), dtype=np.int16), sizes),
            "qid": people["qid"].to_numpy()[np.concatenate(members)],
            "x": np.concatenate([lay[:, 0] for lay in layouts]),
            "y": np.concatenate([lay[:, 1] for lay in layouts]),
        }
    )
    write_parquet_atomic(out, config.layout_parquet())
    log(f"Wrote {config.layout_parquet()} ({len(out):,} rows)")

    d = displacement(layouts, members)
    log(f"Motion between neighbouring windows (x RMS radius): median {np.median(d):.3f}, "
        f"p90 {np.quantile(d, 0.9):.3f}, p99 {np.quantile(d, 0.99):.3f}, max {d.max():.2f}")  # fmt: skip
    fid = [knn_fidelity(emb[m], lay) for m, lay in zip(members[::4], layouts[::4])]
    log(f"15-NN fidelity (every 4th window): mean {np.mean(fid):.3f}, range {min(fid):.3f}-{max(fid):.3f}")
    sex = binary_gender(people["gender"])
    rows = []
    for w, (m, lay) in enumerate(zip(members, layouts)):
        share, chance = same_share(sex[m], knn_2d(lay))
        span = people["birth_year_est"].iloc[m]
        rows.append({"window": w, "born": f"{span.min()} to {span.max()}", "same gender": round(share, 3),
                     "chance": round(chance, 3), "female": round(float(np.mean(sex[m] == "female")), 3)})  # fmt: skip
    log(f"2-d same-gender neighbour share by window:\n{pd.DataFrame(rows).to_string(index=False)}")


if __name__ == "__main__":
    main()

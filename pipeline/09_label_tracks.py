"""Stage 09: label tracks -- each window's layout clustered on its own, the clusters linked across windows.

Settled 2026-11-08 (CLAUDE.md, "label tracks"; scoping/label_tracks.py).

Layers. Each slider window's 2-d layout (stage 08) is clustered with
Toponymy's own primitives -- fast_hdbscan's KD-tree, Boruvka MST with
min_samples config.LABEL_MIN_SAMPLES, condensed tree, leaf clusters -- once per
layer, at one min cluster size per layer. The sizes are the same in every
window, so a layer means the same thing across the slider: each is the size,
from SIZE_GRID, whose windows show config.LABEL_TARGETS labels on average.
Layer 0 is the finest. Within a window the layers nest strictly (one density
tree cut at growing sizes); the stage checks this.

Tracks. Per layer, the clusters of neighbouring windows are matched by the
people they share: a mutual best match covering at least TRACK_LINK_MIN of the
smaller side's shared people continues the track. With TRACK_BRIDGE, a track
that ended is also matched against clusters starting two windows on (bridging
a one-window gap). Anything unmatched starts a new track.

Tree. Each layer is linked on its own, so a track can sit under different
parents over its life. Its parent is the track one layer up holding most of
its person-windows (counting those under a named cluster there); a track never
under one goes to the layer above that, else to Toponymy's synthetic root,
(n_layers, 0).

Outputs:
  data/tracks_<MAP_EMBEDDING>.parquet: window, qid, track_0 .. track_<n-1>
    (int32; -1 = unlabelled; 0 = finest), rows in stage 08's order.
  data/track_tree_<MAP_EMBEDDING>.parquet: layer, track, min_cluster_size,
    parent_layer, parent_track, parent_share (share of its person-windows
    under a named cluster at the parent's layer that are under the parent),
    first_window, last_window, n_windows, n_people, peak_size.
Logs per layer: labels per window, tracks, unlabelled share, lifetimes, label
events per step, how often a labelled person's label switches and why
(boundary crossing vs merge), label motion; then the tree's fit.
"""

import time

import numba
import numpy as np
import pandas as pd
from fast_hdbscan.boruvka import parallel_boruvka
from fast_hdbscan.cluster_trees import condense_tree, extract_leaves, get_cluster_label_vector, mst_to_linkage_tree
from fast_hdbscan.numba_kdtree import build_kdtree

import config
from util import log, write_parquet_atomic

SIZE_GRID = sorted({int(v) for v in np.geomspace(5, 800, 70)})


def window_slices(win: np.ndarray) -> list[slice]:
    """Row ranges of each window; the layout must be in window order."""
    if (np.diff(win) < 0).any():
        raise SystemExit("layout rows are not in window order -- rerun stage 08")
    bounds = np.searchsorted(win, np.arange(int(win.max()) + 2))
    return [slice(int(a), int(b)) for a, b in zip(bounds, bounds[1:])]


def linkage(xy: np.ndarray) -> np.ndarray:
    """Single-linkage tree over mutual reachability distances, as Toponymy builds it (reproducible Boruvka)."""
    tree = build_kdtree(np.ascontiguousarray(xy, dtype=np.float32))
    edges, _, _ = parallel_boruvka(tree, numba.get_num_threads(), min_samples=config.LABEL_MIN_SAMPLES,
                                   sample_weights=np.zeros(1, dtype=np.float32), reproducible=True)  # fmt: skip
    return mst_to_linkage_tree(edges[np.argsort(edges.T[2])])


def leaf_labels(link: np.ndarray, min_size: int, n: int) -> np.ndarray:
    tree = condense_tree(link, int(min_size))
    return get_cluster_label_vector(tree, extract_leaves(tree), 0.0, n)


def closest(options: dict[int, float], target: float) -> int:
    """The key whose value is nearest the target (ties: the smaller key)."""
    return min(options, key=lambda k: (abs(options[k] - target), k))


def layer_sizes(links: list, ns: list[int], targets=config.LABEL_TARGETS) -> tuple[list[int], dict[int, float]]:
    """One min cluster size per layer: the grid size whose windows average closest to the target label count."""
    mean = {m: float(np.mean([leaf_labels(link, m, n).max() + 1 for link, n in zip(links, ns)])) for m in SIZE_GRID}
    sizes = [closest(mean, t) for t in targets]
    if any(b <= a for a, b in zip(sizes, sizes[1:])):
        raise SystemExit(f"min cluster sizes {sizes} for targets {targets} are not increasing -- spread LABEL_TARGETS")
    return sizes, mean


def match(lab_a, pid_a, lab_b, pid_b, cand_a=None, cand_b=None, link_min=config.TRACK_LINK_MIN) -> dict[int, int]:
    """Mutual-best matches between the clusters of two windows by shared people: {cluster in b: cluster in a}.

    cand_a / cand_b restrict the clusters that may match. A pair needs >= link_min of the smaller side's
    people present in both windows (and labelled there).
    """
    na, nb = int(lab_a.max()) + 1, int(lab_b.max()) + 1
    if na <= 0 or nb <= 0:
        return {}
    _, ia, ib = np.intersect1d(pid_a, pid_b, assume_unique=True, return_indices=True)
    la, lb = lab_a[ia], lab_b[ib]
    size_a = np.bincount(la[la >= 0], minlength=na)
    size_b = np.bincount(lb[lb >= 0], minlength=nb)
    both = (la >= 0) & (lb >= 0)
    n = np.zeros((na, nb))
    np.add.at(n, (la[both], lb[both]), 1)
    if cand_a is not None:
        n[~np.isin(np.arange(na), list(cand_a))] = 0
    if cand_b is not None:
        n[:, ~np.isin(np.arange(nb), list(cand_b))] = 0
    best_b, best_a = n.argmax(1), n.argmax(0)
    out = {}
    for a in range(na):
        b = best_b[a]
        if n[a, b] > 0 and best_a[b] == a and n[a, b] >= link_min * min(size_a[a], size_b[b]):
            out[int(b)] = a
    return out


def link_tracks(
    local: list[np.ndarray], pids: list[np.ndarray], bridge: bool = config.TRACK_BRIDGE
) -> list[np.ndarray]:
    """Track id per point of each window (-1 = unclustered), from each window's cluster labels and person ids."""
    tid: list[np.ndarray] = []
    next_id = 0
    for w, lab in enumerate(local):
        ids = np.full(int(lab.max()) + 1, -1)
        if w > 0:
            for b, a in match(local[w - 1], pids[w - 1], lab, pids[w]).items():
                ids[b] = tid[w - 1][a]
            if bridge and w > 1:
                prev = set(tid[w - 1].tolist())
                ended = {c for c, t in enumerate(tid[w - 2]) if t not in prev}
                open_ = set(np.flatnonzero(ids < 0).tolist())
                if ended and open_:
                    for b, a in match(local[w - 2], pids[w - 2], lab, pids[w], ended, open_).items():
                        ids[b] = tid[w - 2][a]
        for c in np.flatnonzero(ids < 0):
            ids[c] = next_id
            next_id += 1
        tid.append(ids)
    return [np.where(lab >= 0, ids[np.maximum(lab, 0)], -1) if len(ids) else np.full(len(lab), -1)
            for lab, ids in zip(local, tid)]  # fmt: skip


def split_share(win: np.ndarray, child: np.ndarray, parent: np.ndarray) -> float:
    """Share of (window, child cluster) pairs whose people fall under more than one parent value (-1 included)."""
    ok = child >= 0
    df = pd.DataFrame({"w": win[ok], "c": child[ok], "p": parent[ok]})
    return float((df.groupby(["w", "c"])["p"].nunique() > 1).mean())


def parent_tree(tracks: np.ndarray) -> pd.DataFrame:
    """Each track's majority parent (rows of `tracks` are person-windows, columns layers, finest first).

    Parent = the track one layer up holding most of the track's rows among those under a named cluster there
    (ties: the lower track id); a track never under one goes to the next layer up, else the root (n_layers, 0).
    """
    n_layers = tracks.shape[1]
    rows = []
    for i in range(n_layers):
        child = tracks[:, i]
        ok = child >= 0
        parent = {int(t): (n_layers, 0, np.nan) for t in np.unique(child[ok])}
        todo = set(parent)
        for j in range(i + 1, n_layers):
            sel = ok & (tracks[:, j] >= 0) & np.isin(child, list(todo))
            if not sel.any():
                continue
            votes = pd.DataFrame({"c": child[sel], "p": tracks[sel, j]}).value_counts().rename("n").reset_index()
            total = votes.groupby("c")["n"].sum()
            best = votes.sort_values(["c", "n", "p"], ascending=[True, False, True]).drop_duplicates("c")
            for r in best.itertuples():
                parent[int(r.c)] = (j, int(r.p), r.n / total[r.c])
                todo.discard(int(r.c))
        rows += [{"layer": i, "track": t, "parent_layer": pl, "parent_track": pt, "parent_share": sh}
                 for t, (pl, pt, sh) in sorted(parent.items())]  # fmt: skip
    return pd.DataFrame(rows)


def track_stats(win: np.ndarray, pid: np.ndarray, tracks: np.ndarray) -> pd.DataFrame:
    out = []
    for i in range(tracks.shape[1]):
        t = tracks[:, i]
        ok = t >= 0
        df = pd.DataFrame({"track": t[ok], "w": win[ok], "p": pid[ok]})
        g = df.groupby("track")
        s = pd.DataFrame({
            "first_window": g["w"].min(),
            "last_window": g["w"].max(),
            "n_windows": g["w"].nunique(),
            "n_people": g["p"].nunique(),
            "peak_size": df.groupby(["track", "w"]).size().groupby(level=0).max(),
        })  # fmt: skip
        out.append(s.reset_index().assign(layer=i))
    return pd.concat(out, ignore_index=True)


def layer_report(win: np.ndarray, pid: np.ndarray, xy: np.ndarray, sl: list[slice], track: np.ndarray) -> dict:
    """Stability diagnostics for one layer's tracks (definitions as in scoping/label_tracks.py)."""
    sets = [set(np.unique(track[s][track[s] >= 0]).tolist()) for s in sl]
    shown = np.array([len(x) for x in sets])
    ok = track >= 0
    df = pd.DataFrame({"w": win[ok], "t": track[ok], "x": xy[ok, 0], "y": xy[ok, 1]})
    pres = df.groupby("t")["w"].agg(["min", "max", "nunique"])
    life = pres["nunique"]
    switch = toggle = both = steps = border = merged = 0
    for w in range(len(sl) - 1):
        a, b = sl[w], sl[w + 1]
        _, ia, ib = np.intersect1d(pid[a], pid[b], assume_unique=True, return_indices=True)
        ta, tb = track[a][ia], track[b][ib]
        sw = (ta >= 0) & (tb >= 0) & (ta != tb)
        switch += int(sw.sum())
        both += int(((ta >= 0) & (tb >= 0)).sum())
        toggle += int(((ta >= 0) != (tb >= 0)).sum())
        steps += len(ia)
        olds, counts = np.unique(ta[sw], return_counts=True)
        for old, k in zip(olds, counts):
            if old in sets[w + 1]:
                border += k  # the old region is still there: the person crossed a boundary
            else:
                dest = tb[(ta == old) & (tb >= 0)]
                merged += k * (np.bincount(dest).max() / len(dest) > 0.5)  # its people mostly joined one region
    rms = np.array([np.sqrt(((xy[s] - xy[s].mean(0)) ** 2).sum(1).mean()) for s in sl])
    cent = df.groupby(["t", "w"])[["x", "y"]].mean()
    nxt = cent.reset_index().assign(w=lambda d: d["w"] - 1).set_index(["t", "w"])
    j = cent.join(nxt, rsuffix="_n", how="inner").reset_index()
    motion = np.hypot(j["x"] - j["x_n"], j["y"] - j["y_n"]).to_numpy() / rms[j["w"].to_numpy() + 1]
    return {
        "labels/window": f"{shown.mean():.1f} ({shown.min()}-{shown.max()})",
        "tracks": len(pres),
        "unlabelled": round(float(np.mean([(track[s] < 0).mean() for s in sl])), 2),
        "life median": int(life.median()),
        "life <= 2": round(float((life <= 2).mean()), 2),
        "gaps": round(float((pres["max"] - pres["min"] + 1 > pres["nunique"]).mean()), 2),
        "events/step": round(float(np.mean([len(x ^ y) for x, y in zip(sets, sets[1:])])), 1),
        "switch": round(switch / max(both, 1), 3),
        "of which merges": round(merged / max(switch, 1), 2),
        "of which boundary": round(border / max(switch, 1), 2),
        "toggle": round(toggle / max(steps, 1), 3),
        "motion median": round(float(np.median(motion)), 3),
        "motion p90": round(float(np.quantile(motion, 0.9)), 3),
    }


def main() -> None:
    t0 = time.monotonic()
    lay = pd.read_parquet(config.layout_parquet())
    win = lay["window"].to_numpy()
    sl = window_slices(win)
    pid = pd.factorize(lay["qid"])[0]
    xy = lay[["x", "y"]].to_numpy(np.float64)
    ns = [s.stop - s.start for s in sl]
    log(f"{config.MAP_EMBEDDING}: {len(lay):,} points in {len(sl)} windows of {min(ns):,}-{max(ns):,}; "
        f"targets {config.LABEL_TARGETS} labels per window (finest first), min_samples {config.LABEL_MIN_SAMPLES}, "
        f"link >= {config.TRACK_LINK_MIN}, bridge one-window gaps: {config.TRACK_BRIDGE}")  # fmt: skip

    links = [linkage(xy[s]) for s in sl]
    sizes, mean = layer_sizes(links, ns)
    log(f"min cluster size per layer: {sizes} -> {[round(mean[m], 1) for m in sizes]} labels per window on average")

    pids = [pid[s] for s in sl]
    tracks = np.full((len(lay), len(sizes)), -1, dtype=np.int32)
    for i, m in enumerate(sizes):
        local = [leaf_labels(link, m, n) for link, n in zip(links, ns)]
        tracks[:, i] = np.concatenate(link_tracks(local, pids))
    for i in range(len(sizes) - 1):
        split = split_share(win, tracks[:, i], tracks[:, i + 1])
        if split > 0:
            log(f"WARNING: {split:.1%} of layer-{i} clusters straddle layer-{i + 1} clusters in a window")

    out = lay[["window", "qid"]].copy()
    for i in range(len(sizes)):
        out[f"track_{i}"] = tracks[:, i]
    write_parquet_atomic(out, config.tracks_parquet())
    log(f"Wrote {config.tracks_parquet()} ({len(out):,} rows)")

    tree = parent_tree(tracks).merge(track_stats(win, pid, tracks), on=["layer", "track"], validate="1:1")
    tree.insert(2, "min_cluster_size", tree["layer"].map(dict(enumerate(sizes))))
    write_parquet_atomic(tree, config.track_tree_parquet())
    log(f"Wrote {config.track_tree_parquet()} ({len(tree):,} tracks)")

    report = pd.DataFrame([{"layer": i, "min size": m} | layer_report(win, pid, xy, sl, tracks[:, i])
                           for i, m in enumerate(sizes)])  # fmt: skip
    pd.set_option("display.width", 250)
    log("Tracks per layer (switch: share of a person's window steps, labelled on both sides, where the label "
        "changes; merges / boundary: why; toggle: steps between named and unlabelled; motion: label centroid "
        f"shift per step in layout radii):\n{report.to_string(index=False)}")  # fmt: skip
    for i in range(len(sizes) - 1):
        t = tree[tree["layer"] == i]
        sh = t["parent_share"].dropna()
        log(f"Tree, layer {i} -> {i + 1}: parent at layer {i + 1} {(t['parent_layer'] == i + 1).mean():.0%}, "
            f"higher {(t['parent_layer'].between(i + 2, len(sizes) - 1)).mean():.0%}, "
            f"root {(t['parent_layer'] == len(sizes)).mean():.0%}; share of a track's person-windows under its "
            f"parent: median {sh.median():.0%}, under half for {(sh < 0.5).mean():.0%} of tracks")  # fmt: skip
    log(f"Done in {time.monotonic() - t0:.0f} s")


if __name__ == "__main__":
    main()

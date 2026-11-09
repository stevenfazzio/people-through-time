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

Lineage check (2026-11-09). Matching neighbouring windows alone lets a track's
id relay to another group in two steps: people from track B cross into track
A's cluster, the next window they regroup with the rest of B, and since they
wore A's label they carry A's id with them ("Thinkers" became the Nazi region
this way, "Novelists" the murderers). So a cluster continues track t only if
its carriers -- the people in both it and t's cluster of the window before --
were mostly in t already: at least config.TRACK_LINEAGE_MIN of those labelled
two windows earlier (checked when t existed then and at least
TRACK_LINEAGE_MIN_PEOPLE carriers were labelled). A failing link is dropped
and the mutual-best matching rerun without it, so the cluster can continue the
track it does descend from. Bridged links need no check: their carriers were in
t two windows earlier by definition.

Drift cut (2026-11-09). A track -- one colour, one ribbon -- claims a region
worth one name, so with config.TRACK_DRIFT_CUT it ends where its content has
moved on, and a new track takes over from the next window ("Classical
Composers" -> "Screen Actors" at window 17). Content is the unit mean of the
members' stage 07b vectors with each window's mean removed (the era drift every
track shares otherwise hides it). Walking a track's windows, it is cut at the
first of two in a row whose content is further from its current segment's first
window than a region typically is from its nearest neighbour in the same window
(the TRACK_DRIFT_QUANTILE of that cosine over the layer). Then each cut moves
so the windows beside it sit with the segment whose mean content they resemble
more: a mixed window at a turn goes with the side it is closer to (window 21's
MLK and Hawking with Bundy and Manson go with the murderers). The linking is
untouched: a cut only renumbers the rest of a track.

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
    first_window, last_window, n_windows, n_people, peak_size, cut_from (the
    track this one took over from at a drift cut; -1 otherwise).
Logs per layer: links the lineage check dropped and what became of them; the
drift bar and every cut;
labels per window, tracks, unlabelled share, lifetimes, label events per step,
how often a labelled person's label switches and why (boundary crossing vs
merge), label motion, the lowest lineage share left; then the tree's fit.
"""

import time
from collections import Counter, defaultdict

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


def match(
    lab_a, pid_a, lab_b, pid_b, cand_a=None, cand_b=None, link_min=config.TRACK_LINK_MIN, banned=()
) -> dict[int, int]:
    """Mutual-best matches between the clusters of two windows by shared people: {cluster in b: cluster in a}.

    cand_a / cand_b restrict the clusters that may match; banned (a, b) pairs may not match each other. A pair
    needs >= link_min of the smaller side's people present in both windows (and labelled there).
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
    for a, b in banned:
        n[a, b] = 0
    best_b, best_a = n.argmax(1), n.argmax(0)
    out = {}
    for a in range(na):
        b = best_b[a]
        if n[a, b] > 0 and best_a[b] == a and n[a, b] >= link_min * min(size_a[a], size_b[b]):
            out[int(b)] = a
    return out


class Lineage:
    """The lineage check for links from window w-1 (cluster a, track t) to window w (cluster b): the carriers --
    people in both a and b -- must mostly have been in t at w-2 already. Without two earlier windows every link
    passes."""

    def __init__(self, lab, pid, prev=None, old=None, min_share=config.TRACK_LINEAGE_MIN,
                 min_people=config.TRACK_LINEAGE_MIN_PEOPLE):  # fmt: skip
        """prev = (labels, person ids, track per cluster) at w-1; old = (person ids, track per person) at w-2."""
        self.min_share, self.min_people = min_share, min_people
        self.total: Counter = Counter()
        self.same: Counter = Counter()
        self.carried: Counter = Counter()
        self.alive: set[int] = set()
        self.track_of_a = np.zeros(0, int)
        self.dropped: list[dict] = []
        if prev is None or old is None:
            return
        lab_p, pid_p, self.track_of_a = prev
        pid_o, trk_o = old
        common, i, j = np.intersect1d(pid, pid_p, assume_unique=True, return_indices=True)
        _, k, m = np.intersect1d(common, pid_o, assume_unique=True, return_indices=True)
        b, a, t_old = lab[i][k], lab_p[j][k], trk_o[m]
        ok = (b >= 0) & (a >= 0) & (t_old >= 0)
        b, a, t_old = b[ok], a[ok], t_old[ok]
        same = t_old == self.track_of_a[a]
        self.total = Counter(zip(a.tolist(), b.tolist()))
        self.same = Counter(zip(a[same].tolist(), b[same].tolist()))
        self.carried = Counter(zip(a.tolist(), b.tolist(), t_old.tolist()))
        self.alive = set(trk_o[trk_o >= 0].tolist())

    def ok(self, a: int, b: int) -> bool:
        """May cluster b continue cluster a's track? Records the links it refuses."""
        t = int(self.track_of_a[a]) if len(self.track_of_a) else -1
        n = self.total[(a, b)]
        if t not in self.alive or n < self.min_people:
            return True
        share = self.same[(a, b)] / n
        if share >= self.min_share:
            return True
        top_t, top_n = max(((tt, c) for (aa, bb, tt), c in self.carried.items() if (aa, bb) == (a, b)),
                           key=lambda x: x[1])  # fmt: skip
        self.dropped.append({"cluster": b, "track": t, "n": n, "share": share, "top_track": top_t,
                             "top_share": top_n / n})  # fmt: skip
        return False


def checked_match(lab_a, pid_a, lab_b, pid_b, lineage: Lineage) -> dict[int, int]:
    """match(), dropping each pair that fails the lineage check and rematching without it, until all pass."""
    banned: set[tuple[int, int]] = set()
    while True:
        m = match(lab_a, pid_a, lab_b, pid_b, banned=banned)
        bad = {(a, b) for b, a in m.items() if not lineage.ok(a, b)}
        if not bad:
            return m
        banned |= bad


def link_tracks(
    local: list[np.ndarray],
    pids: list[np.ndarray],
    bridge: bool = config.TRACK_BRIDGE,
    lineage_min: float = config.TRACK_LINEAGE_MIN,
    lineage_people: int = config.TRACK_LINEAGE_MIN_PEOPLE,
    dropped: list | None = None,
) -> list[np.ndarray]:
    """Track id per point of each window (-1 = unclustered), from each window's cluster labels and person ids.

    Links refused by the lineage check are appended to `dropped` (window, cluster, the track it would have
    continued, its carriers' lineage share, their largest track two windows back, and what it became).
    """
    tid: list[np.ndarray] = []
    trk: list[np.ndarray] = []
    next_id = 0
    for w, lab in enumerate(local):
        ids = np.full(int(lab.max()) + 1, -1)
        if w > 0:
            prev = (local[w - 1], pids[w - 1], tid[w - 1])
            old = (pids[w - 2], trk[w - 2]) if w > 1 else None
            lineage = Lineage(lab, pids[w], prev, old, lineage_min, lineage_people)
            for b, a in checked_match(local[w - 1], pids[w - 1], lab, pids[w], lineage).items():
                ids[b] = tid[w - 1][a]
            if bridge and w > 1:
                alive = set(tid[w - 1].tolist())
                ended = {c for c, t in enumerate(tid[w - 2]) if t not in alive}
                open_ = set(np.flatnonzero(ids < 0).tolist())
                if ended and open_:
                    for b, a in match(local[w - 2], pids[w - 2], lab, pids[w], ended, open_).items():
                        ids[b] = tid[w - 2][a]
        else:
            lineage = Lineage(lab, pids[w])
        first_new = next_id
        for c in np.flatnonzero(ids < 0):
            ids[c] = next_id
            next_id += 1
        tid.append(ids)
        trk.append(np.where(lab >= 0, ids[np.maximum(lab, 0)], -1) if len(ids) else np.full(len(lab), -1))
        if dropped is not None:
            for d in lineage.dropped:
                became = int(ids[d["cluster"]])
                dropped.append(d | {"window": w, "became": became, "new": became >= first_new,
                                    "track_continues": d["track"] in set(ids.tolist())})  # fmt: skip
    return trk


def lineage_shares(pid: np.ndarray, sl: list[slice], track: np.ndarray,
                   min_people: int = config.TRACK_LINEAGE_MIN_PEOPLE) -> np.ndarray:  # fmt: skip
    """What the lineage check tests, over the final tracks: for each track in windows w-2, w-1 and w, the share
    of its carriers (in it at w-1 and w) labelled at w-2 who were in it then (with at least min_people)."""
    out = []
    for w in range(2, len(sl)):
        s0, s1, s2 = sl[w - 2], sl[w - 1], sl[w]
        common, i, j = np.intersect1d(pid[s2], pid[s1], assume_unique=True, return_indices=True)
        _, k, m = np.intersect1d(common, pid[s0], assume_unique=True, return_indices=True)
        t2, t1, t0 = track[s2][i][k], track[s1][j][k], track[s0][m]
        car = (t2 >= 0) & (t2 == t1) & (t0 >= 0)
        alive = set(track[s0][track[s0] >= 0].tolist())
        for t in np.unique(t2[car]):
            sel = car & (t2 == t)
            if t in alive and sel.sum() >= min_people:
                out.append(float((t0[sel] == t).mean()))
    return np.array(out)


def track_centroids(vectors: np.ndarray, rows: np.ndarray, sl: list[slice], tracks: np.ndarray) -> list[dict]:
    """Per layer: {(track, window): unit mean of its members' vectors, each window's mean removed first}.
    rows[i] is layout point i's row in `vectors`."""
    out: list[dict] = [{} for _ in range(tracks.shape[1])]
    for w, s in enumerate(sl):
        X = vectors[rows[s]].astype(np.float64)
        X -= X.mean(0)
        for i in range(tracks.shape[1]):
            t = tracks[s, i]
            ks = np.unique(t[t >= 0])
            if not len(ks):
                continue
            C = (t[None, :] == ks[:, None]).astype(np.float64) @ X
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            for k, c in zip(ks.tolist(), C):
                out[i][(k, w)] = c
    return out


def neighbour_similarity(cents: dict) -> np.ndarray:
    """For each (track, window): cosine to the most similar other track in the same window."""
    by_window = defaultdict(list)
    for (_, w), c in cents.items():
        by_window[w].append(c)
    out = []
    for cs in by_window.values():
        if len(cs) > 1:
            U = np.array(cs)
            S = U @ U.T
            np.fill_diagonal(S, -np.inf)
            out.extend(S.max(1).tolist())
    return np.array(out)


def segment_starts(vecs: list[np.ndarray], bar: float) -> list[tuple[int, float]]:
    """Where a track's run of unit vectors is cut: (index, cosine to the previous segment's first vector) at the
    first of two in a row below `bar` against the current segment's first vector, which then anchors the next."""
    starts = []
    anchor = vecs[0]
    for i in range(1, len(vecs) - 1):
        c = float(vecs[i] @ anchor)
        if c < bar and float(vecs[i + 1] @ anchor) < bar:
            starts.append((i, c))
            anchor = vecs[i]
    return starts


def refine_bounds(vecs: list[np.ndarray], starts: list[int], max_rounds: int = 50) -> list[int]:
    """Move each cut so the windows beside it sit with the segment whose mean content they resemble more (a mixed
    window at a turn goes with the side it is closer to); every segment keeps at least one window."""
    b = [0, *starts, len(vecs)]

    def mean(lo: int, hi: int) -> np.ndarray:
        v = np.sum(vecs[lo:hi], 0)
        return v / np.linalg.norm(v)

    for _ in range(max_rounds):
        changed = False
        for j in range(1, len(b) - 1):
            lo, i, hi = b[j - 1], b[j], b[j + 1]
            if i - 1 > lo and vecs[i - 1] @ mean(i, hi) > vecs[i - 1] @ mean(lo, i - 1):
                b[j], changed = i - 1, True
            elif i + 1 < hi and vecs[i] @ mean(lo, i) > vecs[i] @ mean(i + 1, hi):
                b[j], changed = i + 1, True
        if not changed:
            break
    return b[1:-1]


def drift_cut(sl: list[slice], track: np.ndarray, cents: dict, bar: float) -> tuple[np.ndarray, list[dict]]:
    """One layer's tracks cut where their content drifts (segment_starts, then refine_bounds); each later
    segment gets a new id. Returns the new track per point and one record per cut."""
    out = track.copy()
    next_id = int(track.max()) + 1
    windows = defaultdict(list)
    for k, w in sorted(cents):
        windows[k].append(w)
    cuts = []
    for t, ws in windows.items():
        vecs = [cents[(t, w)] for w in ws]
        found = segment_starts(vecs, bar)
        starts = refine_bounds(vecs, [i for i, _ in found])
        prev = t
        for j, (i, (i0, c)) in enumerate(zip(starts, found)):
            end = starts[j + 1] if j + 1 < len(starts) else len(ws)
            for w in ws[i:end]:
                seg = out[sl[w]]
                seg[track[sl[w]] == t] = next_id
            cuts.append({"track": t, "from": prev, "new": next_id, "window": ws[i], "found": ws[i0], "cos": c})
            prev = next_id
            next_id += 1
    return out, cuts


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
    lin = lineage_shares(pid, sl, track)
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
        "lineage min": round(float(lin.min()), 2) if len(lin) else np.nan,
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
        f"link >= {config.TRACK_LINK_MIN}, bridge one-window gaps: {config.TRACK_BRIDGE}, carriers' lineage >= "
        f"{config.TRACK_LINEAGE_MIN} (when >= {config.TRACK_LINEAGE_MIN_PEOPLE} carriers)")  # fmt: skip

    links = [linkage(xy[s]) for s in sl]
    sizes, mean = layer_sizes(links, ns)
    log(f"min cluster size per layer: {sizes} -> {[round(mean[m], 1) for m in sizes]} labels per window on average")

    pids = [pid[s] for s in sl]
    tracks = np.full((len(lay), len(sizes)), -1, dtype=np.int32)
    for i, m in enumerate(sizes):
        local = [leaf_labels(link, m, n) for link, n in zip(links, ns)]
        drops: list[dict] = []
        tracks[:, i] = np.concatenate(link_tracks(local, pids, dropped=drops))
        log(f"layer {i}: lineage check dropped {len(drops)} links (cluster at window w would have continued "
            f"track t; share of its carriers labelled at w-2 who were in t; their largest track then):")  # fmt: skip
        for d in drops:
            fate = "a new track" if d["new"] else f"continues t{d['became']}"
            log(f"  w{d['window']:2d} t{d['track']}: {d['share']:.0%} of {d['n']} (t{d['top_track']} "
                f"{d['top_share']:.0%}) -> {fate}; t{d['track']} "
                f"{'continues elsewhere' if d['track_continues'] else 'ends'}")  # fmt: skip

    cut_rows = []
    if config.TRACK_DRIFT_CUT:
        z = np.load(config.embeddings_npz(config.MAP_EMBEDDING))
        row_of = pd.Series(np.arange(len(z["qid"])), index=z["qid"].astype(str))
        rows = row_of.reindex(lay["qid"].astype(str)).to_numpy()
        if np.isnan(rows.astype(float)).any():
            raise SystemExit("layout people missing from the embeddings -- rerun stage 07b")
        cents = track_centroids(z["vectors"], rows.astype(int), sl, tracks)
        for i in range(len(sizes)):
            near = neighbour_similarity(cents[i])
            bar = float(np.quantile(near, config.TRACK_DRIFT_QUANTILE))
            tracks[:, i], cuts = drift_cut(sl, tracks[:, i], cents[i], bar)
            q = np.quantile(near, [0.25, 0.5, 0.75])
            log(f"layer {i}: drift bar {bar:.3f} (nearest other region's cosine, window-centred: p25 {q[0]:.2f}, "
                f"median {q[1]:.2f}, p75 {q[2]:.2f}); {len(cuts)} cuts in {len({c['track'] for c in cuts})} "
                "tracks (track: window of the cut, cosine to the segment start -> new track)")  # fmt: skip
            by_track = defaultdict(list)
            for c in cuts:
                moved = f" (found at w{c['found']})" if c["found"] != c["window"] else ""
                by_track[c["track"]].append(f"w{c['window']}{moved} {c['cos']:.2f} -> t{c['new']}")
            for t, parts in by_track.items():
                log(f"  t{t}: " + "; ".join(parts))
            cut_rows += [{"layer": i, "track": c["new"], "cut_from": c["from"]} for c in cuts]

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
    cut_from = pd.DataFrame(cut_rows, columns=["layer", "track", "cut_from"]).astype(int)
    tree = tree.merge(cut_from, on=["layer", "track"], how="left", validate="1:1")
    tree["cut_from"] = tree["cut_from"].fillna(-1).astype(np.int32)
    write_parquet_atomic(tree, config.track_tree_parquet())
    log(f"Wrote {config.track_tree_parquet()} ({len(tree):,} tracks)")

    report = pd.DataFrame([{"layer": i, "min size": m} | layer_report(win, pid, xy, sl, tracks[:, i])
                           for i, m in enumerate(sizes)])  # fmt: skip
    pd.set_option("display.width", 250)
    log("Tracks per layer (switch: share of a person's window steps, labelled on both sides, where the label "
        "changes; merges / boundary: why; toggle: steps between named and unlabelled; motion: label centroid "
        "shift per step in layout radii; lineage min: lowest share of a track's carriers two windows back who "
        f"were in it then):\n{report.to_string(index=False)}")  # fmt: skip
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

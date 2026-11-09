"""Label tracks across the slider windows: per-window clustering + linking vs one 3-d clustering (one-off, 2026-11-08).

Region labels have to persist as the slider moves: a *track* is one region
followed across windows. Two ways to get tracks from stage 08's layout, both
with Toponymy's clustering primitives (fast_hdbscan: KD-tree, Boruvka MST with
min_samples 5, condensed tree, leaf clusters), no API calls:

  A   per window: cluster each window's 2-d layout on its own, then link the
      clusters of neighbouring windows by shared people (mutual best match,
      overlap >= LINK_MIN of the smaller side). A+ also bridges one-window gaps
      (a track that ends is matched against clusters that start two windows on).
  B   stacked: one clustering of all 137k (window, person) points in 3-d,
      (x, y, lambda * window). A cluster is a tube through the windows, i.e. a
      track. lambda = the distance one window step counts for, swept.

Layers are matched on granularity, not parameters: each approach's min cluster
size per layer is chosen so a window shows TARGETS labels on average. A label
is shown in a window if its track has at least the A layer's min cluster size
there (always true for A); points of a track not shown count as unlabelled.

Measures per layer:
  labels/window, tracks, unlabelled share (as displayed)
  lifetime: windows a track is shown in; blinks: tracks shown in <= 2 windows;
    gaps: tracks shown, hidden, shown again
  label events: labels appearing + disappearing per window step
  switch: of a person's window steps with a label on both sides, the share where
    it changes to another region; toggle: share of all steps between named and
    unlabelled
  label motion: track centroid shift per step, in the window's RMS radius
    (stage 08's unit; people move a median 0.058)
  2-d cohesion: share of a labelled point's 15 nearest 2-d neighbours with its label
  split slices: share of (track, window) slices whose members fall apart into
    pieces at link radius SPLIT_RADIUS (largest piece < 90%) -- the label,
    placed at the centroid, would sit between them
  drift: a track named once must mean the same thing throughout. Cosine of the
    track's mean embedding (stage 07b vectors) in its first and last window,
    for tracks shown in >= 3 windows; "drifted" = below the median cosine
    between two different tracks in the same window, i.e. the track ends as
    far from where it began as two separate regions are from each other

Then, for A+, why a labelled person's label switches between windows: their
old region continues (they crossed a boundary), or their old track ended and
its people mostly joined one other region (a merge, often undone a window or
two later), or it ended and scattered.

Output: scoping/results/label_tracks.txt (stdout), data/scoping/label_tracks_{A,B}.html
"""

import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from fast_hdbscan.boruvka import parallel_boruvka
from fast_hdbscan.cluster_trees import condense_tree, extract_leaves, get_cluster_label_vector, mst_to_linkage_tree
from fast_hdbscan.numba_kdtree import build_kdtree
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_variants as ev  # noqa: E402

config = ev.config

MIN_SAMPLES = 5  # Toponymy's default
TARGETS = {"fine": 40, "mid": 20, "coarse": 10}  # labels shown per window, on average
LAMBDAS = [0.02, 0.04, 0.08, 0.16]  # layout units per window step; median 5-NN distance in a window is ~0.08
LINK_MIN = 0.3
SPLIT_RADIUS = 0.3  # ~2x the median 15-NN distance within a window (0.145-0.188)
K = 15
PREVIEW_LAMBDA = 0.04


def linkage(data: np.ndarray) -> np.ndarray:
    tree = build_kdtree(np.ascontiguousarray(data, dtype=np.float32))
    edges, _, _ = parallel_boruvka(tree, numba.get_num_threads(), min_samples=MIN_SAMPLES,
                                   sample_weights=np.zeros(1, dtype=np.float32), reproducible=True)  # fmt: skip
    return mst_to_linkage_tree(edges[np.argsort(edges.T[2])])


def leaf_labels(link: np.ndarray, min_size: int, n: int) -> np.ndarray:
    tree = condense_tree(link, int(min_size))
    return get_cluster_label_vector(tree, extract_leaves(tree), 0.0, n)


class Layout:
    def __init__(self, lay: pd.DataFrame):
        self.win = lay["window"].to_numpy()
        self.xy = lay[["x", "y"]].to_numpy(np.float64)
        self.pid, self.qids = pd.factorize(lay["qid"])
        self.n_windows = int(self.win.max()) + 1
        self.bounds = np.searchsorted(self.win, np.arange(self.n_windows + 1))
        self.sl = [slice(self.bounds[w], self.bounds[w + 1]) for w in range(self.n_windows)]
        xy = [self.xy[s] for s in self.sl]
        self.rms = np.array([np.sqrt(((a - a.mean(0)) ** 2).sum(1).mean()) for a in xy])
        self.knn = [NearestNeighbors(n_neighbors=K + 1).fit(a).kneighbors(return_distance=False)[:, 1:] for a in xy]


def per_window_labels(L: Layout, links: list, m: int) -> list[np.ndarray]:
    return [leaf_labels(link, m, s.stop - s.start) for link, s in zip(links, L.sl)]


def match(lab_a, pid_a, lab_b, pid_b, cand_a=None, cand_b=None) -> dict[int, int]:
    """Mutual-best matches between the clusters of two windows by shared people: {cluster in b: cluster in a}."""
    na, nb = lab_a.max() + 1, lab_b.max() + 1
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
        if n[a, b] > 0 and best_a[b] == a and n[a, b] >= LINK_MIN * min(size_a[a], size_b[b]):
            out[int(b)] = a
    return out


def link_tracks(L: Layout, local: list[np.ndarray], bridge: bool) -> np.ndarray:
    """Approach A: track id per point (-1 = unclustered) from per-window cluster labels."""
    pids = [L.pid[s] for s in L.sl]
    tid: list[np.ndarray] = []
    next_id = 0
    for w, lab in enumerate(local):
        ids = np.full(lab.max() + 1, -1)
        if w > 0:
            for b, a in match(local[w - 1], pids[w - 1], lab, pids[w]).items():
                ids[b] = tid[w - 1][a]
            if bridge and w > 1:
                prev = set(tid[w - 1])
                ended = {c for c, t in enumerate(tid[w - 2]) if t not in prev}
                open_ = {c for c in range(len(ids)) if ids[c] < 0}
                if ended and open_:
                    for b, a in match(local[w - 2], pids[w - 2], lab, pids[w], ended, open_).items():
                        ids[b] = tid[w - 2][a]
        for c in np.flatnonzero(ids < 0):
            ids[c] = next_id
            next_id += 1
        tid.append(ids)
    return np.concatenate([np.where(lab >= 0, ids[lab], -1) if len(ids) else lab for lab, ids in zip(local, tid)])


def displayed(L: Layout, labels: np.ndarray, show_min: int) -> np.ndarray:
    """Keep a track's points in a window only where the track has >= show_min points there."""
    ok = labels >= 0
    key = pd.Series(L.win[ok].astype(np.int64) * 1_000_000 + labels[ok])
    counts = key.map(key.value_counts())
    out = np.full(len(labels), -1)
    idx = np.flatnonzero(ok)
    keep = counts.to_numpy() >= show_min
    out[idx[keep]] = labels[idx[keep]]
    return out


def shown_per_window(L: Layout, track: np.ndarray) -> np.ndarray:
    return np.array([len(np.unique(track[s][track[s] >= 0])) for s in L.sl])


def metrics(L: Layout, track: np.ndarray, emb: np.ndarray) -> dict:
    df = pd.DataFrame({"w": L.win, "t": track, "x": L.xy[:, 0], "y": L.xy[:, 1]})
    lab = df[df.t >= 0]
    shown = shown_per_window(L, track)
    pres = lab.groupby("t")["w"].agg(["min", "max", "nunique"])
    life = pres["nunique"]
    gaps = (pres["max"] - pres["min"] + 1 > pres["nunique"]).mean()
    sets = [set(np.unique(track[s][track[s] >= 0])) for s in L.sl]
    events = np.mean([len(a ^ b) for a, b in zip(sets, sets[1:])])
    switch = toggle = steps = both = 0
    for w in range(L.n_windows - 1):
        a, b = L.sl[w], L.sl[w + 1]
        _, ia, ib = np.intersect1d(L.pid[a], L.pid[b], assume_unique=True, return_indices=True)
        ta, tb = track[a][ia], track[b][ib]
        switch += np.sum((ta >= 0) & (tb >= 0) & (ta != tb))
        toggle += np.sum((ta >= 0) != (tb >= 0))
        both += np.sum((ta >= 0) & (tb >= 0))
        steps += len(ia)
    cent = lab.groupby(["t", "w"])[["x", "y"]].mean()
    nxt = cent.reset_index().assign(w=lambda d: d.w - 1).set_index(["t", "w"])
    j = cent.join(nxt, rsuffix="_n", how="inner").reset_index()
    motion = np.hypot(j.x - j.x_n, j.y - j.y_n).to_numpy() / L.rms[j.w.to_numpy() + 1]
    coh = []
    for w, s in enumerate(L.sl):
        t = track[s]
        m = t >= 0
        coh.append((t[L.knn[w]][m] == t[m, None]).mean(1))
    split, n_slices = 0, 0
    for (_, _w), g in lab.groupby(["t", "w"]):
        n_slices += 1
        if len(g) < 3:
            continue
        graph = radius_neighbors_graph(g[["x", "y"]].to_numpy(), SPLIT_RADIUS)
        _, comp = connected_components(graph, directed=False)
        split += np.bincount(comp).max() / len(g) < 0.9
    cos_first_last, drifted = drift(L, track, emb)
    return {
        "labels/win": f"{shown.mean():.1f} ({shown.min()}-{shown.max()})",
        "tracks": len(pres),
        "unlab": round(float(np.mean([(track[s] < 0).mean() for s in L.sl])), 2),
        "life med": int(life.median()),
        "blinks": round(float((life <= 2).mean()), 2),
        "gaps": round(float(gaps), 2),
        "events/step": round(float(events), 1),
        "switch": round(switch / both, 3),
        "toggle": round(toggle / steps, 3),
        "motion med": round(float(np.median(motion)), 3),
        "motion p90": round(float(np.quantile(motion, 0.9)), 3),
        "cohesion": round(float(np.concatenate(coh).mean()), 3),
        "split": round(split / n_slices, 3),
        "first-last cos": round(cos_first_last, 3),
        "drifted": round(drifted, 2),
    }


def drift(L: Layout, track: np.ndarray, emb: np.ndarray) -> tuple[float, float]:
    """Median first-to-last-window centroid cosine of tracks shown in >= 3 windows, and the share that drifted."""
    ok = track >= 0
    df = pd.DataFrame({"t": track[ok], "w": L.win[ok], "p": L.pid[ok]})
    cent = {}
    for key, g in df.groupby(["t", "w"]):
        c = emb[g["p"].to_numpy()].mean(0)
        cent[key] = c / np.linalg.norm(c)
    between = []
    for w in range(L.n_windows):
        cs = np.array([c for (t, ww), c in cent.items() if ww == w])
        if len(cs) > 1:
            sim = cs @ cs.T
            between.append(sim[np.triu_indices(len(cs), 1)])
    base = float(np.median(np.concatenate(between)))
    pres = df.groupby("t")["w"].agg(["min", "max", "nunique"])
    long = pres[pres["nunique"] >= 3]
    cos = np.array([cent[t, r["min"]] @ cent[t, r["max"]] for t, r in long.iterrows()])
    return float(np.median(cos)), float((cos < base).mean())


def switch_breakdown(L: Layout, track: np.ndarray) -> str:
    border = merged = scattered = 0
    for w in range(L.n_windows - 1):
        a, b = L.sl[w], L.sl[w + 1]
        _, ia, ib = np.intersect1d(L.pid[a], L.pid[b], assume_unique=True, return_indices=True)
        ta, tb = track[a][ia], track[b][ib]
        alive = set(track[b][track[b] >= 0])
        sw = (ta >= 0) & (tb >= 0) & (ta != tb)
        for old in ta[sw]:
            if old in alive:
                border += 1
                continue
            dest = tb[(ta == old) & (tb >= 0)]
            if np.bincount(dest).max() / len(dest) > 0.5:
                merged += 1
            else:
                scattered += 1
    n = border + merged + scattered
    return (f"{n:,} switches: old region continues {border / n:.0%}, old track ended and its people mostly "
            f"joined one region {merged / n:.0%}, ended and scattered {scattered / n:.0%}")  # fmt: skip


def closest(options: dict[int, float], target: float) -> int:
    return min(options, key=lambda m: (abs(options[m] - target), m))


def track_summary(L: Layout, track: np.ndarray, people: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """One row per track: windows, size, birth years, top roles, best-known members."""
    df = pd.DataFrame({"w": L.win, "t": track, "qid": L.qids[L.pid]}).query("t >= 0")
    df = df.merge(people[["qid", "label", "rank", "birth_year_est", "role"]], on="qid")
    rows = []
    for t, g in df.groupby("t"):
        u = g.drop_duplicates("qid").sort_values("rank")
        roles = u["role"][~u["role"].isin(["other", "unknown"])].value_counts(normalize=True)
        rows.append({
            "track": t,
            "windows": f"{g.w.min()}-{g.w.max()}",
            "people": len(u),
            "born": f"{int(u.birth_year_est.quantile(0.1))}..{int(u.birth_year_est.quantile(0.9))}",
            "roles": ", ".join(f"{r} {p:.0%}" for r, p in roles.head(2).items()),
            "best known": "; ".join(u.label.astype(str).head(n)),
        })  # fmt: skip
    return pd.DataFrame(rows)


def provisional_names(L: Layout, track: np.ndarray, people: pd.DataFrame, prefix: str) -> dict[int, str]:
    s = track_summary(L, track, people, n=1)
    out = {}
    for r in s.itertuples():
        role = r.roles.split(",")[0].rsplit(" ", 1)[0] if r.roles else "?"
        out[r.track] = f"{prefix}{r.track} {role}"
    return out


def write_html(L: Layout, lay: pd.DataFrame, people: pd.DataFrame, coarse: np.ndarray, mid: np.ndarray,
               title: str, out: Path) -> None:  # fmt: skip
    import plotly.express as px
    import plotly.graph_objects as go

    palette = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    color = np.where(coarse >= 0, np.array(palette, dtype=object)[np.maximum(coarse, 0) % len(palette)], "#d9d9d9")
    p = people.set_index("qid")
    names = p["label"].reindex(L.qids).fillna(pd.Series(L.qids, index=L.qids)).astype(str).to_numpy()
    years = p["birth_year_est"].reindex(L.qids).to_numpy()
    person = np.array([f"{n} ({y})" for n, y in zip(names, years)], dtype=object)[L.pid]
    hover = np.array([f"{h}<br>C{c} / m{m}" for h, c, m in zip(person, coarse, mid)], dtype=object)
    cname = provisional_names(L, coarse, people, "C")
    mname = provisional_names(L, mid, people, "m")
    ids = np.asarray(L.qids[L.pid], dtype=str)

    def label_trace(track, s, names_, size, weight):
        t = track[s]
        df = pd.DataFrame({"t": t, "x": L.xy[s, 0], "y": L.xy[s, 1]}).query("t >= 0").groupby("t").mean()
        return go.Scatter(x=df.x.round(3), y=df.y.round(3), ids=[f"L{size}_{i}" for i in df.index], mode="text",
                          text=[names_[i] for i in df.index], hoverinfo="skip", showlegend=False,
                          textfont={"size": size, "color": "#111", "weight": weight})  # fmt: skip

    frames, frame_names = [], []
    spans = people.set_index("qid")["birth_year_est"]
    for w, s in enumerate(L.sl):
        yrs = spans.reindex(L.qids[L.pid[s]])
        name = f"w{w}: born {int(yrs.min())} to {int(yrs.max())}"
        pts = go.Scatter(x=L.xy[s, 0].round(3), y=L.xy[s, 1].round(3), ids=ids[s], mode="markers",
                         marker={"size": 4, "color": color[s], "opacity": 0.7}, hovertext=hover[s],
                         hoverinfo="text", showlegend=False)  # fmt: skip
        data = [pts, label_trace(mid, s, mname, 9, "normal"), label_trace(coarse, s, cname, 14, "bold")]
        frames.append(go.Frame(data=data, name=name))
        frame_names.append(name)
    step = {"frame": {"duration": 900, "redraw": False}, "transition": {"duration": 700}, "mode": "immediate"}
    pad = 0.03 * (lay.x.max() - lay.x.min())
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=title,
            height=950,
            xaxis={"range": [lay.x.min() - pad, lay.x.max() + pad], "showticklabels": False},
            yaxis={"range": [lay.y.min() - pad, lay.y.max() + pad], "showticklabels": False, "scaleanchor": "x"},
            updatemenus=[
                {
                    "type": "buttons",
                    "x": 0.0,
                    "y": -0.06,
                    "xanchor": "left",
                    "buttons": [
                        {"label": "Play", "method": "animate", "args": [None, {**step, "fromcurrent": True}]},
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "x": 0.12,
                    "len": 0.88,
                    "y": -0.02,
                    "currentvalue": {"prefix": ""},
                    "steps": [
                        {"label": n.split(":")[0], "method": "animate", "args": [[n], step]} for n in frame_names
                    ],
                }
            ],
        ),  # fmt: skip
    )
    fig.write_html(out, include_plotlyjs="cdn", auto_play=False)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 08's layout (rows in window order) and the roster with a role keyword per person."""
    lay = pd.read_parquet(config.layout_parquet())
    assert (np.diff(lay["window"].to_numpy()) >= 0).all(), "layout rows must be in window order"
    cols = ["qid", "label", "description", "rank", "birth_year_est"]
    people = pd.read_parquet(config.MAP_PEOPLE_PARQUET, columns=cols)
    leads = pd.read_parquet(config.LEADS_PARQUET, columns=["qid", "enwiki_description"])
    people = people.merge(leads, on="qid")
    short = people["enwiki_description"]
    people["role"] = short.where(short.notna(), people["description"].astype(object)).map(ev.role_of)
    return lay, people


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    lay, people = load()
    L = Layout(lay)
    z = np.load(config.embeddings_npz(config.MAP_EMBEDDING))
    row = pd.Series(np.arange(len(z["qid"])), index=z["qid"].astype(str))
    emb = z["vectors"][row.reindex(np.asarray(L.qids, dtype=str)).to_numpy()]  # rows in L.pid order
    print(f"{len(lay):,} points, {L.n_windows} windows; targets (labels/window) {TARGETS}; min_samples {MIN_SAMPLES}")

    # A: per-window trees once, then a grid of min cluster sizes
    links2 = [linkage(L.xy[s]) for s in L.sl]
    grid2 = sorted({int(v) for v in np.geomspace(5, 800, 70)})
    local = {m: per_window_labels(L, links2, m) for m in grid2}
    mean2 = {m: float(np.mean([lab.max() + 1 for lab in local[m]])) for m in grid2}
    m2 = {layer: closest(mean2, t) for layer, t in TARGETS.items()}
    print(f"A min cluster size per layer: {m2}")

    rows, tracks = [], {}
    for layer, m in m2.items():
        for name, bridge in [("A", False), ("A+", True)]:
            t = link_tracks(L, local[m], bridge)
            tracks[name, layer] = t
            rows.append({"approach": name, "layer": layer, "min size": m} | metrics(L, t, emb))

    grid3 = sorted({int(v) for v in np.geomspace(20, 20000, 80)})
    reach = {}
    for lam in LAMBDAS:
        link3 = linkage(np.c_[L.xy, lam * L.win])
        raw = {m: leaf_labels(link3, m, len(L.win)) for m in grid3}
        for layer, t in TARGETS.items():
            disp = {m: displayed(L, raw[m], m2[layer]) for m in grid3}
            mean3 = {m: float(shown_per_window(L, d).mean()) for m, d in disp.items()}
            m3 = closest(mean3, t)
            reach[f"B {lam}", layer] = round(max(mean3.values()), 1)
            tracks[f"B {lam}", layer] = disp[m3]
            rows.append({"approach": f"B {lam}", "layer": layer, "min size": m3} | metrics(L, disp[m3], emb))
    table = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(TARGETS)}
    table = table.sort_values(["layer", "approach"], key=lambda c: c.map(order) if c.name == "layer" else c)
    print("\n== Tracks by approach and layer ==")
    print("(min size: 2-d per window for A, 3-d over the stack for B; a B label is shown where it has >= A's size)")
    print(table.to_string(index=False))
    print(f"\nMost labels/window B can show at each layer's size bar: {reach}")

    print("\n== Why A+ labels switch ==")
    for layer in TARGETS:
        print(f"{layer}: {switch_breakdown(L, tracks['A+', layer])}")

    for key in [("A+", "coarse"), (f"B {PREVIEW_LAMBDA}", "coarse")]:
        print(f"\n== Coarse tracks, {key[0]} ==")
        print(track_summary(L, tracks[key], people).to_string(index=False))

    out_dir = config.DATA_DIR / "scoping"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_html(L, lay, people, tracks["A+", "coarse"], tracks["A+", "mid"],
               "A+: per-window clusters linked into tracks (coarse bold, mid small; colour = coarse track)",
               out_dir / "label_tracks_A.html")  # fmt: skip
    b = f"B {PREVIEW_LAMBDA}"
    write_html(L, lay, people, tracks[b, "coarse"], tracks[b, "mid"],
               f"B: one 3-d clustering of the stacked windows, lambda {PREVIEW_LAMBDA} (coarse bold, mid small)",
               out_dir / "label_tracks_B.html")  # fmt: skip


if __name__ == "__main__":
    main()

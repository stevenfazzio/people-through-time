"""People per label in each slider window: per-window tracks vs one 3-d clustering (one-off, 2026-11-08).

Companion to label_tracks.py, with the same tracks at the sizes it calibrated
(scoping/results/label_tracks.txt): A = per-window 2-d clusters (A and A+ have
the same clusters; bridging only changes track ids), B = the 3-d clustering of
the stacked windows at lambda 0.04, the one in its preview, showing a label only
where its track has at least A's min cluster size in that window.

data/scoping/cluster_sizes.html: one panel per layer, one split violin per
window (A left half, B right half), every cluster also drawn as a point (hover
for its provisional name). Each half's width scales with its number of
labels, within a layer, so B's halves are thinner where it shows fewer. Log
scale, so the density is of log size. The dashed line is the layer's size bar (A's min cluster size). The x labels give
each window's median birth year.

Summary on stdout -> scoping/results/cluster_sizes.txt: size quantiles per
approach and layer, how much of a window its largest label covers, and the
biggest clusters.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_tracks as lt  # noqa: E402

A_SIZES = {"fine": 39, "mid": 76, "coarse": 158}  # 2-d min cluster size per window, from label_tracks.txt
B_SIZES = {"fine": 163, "mid": 358, "coarse": 1218}  # 3-d min cluster size at LAMBDA, from label_tracks.txt
LAMBDA = 0.04
COLORS = {"A": "#3b6fb6", "B": "#d4553a"}
TICKS = [10, 20, 50, 100, 200, 500, 1000, 2000]


def slice_sizes(L: lt.Layout, track: np.ndarray, names: dict[int, str]) -> pd.DataFrame:
    """One row per (window, shown track): its size and the window's size."""
    df = pd.DataFrame({"w": L.win, "t": track}).query("t >= 0")
    s = df.groupby(["w", "t"]).size().rename("size").reset_index()
    s["name"] = s["t"].map(names)
    s["window_n"] = np.diff(L.bounds)[s["w"]]
    return s


def summary(sizes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (approach, layer), g in sizes.groupby(["approach", "layer"], sort=False):
        top = g.loc[g.groupby("w")["size"].idxmax()]
        share = top["size"] / top["window_n"]
        q = g["size"].quantile([0.1, 0.5, 0.9])
        rows.append({
            "approach": approach,
            "layer": layer,
            "labels": len(g),
            "size p10": int(q[0.1]),
            "median": int(q[0.5]),
            "p90": int(q[0.9]),
            "max": int(g["size"].max()),
            "p90/p10": round(q[0.9] / q[0.1], 1),
            "largest label / window: median": round(float(share.median()), 2),
            "largest label / window: max": round(float(share.max()), 2),
        })  # fmt: skip
    return pd.DataFrame(rows)


def build(sizes: pd.DataFrame, tick_text: list[str], out: Path) -> None:
    layers = list(A_SIZES)
    titles = [f"{layer} layer (~{lt.TARGETS[layer]} labels per window)" for layer in layers]
    fig = make_subplots(rows=len(layers), cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=titles)
    for r, layer in enumerate(layers, 1):
        for approach, side in [("A", "negative"), ("B", "positive")]:
            g = sizes[(sizes["layer"] == layer) & (sizes["approach"] == approach)]
            fig.add_trace(
                go.Violin(
                    x=g["w"],
                    y=np.log10(g["size"]),
                    side=side,
                    name="A: per-window clusters" if approach == "A" else f"B: one 3-d clustering (lambda {LAMBDA})",
                    legendgroup=approach,
                    showlegend=r == 1,
                    line={"color": COLORS[approach], "width": 1},
                    fillcolor=COLORS[approach],
                    opacity=0.55,
                    spanmode="hard",
                    scalegroup=layer,
                    scalemode="count",
                    width=0.9,
                    meanline={"visible": False},
                    box={"visible": False},
                    points="all",
                    pointpos=-0.45 if side == "negative" else 0.45,
                    jitter=0.25,
                    marker={"size": 3, "color": COLORS[approach], "opacity": 0.8},
                    hovertext=[
                        f"w{w} {n}: {s:,} people ({s / wn:.0%} of the window)"
                        for w, n, s, wn in zip(g["w"], g["name"], g["size"], g["window_n"])
                    ],  # fmt: skip
                    hoveron="points",
                    hoverinfo="text",
                ),
                row=r,
                col=1,
            )
        fig.add_hline(y=np.log10(A_SIZES[layer]), line={"dash": "dash", "color": "#555", "width": 1}, row=r, col=1)
        top = np.log10(sizes.loc[sizes["layer"] == layer, "size"].max())
        fig.update_yaxes(title_text="people per label", tickvals=np.log10(TICKS), ticktext=[f"{t:,}" for t in TICKS],
                         range=[np.log10(A_SIZES[layer]) - 0.08, top + 0.05], row=r, col=1)  # fmt: skip
    fig.update_xaxes(tickvals=list(range(len(tick_text))), ticktext=tick_text, tickfont={"size": 9})
    fig.update_xaxes(title_text="slider window (median birth year)", row=len(layers), col=1)
    fig.update_layout(
        title="People per label in each slider window: per-window clusters (A, left) vs one 3-d clustering (B, right)",
        violinmode="overlay",
        violingap=0,
        height=1350,
        width=1700,
        template="plotly_white",
        legend={"orientation": "h", "y": 1.04, "x": 0},
    )
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"\nWrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    pd.set_option("display.width", 250)
    lay, people = lt.load()
    L = lt.Layout(lay)
    links2 = [lt.linkage(L.xy[s]) for s in L.sl]
    link3 = lt.linkage(np.c_[L.xy, LAMBDA * L.win])
    parts = []
    for layer in A_SIZES:
        a = lt.link_tracks(L, lt.per_window_labels(L, links2, A_SIZES[layer]), bridge=True)
        b = lt.displayed(L, lt.leaf_labels(link3, B_SIZES[layer], len(L.win)), A_SIZES[layer])
        for approach, track in [("A", a), ("B", b)]:
            names = lt.provisional_names(L, track, people, approach)
            parts.append(slice_sizes(L, track, names).assign(approach=approach, layer=layer))
    sizes = pd.concat(parts, ignore_index=True)
    per_window = sizes.groupby(["approach", "layer"], sort=False)["w"].value_counts().groupby(level=[0, 1]).mean()
    print("Labels per window (mean), as in label_tracks.txt:", per_window.round(1).to_dict())

    print("\n== People per label (all windows) ==")
    print(summary(sizes).to_string(index=False))
    print("\n== Biggest labels per approach and layer ==")
    for (approach, layer), g in sizes.groupby(["approach", "layer"], sort=False):
        top = g.nlargest(3, "size")
        print(f"{approach} {layer:6}: " + "; ".join(f"w{r.w} {r.name} {r.size:,} ({r.size / r.window_n:.0%})"
                                                    for r in top.itertuples()))  # fmt: skip

    years = people.set_index("qid")["birth_year_est"]
    med = [int(years.reindex(L.qids[L.pid[s]]).median()) for s in L.sl]
    tick_text = [f"{w}<br>{y}" for w, y in enumerate(med)]
    out_dir = lt.config.DATA_DIR / "scoping"
    build(sizes, tick_text, out_dir / "cluster_sizes.html")


if __name__ == "__main__":
    main()

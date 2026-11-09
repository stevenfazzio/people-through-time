"""How many people each track holds at each slider step: stacked areas, per-window tracks vs one 3-d clustering
(one-off, 2026-11-08).

Same tracks as label_tracks.py and cluster_sizes.py, at the coarse layer (~10 labels per window): A+ = per-window
2-d clusters linked across windows (one-window gaps bridged), B = one 3-d clustering of the stacked windows at
lambda 0.04, a label shown only where its track has at least A's min cluster size.

data/scoping/track_sizes.html: one panel per approach, x = slider window, y = people. Each coloured band is one
track; its thickness at a window is how many of that window's people it holds. Grey at the bottom = people with
no label there, so each column adds up to the whole window (~4,300). Bands are stacked in order of first
appearance (oldest at the bottom), so a band that ends lets everything above it drop. Bands are named (track id +
most common role keyword) at their peak if they ever hold PEAK_LABEL people or more; hover for the rest.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_sizes as cs  # noqa: E402
import label_tracks as lt  # noqa: E402

LAYER = "coarse"
PEAK_LABEL = 350
UNLABELLED = "#d4d4d4"


def size_table(L: lt.Layout, track: np.ndarray) -> pd.DataFrame:
    """Rows = tracks in order of first appearance (then by peak size), columns = windows, values = people."""
    df = pd.DataFrame({"w": L.win, "t": track}).query("t >= 0")
    tab = df.groupby(["t", "w"]).size().unstack(fill_value=0).reindex(columns=range(L.n_windows), fill_value=0)
    first = (tab > 0).idxmax(axis=1)
    order = pd.DataFrame({"first": first, "peak": tab.max(axis=1)})
    return tab.loc[order.sort_values(["first", "peak"], ascending=[True, False]).index]


def hover_text(name: str, w: int, year: int, n: int, window_n: int) -> str:
    return f"{name}<br>window {w} (median birth year {year}): {n:,} people ({n / window_n:.0%})"


def add_panel(fig, row: int, L: lt.Layout, track: np.ndarray, names: dict[int, str], med_year: list[int]) -> None:
    """One stacked-area panel. A track's trace covers only the windows it holds people in (the stack fills the
    rest with zeros, stackgaps "infer zero"). The bands themselves take no hover: plotly would offer one on those
    filled-in zeros too. Hover comes from invisible markers at the middle of each real band segment instead."""
    tab = size_table(L, track)
    windows = np.arange(L.n_windows)
    window_n = np.diff(L.bounds)
    unlabelled = window_n - tab.sum(axis=0).to_numpy()
    group = f"g{row}"
    fig.add_trace(
        go.Scatter(x=windows, y=unlabelled, name="unlabelled", stackgroup=group, stackgaps="infer zero",
                   mode="lines", line={"width": 0.5, "color": UNLABELLED}, fillcolor=UNLABELLED,
                   legendgroup="unlabelled", showlegend=row == 1, hoverinfo="skip"),
        row=row, col=1,
    )  # fmt: skip
    hx = list(windows)
    hy = list(unlabelled / 2)
    ht = [hover_text("unlabelled", w, med_year[w], unlabelled[w], window_n[w]) for w in windows]
    palette = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    base = unlabelled.astype(float)
    for i, (t, sizes) in enumerate(tab.iterrows()):
        y = sizes.to_numpy()
        on = np.flatnonzero(y > 0)
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(x=on, y=y[on], name=names[t], stackgroup=group, mode="lines", showlegend=False,
                       line={"width": 0.5, "color": "white"}, fillcolor=color, hoverinfo="skip"),
            row=row, col=1,
        )  # fmt: skip
        hx += list(on)
        hy += list(base[on] + y[on] / 2)
        ht += [hover_text(names[t], w, med_year[w], y[w], window_n[w]) for w in on]
        if y.max() >= PEAK_LABEL:
            w = int(y.argmax())
            fig.add_annotation(x=w, y=base[w] + y[w] / 2, text=names[t], showarrow=False, font={"size": 9},
                               bgcolor="rgba(255,255,255,0.6)", row=row, col=1)  # fmt: skip
        base += y
    fig.add_trace(
        go.Scatter(x=hx, y=hy, mode="markers", marker={"size": 14, "opacity": 0}, text=ht,
                   hovertemplate="%{text}<extra></extra>", showlegend=False),
        row=row, col=1,
    )  # fmt: skip


def main() -> None:
    lay, people = lt.load()
    L = lt.Layout(lay)
    links2 = [lt.linkage(L.xy[s]) for s in L.sl]
    link3 = lt.linkage(np.c_[L.xy, cs.LAMBDA * L.win])
    size_a, size_b = cs.A_SIZES[LAYER], cs.B_SIZES[LAYER]
    tracks = {
        "A": lt.link_tracks(L, lt.per_window_labels(L, links2, size_a), bridge=True),
        "B": lt.displayed(L, lt.leaf_labels(link3, size_b, len(L.win)), size_a),
    }
    years = people.set_index("qid")["birth_year_est"]
    med = [int(years.reindex(L.qids[L.pid[s]]).median()) for s in L.sl]
    tick_text = [f"{w}<br>{y}" for w, y in enumerate(med)]
    n = {k: len(np.unique(t[t >= 0])) for k, t in tracks.items()}
    titles = [
        f"A: each window clustered on its own, clusters linked across windows ({n['A']} tracks)",
        f"B: one clustering of all windows stacked in 3-d, lambda {cs.LAMBDA} ({n['B']} tracks)",
    ]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, subplot_titles=titles)
    for row, (approach, track) in enumerate(tracks.items(), 1):
        names = lt.provisional_names(L, track, people, approach)
        add_panel(fig, row, L, track, names, med)
        v = size_table(L, track).to_numpy()
        unl = 1 - v.sum(0).mean() / np.diff(L.bounds).mean()
        print(f"{approach}: {len(v)} tracks; people per track per window: median {int(np.median(v[v > 0]))}, "
              f"max {v.max():,}; unlabelled {unl:.0%} of a window on average")  # fmt: skip
    fig.update_xaxes(tickvals=list(range(L.n_windows)), ticktext=tick_text, tickfont={"size": 9}, range=[-0.3, 31.3])
    fig.update_xaxes(title_text="slider window (median birth year)", row=2, col=1)
    fig.update_yaxes(title_text="people in the window")
    fig.update_layout(
        title=f"People per track at each slider step, {LAYER} layer (~{lt.TARGETS[LAYER]} labels per window)",
        height=1300,
        width=1700,
        template="plotly_white",
        hovermode="closest",
        legend={"orientation": "h", "y": 1.03, "x": 0},
    )
    out = lt.config.DATA_DIR / "scoping" / "track_sizes.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

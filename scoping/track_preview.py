"""Animated preview of stage 09's label tracks, for judging label flicker (one-off, 2026-11-08).

Replaces the animations from label_tracks.py (data/scoping/label_tracks_{A,B}.html) for this purpose. Those
added a label to a frame only while its track existed, and plotly.js (4.1.1, checked in the browser) creates
a text element that enters a frame at (0,0) -- the plot's upper-left corner -- and animates it from there, so
every new label flew in from the corner. Here every track of a layer is in every frame, with the same id:
where a track is absent its text is a single space (plotly deletes a text element whose text is empty, and
recreates it at the corner) and it waits at the nearest window's position, so a label pops in and out in place
and glides with its points while it exists.

Points: coloured by coarse track (grey = unlabelled at the coarse layer). Labels: coarse bold, mid small,
named by track number + most common role keyword (placeholders until Toponymy). The box at the top left lists
the coarse labels that appeared and disappeared since the previous window. Hover a point for the person.

Input: stage 09's tracks and stage 08's layout. Output: data/scoping/track_preview.html
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_tracks as lt  # noqa: E402

config = lt.config
COARSE, MID = 2, 1  # stage 09 layers, 0 = finest
GREY = "#d9d9d9"


def label_positions(win: np.ndarray, xy: np.ndarray, track: np.ndarray, n_windows: int):
    """Per (track, window): centroid x, y, and whether the track is there; absent windows take the nearest
    present window's position (carried forward, then backward)."""
    ok = track >= 0
    df = pd.DataFrame({"t": track[ok], "w": win[ok], "x": xy[ok, 0], "y": xy[ok, 1]})
    cent = df.groupby(["t", "w"])[["x", "y"]].mean()
    tracks = np.unique(df["t"])
    cols = pd.Index(range(n_windows), name="w")
    px_ = cent["x"].unstack().reindex(index=tracks, columns=cols)
    py_ = cent["y"].unstack().reindex(index=tracks, columns=cols)
    present = px_.notna()
    px_ = px_.ffill(axis=1).bfill(axis=1)
    py_ = py_.ffill(axis=1).bfill(axis=1)
    return tracks, px_.to_numpy(), py_.to_numpy(), present.to_numpy()


def change_note(names: dict[int, str], tracks: np.ndarray, present: np.ndarray, w: int) -> str:
    if w == 0:
        return "window 0"
    came = [names[t] for t in tracks[present[:, w] & ~present[:, w - 1]]]
    went = [names[t] for t in tracks[~present[:, w] & present[:, w - 1]]]
    return (f"coarse labels since window {w - 1}:<br>appeared: {', '.join(came) or 'none'}"
            f"<br>disappeared: {', '.join(went) or 'none'}")  # fmt: skip


def main() -> None:
    lay, people = lt.load()
    tr = pd.read_parquet(config.tracks_parquet())
    if not (tr["qid"].to_numpy() == lay["qid"].to_numpy()).all():
        raise SystemExit("tracks are not row-aligned with the layout -- rerun stage 09")
    L = lt.Layout(lay)
    coarse = tr[f"track_{COARSE}"].to_numpy()
    mid = tr[f"track_{MID}"].to_numpy()
    names = {"C": lt.provisional_names(L, coarse, people, "C"), "m": lt.provisional_names(L, mid, people, "m")}
    pos = {"C": label_positions(L.win, L.xy, coarse, L.n_windows), "m": label_positions(L.win, L.xy, mid, L.n_windows)}

    palette = np.array(px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24)
    color = np.where(coarse >= 0, palette[np.maximum(coarse, 0) % len(palette)], GREY)
    p = people.set_index("qid")
    who = np.array([f"{p.at[q, 'label']} ({p.at[q, 'birth_year_est']})" for q in L.qids], dtype=object)[L.pid]
    hover = np.array([f"{h}<br>C{c} / m{m}" for h, c, m in zip(who, coarse, mid)], dtype=object)
    ids = np.asarray(L.qids[L.pid], dtype=str)
    years = p["birth_year_est"]
    x0, x1 = lay["x"].min(), lay["x"].max()
    y0, y1 = lay["y"].min(), lay["y"].max()
    pad = 0.03 * (x1 - x0)

    def labels(key: str, w: int, size: int, weight: str) -> go.Scatter:
        tracks, lx, ly, present = pos[key]
        text = [names[key][t] if present[i, w] else " " for i, t in enumerate(tracks)]
        return go.Scatter(x=lx[:, w].round(3), y=ly[:, w].round(3), ids=[f"{key}{t}" for t in tracks], text=text,
                          mode="text", hoverinfo="skip", showlegend=False,
                          textfont={"size": size, "color": "#111", "weight": weight})  # fmt: skip

    frames, frame_names = [], []
    for w, s in enumerate(L.sl):
        yrs = years.reindex(L.qids[L.pid[s]])
        name = f"w{w}: born {int(yrs.min())} to {int(yrs.max())}"
        pts = go.Scatter(x=L.xy[s, 0].round(3), y=L.xy[s, 1].round(3), ids=ids[s], mode="markers",
                         marker={"size": 4, "color": color[s], "opacity": 0.7}, hovertext=hover[s],
                         hoverinfo="text", showlegend=False)  # fmt: skip
        note = go.Scatter(x=[x0 - pad], y=[y1 + pad], ids=["note"], mode="text", textposition="bottom right",
                          text=[change_note(names["C"], pos["C"][0], pos["C"][3], w)], hoverinfo="skip",
                          showlegend=False, textfont={"size": 11, "color": "#444"})  # fmt: skip
        frames.append(go.Frame(data=[pts, labels("m", w, 9, "normal"), labels("C", w, 14, "bold"), note], name=name))
        frame_names.append(name)

    step = {"frame": {"duration": 900, "redraw": False}, "transition": {"duration": 700}, "mode": "immediate"}
    play = {"label": "Play", "method": "animate", "args": [None, {**step, "fromcurrent": True}]}
    pause = {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title="Stage 09 label tracks (coarse bold, mid small; colour = coarse track; names are placeholders)",
            height=950,
            xaxis={"range": [x0 - pad, x1 + pad], "showticklabels": False},
            yaxis={"range": [y0 - pad, y1 + 2 * pad], "showticklabels": False, "scaleanchor": "x"},
            updatemenus=[{"type": "buttons", "x": 0.0, "y": -0.06, "xanchor": "left", "buttons": [play, pause]}],
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
            ],  # fmt: skip
        ),
    )
    out = config.DATA_DIR / "scoping" / "track_preview.html"
    fig.write_html(out, include_plotlyjs="cdn", auto_play=False)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

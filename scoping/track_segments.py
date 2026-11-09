"""Stage 09's tracks named whole vs cut into segments where their content drifts (one-off, 2026-11-08).

Naming files each person under their most common track, so a track is named for what it holds most of the
time. A track whose content drifts (a coarse track that starts with Mozart, Beethoven and Bach and ends with film
actors) would be named for its majority, and its early windows would carry the wrong name. Cutting it into
segments, each named on its own, fixes that at the cost of the label's text changing at the cut.

Drift, per layer, in the stage 07b vectors (the space the map is laid out from): a (track, window) slice's
centroid is the unit mean of its people's vectors. The bar is the median cosine between two *different*
tracks' centroids in the same window: content that has moved that far is as different as two separate regions.
Walking a track's windows in order, each segment is anchored at its first window's centroid (the measure from
label_tracks.py's drift column, applied as we go); the track is cut at window w when the slices at w and at the
next window both fall below the bar against the anchor (one window below it is treated as noise, so no segment
is shorter than two windows except at a track's start). A running centroid of the whole segment was tried first
and cut nothing: it follows gradual drift instead of measuring it.

Nothing here is named by an LLM yet: each version is described by what a name would summarize -- its windows,
birth years, most common role keywords and best-known people.

Output: scoping/results/track_segments.txt (stdout), data/scoping/track_segments.html
"""

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_tracks as lt  # noqa: E402

config = lt.config
LAYER_NAMES = {0: "fine", 1: "mid", 2: "coarse"}


def slice_centroids(rows: pd.DataFrame, emb: np.ndarray) -> dict[tuple[int, int], tuple[np.ndarray, int]]:
    """(track, window) -> (sum of its people's vectors, count)."""
    out = {}
    for key, g in rows.groupby(["t", "w"]):
        v = emb[g["e"].to_numpy()]
        out[key] = (v.sum(0), len(v))
    return out


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def between_track_bar(cents: dict, n_windows: int) -> float:
    """Median cosine between the centroids of two different tracks in the same window."""
    sims = []
    for w in range(n_windows):
        c = np.array([unit(s) for (t, ww), (s, n) in cents.items() if ww == w])
        if len(c) > 1:
            m = c @ c.T
            sims.append(m[np.triu_indices(len(c), 1)])
    return float(np.median(np.concatenate(sims)))


def segment(windows: list[int], cents: dict, t: int, bar: float) -> list[list[int]]:
    """Cut a track's windows where its slices fall below `bar` against the segment's first window, at the first
    of two windows in a row that do."""
    segs = [[windows[0]]]
    anchor = unit(cents[t, windows[0]][0])
    for i, w in enumerate(windows[1:], 1):
        here = unit(cents[t, w][0]) @ anchor < bar
        nxt = i + 1 < len(windows) and unit(cents[t, windows[i + 1]][0]) @ anchor < bar
        if here and nxt:
            segs.append([w])
            anchor = unit(cents[t, w][0])
        else:
            segs[-1].append(w)
    return segs


def describe(rows: pd.DataFrame, people: pd.DataFrame, n: int = 5) -> dict:
    u = people.loc[rows["qid"].unique()].sort_values("rank")
    roles = u["role"][~u["role"].isin(["other", "unknown"])].value_counts(normalize=True)
    w = rows["w"]
    return {
        "windows": f"{w.min()}–{w.max()}",
        "people": len(u),
        "born": f"{int(u['birth_year_est'].quantile(0.1))} to {int(u['birth_year_est'].quantile(0.9))}",
        "roles": ", ".join(f"{r} {p:.0%}" for r, p in roles.head(3).items()),
        "best known": "; ".join(u["label"].astype(str).head(n)),
    }


def page(sections: list[str], summary: str) -> str:
    style = """
    :root { --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#e3e3e6; --whole:#f4f4f6; --seg:#fff; --accent:#3b6fb6; }
    @media (prefers-color-scheme: dark) { :root { --bg:#161618; --fg:#ececef; --muted:#9a9aa1; --line:#2c2c30;
      --whole:#202024; --seg:#161618; --accent:#7aa5e8; } }
    body { background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system, system-ui, sans-serif;
      margin:0 auto; max-width:1180px; padding:24px 16px 64px; }
    h1 { font-size:20px; margin:0 0 6px; } h2 { font-size:16px; margin:32px 0 8px; }
    p.note { color:var(--muted); margin:4px 0 14px; max-width:860px; }
    table { border-collapse:collapse; width:100%; margin:10px 0 18px; table-layout:fixed; }
    th, td { text-align:left; vertical-align:top; padding:6px 8px; border-bottom:1px solid var(--line); }
    th { color:var(--muted); font-weight:500; font-size:12px; }
    tr.whole td { background:var(--whole); font-weight:500; }
    tr.whole td.tag { color:var(--accent); }
    tr.main td { border-bottom:none; padding-bottom:2px; }
    tr.who td { padding-top:0; color:var(--muted); font-size:13px; }
    td.tag { color:var(--muted); } col.t { width:96px; } col.w { width:70px; } col.n { width:60px; }
    col.b { width:120px; }
    """
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            f"<title>Track segments</title><style>{style}</style></head><body>"
            f"<h1>Tracks named whole vs cut where their content drifts</h1>{summary}{''.join(sections)}"
            f"</body></html>")  # fmt: skip


def track_table(label: str, whole: dict, segs: list[dict]) -> str:
    head = ("<table><colgroup><col class='t'><col class='w'><col class='n'><col class='b'><col></colgroup>"
            "<tr><th></th><th>windows</th><th>people</th><th>born (10–90%)</th><th>roles</th></tr>")  # fmt: skip

    def row(cls: str, tag: str, d: dict) -> str:
        cells = [d["windows"], str(d["people"]), d["born"], d["roles"]]
        main = f"<tr class='{cls} main'><td class='tag'>{html.escape(tag)}</td>" + "".join(
            f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>"  # fmt: skip
        who = f"<tr class='{cls} who'><td></td><td colspan='4'>Best known: {html.escape(d['best known'])}</td></tr>"
        return main + who

    body = row("whole", f"{label} whole", whole)
    body += "".join(row("seg", f"segment {i + 1}", d) for i, d in enumerate(segs))
    return head + body + "</table>"


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    lay, people = lt.load()
    tr = pd.read_parquet(config.tracks_parquet())
    z = np.load(config.embeddings_npz(config.MAP_EMBEDDING))
    row_of = pd.Series(np.arange(len(z["qid"])), index=z["qid"].astype(str))
    emb = z["vectors"].astype(np.float64)
    people = people.set_index("qid")
    n_windows = int(tr["window"].max()) + 1
    e = row_of.reindex(tr["qid"].astype(str)).to_numpy()

    sections, summary_rows = [], []
    for layer in (2, 1, 0):  # coarse first: the most visible labels
        t = tr[f"track_{layer}"].to_numpy()
        ok = t >= 0
        rows = pd.DataFrame({"t": t[ok], "w": tr["window"].to_numpy()[ok], "qid": tr["qid"].to_numpy()[ok], "e": e[ok]})
        cents = slice_centroids(rows, emb)
        bar = between_track_bar(cents, n_windows)
        cut = []
        for tid, g in rows.groupby("t"):
            windows = sorted(g["w"].unique())
            segs = segment(windows, cents, tid, bar)
            if len(segs) > 1:
                cut.append((tid, g, segs))
        n_tracks = rows["t"].nunique()
        n_segs = sum(len(s) for _, _, s in cut)
        summary_rows.append({"layer": f"{layer} ({LAYER_NAMES[layer]})", "tracks": n_tracks,
                             "bar (cosine)": round(bar, 3), "tracks cut": len(cut), "segments from them": n_segs,
                             "names needed": n_tracks - len(cut) + n_segs})  # fmt: skip
        print(f"\n== Layer {layer} ({LAYER_NAMES[layer]}): {len(cut)} of {n_tracks} tracks cut into {n_segs} segments "
              f"(bar: cosine {bar:.3f}) ==")  # fmt: skip
        tables = []
        for tid, g, segs in sorted(cut, key=lambda c: -c[1]["qid"].nunique()):
            whole = describe(g, people)
            parts = [describe(g[g["w"].isin(s)], people) for s in segs]
            label = f"{'fmc'[layer]}{tid}"
            print(f"{label} whole: {whole}")
            for i, d in enumerate(parts):
                print(f"   segment {i + 1}: {d}")
            tables.append(track_table(label, whole, parts))
        sections.append(f"<h2>Layer {layer} ({LAYER_NAMES[layer]}): {len(cut)} of {n_tracks} tracks cut into "
                        f"{n_segs} segments</h2>" + "".join(tables))  # fmt: skip

    summary = pd.DataFrame(summary_rows)
    print(f"\n{summary.to_string(index=False)}")
    note = (
        "<p class='note'>Named whole, a track gets one name for what it holds most of the time. Cut, each segment "
        "gets its own name and the label's text changes at the cut. A track is cut where its people's average "
        "embedding has drifted as far from where the segment began as two separate regions in the same window "
        "are from each other (the median cosine between different tracks), for two windows in a row. Shaded rows are "
        "the whole track; the rows under them are its segments. Track numbers are stage 09's (f = fine, m = mid, "
        "c = coarse). Roles are keywords from the short descriptions; nothing is LLM-named yet.</p>"
    )
    cells = "".join(f"<th>{html.escape(c)}</th>" for c in summary.columns)
    srows = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>" for r in summary.values)
    out = config.DATA_DIR / "scoping" / "track_segments.html"
    out.write_text(page(sections, note + f"<table><tr>{cells}</tr>{srows}</table>"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

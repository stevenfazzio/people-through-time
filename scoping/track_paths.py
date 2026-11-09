"""Each track's path of per-window names from stage 10, for reading before stage 11 (one-off, 2026-11-09).

Stage 11 will turn each path into one name for the whole track. This page shows the paths as stage 10 left
them: one block per track, coarse layer first, with the name Toponymy gave it in each window it appears in
(runs of windows with the same name merged), the window's median birth year, and the track's size there.

Output: data/scoping/track_paths.html
"""

import html
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

LAYER_NAMES = ("fine", "mid", "coarse")


def runs(path: pd.DataFrame) -> list[dict]:
    """Consecutive windows with the same name, merged."""
    out = []
    for r in path.itertuples():
        if out and out[-1]["name"] == r.name and out[-1]["last"] == r.window - 1:
            out[-1]["last"] = r.window
            out[-1]["size"] = max(out[-1]["size"], r.size)
        else:
            out.append({"first": r.window, "last": r.window, "name": r.name, "size": r.size})
    return out


def main() -> None:
    names = pd.read_parquet(config.window_names_parquet())
    final_path = config.track_names_parquet()
    final = pd.read_parquet(final_path).set_index(["layer", "track"]) if final_path.exists() else None
    tree = pd.read_parquet(config.track_tree_parquet()).set_index(["layer", "track"])
    tr = pd.read_parquet(config.tracks_parquet(), columns=["window", "qid"])
    years = pd.read_parquet(config.MAP_PEOPLE_PARQUET, columns=["qid", "birth_year_est"])
    med = tr.merge(years, on="qid").groupby("window")["birth_year_est"].median().round().astype(int)
    sections = []
    for layer in (2, 1, 0):
        part = names[names["layer"] == layer]
        tracks = part.groupby("track")["window"].min().sort_values().index
        blocks, distinct = [], []
        for t in tracks:
            path = part[part["track"] == t].sort_values("window")
            rs = runs(path)
            distinct.append(path["name"].nunique() / len(path))
            info = tree.loc[(layer, t)]
            name = ""
            if final is not None:
                fr = final.loc[(layer, t)]
                flag = " <span class='meta'>(renamed apart from a twin)</span>" if fr["renamed"] else ""
                name = f"<div class='final'>{html.escape(fr['name'])}{flag}</div>"
            items = "".join(
                f"<li><span class='w'>w{r['first']}{'' if r['first'] == r['last'] else '–' + str(r['last'])}"
                f" · {med[r['first']]}{'' if r['first'] == r['last'] else '–' + str(med[r['last']])}</span> "
                f"{html.escape(r['name'])} <span class='n'>({r['size']})</span></li>"
                for r in rs
            )
            blocks.append(f"<div class='track'>{name}<h3>{'fmc'[layer]}{t} <span class='meta'>windows "
                          f"{int(info['first_window'])}–{int(info['last_window'])}, {int(info['n_people']):,} people, "
                          f"{path['name'].nunique()} distinct names in {len(path)} windows</span></h3>"
                          f"<ul>{items}</ul></div>")  # fmt: skip
        summary = (f"{len(tracks)} tracks; distinct names per window shown: median "
                   f"{pd.Series(distinct).median():.0%}")  # fmt: skip
        open_ = " open" if layer == 2 else ""
        sections.append(f"<details{open_}><summary>Layer {layer} ({LAYER_NAMES[layer]}): {summary}</summary>"
                        f"{''.join(blocks)}</details>")  # fmt: skip
    style = """
    :root { --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#e3e3e6; --accent:#3b6fb6; }
    @media (prefers-color-scheme: dark) { :root { --bg:#161618; --fg:#ececef; --muted:#9a9aa1; --line:#2c2c30;
      --accent:#7aa5e8; } }
    body { background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system, system-ui, sans-serif;
      margin:0 auto; max-width:1100px; padding:24px 16px 64px; }
    h1 { font-size:20px; margin:0 0 6px; } p.note { color:var(--muted); margin:4px 0 16px; }
    summary { font-size:16px; font-weight:600; cursor:pointer; margin:18px 0 8px; }
    .track { border-top:1px solid var(--line); padding:8px 0 4px; }
    h3 { font-size:13px; margin:0 0 4px; color:var(--accent); font-weight:500; }
    .final { font-size:16px; font-weight:600; margin:2px 0; } .meta { color:var(--muted); font-weight:400; }
    ul { margin:0; padding-left:18px; } li { margin:1px 0; }
    .w { color:var(--muted); font-variant-numeric:tabular-nums; display:inline-block; min-width:130px; }
    .n { color:var(--muted); font-size:12px; }
    """
    page = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            f"<title>Track name paths</title><style>{style}</style></head><body>"
            f"<h1>Track names (stage 11) and their per-window paths (stage 10)</h1><p class='note'>Bold: the one "
            f"name stage 11 gave the track. Below it, the names Toponymy gave it in each window on its own. "
            f"Window number · median birth year, then the "
            f"name, then the track's size in that window (largest, for merged runs). Coarse layer first; tracks "
            f"in order of first appearance.</p>{''.join(sections)}</body></html>")  # fmt: skip
    out = config.DATA_DIR / "scoping" / "track_paths.html"
    out.write_text(page)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

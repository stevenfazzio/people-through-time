"""Opus 5.5 vs Sonnet 5.5 names for the same window, side by side (one-off, 2026-11-09).

Both runs come from scoping/name_window.py on the same clusters, keyphrases and prompts; only the model
differs. One row per cluster: its stage 09 track, size, each model's name and the best-known people.

Output: data/scoping/namer_comparison.html
"""

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

WINDOW = 16
LAYER_NAMES = ("fine", "mid", "coarse")


def main() -> None:
    runs = {m: json.loads((config.DATA_DIR / "scoping" / f"name_window_{WINDOW}_{m}.json").read_text())
            for m in ("opus", "sonnet")}  # fmt: skip
    lo, hi = runs["opus"]["born"]
    sections = []
    for i in (2, 1, 0):
        a = {e["track"]: e for e in runs["opus"]["layers"][i]}
        b = {e["track"]: e for e in runs["sonnet"]["layers"][i]}
        rows = []
        same = 0
        for t in sorted(a, key=lambda t: -a[t]["size"]):
            na, nb = a[t]["name"], b.get(t, {}).get("name", "")
            same += na == nb
            cls = " class='same'" if na == nb else ""
            rows.append(f"<tr{cls}><td class='t'>{'fmc'[i]}{t}</td><td class='n'>{a[t]['size']}</td>"
                        f"<td>{html.escape(na)}</td><td>{html.escape(nb)}</td>"
                        f"<td class='who'>{html.escape(a[t]['best_known'])}</td></tr>")  # fmt: skip
        wa = sorted(len(e["name"].split()) for e in a.values())
        wb = sorted(len(e["name"].split()) for e in b.values())
        sections.append(
            f"<h2>Layer {i} ({LAYER_NAMES[i]}): {len(a)} clusters</h2><p class='note'>Median words: Opus "
            f"{wa[len(wa) // 2]}, Sonnet {wb[len(wb) // 2]}; identical names: {same}.</p>"
            "<table><colgroup><col class='t'><col class='n'><col><col><col class='who'></colgroup>"
            "<tr><th>track</th><th>people</th><th>Opus 5.5</th><th>Sonnet 5.5</th><th>best known</th></tr>"
            + "".join(rows)
            + "</table>"
        )
    style = """
    :root { --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#e3e3e6; --same:#f4f4f6; }
    @media (prefers-color-scheme: dark) { :root { --bg:#161618; --fg:#ececef; --muted:#9a9aa1; --line:#2c2c30;
      --same:#202024; } }
    body { background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system, system-ui, sans-serif;
      margin:0 auto; max-width:1280px; padding:24px 16px 64px; }
    h1 { font-size:20px; margin:0 0 6px; } h2 { font-size:16px; margin:28px 0 4px; }
    p.note { color:var(--muted); margin:2px 0 10px; }
    table { border-collapse:collapse; width:100%; table-layout:fixed; }
    th, td { text-align:left; vertical-align:top; padding:6px 8px; border-bottom:1px solid var(--line); }
    th { color:var(--muted); font-weight:500; font-size:12px; }
    col.t { width:64px; } col.n { width:60px; } col.who { width:28%; }
    td.t, td.n, td.who { color:var(--muted); } tr.same td { background:var(--same); }
    """
    usage = []
    for m, log in (("Opus 5.5", "name_window_opus.txt"), ("Sonnet 5.5", "name_window_sonnet.txt")):
        path = config.REPO_ROOT / "scoping" / "results" / log
        line = (
            next((ln for ln in path.read_text().splitlines() if ln.startswith("Naming:")), "") if path.exists() else ""
        )
        usage.append(f"<li><b>{m}</b>: {html.escape(line.removeprefix('Naming: '))}</li>")
    page = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            f"<title>Namer comparison</title><style>{style}</style></head><body>"
            f"<h1>Window {WINDOW} (born {lo}–{hi}): Opus 5.5 vs Sonnet 5.5 names</h1>"
            f"<p class='note'>Same clusters, keyphrases and prompts; only the model differs. Shaded rows: the "
            f"two names are identical. Coarse layer first.</p><ul>{''.join(usage)}</ul>{''.join(sections)}"
            f"</body></html>")  # fmt: skip
    out = config.DATA_DIR / "scoping" / "namer_comparison.html"
    out.write_text(page)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

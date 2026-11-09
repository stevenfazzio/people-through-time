"""Stage 12: export what the page in docs/ loads.

Decided 2026-11-09 (CLAUDE.md, "Frontend"): a custom deck.gl page, not datamapplot (0.7.3 can only animate
colours, and its search and hover are bound to one fixed point set). A slider over the 32 windows moves the
map continuously -- people in both neighbouring windows glide between their two positions, the others fade in
or out where they stand -- and settles on a window when released. The slider is itself a chart: a ThemeRiver
of the coarse tracks, each ribbon as thick as the track's people in each window, named inside where it fits.

Written to config.FRONTEND_DATA_DIR:
- points.bin: for every (person, window), person by person in rank order and each person's windows in order:
  x, y as uint16 (2N values), then the coarse track as uint8 (N values; 255 = none). With
  NORMALIZE_WINDOWS each window is first centred and scaled to RMS radius 1 (a similarity transform per
  window: nobody's place in their window changes, but the layout's breathing between windows goes away).
- core.json, loaded with it: per person name, birth year and its label, bin, first window and window count
  (a person in bin b is in windows b-4..b, clipped), marker radius; per window its people's birth-year range;
  every track's name (and short name for the coarse layer), lifetime and colour; label centroids per (window,
  track) with the track's size there (the ribbons' thickness); the coarse ribbons' order; the palette.
- details.json, fetched after the first paint: short description, lead excerpt, thumbnail, article, the
  language a translated lead came from, death year, monthly views.

Colours: coarse tracks get palette colours by interval colouring with config.COLOR_BUFFER_WINDOWS of slack, so
no two tracks alive within a window of each other share one; the palette (glasbey, CVD-safe) has exactly as
many colours as the most such tracks at once. Ribbon order: one fixed order, so ribbons never cross; it starts
from the tracks' mean height on the map and moves tracks so those exchanging people between windows (merges,
splits) sit next to each other.

Inputs: stage 05 (people), 06 (leads), 04 (death precision), 08 (layout), 09 (tracks), 11 (names), 11b (short
names). Output: docs/data/{core.json, points.bin, details.json}; logs every file's raw and gzipped size.
"""

import gzip
import json
import re
import time
from urllib.parse import quote

import glasbey
import numpy as np
import pandas as pd

import config
from util import log, write_bytes_atomic

N_LAYERS = 3
COARSE = N_LAYERS - 1
NO_TRACK = 255
LANGUAGE_NAMES = {  # every edition stage 06 translated from (2026-11-08); others fall back to the code
    "ja": "Japanese", "zh": "Chinese", "ar": "Arabic", "ru": "Russian", "de": "German", "fa": "Persian",
    "fr": "French", "es": "Spanish", "tr": "Turkish", "pl": "Polish", "vi": "Vietnamese", "kk": "Kazakh",
    "ko": "Korean", "hi": "Hindi", "it": "Italian",
}  # fmt: skip
THUMB_HOSTS = {"T/": "https://thumb.wikimedia.org/wikipedia/", "U/": "https://upload.wikimedia.org/wikipedia/"}
ESTIMATED_SOURCES = {"death", "floruit", "work_start"}  # birth years stage 04 inferred from another date
YEAR_PRECISION = 9  # Wikidata: 9 = year, 8 = decade, 7 = century, ...

# --- text ---------------------------------------------------------------------------------------------------

PAREN = re.compile(r"\s*[(\[][^()\[\]]*[)\]]")
ABBREVIATIONS = {
    "c", "ca", "fl", "r", "d", "b", "St", "Ste", "Jr", "Sr", "Dr", "Mr", "Mrs", "Ms", "Gen", "Col", "Lt", "Capt",
    "Sgt", "Rev", "Fr", "Mt", "No", "vs", "Inc", "Ltd", "Co", "Bros", "Corp", "approx", "Hon", "Sen", "Rep",
    "Gov", "Pres", "Prof", "Adm", "Maj", "Cmdr", "Ft", "Ave", "Blvd",
}  # fmt: skip
SENTENCE_END = re.compile(r"[.!?][\"”’']?(?=\s+[A-Z0-9\"“‘])")


def strip_parentheticals(text: str) -> str:
    """Drop (...) and [...] groups, innermost first -- native names, pronunciations and dates in a lead's
    first sentence -- and tidy the spaces they leave before punctuation."""
    prev = None
    while prev != text:
        prev, text = text, PAREN.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    return re.sub(r"\s+([,.;:!?])", r"\1", text).strip()


def sentence_ends(text: str) -> list[int]:
    """Indices just past each sentence-ending punctuation mark, skipping abbreviations and initials."""
    ends = []
    for m in SENTENCE_END.finditer(text):
        if text[m.start()] == ".":
            word = re.search(r"(\S+)$", text[: m.start()])
            w = word.group(1).lstrip("(\"“'") if word else ""
            if w in ABBREVIATIONS or (len(w) == 1 and w.isalpha()) or re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", w):
                continue
        ends.append(m.end())
    return ends


def excerpt(lead: str, max_chars: int = config.EXCERPT_MAX_CHARS) -> str:
    """Whole sentences of the lead's first paragraph, parentheticals dropped, up to max_chars; a first sentence
    longer than that is cut at a word and ends in an ellipsis."""
    para = next((p for p in (lead or "").split("\n") if p.strip()), "")
    para = strip_parentheticals(para)
    if len(para) <= max_chars:
        return para
    fits = [e for e in sentence_ends(para) if e <= max_chars]
    if fits:
        return para[: fits[-1]].strip()
    return para[:max_chars].rsplit(" ", 1)[0].rstrip(",;:—–- ") + "…"


def format_year(year: int) -> str:
    """Astronomical year -> '356 BC', '10,546 BC', 'AD 451', '1879' (WDQS years are astronomical: 0 = 1 BC)."""
    year = int(year)
    if year <= 0:
        bc = 1 - year
        return f"{bc:,} BC" if bc >= 10_000 else f"{bc} BC"
    return f"AD {year}" if year < 1000 else str(year)


def born_label(year, precision, source) -> str:
    """'c. 356 BC' when Wikidata gives the birth at decade precision or coarser, 'c. 1450 (est.)' when stage 04
    inferred it from another date, the plain year otherwise."""
    text = format_year(year)
    if source in ESTIMATED_SOURCES:
        return f"c. {text} (est.)"
    if pd.notna(precision) and int(precision) < YEAR_PRECISION:
        return f"c. {text}"
    return text


def died_label(year, precision) -> str:
    if pd.isna(year):
        return ""
    text = format_year(year)
    return f"c. {text}" if pd.notna(precision) and int(precision) < YEAR_PRECISION else text


def display_name(label, enwiki_title, top_title) -> str:
    """Wikidata's label, else the enwiki title, else the top edition's title (CLAUDE.md "Missing labels")."""
    for name in (label, enwiki_title, top_title):
        if isinstance(name, str) and name.strip():
            return name.replace("_", " ").strip()
    return ""


def article_ref(site, title, qid: str) -> str:
    """'<lang>:<Title_with_underscores>' for the article the lead came from, or 'wd:<qid>' for the three leads
    that are Wikidata descriptions. The page builds the URL."""
    if isinstance(site, str) and isinstance(title, str) and site.endswith("wiki"):
        return f"{site[:-4].replace('_', '-')}:{title.replace(' ', '_')}"
    return f"wd:{qid}"


def article_url(ref: str) -> str:
    """The same URL the page builds from article_ref's output (used by tests and the log)."""
    prefix, title = ref.split(":", 1)
    if prefix == "wd":
        return f"https://www.wikidata.org/wiki/{title}"
    return f"https://{prefix}.wikipedia.org/wiki/{quote(title, safe='')}"


def thumb_code(url) -> str:
    """A thumbnail URL without its query string, its Wikimedia host shortened to 'T/' or 'U/'."""
    if not isinstance(url, str) or not url:
        return ""
    url = url.split("?", 1)[0]
    for code, host in THUMB_HOSTS.items():
        if url.startswith(host):
            return code + url[len(host) :]
    return url


def translated_from(source, site) -> str:
    if source != "translated" or not isinstance(site, str):
        return ""
    code = site[:-4].replace("_", "-")
    return LANGUAGE_NAMES.get(code, code)


# --- geometry -----------------------------------------------------------------------------------------------


def normalize_windows(lay: pd.DataFrame) -> pd.DataFrame:
    """Centre each window on its mean and scale it to RMS radius 1."""
    out = lay.copy()
    g = out.groupby("window")
    cx, cy = g["x"].transform("mean"), g["y"].transform("mean")
    dx, dy = out["x"] - cx, out["y"] - cy
    rms = np.sqrt((dx**2 + dy**2).groupby(out["window"]).transform("mean"))
    out["x"], out["y"] = (dx / rms).astype("float32"), (dy / rms).astype("float32")
    return out


def quantize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.round((values - lo) / (hi - lo) * 65535).astype("<u2")


def dequantize(q: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + q.astype(float) / 65535 * (hi - lo)


def person_windows(lay: pd.DataFrame, qids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """First window and window count per person (in qids order); each person's windows must be consecutive."""
    g = lay.groupby("qid")["window"].agg(["min", "max", "count"]).reindex(qids)
    if g["count"].isna().any():
        raise SystemExit(f"{int(g['count'].isna().sum())} roster people have no layout point")
    if not (g["max"] - g["min"] + 1 == g["count"]).all():
        raise SystemExit("a person's windows are not consecutive")
    return g["min"].to_numpy("uint8"), g["count"].to_numpy("uint8")


# --- colour and order -----------------------------------------------------------------------------------------


def interval_colors(first: np.ndarray, last: np.ndarray, buffer: int) -> np.ndarray:
    """Colour index per track such that no two tracks whose lifetimes, widened by `buffer` windows on each
    side, overlap share one. Greedy by start time uses exactly as many colours as the most such tracks at once
    (interval graphs are perfect), so the palette needs no more."""
    lo, hi = first - buffer, last + buffer
    color = np.full(len(first), -1)
    for i in np.lexsort((hi, lo)):
        taken = {color[j] for j in range(len(first)) if color[j] >= 0 and lo[j] <= hi[i] and lo[i] <= hi[j]}
        color[i] = next(c for c in range(len(first)) if c not in taken)
    return color


def max_overlap(first: np.ndarray, last: np.ndarray, buffer: int) -> int:
    lo, hi = first - buffer, last + buffer
    return int(max(((lo <= w) & (hi >= w)).sum() for w in range(lo.min(), hi.max() + 1)))


def palette(n: int) -> list[str]:
    """n glasbey colours, CVD-safe, light enough for a dark background, kept apart from the unlabelled grey."""
    avoid = [config.UNLABELLED_COLOR, "#11151b"]
    colors = glasbey.extend_palette(
        avoid,
        palette_size=len(avoid) + n,
        colorblind_safe=True,
        cvd_severity=50.0,
        lightness_bounds=config.COLOR_LIGHTNESS,
        chroma_bounds=(25, 90),
        hue_bounds=(0, 360),
    )
    return [c.lower() for c in colors[len(avoid) :]]


def order_cost(pos: np.ndarray, fa: np.ndarray, fb: np.ndarray, fw: np.ndarray, alive: np.ndarray) -> float:
    """Sum over flows of people x the number of tracks alive at the flow's step stacked between its two ends."""
    lo = np.minimum(pos[fa], pos[fb])[:, None]
    hi = np.maximum(pos[fa], pos[fb])[:, None]
    between = alive & (pos[None, :] > lo) & (pos[None, :] < hi)
    return float((between.sum(1) * fw).sum())


def ribbon_order(start: list[int], flows: pd.DataFrame, alive_steps: np.ndarray) -> list[int]:
    """Local search from `start` (track ids 0..T-1): move single tracks to whichever position lowers
    order_cost, until no move does. flows: [a, b, step, people] (step w = windows w -> w + 1);
    alive_steps[step, track] = the track is alive at w or w + 1."""
    order = list(start)
    if flows.empty:
        return order
    fa, fb = flows["a"].to_numpy(), flows["b"].to_numpy()
    fw, alive = flows["people"].to_numpy(float), alive_steps[flows["step"].to_numpy()]

    def cost(o):
        return order_cost(positions(o, alive_steps.shape[1]), fa, fb, fw, alive)

    best = cost(order)
    improved = True
    while improved:
        improved = False
        for t in list(order):
            rest = [u for u in order if u != t]
            for k in range(len(rest) + 1):
                cand = rest[:k] + [t] + rest[k:]
                c = cost(cand)
                if c < best - 1e-9:
                    order, best, improved = cand, c, True
    return order


def positions(order: list[int], n: int) -> np.ndarray:
    pos = np.full(n, -1)
    pos[order] = np.arange(len(order))
    return pos


def coarse_flows(tracks: pd.DataFrame) -> pd.DataFrame:
    """People in coarse track a at window w and coarse track b at w + 1 (a != b), per step."""
    t = tracks[["window", "qid", f"track_{COARSE}"]].rename(columns={f"track_{COARSE}": "t"})
    nxt = t.assign(window=t["window"] - 1)
    m = t.merge(nxt, on=["window", "qid"], suffixes=("_a", "_b"))
    m = m[(m["t_a"] >= 0) & (m["t_b"] >= 0) & (m["t_a"] != m["t_b"])]
    f = m.groupby(["t_a", "t_b", "window"]).size().reset_index(name="people")
    return f.rename(columns={"t_a": "a", "t_b": "b", "window": "step"})


# --- main -----------------------------------------------------------------------------------------------------


def load_people() -> pd.DataFrame:
    mp = pd.read_parquet(config.MAP_PEOPLE_PARQUET).sort_values("rank").reset_index(drop=True)
    if not (mp["rank"].to_numpy() == np.arange(1, len(mp) + 1)).all():
        raise SystemExit("map_people ranks are not 1..N")
    ld = pd.read_parquet(config.LEADS_PARQUET)
    facts = pd.read_parquet(config.PEOPLE_FACTS_PARQUET, columns=["qid", "death_precision"])
    df = mp.merge(ld, on="qid", how="left", validate="1:1").merge(facts, on="qid", how="left", validate="1:1")
    edges = config.bin_edges(df["birth_year_est"].astype(float))
    if not (np.digitize(df["birth_year_est"].astype(float), edges[1:-1]) == df["bin"]).all():
        raise SystemExit("config.bin_edges() no longer reproduces stage 05's bins")
    return df


def sizes_line(paths) -> tuple[str, int, int]:
    raw = gz = 0
    parts = []
    for p in paths:
        data = p.read_bytes()
        r, z = len(data), len(gzip.compress(data, 6))
        raw, gz = raw + r, gz + z
        parts.append(f"{p.relative_to(config.REPO_ROOT)} {r / 1e6:.2f} MB ({z / 1e6:.2f} MB gzipped)")
    return "; ".join(parts), raw, gz


def main() -> None:
    t0 = time.monotonic()
    people = load_people()
    lay = pd.read_parquet(config.layout_parquet())
    tracks = pd.read_parquet(config.tracks_parquet())
    names = pd.read_parquet(config.track_names_parquet())
    shorts = pd.read_parquet(config.short_names_parquet())
    log(f"{len(people):,} people, {len(lay):,} layout points, {len(names)} tracks")

    if config.NORMALIZE_WINDOWS:
        lay = normalize_windows(lay)
    idx = pd.Series(np.arange(len(people)), index=people["qid"])
    lay = lay.merge(tracks, on=["window", "qid"], how="left", validate="1:1")
    lay["person"] = idx.reindex(lay["qid"]).to_numpy()
    lay = lay.sort_values(["person", "window"]).reset_index(drop=True)
    w0, nw = person_windows(lay, people["qid"])
    expected_w0 = np.clip(people["bin"].to_numpy() - (config.WINDOW_BINS - 1), 0, config.N_WINDOWS - 1)
    if not (w0 == expected_w0).all():
        log("WARNING: some people's first window isn't bin - 4; the page uses the exported w0, so this is harmless")

    # --- tracks: names, lifetimes, colours --------------------------------------------------------------------
    names = names.sort_values(["layer", "track"]).reset_index(drop=True)
    names = names.merge(shorts, on=["layer", "track"], how="left")
    names["short_name"] = names["short_name"].fillna("")
    gid = {(int(r.layer), int(r.track)): i for i, r in enumerate(names.itertuples())}
    coarse = names[names["layer"] == COARSE]
    cfirst, clast = coarse["first_window"].to_numpy(), coarse["last_window"].to_numpy()
    n_colors = max_overlap(cfirst, clast, config.COLOR_BUFFER_WINDOWS)
    cidx = interval_colors(cfirst, clast, config.COLOR_BUFFER_WINDOWS)
    colors = palette(n_colors)
    names["color"] = -1
    names.loc[coarse.index, "color"] = cidx

    # --- labels: centroid and size per (window, track), in the normalized layout ----------------------------
    labels = []
    for layer in range(N_LAYERS):
        col = f"track_{layer}"
        g = lay[lay[col] >= 0].groupby(["window", col])
        c = g.agg(x=("x", "mean"), y=("y", "mean"), n=("x", "size")).reset_index()
        c["g"] = [gid[layer, int(t)] for t in c[col]]
        labels.append(c[["window", "g", "x", "y", "n"]])
    labels = pd.concat(labels).sort_values(["window", "g"]).reset_index(drop=True)

    # --- ribbon order (on coarse track ids, which are 0..T-1) ------------------------------------------------
    n_coarse = len(coarse)
    if sorted(coarse["track"]) != list(range(n_coarse)):
        raise SystemExit("coarse track ids are not 0..T-1")
    cl = labels[labels["g"].isin(coarse.index)].assign(t=lambda d: names.loc[d["g"], "track"].to_numpy())
    mean_y = cl.groupby("t")["y"].mean()  # flipY: larger y is lower on screen, and lower in the chart
    start = sorted(range(n_coarse), key=lambda t: mean_y[t])
    alive_w = np.zeros((config.N_WINDOWS, n_coarse), bool)
    alive_w[cl["window"].to_numpy(), cl["t"].to_numpy()] = True
    alive_steps = alive_w[:-1] | alive_w[1:]
    flows = coarse_flows(lay)
    order = ribbon_order(start, flows, alive_steps)
    fa, fb, fw = flows["a"].to_numpy(), flows["b"].to_numpy(), flows["people"].to_numpy(float)
    al = alive_steps[flows["step"].to_numpy()]
    c0 = order_cost(positions(start, n_coarse), fa, fb, fw, al)
    c1 = order_cost(positions(order, n_coarse), fa, fb, fw, al)
    log(f"ribbon order: {len(flows)} coarse flows ({int(fw.sum()):,} person-steps); people x tracks stacked "
        f"between flow ends {c0:,.0f} by map height -> {c1:,.0f} after local search")  # fmt: skip

    # --- per person -----------------------------------------------------------------------------------------
    score = np.log10(people["score"].to_numpy(float))
    rlo, rhi = config.MARKER_RADIUS_PX
    radius = rlo + (rhi - rlo) * (score - score.min()) / (score.max() - score.min())
    name = [display_name(*a) for a in zip(people["label"].astype(object), people["enwiki_title"], people["top_title"])]
    born = [born_label(*a) for a in zip(people["birth_year_est"], people["birth_precision"], people["birth_source"])]
    year = people["birth_year_est"].astype(int)
    wy = lay.assign(year=year.to_numpy()[lay["person"].to_numpy()]).groupby("window")["year"]
    bin_years = [int(year.min()), *[int(e) for e in config.bin_edges(year.astype(float))[1:-1]], int(year.max()) + 1]

    # --- points.bin -------------------------------------------------------------------------------------------
    pad = 0.02
    xlo, xhi = float(lay["x"].min()) - pad, float(lay["x"].max()) + pad
    ylo, yhi = float(lay["y"].min()) - pad, float(lay["y"].max()) + pad
    qxy = np.empty(2 * len(lay), "<u2")
    qxy[0::2], qxy[1::2] = quantize(lay["x"].to_numpy(), xlo, xhi), quantize(lay["y"].to_numpy(), ylo, yhi)
    ctrack = lay[f"track_{COARSE}"].to_numpy()
    ptrack = np.where(ctrack >= 0, ctrack, NO_TRACK).astype("u1")
    if ctrack.max() >= NO_TRACK:
        raise SystemExit("more coarse tracks than a uint8 holds")
    err = np.abs(dequantize(qxy[0::2], xlo, xhi) - lay["x"].to_numpy()).max()

    core = {
        "version": 1,
        "n_windows": config.N_WINDOWS,
        "n_bins": config.N_BINS,
        "window_bins": config.WINDOW_BINS,
        "n_points": len(lay),
        "bounds": [round(xlo, 5), round(xhi, 5), round(ylo, 5), round(yhi, 5)],
        "bin_years": bin_years,
        "windows": {
            "lo": wy.min().astype(int).tolist(),
            "hi": wy.max().astype(int).tolist(),
            "n": wy.size().astype(int).tolist(),
        },  # fmt: skip
        "people": {
            "name": name,
            "born": born,
            "year": year.tolist(),
            "bin": people["bin"].astype(int).tolist(),
            "w0": w0.astype(int).tolist(),
            "nw": nw.astype(int).tolist(),
            "r": np.round(radius * 10).astype(int).tolist(),
        },
        "tracks": {
            "layer": names["layer"].astype(int).tolist(),
            "id": names["track"].astype(int).tolist(),
            "name": names["name"].tolist(),
            "short": names["short_name"].tolist(),
            "first": names["first_window"].astype(int).tolist(),
            "last": names["last_window"].astype(int).tolist(),
            "color": names["color"].astype(int).tolist(),
        },
        "coarse": {
            "order": [gid[COARSE, int(t)] for t in order],
            "palette": colors,
            "unlabelled": config.UNLABELLED_COLOR,
        },
        "labels": {
            "window": labels["window"].astype(int).tolist(),
            "track": labels["g"].astype(int).tolist(),
            "x": labels["x"].round(4).tolist(),
            "y": labels["y"].round(4).tolist(),
            "n": labels["n"].astype(int).tolist(),
        },
        "charset": "".join(sorted({ch for s in names["name"] for ch in s})),
        "built": time.strftime("%Y-%m-%d"),
    }

    details = {
        "desc": [
            d if isinstance(d, str) else (w if isinstance(w, str) else "")
            for d, w in zip(people["enwiki_description"], people["description"].astype(object))
        ],  # fmt: skip
        "excerpt": [excerpt(t) for t in people["lead"]],
        "thumb": [thumb_code(u) for u in people["thumb_url"]],
        "ref": [article_ref(s, t, q) for s, t, q in zip(people["lead_site"], people["lead_title"], people["qid"])],
        "from": [translated_from(s, site) for s, site in zip(people["lead_source"], people["lead_site"])],
        "died": [died_label(y, p) for y, p in zip(people["death_year"], people["death_precision"])],
        "views": people["median_views"].round().astype(int).tolist(),
    }

    out = config.FRONTEND_DATA_DIR
    write_bytes_atomic(qxy.tobytes() + ptrack.tobytes(), out / "points.bin")
    write_bytes_atomic(json.dumps(core, ensure_ascii=False, separators=(",", ":")).encode(), out / "core.json")
    write_bytes_atomic(json.dumps(details, ensure_ascii=False, separators=(",", ":")).encode(), out / "details.json")

    log(f"{n_colors} colours for {len(coarse)} coarse tracks ({config.COLOR_BUFFER_WINDOWS}-window buffer): "
        f"{' '.join(colors)}")  # fmt: skip
    log(f"quantization error at most {err:.5f} (layout span {xhi - xlo:.2f})")
    ex = pd.Series(details["excerpt"]).str.len()
    log(f"excerpts: median {ex.median():.0f} chars, {(ex == 0).sum()} empty, "
        f"{pd.Series(details['excerpt']).str.endswith('…').sum()} cut mid-sentence; thumbnails "
        f"{sum(map(bool, details['thumb'])):,}; translated {sum(map(bool, details['from']))}; names from a title "
        f"{int(people['label'].isna().sum())}")  # fmt: skip
    unknown = {f for f in details["from"] if f and f not in LANGUAGE_NAMES.values()}
    if unknown:
        log(f"WARNING: no language name for {sorted(unknown)}")
    data_line, raw, gz = sizes_line([out / "core.json", out / "points.bin", out / "details.json"])
    log(f"Wrote {data_line}")
    page = [p for p in (config.FRONTEND_DIR / f for f in ("index.html", "app.js", "style.css")) if p.exists()]
    if page:
        page_line, praw, pgz = sizes_line(page)
        log(f"Page: {page_line}")
        raw, gz = raw + praw, gz + pgz
    log(f"Total {raw / 1e6:.2f} MB ({gz / 1e6:.2f} MB gzipped), not counting thumbnails, fonts and deck.gl; "
        f"{(time.monotonic() - t0):.0f} s")  # fmt: skip


if __name__ == "__main__":
    main()

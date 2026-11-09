"""Stage 10: name every window's label clusters with Toponymy and Claude.

Settled 2026-11-08 / 09 (CLAUDE.md, "naming"; scoping/name_window.py tested window 16). Geometry decides
the map's structure (stage 09); this stage only describes it, one window at a time, as Toponymy was built
to work: within a window the three layers nest strictly, so Toponymy's own tree is right, and its passes
that keep names distinct run among exactly the labels on screen together. Stage 11 then names each track
from its path of per-window names.

Per window: stage 09's clusters handed to Toponymy 0.5.4 as a pre-fitted clusterer (each layer's track ids
renumbered 0..k-1, -1 unlabelled), so it does no clustering; documents = the people's English leads
(stage 06) with their stage 07b vectors; keyphrases and candidate names embedded by stage 07's Qwen3
embedder without the role instruction and with the gender direction removed, in batches of
EMBED_SHORT_BATCH_ITEMS; names from config.NAME_MODEL via namer.py (every response cached), with
config.NAME_INSTRUCTIONS appended to each prompt; detail levels config.NAME_DETAIL_LEVELS.

Resumable: each named window is saved to config.WINDOW_NAMES_DIR/<window>.json and skipped on a rerun. A
window with an empty name (Toponymy turns a failed call into an empty name and carries on) is not saved, so
a rerun retries it -- the successful prompts come from the cache.

Output: data/window_names_<MAP_EMBEDDING>.parquet: window (int16), layer (int8, 0 = finest), track (int32,
stage 09's id), size (int32, people in the window), name. Written only once every window is named.
Logs per window: clusters, Claude calls and cost, and names to review: gendered words, names not in title
case, and a name repeated within a layer.
"""

import time

import numpy as np
import pandas as pd

import config
from embedder import DirectionRemovedEmbedder, OpenRouterEmbedder
from namer import AsyncClaudeNamer
from naming import not_title_case, review  # noqa: F401  (re-exported for tests)
from util import log, write_json_atomic, write_parquet_atomic

N_LAYERS = 3
EXEMPLAR_DELIMITERS = ['    * "', '"\n']  # Toponymy's default


def local_labels(track: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Track ids -> 0..k-1 in track order (Toponymy's cluster numbering), -1 kept; and each local cluster's track."""
    out = np.full(len(track), -1)
    ok = track >= 0
    codes, uniques = pd.factorize(track[ok], sort=True)
    out[ok] = codes
    return out, np.asarray(uniques)


def name_window(w: int, rows: pd.DataFrame, xy: np.ndarray, texts: list[str], emb: np.ndarray, years: np.ndarray,
                namer: AsyncClaudeNamer, embedder) -> list[dict]:  # fmt: skip
    """Toponymy over one window's pre-fitted clusters; one entry per (layer, track)."""
    from toponymy import Toponymy, ToponymyClusterer
    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.clustering import build_cluster_tree, centroids_from_labels
    from toponymy.templates import PROMPT_TEMPLATES

    labels, track_of, layers = [], [], []
    for i in range(N_LAYERS):
        lab, tracks = local_labels(rows[f"track_{i}"].to_numpy())
        labels.append(lab)
        track_of.append(tracks)
        layer = ClusterLayerText(lab, centroids_from_labels(lab, emb), layer_id=i,
                                 exemplar_delimiters=EXEMPLAR_DELIMITERS, prompt_template=PROMPT_TEMPLATES)  # fmt: skip
        layers.append(layer)
    clusterer = ToponymyClusterer()
    clusterer.cluster_layers_ = layers
    clusterer.cluster_tree_ = build_cluster_tree(labels)
    lo, hi = int(years.min()), int(years.max())
    model = Toponymy(
        llm_wrapper=namer,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description=config.NAME_OBJECT_DESCRIPTION,
        corpus_description=config.NAME_CORPUS_DESCRIPTION.format(lo=lo, hi=hi),
        lowest_detail_level=config.NAME_DETAIL_LEVELS[0],
        highest_detail_level=config.NAME_DETAIL_LEVELS[1],
        verbose=False,
    )
    # Toponymy.fit takes (objects, high-D embeddings, low-D map) -- the reverse of the clusterer's order -- so
    # pass by keyword. The map only matters for clustering, which the pre-fitted clusterer skips.
    model.fit(texts, embedding_vectors=emb, clusterable_vectors=xy)
    entries = []
    for i, names in enumerate(model.topic_names_):
        sizes = np.bincount(labels[i][labels[i] >= 0], minlength=len(names))
        for k, name in enumerate(names):
            entries.append({"window": w, "layer": i, "track": int(track_of[i][k]), "size": int(sizes[k]),
                            "name": name.strip()})  # fmt: skip
    return entries


def main() -> None:
    t0 = time.monotonic()
    tr = pd.read_parquet(config.tracks_parquet())
    lay = pd.read_parquet(config.layout_parquet(), columns=["window", "qid", "x", "y"])
    if not (tr["qid"].to_numpy() == lay["qid"].to_numpy()).all():
        raise SystemExit("tracks are not row-aligned with the layout -- rerun stage 09")
    people = pd.read_parquet(config.MAP_PEOPLE_PARQUET, columns=["qid", "birth_year_est"]).set_index("qid")
    leads = pd.read_parquet(config.LEADS_PARQUET, columns=["qid", "lead"]).set_index("qid")["lead"]
    z = np.load(config.embeddings_npz(config.MAP_EMBEDDING))
    row_of = pd.Series(np.arange(len(z["qid"])), index=z["qid"].astype(str))
    windows = [int(w) for w in sorted(tr["window"].unique())]  # plain ints: they go into JSON
    log(f"{len(windows)} windows; {config.NAME_MODEL} at effort {config.NAME_EFFORT}, detail levels "
        f"{config.NAME_DETAIL_LEVELS}; cache {config.NAME_CACHE}")  # fmt: skip

    namer = AsyncClaudeNamer()
    inner = OpenRouterEmbedder(instruction=None, batch_items=config.EMBED_SHORT_BATCH_ITEMS)
    embedder = DirectionRemovedEmbedder(inner, z["direction"])
    all_entries, missing = [], []
    for w in windows:
        path = config.WINDOW_NAMES_DIR / f"{w:02d}.json"
        if path.exists():
            all_entries += pd.read_json(path).to_dict("records")
            continue
        tw = time.monotonic()
        before = dict(namer.usage)
        sel = (tr["window"] == w).to_numpy()
        rows = tr[sel].reset_index(drop=True)
        qids = rows["qid"].astype(str)
        texts = leads.reindex(qids).tolist()
        if any(not isinstance(t, str) or not t for t in texts):
            raise SystemExit(f"window {w}: people without a lead -- rerun stage 06")
        emb = z["vectors"][row_of.reindex(qids).to_numpy()].astype(np.float32)
        years = people["birth_year_est"].reindex(qids).to_numpy(dtype=float)
        xy = lay.loc[sel, ["x", "y"]].to_numpy(np.float32)
        entries = name_window(w, rows, xy, texts, emb, years, namer, embedder)
        calls = namer.usage["calls"] - before["calls"]
        notes = review(entries)
        counts = [sum(e["layer"] == i for e in entries) for i in range(N_LAYERS)]
        log(f"window {w:2d} (born {int(np.nanmin(years))} to {int(np.nanmax(years))}): {counts} clusters, "
            f"{calls} Claude calls, {time.monotonic() - tw:.0f} s; running cost ${namer.cost_usd():.2f}")  # fmt: skip
        for n in notes:
            log(f"  review: {n}")
        if any(not e["name"] for e in entries):
            log(f"WARNING: window {w} has empty names (failed calls); not saved -- rerun to retry")
            missing.append(w)
            continue
        write_json_atomic(entries, path)
        all_entries += entries

    u = namer.usage
    log(f"Claude: {u['calls']} calls this run ({u['cached']} from cache), {u['input_tokens']:,} input / "
        f"{u['output_tokens']:,} output tokens = ${namer.cost_usd():.2f}; embedding {inner.tokens:,} tokens, "
        f"${inner.cost_usd:.4f}")  # fmt: skip
    if missing:
        raise SystemExit(f"windows {missing} not fully named -- rerun to retry them (cached calls are free)")
    out = pd.DataFrame(all_entries).astype({"window": "int16", "layer": "int8", "track": "int32", "size": "int32"})
    out = out.sort_values(["window", "layer", "track"]).reset_index(drop=True)
    expected = tr.melt(id_vars="window", value_vars=[f"track_{i}" for i in range(N_LAYERS)])
    expected = expected[expected["value"] >= 0].drop_duplicates(["window", "variable", "value"])
    if len(out) != len(expected):
        raise SystemExit(f"{len(out)} names for {len(expected)} (window, layer, track) clusters")
    write_parquet_atomic(out, config.window_names_parquet())
    notes = review(out.to_dict("records"))
    log(f"Wrote {config.window_names_parquet()} ({len(out):,} names; {len(notes)} to review over all windows) "
        f"in {(time.monotonic() - t0) / 60:.1f} min")  # fmt: skip


if __name__ == "__main__":
    main()

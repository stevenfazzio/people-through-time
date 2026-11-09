"""Name one slider window with Toponymy: cost per window and a first look at the names (one-off, 2026-11-08).

The naming decision (CLAUDE.md, 2026-11-08): Toponymy names each window on its own, then a second step names
each track from its path of per-window names. This runs the first step on WINDOW only, to measure what a
window costs before naming all 32.

Toponymy gets stage 09's clusters as a pre-fitted clusterer (so it does no clustering of its own): per layer,
the window's track ids renumbered 0..k-1 (-1 = unlabelled). Within a window the layers nest strictly, so
Toponymy's own build_cluster_tree gives the right tree. Documents are the people's English leads (stage 06);
their vectors are stage 07b's (gender removed); keyphrases and candidate names are embedded with stage 07's
Qwen3 embedder without the role instruction and with the gender direction removed (config, stage 10).

Output: scoping/results/name_window_<model>.txt (stdout), data/scoping/name_window_<WINDOW>_<model>.json
(model: opus or sonnet, from config.NAME_MODEL; Opus 5.5 run 2026-11-08, Sonnet 5.5 2026-11-09)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402
from embedder import DirectionRemovedEmbedder, OpenRouterEmbedder  # noqa: E402
from namer import AsyncClaudeNamer  # noqa: E402

WINDOW = 16  # median birth year 1895: late-Romantic composers through early cinema; ~40 / 20 / 10 labels
LAYER_NAMES = ("fine", "mid", "coarse")
MODEL_TAG = {"claude-opus-5-5": "opus", "claude-sonnet-5-5": "sonnet"}


def local_labels(track: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Track ids -> 0..k-1 (Toponymy's cluster numbering), -1 kept; and each local cluster's track id."""
    out = np.full(len(track), -1)
    ok = track >= 0
    codes, uniques = pd.factorize(track[ok], sort=True)
    out[ok] = codes
    return out, np.asarray(uniques)


def main() -> None:
    from toponymy import Toponymy, ToponymyClusterer
    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.clustering import build_cluster_tree, centroids_from_labels
    from toponymy.templates import PROMPT_TEMPLATES

    t0 = time.monotonic()
    tr = pd.read_parquet(config.tracks_parquet())
    lay = pd.read_parquet(config.layout_parquet())
    sel = (tr["window"] == WINDOW).to_numpy()
    rows, xy = tr[sel].reset_index(drop=True), lay.loc[sel, ["x", "y"]].to_numpy(np.float32)
    people = pd.read_parquet(config.MAP_PEOPLE_PARQUET, columns=["qid", "label", "rank", "birth_year_est"])
    leads = pd.read_parquet(config.LEADS_PARQUET, columns=["qid", "lead"])
    p = rows[["qid"]].merge(people, on="qid", how="left").merge(leads, on="qid", how="left")
    if p["lead"].isna().any():
        raise SystemExit(f"{p['lead'].isna().sum()} people without a lead -- rerun stage 06")
    z = np.load(config.embeddings_npz(config.MAP_EMBEDDING))
    row_of = pd.Series(np.arange(len(z["qid"])), index=z["qid"].astype(str))
    emb = z["vectors"][row_of.reindex(p["qid"].astype(str)).to_numpy()].astype(np.float32)
    lo, hi = int(p["birth_year_est"].min()), int(p["birth_year_est"].max())
    print(f"window {WINDOW}: {len(p):,} people born {lo} to {hi}")

    labels, track_of, layers = [], [], []
    for i in range(3):
        lab, tracks = local_labels(rows[f"track_{i}"].to_numpy())
        labels.append(lab)
        track_of.append(tracks)
        layer = ClusterLayerText(lab, centroids_from_labels(lab, emb), layer_id=i,
                                 exemplar_delimiters=['    * "', '"\n'], prompt_template=PROMPT_TEMPLATES)  # fmt: skip
        layers.append(layer)
        print(f"  layer {i} ({LAYER_NAMES[i]}): {len(tracks)} clusters, {(lab < 0).mean():.0%} unlabelled")
    clusterer = ToponymyClusterer()
    clusterer.cluster_layers_ = layers
    clusterer.cluster_tree_ = build_cluster_tree(labels)

    namer = AsyncClaudeNamer()
    inner = OpenRouterEmbedder(instruction=None)
    embedder = DirectionRemovedEmbedder(inner, z["direction"])
    model = Toponymy(
        llm_wrapper=namer,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="Wikipedia biographies (the opening of each person's article)",
        corpus_description=(
            f"the most famous people born between {lo} and {hi}, from all Wikipedias -- one era "
            "of a map of the most famous people of every era"
        ),  # fmt: skip
        lowest_detail_level=config.NAME_DETAIL_LEVELS[0],
        highest_detail_level=config.NAME_DETAIL_LEVELS[1],
    )
    # Toponymy.fit takes (objects, high-D embeddings, low-D map) -- the reverse of the clusterer's order -- so
    # pass by keyword. The map only matters for clustering, which the pre-fitted clusterer skips.
    model.fit(p["lead"].tolist(), embedding_vectors=emb, clusterable_vectors=xy)
    elapsed = time.monotonic() - t0

    out = {"window": WINDOW, "born": [lo, hi], "layers": []}
    by_rank = p.sort_values("rank")
    for i, names in enumerate(model.topic_names_):
        print(f"\n== Layer {i} ({LAYER_NAMES[i]}): {len(names)} names ==")
        entries = []
        for k, name in enumerate(names):
            members = by_rank[labels[i][by_rank.index] == k]
            best = "; ".join(members["label"].astype(str).head(4))
            entries.append({"cluster": k, "track": int(track_of[i][k]), "size": len(members), "name": name,
                            "best_known": best})  # fmt: skip
        for e in sorted(entries, key=lambda e: -e["size"]):
            print(f"  {'fmc'[i]}{e['track']:<4} {e['size']:>4}  {e['name'] or '(EMPTY)'}  |  {e['best_known']}")
        out["layers"].append(entries)

    empty = sum(1 for layer in out["layers"] for e in layer if not e["name"])
    u = namer.usage
    calls = u["calls"]
    cost = namer.cost_usd()
    print(f"\nNaming: {calls} Claude calls ({u['cached']} more from cache), {u['input_tokens']:,} input / "
          f"{u['output_tokens']:,} output tokens = ${cost:.2f} ({config.NAME_MODEL}, effort {config.NAME_EFFORT}); "
          f"{empty} empty names")  # fmt: skip
    if calls:
        print(f"  per call: {u['input_tokens'] / calls:,.0f} input, {u['output_tokens'] / calls:,.0f} output tokens")
    print(f"Embedding (keyphrases, names): {inner.tokens:,} tokens, ${inner.cost_usd:.4f}")
    print(f"Wall clock: {elapsed / 60:.1f} min")
    n_windows = int(tr["window"].max()) + 1
    print(f"All {n_windows} windows at this rate: ~${cost * n_windows:.0f}; every second window: "
          f"~${cost * n_windows / 2:.0f} (before the per-track consolidation step)")  # fmt: skip
    path = config.DATA_DIR / "scoping" / f"name_window_{WINDOW}_{MODEL_TAG[config.NAME_MODEL]}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

"""Stage 07: embed every roster person's lead with Qwen3-Embedding-8B, once per instruction variant.

Text = stage 06's English lead (the article opening, which already starts with
the person's name). Variants = config.EMBED_INSTRUCTIONS: "plain" (the text as
is) and "role" (Qwen3's instruction format, asking what the person is known
for: role, field and society). scoping/embed_variants.py compares them; later
stages read config.EMBED_VARIANT (through stage 07b's gender erasure, config.MAP_EMBEDDING).

Vectors come through embedder.py's SQLite cache (data/raw/embedding_cache.sqlite),
so a rerun pays only for texts never sent, and changing a lead re-embeds just
that person.

Output: data/embeddings_<variant>.npz -- qid (str), vectors (float32, n x 4096,
as returned), model, provider, instruction; rows in map_people.parquet order.
"""

import time

import numpy as np
import pandas as pd

import config
from embedder import OpenRouterEmbedder
from util import log, save_npz_atomic


def main() -> None:
    people = pd.read_parquet(config.MAP_PEOPLE_PARQUET, columns=["qid"])
    leads = pd.read_parquet(config.LEADS_PARQUET, columns=["qid", "lead"])
    df = people.merge(leads, on="qid", how="left", validate="1:1")
    if df["lead"].isna().any():
        raise SystemExit(f"{df['lead'].isna().sum()} roster people have no lead -- rerun stage 06")
    texts = df["lead"].tolist()
    log(f"{len(texts):,} leads, {sum(map(len, texts)):,} characters")

    for variant, instruction in config.EMBED_INSTRUCTIONS.items():
        t0 = time.monotonic()
        embedder = OpenRouterEmbedder(instruction=instruction)
        vectors = embedder.encode(texts, show_progress_bar=True)
        norms = np.linalg.norm(vectors, axis=1)
        log(f"{variant}: {vectors.shape} in {time.monotonic() - t0:.0f}s via {embedder.provider}; "
            f"{embedder.tokens:,} new tokens, ${embedder.cost_usd:.4f}; "
            f"norms {norms.min():.3f}-{norms.max():.3f}")  # fmt: skip
        if not np.isfinite(vectors).all():
            raise SystemExit(f"{variant}: non-finite values in the vectors")
        save_npz_atomic(
            config.embeddings_npz(variant),
            qid=df["qid"].to_numpy(dtype=str),
            vectors=vectors,
            model=np.array(config.EMBED_MODEL),
            provider=np.array(embedder.provider),
            instruction=np.array(instruction or ""),
        )
        log(f"Wrote {config.embeddings_npz(variant)}")


if __name__ == "__main__":
    main()

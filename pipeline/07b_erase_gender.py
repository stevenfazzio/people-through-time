"""Stage 07b: project gender out of the map's embeddings (one global centroid direction).

Decided 2026-11-08 after the AlignedUMAP preview: women clustered almost
completely in every window (2-d same-gender neighbour share 0.96-0.98 before
~1500, 0.87-0.93 after, against chance 0.52-0.76). Method from
scoping/gender_erasure.py: d = unit(mean female vector - mean male vector)
over the roster's male/female people (the other genders are too few to
estimate a direction from); every vector x -> unit(x - (x.d) d). LEACE was
indistinguishable; per-window erasers moved 2-d purity only 0.752 -> 0.735.
Linear only: the text still says "actress", "queen", "she".

Output: data/embeddings_<EMBED_VARIANT>_nogender.npz -- qid, vectors (unit,
float32, map_people order), direction (the erased unit vector: project it out
of anything else embedded into this space, e.g. Toponymy's keyphrases), and
the source file's model / provider / instruction.
Logs: held-out linear-probe AUC for gender (split first, direction fit on the
train split only) and within-window 15-NN same-gender share, before and after.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import config
from util import log, save_npz_atomic

K = 15


def gender_direction(x: np.ndarray, female: np.ndarray, male: np.ndarray) -> np.ndarray:
    d = x[female].mean(axis=0) - x[male].mean(axis=0)
    return (d / np.linalg.norm(d)).astype(np.float32)


def erase(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Project d out of every row, then renormalize (the layout uses cosine distance)."""
    out = x - np.outer(x @ d, d)
    return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)


def probe_auc(x: np.ndarray, y: np.ndarray, erased: bool, seed: int = 42) -> float:
    """Held-out logistic probe; when erased, the direction is fit on the train split only (living-people-map:
    fitting it on everything first makes the held-out AUC anti-predictive)."""
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)
    if erased:
        d = gender_direction(x_tr, y_tr == 1, y_tr == 0)
        x_tr, x_te = erase(x_tr, d), erase(x_te, d)
    clf = LogisticRegression(max_iter=3000).fit(x_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1]))


def same_gender_share(x: np.ndarray, sex: np.ndarray, k: int = K) -> float:
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argpartition(-sim, k, axis=1)[:, :k]
    known = sex != ""
    shares = [np.mean(sex[nn[i]][known[nn[i]]] == sex[i]) for i in np.flatnonzero(known) if known[nn[i]].any()]
    return float(np.mean(shares))


def main() -> None:
    people = pd.read_parquet(config.MAP_PEOPLE_PARQUET, columns=["qid", "gender", "bin"])
    src = np.load(config.embeddings_npz(config.EMBED_VARIANT))
    if not (src["qid"] == people["qid"].to_numpy(dtype=str)).all():
        raise SystemExit("embeddings are not row-aligned with map_people -- rerun stage 07")
    x = src["vectors"].astype(np.float32)
    g = people["gender"].astype(object)
    female, male = (g == "female").to_numpy(bool), (g == "male").to_numpy(bool)
    log(f"{config.EMBED_VARIANT}: {len(x):,} vectors; {female.sum():,} female, {male.sum():,} male, "
        f"{len(x) - female.sum() - male.sum()} other or unknown (erased, not used to fit)")  # fmt: skip

    d = gender_direction(x, female, male)
    out = erase(x, d)
    gap = float((out[female].mean(0) - out[male].mean(0)) @ d)
    log(f"Mean female-male gap along the direction: {float((x[female].mean(0) - x[male].mean(0)) @ d):.4f} "
        f"-> {gap:.4f} (after renormalizing)")  # fmt: skip

    labelled = female | male
    y = female[labelled].astype(np.int64)
    log(f"Held-out gender probe AUC: {probe_auc(x[labelled], y, False):.3f} -> {probe_auc(x[labelled], y, True):.3f}")
    sex = np.where(female, "female", np.where(male, "male", ""))
    b = people["bin"].to_numpy()
    for w in range(0, config.N_WINDOWS, 8):
        m = np.flatnonzero((b >= w) & (b < w + config.WINDOW_BINS))
        log(f"  window {w}: within-window 15-NN same-gender share {same_gender_share(x[m], sex[m]):.3f} -> "
            f"{same_gender_share(out[m], sex[m]):.3f}")  # fmt: skip

    path = config.embeddings_npz(config.MAP_EMBEDDING)
    save_npz_atomic(path, qid=src["qid"], vectors=out, direction=d, model=src["model"], provider=src["provider"],
                    instruction=src["instruction"])  # fmt: skip
    log(f"Wrote {path}")


if __name__ == "__main__":
    if not config.ERASE_GENDER:
        raise SystemExit("config.ERASE_GENDER is off: stage 08 reads the unerased vectors; nothing to do")
    main()

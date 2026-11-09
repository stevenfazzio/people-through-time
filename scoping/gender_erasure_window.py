"""Global vs per-window gender erasure, and re-erasing after renormalizing (one-off, 2026-11-08).

Follow-up to gender_erasure.py, which found that a centroid eraser fit on all
31k leaves gender linearly decodable within each window (held-out probe AUC
0.98-0.995): the gender direction differs by era. Role variant, windows 1, 12
and 27, centroid erasure:

  none                  the vectors as embedded
  global x1             one global centroid projection, renormalized
  global x1, no renorm  the same without renormalizing (is the norm the leak?)
  global x3             erase, renormalize, refit: three rounds
  window x1 / x3        the same, fit on the window's own people

The window erasers are fit on the whole window, so their in-window probe AUC
is biased low (the split-first artifact); the neighbourhood purities are what
matters for the map.

Output: scoping/results/gender_erasure_window.txt
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

spec = importlib.util.spec_from_file_location("gender_erasure", Path(__file__).resolve().parent / "gender_erasure.py")
ge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ge)
ev = ge.ev


def centroid_rounds(x: np.ndarray, labelled: np.ndarray, y: np.ndarray, rounds: int, renorm: bool = True):
    for _ in range(rounds):
        d = x[labelled & (y == 1)].mean(0) - x[labelled & (y == 0)].mean(0)
        d /= np.linalg.norm(d)
        x = x - np.outer(x @ d, d)
        if renorm:
            x = ge.unit(x)
    return x


def probe(x: np.ndarray, y: np.ndarray) -> float:
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=3000).fit(x_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1]))


def main() -> None:
    import umap

    warnings.filterwarnings("ignore")
    df, raw = ev.load()
    wins = ev.window_members(df)
    labelled = df["sex"].notna().to_numpy()
    y = df["sex"].eq("female").fillna(False).to_numpy(np.int64)
    x0 = raw["role"]
    glob = {
        "global x1": centroid_rounds(x0, labelled, y, 1),
        "global x1, no renorm": centroid_rounds(x0, labelled, y, 1, renorm=False),
        "global x3": centroid_rounds(x0, labelled, y, 3),
    }
    rows = []
    for w in ge.WINDOWS_2D:
        m = wins[w]
        sub = df.iloc[m]
        lw, yw = labelled[m], y[m]
        sexes = sub["sex"].to_numpy(dtype=object)
        roles = sub["role"].to_numpy(dtype=object)
        roles = np.where(np.isin(roles, ["other", "unknown"]), None, roles)
        candidates = {"none": x0[m]} | {k: v[m] for k, v in glob.items()}
        candidates |= {"window x1": centroid_rounds(x0[m], lw, yw, 1), "window x3": centroid_rounds(x0[m], lw, yw, 3)}
        for name, x in candidates.items():
            nn = ev.knn(ge.unit(x))
            row = {"window": w, "method": name, "probe AUC in window": probe(x[lw], yw[lw])}
            row["gender kNN"], row["chance"] = ev.purity(sexes, nn)
            row["role kNN"] = ev.purity(roles, nn)[0]
            if name != "global x1, no renorm":  # UMAP's cosine metric renormalizes anyway
                xy = umap.UMAP(n_neighbors=15, min_dist=0.05, metric="cosine", random_state=42).fit_transform(x)
                row["gender 2d"] = ge.purity_2d(xy, sexes)[0]
                row["role 2d"] = ge.purity_2d(xy, roles)[0]
            rows.append(row)
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nMean over the three windows:")
    print(table.groupby("method", sort=False).mean(numeric_only=True).drop(columns="window").round(3).to_string())


if __name__ == "__main__":
    main()

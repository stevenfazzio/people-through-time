"""Gender erasure on the Qwen3 lead embeddings (one-off, 2026-11-08).

scoping/embed_variants.py found gender a strong organizer in both variants
(within-window 15-NN gender purity 0.908 plain, 0.945 role, chance 0.663),
and stronger with the role instruction. living-people-map erased it
(2026-07-09): centroid and LEACE both took a held-out probe from AUC 1.00 to
~0.49 and 2-d gender purity from 0.904 to ~0.65, nearly identical because a
binary concept is rank-1 for LEACE. Here, for plain and role:

  none      the vectors as embedded
  centroid  project out the unit (female mean - male mean) direction
  leace     concept_erasure.LeaceEraser (no linear probe can beat chance)

Both fit on male/female rows over all 31k (others are too few) and applied to
everyone. Measures:
1. Held-out linear probe AUC for gender, split-first (the eraser is fit on the
   train split only -- fitting on everything first makes held-out AUC
   anti-predictive, living-people-map). Globally, and within three windows,
   where a globally fit eraser can leave era-specific gender directions.
2. Within-window 15-NN purity (as in embed_variants.py): gender, role, top
   language, translated; and the share of neighbours kept from "none".
3. 2-d purity (UMAP per window, 15-NN in the layout) for three windows.
4. Probes: nearest neighbours of a few people, role variant.
5. data/scoping/gender_erasure.html: role variant, three windows x method, by gender.

Output: scoping/results/gender_erasure.txt
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from concept_erasure import LeaceEraser
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("embed_variants", ROOT / "embed_variants.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
config = ev.config

METHODS = ["none", "centroid", "leace"]
WINDOWS_2D = ev.HTML_WINDOWS
PROBES = ["Cleopatra", "Hildegard of Bingen", "Murasaki Shikibu", "Napoleon", "Marie Curie", "Frida Kahlo",
          "Taylor Swift"]  # fmt: skip


def fit(method: str, x: np.ndarray, y: np.ndarray):
    """An erasure function fit on (x, y), y = 1 for female."""
    if method == "none":
        return lambda v: v
    if method == "centroid":
        d = x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0)
        d /= np.linalg.norm(d)
        return lambda v: v - np.outer(v @ d, d)
    eraser = LeaceEraser.fit(torch.from_numpy(x), torch.from_numpy(y))
    return lambda v: eraser(torch.from_numpy(v)).numpy().astype(np.float32)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def probe_auc(x: np.ndarray, y: np.ndarray, method: str, seed: int = 42) -> float:
    """Split first, fit the eraser on train only, then a logistic probe on the erased, re-normalized vectors."""
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)
    erase = fit(method, x_tr, y_tr)
    x_tr, x_te = unit(erase(x_tr)), unit(erase(x_te))
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(x_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1]))


def purity_2d(xy: np.ndarray, values: np.ndarray, k: int = ev.K) -> tuple[float, float]:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    return ev.purity(values, nn)


def main() -> None:
    import umap

    df, raw = ev.load()
    wins = ev.window_members(df)
    labelled = df["sex"].notna().to_numpy()
    y_all = df["sex"].eq("female").fillna(False).to_numpy(dtype=np.int64)
    pd.set_option("display.width", 220)
    print(f"{labelled.sum():,} of {len(df):,} people male/female ({y_all.sum():,} female)")

    erased = {}
    for v in ev.VARIANTS:
        for m in METHODS:
            erase = fit(m, raw[v][labelled], y_all[labelled])
            erased[v, m] = unit(erase(raw[v]))

    print("\n== 1. Held-out gender probe AUC (0.5 = nothing linear left) ==")
    rows = []
    for v in ev.VARIANTS:
        for m in METHODS:
            row = {"variant": v, "method": m, "global": round(probe_auc(raw[v][labelled], y_all[labelled], m), 3)}
            for w in WINDOWS_2D:
                idx = wins[w][labelled[wins[w]]]
                row[f"window {w}"] = round(probe_auc(raw[v][idx], y_all[idx], m), 3)
            rows.append(row)
            print(row, flush=True)
    print(pd.DataFrame(rows).to_string(index=False))
    print("(window columns: the eraser refit on that window's train split -- the per-window ceiling)")
    print("\nWithin-window probe on GLOBALLY erased vectors (what the map would get):")
    for v in ev.VARIANTS:
        for m in METHODS:
            aucs = []
            for w in WINDOWS_2D:
                idx = wins[w][labelled[wins[w]]]
                x_tr, x_te, y_tr, y_te = train_test_split(
                    erased[v, m][idx], y_all[idx], test_size=0.2, random_state=42, stratify=y_all[idx]
                )
                clf = LogisticRegression(max_iter=3000).fit(x_tr, y_tr)
                aucs.append(roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1]))
            print(f"  {v:5} {m:8} " + "  ".join(f"w{w} {a:.3f}" for w, a in zip(WINDOWS_2D, aucs)))

    print("\n== 2. Within-window 15-NN purity in the embedding space ==")
    rows = []
    for v in ev.VARIANTS:
        base_nn = [ev.knn(erased[v, "none"][m]) for m in wins]
        for meth in METHODS:
            acc: dict[str, list] = {}
            for m, nn0 in zip(wins, base_nn):
                nn = nn0 if meth == "none" else ev.knn(erased[v, meth][m])
                sub = df.iloc[m]
                for name, col in [("gender", "sex"), ("role", "role"), ("top language", "top_lang")]:
                    vals = sub[col].to_numpy(dtype=object)
                    if col == "role":
                        vals = np.where(np.isin(vals, ["other", "unknown"]), None, vals)
                    acc.setdefault(name, []).append(ev.purity(vals, nn)[0])
                tr = sub["translated"].to_numpy()
                if tr.sum() >= 5:
                    acc.setdefault("translated", []).append(float(tr[nn[tr]].mean()))
                acc.setdefault("kept neighbours", []).append(
                    np.mean([len(set(a) & set(b)) / ev.K for a, b in zip(nn, nn0)])
                )
            rows.append({"variant": v, "method": meth} | {k: round(float(np.mean(x)), 3) for k, x in acc.items()})
    print(pd.DataFrame(rows).to_string(index=False))
    print("chance: gender 0.663, role 0.255, top language 0.564, translated 0.010")

    print(f"\n== 3. 2-d purity, UMAP per window (windows {WINDOWS_2D}) ==")
    rows, layouts = [], {}
    for v in ev.VARIANTS:
        for meth in METHODS:
            row = {"variant": v, "method": meth}
            g, r = [], []
            for w in WINDOWS_2D:
                m = wins[w]
                xy = umap.UMAP(n_neighbors=15, min_dist=0.05, metric="cosine", random_state=42).fit_transform(
                    erased[v, meth][m]
                )
                layouts[v, meth, w] = xy
                sub = df.iloc[m]
                g.append(purity_2d(xy, sub["sex"].to_numpy(dtype=object))[0])
                roles = sub["role"].to_numpy(dtype=object)
                r.append(purity_2d(xy, np.where(np.isin(roles, ["other", "unknown"]), None, roles))[0])
            row |= {"gender 2d": round(float(np.mean(g)), 3), "role 2d": round(float(np.mean(r)), 3)}
            rows.append(row)
            print(row, flush=True)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n== 4. Probes (role variant): 8 nearest neighbours in the central window ==")
    for name in PROBES:
        hit = df.index[df["label"] == name]
        if len(hit) == 0:
            continue
        i = hit[0]
        w = int(np.clip(df.at[i, "bin"] - config.WINDOW_BINS // 2, 0, len(wins) - 1))
        m = wins[w]
        local = int(np.flatnonzero(m == i)[0])
        print(f"\n{name} ({df.at[i, 'birth_year_est']}), window {w}:")
        for meth in METHODS:
            x = erased["role", meth][m]
            sims = x @ x[local]
            sims[local] = -np.inf
            nb = df.iloc[m[np.argsort(-sims)[:8]]]
            tag = {"female": "F", "male": "M"}
            print(f"  {meth:8}: " + "; ".join(f"{r.label} ({tag.get(r.sex, '?')})" for r in nb.itertuples()))

    write_html(df, wins, layouts)


def write_html(df, wins, layouts) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    colors = {"male": "#3b6fb6", "female": "#d4553a", "other": "#9a9a9a"}
    titles = [f"window {w}: role, {m}" for w in WINDOWS_2D for m in METHODS]
    fig = make_subplots(rows=len(WINDOWS_2D), cols=len(METHODS), subplot_titles=titles, horizontal_spacing=0.02,
                        vertical_spacing=0.04)  # fmt: skip
    for row, w in enumerate(WINDOWS_2D, 1):
        sub = df.iloc[wins[w]]
        g = sub["sex"].fillna("other").to_numpy()
        hover = (sub["label"].astype(str) + " (" + sub["birth_year_est"].astype(str) + ")<br>" + sub["role"] + ": "
                 + sub["desc"].astype(str)).to_numpy()  # fmt: skip
        for col, meth in enumerate(METHODS, 1):
            xy = layouts["role", meth, w]
            for gender, color in colors.items():
                s = g == gender
                fig.add_trace(
                    go.Scattergl(
                        x=xy[s, 0],
                        y=xy[s, 1],
                        mode="markers",
                        name=gender,
                        legendgroup=gender,
                        showlegend=(row == 1 and col == 1),
                        marker={"size": 4, "color": color, "opacity": 0.7},
                        hovertext=hover[s],
                        hoverinfo="text",
                    ),  # fmt: skip
                    row=row,
                    col=col,
                )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(height=600 * len(WINDOWS_2D), width=1650,
                      title="Role-instructed Qwen3 embeddings, gender erased: none / centroid / LEACE")  # fmt: skip
    out = config.DATA_DIR / "scoping" / "gender_erasure.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

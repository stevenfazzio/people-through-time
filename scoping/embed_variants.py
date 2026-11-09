"""Plain vs role-instructed Qwen3 embeddings of the roster's leads (one-off, 2026-11-08).

Which variant should the map use? Stage 07 embedded every lead twice: "plain"
(the text as is) and "role" (Qwen3's instruction format, asking what the person
is known for: role, field and society). The map lays out each slider window on
its own, so everything is measured within windows (k = 15 cosine neighbours,
averaged over a person's windows):

1. kNN purity of gender, role (a keyword from the short description), top
   language (a culture proxy) and translated leads, each against its chance
   rate. Wanted: role and culture high, gender and translation low.
2. Birth-year locality: mean |birth-year rank difference| to neighbours, as a
   share of the window (0 = same time, ~0.33 = no time order).
3. How many neighbours the two variants share.
4. Probes: the 8 nearest neighbours of well-known people in their central window.
5. data/scoping/embed_variants.html: UMAP of three windows per variant, by role.

Output: scoping/results/embed_variants.txt
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

K = 15
VARIANTS = list(config.EMBED_INSTRUCTIONS)
# Role from the earliest of these words in the short description ("American actor
# and politician" -> film & TV). Order within the dict doesn't matter.
ROLES = {
    "royalty & nobility": r"emperor|empress|king|queen|pharaoh|sultan|sultana|caliph|shah|khan|tsar|tsarina|czar"
    r"|monarch|ruler|prince|princess|duke|duchess|count|countess|consort|regent|noble\w*|aristocrat|chieftain"
    r"|lord|lady|dauphin|infanta|royal|nobleman|noblewoman|tyrant|satrap|maharaja|rani|raja|khagan|elector|tlatoani|emir|amir|doge|shogun|sovereign",
    "religion": r"saint|pope|bishop|archbishop|priest|monk|nun|theologian|prophet|rabbi|imam|missionary|martyr"
    r"|apostle|patriarch|cleric|abbot|abbess|guru|mystic|preacher|sahaba|companion of|religious|deity|evangelist"
    r"|taoist|buddhist|catholic church|sufi|zen master",
    "politics & law": r"politician|president|prime minister|statesman|stateswoman|diplomat|senator|consul|chancellor"
    r"|governor|minister|revolutionary|activist|lawyer|judge|jurist|official|mayor|dictator|vizier|bureaucrat"
    r"|civil servant|first lady|leader|prefect|secretary|tribune|magistrate|attorney general|surgeon general",
    "military": r"general|military|soldier|admiral|commander|warrior|officer|samurai|conqueror|warlord|marshal"
    r"|knight|mercenary|centurion|legionary|daimyo|daimy[oō]",
    "thought & science": r"philosopher|scientist|physicist|chemist|mathematician|astronomer|biologist|physician"
    r"|economist|inventor|engineer|historian|scholar|psychologist|sociologist|linguist|geographer|naturalist"
    r"|botanist|polymath|astrologer|alchemist|anatomist|doctor|nurse|surgeon|archaeologist|anthropologist"
    r"|researcher|academic|professor|explorer|navigator|\w+ologist",
    "writing": r"writer|poet|poetess|novelist|author|playwright|dramatist|journalist|essayist|screenwriter|lyricist"
    r"|translator|chronicler|biographer|mangaka|satirist",
    "visual arts": r"painter|sculptor|artist|architect|photographer|illustrator|designer|cartoonist|engraver"
    r"|calligrapher|potter",
    "music": r"singer|musician|composer|rapper|songwriter|pianist|guitarist|drummer|band|dj|violinist|conductor"
    r"|vocalist|idol|record producer|organist|cellist|bassist",
    "film & tv": r"actor|actress|film director|filmmaker|director|comedian|television|presenter|host|youtuber"
    r"|model|influencer|pornographic|streamer|tarento|voice actor|seiyu|announcer|personality|dancer|choreographer",
    "sport": r"footballer|player|athlete|boxer|wrestler|tennis|cricketer|racing driver|driver|swimmer|coach"
    r"|olympic|martial artist|chess|cyclist|golfer|skater|gymnast|fighter|sprinter|runner|jockey|mountaineer"
    r"|bodybuilder|referee|sumo|rikishi|skier|basketball|baseball|hockey|skateboarder|stuntman|swordsman|surfer|climber|archer|fencer",
    "business": r"businessman|businesswoman|businessperson|entrepreneur|investor|executive|founder|banker"
    r"|industrialist|merchant|magnate|philanthropist|billionaire",
    "crime": r"criminal|serial killer|murderer|gangster|pirate|terrorist|outlaw|assassin|mobster|fraudster"
    r"|drug lord|robber|spy",
    "remains": r"mummy|bog body|skeleton|remains|skull",
    "family of the famous": r"daughter|son|wife|mother|husband|father|sister|brother|widow|mistress|concubine"
    r"|grandson|granddaughter|niece|nephew|heir|child",
}
ROLE_RES = {k: re.compile(rf"\b(?:{v})\b", re.IGNORECASE) for k, v in ROLES.items()}
PROBES = [
    "Confucius", "Cleopatra", "Avicenna", "Murasaki Shikibu", "Hildegard of Bingen", "Genghis Khan",
    "Leonardo da Vinci", "Napoleon", "Marie Curie", "Frida Kahlo", "Pelé", "Taylor Swift",
]  # fmt: skip
HTML_WINDOWS = [1, 12, 27]


def role_of(desc) -> str:
    if not isinstance(desc, str):
        return "unknown"
    best, pos = "other", len(desc) + 1
    for role, rx in ROLE_RES.items():
        m = rx.search(desc)
        if m and m.start() < pos:
            best, pos = role, m.start()
    return best


def load() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    people = pd.read_parquet(config.MAP_PEOPLE_PARQUET)
    leads = pd.read_parquet(config.LEADS_PARQUET, columns=["qid", "lead_source", "enwiki_description"])
    df = people.merge(leads, on="qid", validate="1:1")
    desc = df["enwiki_description"].where(df["enwiki_description"].notna(), df["description"].astype(object))
    df["desc"] = desc
    df["role"] = desc.map(role_of)
    df["sex"] = df["gender"].where(df["gender"].isin(["male", "female"]))
    df["translated"] = df["lead_source"].eq("translated")
    vecs = {}
    for v in VARIANTS:
        z = np.load(config.embeddings_npz(v))
        assert (z["qid"] == df["qid"].to_numpy(dtype=str)).all(), f"{v}: rows out of order"
        x = z["vectors"].astype(np.float32)
        vecs[v] = x / np.linalg.norm(x, axis=1, keepdims=True)
    return df, vecs


def window_members(df: pd.DataFrame) -> list[np.ndarray]:
    n_windows = config.N_BINS - config.WINDOW_BINS + 1
    b = df["bin"].to_numpy()
    return [np.flatnonzero((b >= w) & (b < w + config.WINDOW_BINS)) for w in range(n_windows)]


def knn(x: np.ndarray, k: int = K) -> np.ndarray:
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argpartition(-sim, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sim, nn, axis=1), axis=1)
    return np.take_along_axis(nn, order, axis=1)


def purity(values: np.ndarray, nn: np.ndarray) -> tuple[float, float]:
    """Mean share of a person's known-valued neighbours sharing their value, and its chance rate."""
    known = pd.notna(values)
    same, chance = [], []
    vals, counts = np.unique(values[known].astype(str), return_counts=True)
    n_of = dict(zip(vals, counts))
    n_known = known.sum()
    for i in np.flatnonzero(known):
        nb = nn[i][known[nn[i]]]
        if len(nb) == 0:
            continue
        same.append(np.mean(values[nb].astype(str) == str(values[i])))
        chance.append((n_of[str(values[i])] - 1) / (n_known - 1))
    return float(np.mean(same)), float(np.mean(chance))


def main() -> None:
    df, vecs = load()
    wins = window_members(df)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    print(f"{len(df):,} people, {len(wins)} windows of {min(map(len, wins)):,}-{max(map(len, wins)):,}; k = {K}")
    print("Instruction (role):", config.EMBED_INSTRUCTIONS["role"])
    print("\nRole categories (earliest keyword in the enwiki short description, else Wikidata's):")
    print(df["role"].value_counts().to_string())

    nns = {v: [knn(vecs[v][m]) for m in wins] for v in VARIANTS}
    rows = []
    for v in VARIANTS:
        acc: dict[str, list] = {}
        for m, nn in zip(wins, nns[v]):
            sub = df.iloc[m]
            for name, col in [("gender", "sex"), ("role", "role"), ("top language", "top_lang")]:
                vals = sub[col].to_numpy(dtype=object)
                if col == "role":
                    vals = np.where(np.isin(vals, ["other", "unknown"]), None, vals)
                acc.setdefault(name, []).append(purity(vals, nn))
            tr = sub["translated"].to_numpy()
            if tr.sum() >= 5:
                acc.setdefault("translated (their neighbours)", []).append(
                    (float(tr[nn[tr]].mean()), float((tr.sum() - 1) / (len(tr) - 1)))
                )
            r = sub["birth_year_est"].rank(pct=True).to_numpy()
            acc.setdefault("birth-year rank distance", []).append((float(np.abs(r[:, None] - r[nn]).mean()), 1 / 3))
        for name, pairs in acc.items():
            p, c = np.mean(pairs, axis=0)
            rows.append({"variant": v, "measure": name, "purity": round(p, 3), "chance": round(c, 3),
                         "lift": round(p / c, 2)})  # fmt: skip
    table = pd.DataFrame(rows).pivot(index="measure", columns="variant", values=["purity", "chance", "lift"])
    print("\n== kNN purity within windows (mean over windows) ==")
    print(table.to_string())

    print("\n== By era: role and top-language purity (lift over chance) ==")
    eras = [(0, 8, "windows 0-7 (to ~1140)"), (8, 16, "8-15 (~990-1830)"), (16, 24, "16-23 (~1830-1950)"),
            (24, 32, "24-31 (1944-2013)")]  # fmt: skip
    for lo, hi, name in eras:
        parts = []
        for v in VARIANTS:
            for col in ["role", "top_lang", "sex"]:
                ps = []
                for w in range(lo, hi):
                    sub = df.iloc[wins[w]]
                    vals = sub[col].to_numpy(dtype=object)
                    if col == "role":
                        vals = np.where(np.isin(vals, ["other", "unknown"]), None, vals)
                    p, c = purity(vals, nns[v][w])
                    ps.append(p / c)
                parts.append(f"{v} {col} {np.mean(ps):.2f}")
        print(f"  {name}: " + "; ".join(parts))

    overlap = np.mean([np.mean([len(set(a) & set(b)) / K for a, b in zip(p, r)])
                       for p, r in zip(nns[VARIANTS[0]], nns[VARIANTS[1]])])  # fmt: skip
    print(f"\nNeighbours shared by the two variants: {overlap:.1%}")

    print("\n== Probes: 8 nearest neighbours in the person's central window ==")
    n_windows = len(wins)
    for name in PROBES:
        hit = df.index[df["label"] == name]
        if len(hit) == 0:
            print(f"\n{name}: not in the roster")
            continue
        i = hit[0]
        w = int(np.clip(df.at[i, "bin"] - config.WINDOW_BINS // 2, 0, n_windows - 1))
        m = wins[w]
        local = int(np.flatnonzero(m == i)[0])
        print(f"\n{name} ({df.at[i, 'birth_year_est']}), window {w}:")
        for v in VARIANTS:
            nb = df.iloc[m[nns[v][w][local][:8]]]
            print(f"  {v:5}: " + "; ".join(f"{r.label} ({r.birth_year_est})" for r in nb.itertuples()))

    write_html(df, vecs, wins)


def write_html(df: pd.DataFrame, vecs: dict[str, np.ndarray], wins: list[np.ndarray]) -> None:
    import plotly.express as px
    import plotly.graph_objects as go
    import umap
    from plotly.subplots import make_subplots

    roles = [r for r in df["role"].value_counts().index if r not in ("other", "unknown")] + ["other", "unknown"]
    palette = px.colors.qualitative.Dark24
    colors = {r: palette[i % len(palette)] for i, r in enumerate(roles[:-2])} | {"other": "#b0b0b0",
                                                                                "unknown": "#e0e0e0"}  # fmt: skip
    titles = []
    for w in HTML_WINDOWS:
        sub = df.iloc[wins[w]]
        span = f"{sub['birth_year_est'].min()} to {sub['birth_year_est'].max()}"
        titles += [f"window {w} ({span}): {v}" for v in VARIANTS]
    fig = make_subplots(rows=len(HTML_WINDOWS), cols=len(VARIANTS), subplot_titles=titles,
                        horizontal_spacing=0.03, vertical_spacing=0.04)  # fmt: skip
    for row, w in enumerate(HTML_WINDOWS, 1):
        m = wins[w]
        sub = df.iloc[m]
        for col, v in enumerate(VARIANTS, 1):
            xy = umap.UMAP(n_neighbors=15, min_dist=0.05, metric="cosine", random_state=42).fit_transform(vecs[v][m])
            for role in roles:
                s = (sub["role"] == role).to_numpy()
                if not s.any():
                    continue
                hover = (sub.loc[s, "label"].astype(str) + " (" + sub.loc[s, "birth_year_est"].astype(str) + ")<br>"
                         + sub.loc[s, "desc"].astype(str))  # fmt: skip
                fig.add_trace(
                    go.Scattergl(
                        x=xy[s, 0],
                        y=xy[s, 1],
                        mode="markers",
                        name=role,
                        legendgroup=role,
                        showlegend=(row == 1 and col == 1),
                        marker={"size": 4, "color": colors[role]},
                        hovertext=hover,
                        hoverinfo="text",
                    ),  # fmt: skip
                    row=row,
                    col=col,
                )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(height=650 * len(HTML_WINDOWS), width=1500, title="Plain vs role-instructed Qwen3 embeddings, "
                      "UMAP per window, coloured by role (from the short description)")  # fmt: skip
    out = config.DATA_DIR / "scoping" / "embed_variants.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

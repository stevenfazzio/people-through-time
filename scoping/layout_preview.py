"""Animated preview of the AlignedUMAP layout across the 32 slider windows (one-off, 2026-11-08).

Not the frontend: a cheap look at two things before labels and the real page:
whether the motion between windows reads well, and whether gender visibly
organizes the map (the condition, set 2026-11-08, for revisiting erasure).

A plotly animation with one frame per window. Points are keyed by QID, so a
person glides between the windows they share and pops in or out at the edges.
Axis ranges are fixed across frames, so drift and scale changes show. Two
pages, the same layout coloured two ways: by role (a keyword from the short
description, as in embed_variants.py) and by gender.

Output: data/scoping/layout_<MAP_EMBEDDING>_by_{role,gender}.html (2026-11-08: role and role_nogender)
"""

import importlib.util
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

spec = importlib.util.spec_from_file_location("embed_variants", Path(__file__).resolve().parent / "embed_variants.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
config = ev.config


def load() -> pd.DataFrame:
    lay = pd.read_parquet(config.layout_parquet())
    people = pd.read_parquet(config.MAP_PEOPLE_PARQUET, columns=["qid", "label", "description", "gender",
                                                                "birth_year_est", "rank"])  # fmt: skip
    leads = pd.read_parquet(config.LEADS_PARQUET, columns=["qid", "enwiki_description"])
    p = people.merge(leads, on="qid")
    desc = p["enwiki_description"].where(p["enwiki_description"].notna(), p["description"].astype(object))
    p["desc"] = desc.fillna("").astype(str)
    p["role"] = desc.map(ev.role_of)
    p["sex"] = p["gender"].where(p["gender"].isin(["male", "female"])).fillna("other").astype(str)
    df = lay.merge(p, on="qid", validate="m:1")
    df["hover"] = (
        df["label"].fillna(df["qid"]).astype(str)
        + " ("
        + df["birth_year_est"].astype(str)
        + ")<br>"
        + df["desc"].str.slice(0, 70)
    )
    return df


def palette(df: pd.DataFrame, by: str) -> dict[str, str]:
    if by == "sex":
        return {"male": "#3b6fb6", "female": "#d4553a", "other": "#9a9a9a"}
    roles = [r for r in df.drop_duplicates("qid")["role"].value_counts().index if r not in ("other", "unknown")]
    colors = px.colors.qualitative.Dark24
    return {r: colors[i % len(colors)] for i, r in enumerate(roles)} | {"other": "#b5b5b5", "unknown": "#dcdcdc"}


def traces(win: pd.DataFrame, by: str, colors: dict[str, str]) -> list[go.Scatter]:
    out = []
    for cat, color in colors.items():
        s = win[win[by] == cat]
        out.append(
            go.Scatter(
                x=s["x"].round(3),
                y=s["y"].round(3),
                ids=s["qid"],
                mode="markers",
                name=cat,
                marker={"size": 4, "color": color, "opacity": 0.75},
                hovertext=s["hover"],
                hoverinfo="text",
            )  # fmt: skip
        )
    return out


def build(df: pd.DataFrame, by: str, title: str, out: Path) -> None:
    colors = palette(df, by)
    windows = sorted(df["window"].unique())
    pad_x = (df["x"].max() - df["x"].min()) * 0.03
    pad_y = (df["y"].max() - df["y"].min()) * 0.03
    spans = df.groupby("window")["birth_year_est"].agg(["min", "max"])
    names = [f"w{w}: born {spans.at[w, 'min']} to {spans.at[w, 'max']}" for w in windows]
    frames = [go.Frame(data=traces(df[df["window"] == w], by, colors), name=n) for w, n in zip(windows, names)]
    step_args = {"frame": {"duration": 900, "redraw": False}, "transition": {"duration": 700}, "mode": "immediate"}
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=title,
            height=900,
            xaxis={"range": [df["x"].min() - pad_x, df["x"].max() + pad_x], "showticklabels": False},
            yaxis={
                "range": [df["y"].min() - pad_y, df["y"].max() + pad_y],
                "showticklabels": False,
                "scaleanchor": "x",
            },  # fmt: skip
            updatemenus=[
                {
                    "type": "buttons",
                    "x": 0.0,
                    "y": -0.06,
                    "xanchor": "left",
                    "buttons": [
                        {"label": "Play", "method": "animate", "args": [None, {**step_args, "fromcurrent": True}]},
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                        },  # fmt: skip
                    ],
                }
            ],
            sliders=[
                {
                    "x": 0.12,
                    "len": 0.88,
                    "y": -0.02,
                    "currentvalue": {"prefix": ""},
                    "steps": [
                        {"label": n.split(":")[0], "method": "animate", "args": [[n], step_args]} for n in names
                    ],  # fmt: skip
                }
            ],
        ),
    )
    fig.write_html(out, include_plotlyjs="cdn", auto_play=False)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    df = load()
    print(f"{len(df):,} points in {df['window'].nunique()} windows")
    out_dir = config.DATA_DIR / "scoping"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = config.MAP_EMBEDDING
    base = f"AlignedUMAP layout of the slider windows ({name} embedding)"
    build(df, "role", f"{base}, coloured by role", out_dir / f"layout_{name}_by_role.html")
    build(df, "sex", f"{base}, coloured by gender", out_dir / f"layout_{name}_by_gender.html")
    shared = df.groupby("qid")["window"].nunique()
    print(f"windows per person: {shared.value_counts().sort_index().to_dict()}")
    print(f"axis extent x {df['x'].min():.1f}..{df['x'].max():.1f}, y {df['y'].min():.1f}..{df['y'].max():.1f}")
    per = df.groupby("window")[["x", "y"]].agg(["mean", "std"]).round(2)
    print(f"per-window centre and spread (drift check):\n{per.iloc[::4].to_string()}")


if __name__ == "__main__":
    main()

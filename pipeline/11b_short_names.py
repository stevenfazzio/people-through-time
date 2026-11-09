"""Stage 11b: a short name per coarse track, for the timeline chart's ribbons.

Decided 2026-11-09 (CLAUDE.md, "Frontend"). The page's slider is a ThemeRiver of the coarse tracks: one ribbon
per track, as thick as the track's people in each window, with its name written inside. Ribbons are thin, so
most of stage 11's names don't fit; the chart uses the full name where it fits, else this short one, else none.
The map's own labels keep stage 11's names.

Each prompt gives the track's full name (stage 11), the distinct per-window names along its path and stage
11's reading of its first and last windows, and asks for a form of at most config.SHORT_NAME_MAX_WORDS words
and config.SHORT_NAME_MAX_CHARS characters that is still true of the whole path. Then up to
config.SHORT_NAME_ROUNDS rounds re-ask, in one prompt per group, for names over the limit and for tracks
alive in the same window that got the same short name.

Model, effort and cache are stages 10-11's (namer.py, config.NAME_*).

Output: data/track_short_names_<MAP_EMBEDDING>.parquet: layer (int8), track (int32), short_name.
"""

import asyncio
import json
import re
import time
from collections import defaultdict

import pandas as pd

import config
from namer import AsyncClaudeNamer
from naming import clean_name, not_title_case
from util import log, write_parquet_atomic

SYSTEM = (
    "You name regions of an interactive map of the most famous people of every era, drawn from all Wikipedias. "
    "A slider picks a window of birth years; nearby people form labelled regions, each followed from window to "
    "window as a track. Under the map, a timeline chart draws each broad region as a ribbon whose thickness is "
    "its number of people, with the region's name written inside the ribbon. The ribbons are thin, so the chart "
    "needs short forms of the names."
)
SHORT_SCHEMA = {
    "type": "object",
    "properties": {"short_name": {"type": "string"}},
    "required": ["short_name"],
    "additionalProperties": False,
}
GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        "names": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"region": {"type": "string"}, "short_name": {"type": "string"}},
                "required": ["region", "short_name"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["names"],
    "additionalProperties": False,
}


def limits() -> str:
    return (
        f"at most {config.SHORT_NAME_MAX_WORDS} words and {config.SHORT_NAME_MAX_CHARS} characters including "
        'spaces, in Title Case; "&" may stand for "and"'
    )


RULES = (
    "It must stay true of every person in the region along its whole path, as the full name is: drop detail "
    "rather than narrow the name to one part of the path. Don't describe the region by gender, and don't hedge "
    '(no "mostly", "and others" or parentheses).'
)


def key(layer: int, track: int) -> str:
    return f"{'fmc'[layer]}{track}"


def describe(r) -> list[str]:
    return [
        f'Full name: "{r.name}"',
        f"Its names window by window, in order: {r.path}",
        f"In its first windows it is: {r.earliest}. In its last: {r.latest}.",
    ]


def short_prompt(r) -> str:
    return "\n".join(
        [
            *describe(r),
            "",
            f"Write a short form of the full name for its ribbon: {limits()}. {RULES}",
            'Reply in JSON: "short_name".',
        ]  # fmt: skip
    )


def group_prompt(rows: list, why: str) -> str:
    lines = [f"{why} Give each region a short form of its full name for its ribbon: {limits()}. {RULES}", ""]
    for r in rows:
        lines += [f"Region {key(r.layer, r.track)}:", *describe(r), ""]
    lines.append('Reply in JSON: "names", one entry per region, each with its "region" id and its "short_name".')
    return "\n".join(lines)


def too_long(name: str) -> bool:
    words = [w for w in name.split() if w != "&"]
    return len(name) > config.SHORT_NAME_MAX_CHARS or len(words) > config.SHORT_NAME_MAX_WORDS


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.casefold().replace("&", " and ")).strip()


def clashes(short: dict[tuple[int, int], str], alive: dict[tuple[int, int], set[int]]) -> list[list[tuple]]:
    """Groups of same-layer tracks alive in a common window that share a short name (connected components)."""
    parent: dict = {}

    def find(k):
        parent.setdefault(k, k)
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    by_name = defaultdict(list)
    for k, name in short.items():
        by_name[(k[0], norm(name))].append(k)
    for keys in by_name.values():
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if alive[a] & alive[b]:
                    parent[find(b)] = find(a)
    groups = defaultdict(list)
    for k in parent:
        groups[find(k)].append(k)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def parse_group(text: str, keys: list[tuple[int, int]]) -> dict[tuple[int, int], str]:
    wanted = {key(*k): k for k in keys}
    out = {}
    for e in json.loads(text)["names"]:
        rid = re.sub(r"^region\s+", "", e["region"].strip(), flags=re.IGNORECASE).lower()
        if rid in wanted and clean_name(e["short_name"]):
            out[wanted[rid]] = clean_name(e["short_name"])
    return out


async def ask_all(namer: AsyncClaudeNamer, prompts: dict, schema: dict) -> dict:
    keys = list(prompts)
    results = await asyncio.gather(*(namer._create(SYSTEM, prompts[k], schema) for k in keys), return_exceptions=True)
    return dict(zip(keys, results))


def main() -> None:
    t0 = time.monotonic()
    names = pd.read_parquet(config.track_names_parquet())
    names = names[names["layer"].isin(config.SHORT_NAME_LAYERS)].reset_index(drop=True)
    per_window = pd.read_parquet(config.window_names_parquet())
    alive = {(int(lay), int(t)): set(g["window"].astype(int))
             for (lay, t), g in per_window.groupby(["layer", "track"])}  # fmt: skip
    rows = {(int(r.layer), int(r.track)): r for r in names.itertuples()}
    log(f"{len(rows)} tracks on layers {config.SHORT_NAME_LAYERS}; {config.NAME_MODEL} at effort {config.NAME_EFFORT}")

    namer = AsyncClaudeNamer(instructions="")
    replies = asyncio.run(ask_all(namer, {k: short_prompt(r) for k, r in rows.items()}, SHORT_SCHEMA))
    failed = [k for k, r in replies.items() if isinstance(r, Exception)]
    if failed:
        raise SystemExit(f"{len(failed)} calls failed (first: {replies[failed[0]]!r}) -- rerun; the rest is cached")
    short = {k: clean_name(json.loads(r)["short_name"]) for k, r in replies.items()}
    # A full name that already fits needs no other form.
    for k, r in rows.items():
        if not too_long(r.name):
            short[k] = r.name

    for rnd in range(1, config.SHORT_NAME_ROUNDS + 1):
        groups = [[k] for k in short if too_long(short[k])]
        reasons = {tuple(g): f"This region's short name was too long ({short[g[0]]!r})." for g in groups}
        for g in clashes(short, alive):
            groups.append(g)
            reasons[tuple(g)] = (f"These regions appear in the same windows but got the same short name, "
                                 f"{short[g[0]]!r}; their ribbons must be told apart.")  # fmt: skip
        log(f"round {rnd}: {sum(len(g) == 1 for g in groups)} too long, {sum(len(g) > 1 for g in groups)} clashes")
        if not groups:
            break
        prompts = {tuple(g): group_prompt([rows[k] for k in g], reasons[tuple(g)]) for g in groups}
        got_all = asyncio.run(ask_all(namer, prompts, GROUP_SCHEMA))
        for g, reply in got_all.items():
            if isinstance(reply, Exception):
                log(f"WARNING: re-asking {[key(*k) for k in g]} failed: {reply!r}")
                continue
            for k, name in parse_group(reply, list(g)).items():
                short[k] = name

    out = pd.DataFrame([{"layer": k[0], "track": k[1], "short_name": short[k]} for k in sorted(short)])
    out = out.astype({"layer": "int8", "track": "int32"})
    if out["short_name"].eq("").any():
        raise SystemExit("empty short names -- rerun")
    write_parquet_atomic(out, config.short_names_parquet())

    u = namer.usage
    log(f"Claude: {u['calls']} calls ({u['cached']} from cache), {u['input_tokens']:,} input / "
        f"{u['output_tokens']:,} output tokens = ${namer.cost_usd():.2f}")  # fmt: skip
    for k in sorted(short):
        flag = (" [too long]" if too_long(short[k]) else "") + (" [case]" if not_title_case(short[k]) else "")
        log(f"  {key(*k):>4} {short[k]!r:<24} <- {rows[k].name}{flag}")
    log(f"{sum(map(too_long, short.values()))} over the limit, {len(clashes(short, alive))} clashes left")
    log(f"Wrote {config.short_names_parquet()} ({len(out)} tracks) in {(time.monotonic() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

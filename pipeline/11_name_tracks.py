"""Stage 11: one name per track, from its path of per-window names.

Settled 2026-11-08 / 09 (CLAUDE.md, "naming"). Stage 09's geometry decides the tracks; stage 10 named each
window's clusters on its own; here an LLM reads a track's names window by window and writes the one name its
label carries at every slider position -- generalizing where the content changes along the path (composers
-> film actors), never cutting the track.

Each prompt gives the track's layer and Toponymy's length tier for it (naming.summary_kind: 4-8 words fine,
2-5 mid, 1-4 coarse, the tiers stage 10 asked for), its windows and birth years, its per-window names with
the window's median birth year and the track's size there, its best-known people over the whole path, and
its share of women and men -- with the rule that a name may name a gender only if 95% of the people share it
(stage 10 named clusters 36-46% women "Leading Men").

The reply is structured output (a JSON schema): what the region is in its first windows and in its last
(with their best-known people in the prompt), a candidate name, whether the candidate is true of each end,
then the final name -- the candidate, or something broader that is. A name may not describe anyone as
something they aren't: where no umbrella term is true of everyone, the parts may be joined with "and". (Tested
2026-11-09 on known tracks: asked for the name only, Sonnet named c30 -- composers, then film actors --
"American Film and TV Actors" whatever the wording or effort, and called a track that drifts from serial
killers to adult-film performers "Infamous Sexual Criminals".)

Then names must differ where they're seen together: tracks of one layer that share a name (ignoring case and
punctuation) in any window are renamed together in one prompt, for up to config.TRACK_NAME_ROUNDS rounds.

Model, effort and cache are stage 10's (namer.py, config.NAME_*), without stage 10's appended instructions
(the rules are in the prompt). A failed call leaves the stage without output; rerun, the rest comes from the
cache.

Output: data/track_names_<MAP_EMBEDDING>.parquet: layer (int8), track (int32), name, first_window,
last_window, n_windows, n_people, female_share, male_share, renamed (bool: changed in a distinctness round),
path (the distinct per-window names in order, " | "-joined), earliest / latest / candidate (from the reply,
for review).
"""

import asyncio
import json
import re
import time
from collections import defaultdict

import pandas as pd

import config
from namer import AsyncClaudeNamer
from naming import LAYER_NAMES, clean_name, gender_mismatch, not_title_case, summary_kind
from util import log, write_parquet_atomic

N_LAYERS = 3
SYSTEM = (
    "You name regions of an interactive map of the most famous people of every era, drawn from all Wikipedias. "
    "A slider picks a window of birth years; each window's people are laid out by what they are known for, and "
    "nearby people form labelled regions at three zoom levels (fine, mid, coarse). A region is followed from "
    "window to window as a track, and in each window it was named on its own. You write the one name its label "
    "carries at every slider position."
)
RULES = (
    "Use Title Case. A name may not describe any of the region's people as something they are not. Don't describe "
    "a region by gender (men, women, male, female, leading men, actresses) unless at least 95% of its people share "
    'that gender. Don\'t hedge (no "mostly", "and others" or parentheses).'
)
ASK_ONE = (
    "Write one name for the region. Its label shows this one name in every window listed, so the name must be "
    "true of the region in each of them, including the first and last windows, however few people the region has "
    "there. Where its content changes along the path, use an umbrella term that is true of all of it, as specific "
    'as that allows; if no umbrella term is true of everyone, joining the parts with "and" is better than a '
    "name that is false for some of them. " + RULES + '\n\nReply in JSON: "earliest": what the region is in '
    'its first windows, in a few words; "latest": what it is in its last windows; "candidate": a name for '
    'the region; "fits_earliest" and "fits_latest": whether the candidate is true of the people in the first '
    'and in the last windows; "name": the final name -- the candidate if it fits both ends, otherwise a name '
    "that does."
)
NAME_SCHEMA = {
    "type": "object",
    "properties": {
        "earliest": {"type": "string"},
        "latest": {"type": "string"},
        "candidate": {"type": "string"},
        "fits_earliest": {"type": "boolean"},
        "fits_latest": {"type": "boolean"},
        "name": {"type": "string"},
    },
    "required": ["earliest", "latest", "candidate", "fits_earliest", "fits_latest", "name"],
    "additionalProperties": False,
}
RENAME_SCHEMA = {
    "type": "object",
    "properties": {
        "names": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"region": {"type": "string"}, "name": {"type": "string"}},
                "required": ["region", "name"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["names"],
    "additionalProperties": False,
}


def track_key(layer: int, track: int) -> str:
    return f"{'fmc'[layer]}{track}"


def track_facts(names: pd.DataFrame, tr: pd.DataFrame, people: pd.DataFrame) -> dict[tuple[int, int], dict]:
    """Everything a prompt says about a track: path, windows, birth years, best known, gender shares."""
    p = people.set_index("qid")
    med = tr[["window", "qid"]].merge(people[["qid", "birth_year_est"]], on="qid")
    med = med.groupby("window")["birth_year_est"].median().round().astype(int)
    facts = {}
    for layer in range(N_LAYERS):
        rows = tr.loc[tr[f"track_{layer}"] >= 0, ["window", "qid", f"track_{layer}"]]
        for t, g in rows.groupby(f"track_{layer}"):
            who = p.loc[g["qid"].unique()].sort_values("rank")
            sex = who["gender"].astype(str)
            path = names[(names["layer"] == layer) & (names["track"] == t)].sort_values("window")
            years = who["birth_year_est"].astype(float)
            first, last = int(path["window"].min()), int(path["window"].max())
            ends = {w: p.loc[g.loc[g["window"] == w, "qid"]].sort_values("rank")["label"].astype(str).head(4).tolist()
                    for w in (first, last)}  # fmt: skip
            facts[layer, int(t)] = {
                "layer": layer,
                "track": int(t),
                "path": [(int(r.window), int(med[r.window]), int(r.size), r.name) for r in path.itertuples()],
                "first": first,
                "last": last,
                "first_best": ends[first],
                "last_best": ends[last],
                "born": (int(years.quantile(0.1)), int(years.quantile(0.9))),
                "best_known": who["label"].astype(str).head(config.TRACK_NAME_BEST_KNOWN).tolist(),
                "n_people": len(who),
                "female_share": float(sex.eq("female").mean()),
                "male_share": float(sex.eq("male").mean()),
            }
    return facts


def describe(f: dict) -> list[str]:
    lines = [
        f"It appears in slider windows {f['first']}–{f['last']}; most of its people were born {f['born'][0]} to "
        f"{f['born'][1]}.",
        "Its names, window by window (window · the window's median birth year · people in the region there: name):",
    ]
    lines += [f"  w{w} · {year} · {size}: {name}" for w, year, size, name in f["path"]]
    lines += [
        f"Its best-known people over the whole path: {'; '.join(f['best_known'])}.",
        f"Best known in its first window (w{f['first']}): {'; '.join(f['first_best'])}. In its last window "
        f"(w{f['last']}): {'; '.join(f['last_best'])}.",
        f"Of its {f['n_people']:,} people, {f['female_share']:.0%} are women and {f['male_share']:.0%} are men.",
    ]
    return lines


def track_prompt(f: dict) -> str:
    layer = f["layer"]
    head = (f"Layer: {LAYER_NAMES[layer]} (about {config.LABEL_TARGETS[layer]} labels per window at this layer). "
            f"The name should be {summary_kind(layer)}.")  # fmt: skip
    return "\n".join([head, *describe(f), "", ASK_ONE])


def rename_prompt(group: list[dict], shared: str) -> str:
    layer = group[0]["layer"]
    lines = [
        f"These regions are on the {LAYER_NAMES[layer]} layer of the map and appear in the same windows, but they "
        f'were given the same name, "{shared}". Give each a distinct name that still fits its whole path; each '
        f"should be {summary_kind(layer)}. {RULES}",
        "",
    ]
    for f in group:
        lines += [f"Region {track_key(f['layer'], f['track'])}:", *describe(f), ""]
    lines.append('Reply in JSON: "names", one entry per region, each with its "region" id and its new "name".')
    return "\n".join(lines)


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()


def collisions(final: dict[tuple[int, int], str], names: pd.DataFrame) -> list[list[tuple[int, int]]]:
    """Groups of same-layer tracks that share a final name in at least one window (connected components)."""
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(k):
        parent.setdefault(k, k)
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for (_w, layer), g in names.groupby(["window", "layer"]):
        by_name = defaultdict(list)
        for t in g["track"]:
            by_name[norm(final[layer, int(t)])].append((layer, int(t)))
        for keys in by_name.values():
            for k in keys[1:]:
                parent[find(k)] = find(keys[0])
    groups = defaultdict(list)
    for k in parent:
        groups[find(k)].append(k)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def parse_renames(text: str, keys: list[tuple[int, int]]) -> dict[tuple[int, int], str]:
    """The rename reply (RENAME_SCHEMA JSON) -> {key: name} for the regions asked about; others are ignored."""
    wanted = {track_key(*k): k for k in keys}
    out = {}
    for e in json.loads(text)["names"]:
        rid = re.sub(r"^region\s+", "", e["region"].strip(), flags=re.IGNORECASE).lower()
        if rid in wanted and clean_name(e["name"]):
            out[wanted[rid]] = clean_name(e["name"])
    return out


async def ask_all(namer: AsyncClaudeNamer, prompts: dict, schema: dict) -> dict:
    """Run prompts concurrently (the namer's semaphore caps it); a failed call maps to its exception."""
    keys = list(prompts)
    calls = (namer._create(SYSTEM, prompts[k], schema) for k in keys)
    results = await asyncio.gather(*calls, return_exceptions=True)
    return dict(zip(keys, results))


def name_tracks(facts: dict, names: pd.DataFrame, namer: AsyncClaudeNamer) -> tuple[dict, set, dict]:
    """First names for every track, then distinctness rounds. Returns ({key: name}, {renamed keys}, {key: the
    first reply's JSON: earliest, latest, candidate, fits_earliest, fits_latest, name})."""
    replies = asyncio.run(ask_all(namer, {k: track_prompt(f) for k, f in facts.items()}, NAME_SCHEMA))
    failed = [k for k, r in replies.items() if isinstance(r, Exception)]
    if failed:
        raise SystemExit(f"{len(failed)} calls failed (first: {replies[failed[0]]!r}) -- rerun; the rest is cached")
    checks = {k: json.loads(r) for k, r in replies.items()}
    final = {k: clean_name(c["name"]) for k, c in checks.items()}
    renamed: set = set()
    for rnd in range(1, config.TRACK_NAME_ROUNDS + 1):
        groups = collisions(final, names)
        log(f"distinctness round {rnd}: {len(groups)} groups of tracks share a name in some window "
            f"({sum(map(len, groups))} tracks)")  # fmt: skip
        if not groups:
            break
        prompts = {tuple(g): rename_prompt([facts[k] for k in g], final[g[0]]) for g in groups}
        replies = asyncio.run(ask_all(namer, prompts, RENAME_SCHEMA))
        for g, reply in replies.items():
            if isinstance(reply, Exception):
                log(f"WARNING: renaming {[track_key(*k) for k in g]} failed: {reply!r}")
                continue
            got = parse_renames(reply, list(g))
            for k in g:
                if k in got and got[k] and got[k] != final[k]:
                    final[k] = got[k]
                    renamed.add(k)
            if len(got) < len(g):
                log(f"WARNING: renaming {[track_key(*k) for k in g]}: got {len(got)} of {len(g)} names")
    return final, renamed, checks


def main() -> None:
    t0 = time.monotonic()
    names = pd.read_parquet(config.window_names_parquet())
    tr = pd.read_parquet(config.tracks_parquet())
    people = pd.read_parquet(config.MAP_PEOPLE_PARQUET, columns=["qid", "label", "rank", "gender", "birth_year_est"])
    facts = track_facts(names, tr, people)
    log(f"{len(facts)} tracks ({[sum(k[0] == i for k in facts) for i in range(N_LAYERS)]} fine / mid / coarse) "
        f"from {len(names):,} per-window names; {config.NAME_MODEL} at effort {config.NAME_EFFORT}")  # fmt: skip
    namer = AsyncClaudeNamer(instructions="")
    final, renamed, checks = name_tracks(facts, names, namer)

    out = pd.DataFrame([{
        "layer": f["layer"], "track": f["track"], "name": final[k], "first_window": f["first"],
        "last_window": f["last"], "n_windows": len(f["path"]), "n_people": f["n_people"],
        "female_share": round(f["female_share"], 3), "male_share": round(f["male_share"], 3),
        "renamed": k in renamed, "path": " | ".join(dict.fromkeys(n for *_, n in f["path"])),
        "earliest": checks[k]["earliest"], "latest": checks[k]["latest"], "candidate": checks[k]["candidate"],
    } for k, f in sorted(facts.items())])  # fmt: skip
    out = out.astype({"layer": "int8", "track": "int32"})
    empty = out["name"].eq("").sum()
    if empty:
        raise SystemExit(f"{empty} empty names -- rerun")
    write_parquet_atomic(out, config.track_names_parquet())

    u = namer.usage
    log(f"Claude: {u['calls']} calls ({u['cached']} from cache), {u['input_tokens']:,} input / "
        f"{u['output_tokens']:,} output tokens = ${namer.cost_usd():.2f}")  # fmt: skip
    words = out["name"].str.split().str.len()
    by_layer = words.groupby(out["layer"]).median().tolist()
    clashes = len(collisions(final, names))
    log(f"Words per name, median by layer (fine / mid / coarse): {by_layer}; {len(renamed)} tracks renamed for "
        f"distinctness; {clashes} name clashes left")  # fmt: skip
    for r in out.itertuples():
        f = facts[r.layer, r.track]
        if gender_mismatch(r.name, f["female_share"]):
            log(f"  review: {track_key(r.layer, r.track)} names a gender ({f['female_share']:.0%} women): {r.name}")
        if not_title_case(r.name):
            log(f"  review: {track_key(r.layer, r.track)} not title case: {r.name}")
    log(f"Wrote {config.track_names_parquet()} ({len(out)} tracks) in {(time.monotonic() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

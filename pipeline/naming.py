"""Checks and conventions shared by the naming stages (10: per-window names, 11: one name per track)."""

import re

import config

GENDERED = re.compile(r"\b(male|female|men|women|man|woman)\b", re.IGNORECASE)
MALE_TERMS = re.compile(r"\b(male|men|man|leading men|sportsmen|kings and princes)\b", re.IGNORECASE)
FEMALE_TERMS = re.compile(r"\b(female|women|woman|actress|actresses|sportswomen)\b", re.IGNORECASE)
GENDER_BAR = 0.95  # a name may name a gender only if at least this share of the region's people share it
SMALL_WORDS = set(
    "a an and as at by for from in into of on or the to vs with under over after before between during through "
    "within without upon via per de del la le von van der da di du y al el".split()
)
LAYER_NAMES = ("fine", "mid", "coarse")


def not_title_case(name: str) -> bool:
    """A word starting with a lower-case letter that isn't a small word ("and", "of", ...) Title Case allows;
    the first word must always be capitalized. Words are whitespace-separated, so "Mid-20th-Century" is one."""
    words = [w.strip("()[]\"'“”‘’,;:") for w in name.split()]
    # an elided particle starts a foreign proper name: "all'Italiana", "d'Or"
    words = [w for w in words if w and w[0].isalpha() and not re.match(r"^(?:all|dell|d|l)['’]", w)]
    if not words:
        return False
    if words[0][0].islower():
        return True
    return any(w[0].islower() and w.lower() not in SMALL_WORDS for w in words[1:])


def gender_mismatch(name: str, female_share: float) -> bool:
    """The name names a gender that fewer than GENDER_BAR of the region's people share."""
    male = MALE_TERMS.search(name) and not re.search(r"\bMad Men\b", name)  # a TV title, not a gender
    return bool((male and 1 - female_share < GENDER_BAR) or (FEMALE_TERMS.search(name) and female_share < GENDER_BAR))


def summary_kind(layer: int, n_layers: int = 3) -> str:
    """Toponymy's length-and-style phrase for a layer at config.NAME_DETAIL_LEVELS, e.g. "specific and detailed
    (4 to 8 word)": the same tier stage 10 asked for, spread evenly from the finest layer to the coarsest."""
    from toponymy.templates import SUMMARY_KINDS

    lo, hi = config.NAME_DETAIL_LEVELS
    level = lo + (hi - lo) * layer / max(n_layers - 1, 1)
    return SUMMARY_KINDS[int(round(level * (len(SUMMARY_KINDS) - 1)))]


def clean_name(text: str) -> str:
    """A model's bare-name reply: first line, without surrounding quotes, markdown or a final period."""
    line = next((ln for ln in text.strip().splitlines() if ln.strip()), "").strip()
    line = re.sub(r"^(?:name\s*:\s*)", "", line, flags=re.IGNORECASE).strip()
    while True:  # peel quotes, markdown and a final period, in whatever order the model nested them
        peeled = line.strip().strip("*_`\"'“”‘’").strip()
        if peeled.endswith(".") and not peeled.endswith(".."):
            peeled = peeled[:-1]
        if peeled == line:
            return line
        line = peeled


def review(entries: list[dict]) -> list[str]:
    """Names to look at: empty, gendered, not title case, or repeated within a layer of one window."""
    notes = []
    for e in entries:
        tag = f"layer {e['layer']} track {e['track']}"
        if not e["name"]:
            notes.append(f"{tag}: EMPTY")
            continue
        if GENDERED.search(e["name"]):
            notes.append(f"{tag}: gendered word: {e['name']}")
        if not_title_case(e["name"]):
            notes.append(f"{tag}: not title case: {e['name']}")
    seen: dict[tuple, int] = {}
    for e in entries:  # the same track keeps a similar name across windows; only clashes within one matter
        key = (e.get("window"), e["layer"], e["name"].casefold())
        if e["name"] and key in seen:
            notes.append(f"window {e.get('window')} layer {e['layer']}: tracks {seen[key]} and {e['track']} share "
                         f"the name {e['name']}")  # fmt: skip
        seen.setdefault(key, e["track"])
    return notes

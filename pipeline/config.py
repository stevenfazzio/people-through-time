"""Central configuration for the people-through-time-map pipeline.

All knobs live here -- no CLI args (family convention shared with
living-people-map / movie-madness-map / steam-atlas / jeopardy-wikipedia-map).
For smoke tests, lower the knobs (fewer months, smaller MAP_N); don't
add flags.

Values marked PROVISIONAL are working defaults, not decisions -- see the
"Open questions" section of CLAUDE.md before relying on them.
"""

from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = "people-through-time-map"
PROJECT_TAGLINE = "A semantic map of the most famous people of every era"

# Wikimedia REQUIRES a descriptive User-Agent with real contact info --
# requests with a generic/placeholder UA get rate-limited (429) quickly.
USER_AGENT = f"{PROJECT_NAME}/0.1 (https://github.com/stevenfazzio/{PROJECT_NAME}; fazzios@gmail.com)"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
LOG_DIR = REPO_ROOT / "logs"

# --- Roster: Wikidata humans (P31=Q5) with >=1 language-Wikipedia article ---
# "Wikipedia" = sites with wikibase:wikiGroup "wikipedia" (on the SITE iri, not
# the sitelink node). That excludes abstract.wikipedia.org, Wikiquote, Commons.
WIKIDATA_HUMAN_QID = "Q5"

# --- Fame metric: median of 12 monthly all-language pageview totals ---
# PROVISIONAL window. Same 12 months as living-people-map (comparable to it) vs.
# the freshest trailing 12 is an open question; this is the fresher option.
PAGEVIEW_MONTHS = [f"{y}-{m:02d}" for y, m in [(2025, 11), (2025, 12)] + [(2026, m) for m in range(1, 11)]]
PAGEVIEW_URL_TEMPLATE = (
    "https://dumps.wikimedia.org/other/pageview_complete/monthly/{year}/{year}-{month}/pageviews-{year}{month}-user.bz2"
)
# Keep the desktop / mobile-web / mobile-app split (living-people-map summed it
# away). Steady high desktop-only traffic is the first bot-filter signal: Ansel
# Adams ran ~1.0-1.4M "user" views every month of 2025-07..2026-06, ~2x
# Elizabeth II. Matters most in the earliest windows, where the floor below is
# only ~10 views/day.
KEEP_ACCESS_METHOD_SPLIT = True

# --- Roster selection: rank everyone by median_views x age (decided 2026-11-08) ---
# Lindy reading: something famous for A years can expect ~A more, so today's
# monthly readership x age ~ estimated lifetime (past + future) attention. One
# formula replaces the per-era fame bars the earlier log-time stratification
# set by hand; on the enwiki proxy the implied bar falls smoothly from ~19k
# views/mo (1980+ births) to ~200 (BCE). Age = years since birth, not years
# famous: AGE_OFFSET=20 is the closer-to-Lindy variant (penalizes young
# moderns harder, gives history more room); 0 keeps the one-sentence story.
FAME_REFERENCE_YEAR = 2026
AGE_OFFSET = 0
MAP_N = 31_000  # PROVISIONAL: revisit with all-language views + an AlignedUMAP benchmark
# Applied BEFORE ranking. Without it the formula admits BCE people at ~190
# views/mo (~6/day) -- inside crawler noise.
MIN_MEDIAN_VIEWS = 300


def fame_score(median_views, birth_year):
    """median_views x max(age - AGE_OFFSET, 1); works on scalars or arrays."""
    return median_views * np.maximum(FAME_REFERENCE_YEAR - birth_year - AGE_OFFSET, 1)


# --- Slider: equal-count bins over birth year, windows of consecutive bins ---
# N_BINS bins holding ~equal numbers of roster people (edges snapped to whole
# years, so a person's bin depends only on their birth year). A window =
# WINDOW_BINS consecutive bins stepping one bin (80% overlap between
# neighbours, so AlignedUMAP relations are dense): N_BINS - WINDOW_BINS + 1 = 32
# windows of ~4.3k people. Single-handle slider; the frontend interpolates
# between precomputed windows.
N_BINS = 36
WINDOW_BINS = 5


def bin_edges(birth_years) -> np.ndarray:
    """Equal-count birth-year bin edges for the roster, snapped to whole years; outer edges are +-inf."""
    q = np.quantile(np.asarray(birth_years, dtype=float), np.linspace(0, 1, N_BINS + 1))
    edges = np.unique(np.floor(q))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


# --- Text to embed ---
# English lead when the person has an enwiki article; otherwise the lead of
# their most-viewed edition, LLM-translated to English (decided 2026-11-07).
TRANSLATE_NON_ENGLISH = True

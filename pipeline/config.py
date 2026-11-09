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

DUMPS_DIR = DATA_DIR / "dumps"  # raw downloads; deleted after parsing unless KEEP_RAW_DUMPS
KEEP_RAW_DUMPS = False

# --- Stage 00: roster = Wikidata humans (P31=Q5) with >=1 language-Wikipedia article ---
# "Wikipedia" = sites with wikibase:wikiGroup "wikipedia" (on the SITE iri, not
# the sitelink node). That excludes abstract.wikipedia.org, Wikiquote, Commons.
QLEVER_URL = "https://qlever.dev/api/wikidata"
QLEVER_PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX schema: <http://schema.org/>
"""
ROSTER_PARQUET = DATA_DIR / "roster.parquet"  # [qid, n_wikipedias]

# --- Stage 01: Wikipedia sitelinks for every roster human ---
WB_ITEMS_PER_SITE_URL = "https://dumps.wikimedia.org/wikidatawiki/latest/wikidatawiki-latest-wb_items_per_site.sql.gz"
SITELINKS_PARQUET = DATA_DIR / "sitelinks.parquet"  # [qid, site, title] (title underscored)

# --- Stage 01b: enwiki redirect map (renames + aliases; non-English redirects NOT resolved) ---
ENWIKI_DUMP_URL = "https://dumps.wikimedia.org/enwiki/latest/{name}"
REDIRECTS_PARQUET = DATA_DIR / "redirects.parquet"  # [source_title, target_title] (underscored)

# --- Stage 02: fame metric = median of 12 monthly all-language pageview totals ---
# Freshest trailing 12 complete months (decided 2026-11-08; the 2026-10 dump
# was published 2026-11-01). living-people-map's 2025-07..2026-06 window would
# only have bought comparability: its all-language views covered living people
# and its raw dumps are gone, so every month had to be re-fetched either way.
PAGEVIEW_MONTHS = [f"{y}-{m:02d}" for y, m in [(2025, 11), (2025, 12)] + [(2026, m) for m in range(1, 11)]]
PAGEVIEW_URL_TEMPLATE = (
    "https://dumps.wikimedia.org/other/pageview_complete/monthly/{year}/{year}-{month}/pageviews-{year}{month}-user.bz2"
)
PAGEVIEWS_DIR = DATA_DIR / "pageviews"
PAGEVIEW_WORKERS = 2  # months processed concurrently; dumps.wikimedia.org allows ~2-3 connections
# Keep the desktop / mobile-web / mobile-app split (living-people-map summed it
# away): stage 03's bot filter works on the desktop:mobile ratio.
ACCESS_METHODS = ["desktop", "mobile-web", "mobile-app"]


def pageview_month_parquet(month: str) -> Path:
    """[qid_int, lang, access, views] for one month."""
    return PAGEVIEWS_DIR / f"pageviews_{month.replace('-', '')}.parquet"


# --- Stage 03: per-person view metrics, with the access-ratio bot filter ---
PEOPLE_VIEWS_PARQUET = DATA_DIR / "people_views.parquet"
# Bot filter (decided 2026-11-08). Real readers move every access method of a
# language together; a crawler counted as "user" inflates one. Per (person,
# language), in its median month: r = log((D + 1) / (M + 1)) - log(that
# language's D/M over every roster human), D = desktop, M = mobile web + app.
# r is clipped to [-log MAX_MOBILE_SKEW, +log MAX_DESKTOP_SKEW] by shrinking the
# over-represented side, by the same fraction in every month. Real audience
# skew stays well inside (academic pages ~+1..2, true crime ~-1.4). Median
# month, so one-month bursts are left to the median of monthly totals.
# Cut, e.g.: Fiann Paul eswiki ~51k desktop / 75 mobile a year, Craig Small en.
MAX_DESKTOP_SKEW = 16
MAX_MOBILE_SKEW = 8

# --- Stage 04: Wikidata facts (dates, gender, label, description) for everyone above the floor ---
WDQS_URL = "https://query.wikidata.org/sparql"
WDQS_CHUNK = 1_000  # QIDs per VALUES query: 3-5 s each on 2026-11-08, far inside WDQS's 60 s
WDQS_SLEEP = 1.0  # between queries; WDQS allows a client ~60 s of query time per minute
FACTS_RAW_DIR = DATA_DIR / "raw" / "wikidata_facts"  # per-chunk raw rows, the resumable checkpoint
PEOPLE_FACTS_PARQUET = DATA_DIR / "people_facts.parquet"
# Fallback birth year when there is no usable P569: the first of baptism, floruit,
# work-period start, death, minus its offset. Offsets = median gap to P569 among
# people with both at year precision (measured on the 2026-11-08 run over 792k
# people; stage 04 prints the medians again on every build).
BIRTH_FALLBACK_OFFSETS = {"baptism": 0, "floruit": 46, "work_start": 20, "death": 71}
MAX_LIFESPAN = 122  # longer, or death before birth (both at year precision): no birth year


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
# PROVISIONAL. First real-data build (2026-11-08, all-language views + Wikidata
# birth years): 52% born before 1900, 19 of 36 bins; AlignedUMAP at this size ~10 min.
MAP_N = 31_000
# Applied BEFORE ranking. Without it the formula admits BCE people at ~190
# views/mo (~6/day) -- inside crawler noise. It is the binding bar for anyone
# born more than ~5,800 years ago, so no age cap is needed (a 5,000-year cap
# changes no roster place; 3,000 years swaps 66).
MIN_MEDIAN_VIEWS = 300
# Hand-curated birth years for historical people whose Wikidata date comes from
# a traditional chronology (the rule is in the file's header). Versioned, not in data/.
BIRTH_YEAR_OVERRIDES_CSV = REPO_ROOT / "curated" / "birth_year_overrides.csv"
MAP_PEOPLE_PARQUET = DATA_DIR / "map_people.parquet"  # stage 05: the roster, one row per person, rank order


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


# --- Stage 06: the text to embed = each person's Wikipedia lead, in English ---
# English lead when the person has an enwiki article; otherwise the lead of
# their most-viewed edition, translated to English by Claude (decided 2026-11-07).
WIKI_API_URL = "https://{lang}.wikipedia.org/w/api.php"  # lang = pageview-dump code (stage 03's top_lang)
LEAD_BATCH = 20  # titles per request: the exlimit maximum for intro extracts
LEAD_WORKERS = 2  # concurrent requests; extracts take ~0.1-1.3 s on enwiki
LEAD_MIN_INTERVAL = 0.4  # s between request starts across workers: <= 150/min of Wikimedia's ~200/min budget
THUMB_SIZE = 320  # px, hovercard thumbnail width
LEADS_RAW_DIR = DATA_DIR / "raw" / "leads"  # <site>.jsonl, one record per requested title: the resumable cache
TRANSLATIONS_DIR = DATA_DIR / "raw" / "translations"  # <qid>.json, reused while source text and prompt are unchanged
TRANSLATE_MODEL = "claude-opus-5-5"
TRANSLATE_EFFORT = "low"  # faithful translation of a few paragraphs needs little reasoning
TRANSLATE_WORKERS = 4
LEADS_PARQUET = DATA_DIR / "leads.parquet"


# --- Stage 07: embeddings (Qwen3-Embedding-8B through OpenRouter; decided 2026-11-08) ---
# The family's current embedder (tv-tropes-map, deity-map); replaces
# living-people-map's Cohere embed-v4.0. $0.01 per million tokens on 2026-11-08.
EMBED_API_URL = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL = "qwen/qwen3-embedding-8b"
# Pinned, fallbacks off (see embedder.py). Nebius is the one provider listing
# bf16; DeepInfra (unstated precision) and SiliconFlow (fp8) also serve it.
# Toponymy's naming stage must use the same provider.
EMBED_PROVIDER = "Nebius"
EMBED_BATCH_ITEMS = 32
EMBED_BATCH_CHARS = 120_000
EMBED_WORKERS = 4
EMBED_CACHE = DATA_DIR / "raw" / "embedding_cache.sqlite"  # keyed by model, provider and the exact text sent
# Qwen3-Embedding takes a task instruction ("Instruct: ...\nQuery:<text>").
# Both variants are embedded; scoping/embed_variants.py compares them.
EMBED_INSTRUCTIONS = {
    "plain": None,
    "role": "Identify what the person described in this Wikipedia article opening is known for: their role, their "
    "field, and the society they lived in",
}
# The variant later stages use (decided 2026-11-08): "role" groups people with
# their peers (Napoleon -> his marshals) where "plain" leans to family and circle.
# Its cost, stronger gender separation, is what stage 07b's erasure takes out.
EMBED_VARIANT = "role"


def embeddings_npz(variant: str = EMBED_VARIANT) -> Path:
    """[qid, vectors] row-aligned with map_people.parquet (rank order)."""
    return DATA_DIR / f"embeddings_{variant}.npz"


# --- Stage 07b: gender erasure (decided 2026-11-08, after the AlignedUMAP preview) ---
# One global centroid projection: the unit (female mean - male mean) direction,
# fit on the roster's male/female people, projected out of every vector, which is
# then renormalized. LEACE was indistinguishable and per-window erasers added
# little in 2-d (scoping/gender_erasure*.py). Linear only: "actress", "queen" and
# pronouns still carry gender. The direction is saved with the vectors and must
# also be projected out of anything else embedded into this space (Toponymy's
# keyphrases and region names).
ERASE_GENDER = True
MAP_EMBEDDING = f"{EMBED_VARIANT}_nogender" if ERASE_GENDER else EMBED_VARIANT  # what stage 08 lays out


# --- Stage 08: AlignedUMAP layout of every slider window ---
# Benchmarked 2026-11-08 (scoping/bench_aligned_umap.py, living-people-map's
# 25k cut into this slider structure): umap-learn 0.5.12's alignment defaults
# moved a shared person a median 0.049 layout radii between neighbouring
# windows, ~5x steadier than independent UMAPs + Procrustes. Revisit
# ALIGNMENT_REGULARISATION only if the frontend shows jumpy motion. min_dist is
# a clustering parameter here: labels are clustered in the 2-d layout.
UMAP_KW = {"n_neighbors": 15, "min_dist": 0.05, "metric": "cosine", "random_state": 42}  # family defaults
ALIGNMENT_REGULARISATION = 1e-2
ALIGNMENT_WINDOW_SIZE = 3
N_WINDOWS = N_BINS - WINDOW_BINS + 1  # window w = bins w .. w + WINDOW_BINS - 1


def layout_parquet(variant: str = MAP_EMBEDDING) -> Path:
    """[window, qid, x, y]: one row per person per window they belong to."""
    return DATA_DIR / f"layout_{variant}.parquet"


# --- Stage 09: label tracks ---
# Settled 2026-11-08 (scoping/label_tracks.py; CLAUDE.md "label tracks"): each
# window's 2-d layout is clustered on its own with Toponymy's primitives
# (fast_hdbscan leaf clusters), and clusters in neighbouring windows are linked
# into tracks by shared people. A layer's min cluster size is the same in every
# window, chosen so a window shows LABEL_TARGETS labels on average (scoping:
# 39 / 76 / 158 people). Layer 0 is the finest, as in Toponymy.
LABEL_TARGETS = (40, 20, 10)  # labels per window, finest layer first
LABEL_MIN_SAMPLES = 5  # Toponymy's default
TRACK_LINK_MIN = 0.3  # a match needs this share of the smaller cluster's people present in both windows
TRACK_BRIDGE = True  # also link a track that ended to a cluster starting two windows on (one-window gap)


def tracks_parquet(variant: str = MAP_EMBEDDING) -> Path:
    """[window, qid, track_0 .. track_<n-1>]: each person's track per layer in each window; -1 = unlabelled."""
    return DATA_DIR / f"tracks_{variant}.parquet"


def track_tree_parquet(variant: str = MAP_EMBEDDING) -> Path:
    """One row per (layer, track): its majority parent, lifetime and size."""
    return DATA_DIR / f"track_tree_{variant}.parquet"


# --- Stage 10: naming (Toponymy per window, then one name per track) ---
# Settled 2026-11-08 (CLAUDE.md "naming"). Toponymy names each window's layers with
# Claude (namer.py); a second step names each track from its path of per-window
# names. Keyphrases and candidate names are embedded with stage 07's embedder
# *without* the role instruction -- Qwen3's query/document pairing: the people were
# embedded as instructed queries, keyphrases are plain documents -- and with stage
# 07b's gender direction removed, so they share the people's space.
# Sonnet 5.5 chosen 2026-11-09 for cost (half Opus 5.5's price); window 16 with
# Opus 5.5 cost $1.20 (scoping/results/name_window_opus.txt).
NAME_MODEL = "claude-sonnet-5-5"  # $2 / $10 per MTok in / out (claude-api skill, 2026-10-06)
NAME_EFFORT = "medium"
NAME_WORKERS = 8  # concurrent Claude requests
NAME_DETAIL_LEVELS = (0.4, 0.8)  # Toponymy lowest/highest detail level: 4-8 words finest, 1-4 coarsest
NAME_CACHE = DATA_DIR / "raw" / "llm_cache.sqlite"  # every response, keyed by model, effort and prompt
# Appended to every naming prompt (2026-11-09), for two slips Sonnet 5.5 made on window 16: names in
# sentence case, and a gender the cluster doesn't have ("... American Male Actors" over Judy Garland).
NAME_INSTRUCTIONS = (
    "\n\nAdditional rules for every name you give: write it in Title Case, and do not describe a group by "
    "gender (male, female, men, women) unless every person in it shares that gender. Keep your response in the "
    "JSON format requested above."
)
NAME_OBJECT_DESCRIPTION = "Wikipedia biographies (the opening of each person's article)"
NAME_CORPUS_DESCRIPTION = (  # {lo}, {hi}: the window's earliest and latest birth year
    "the most famous people born between {lo} and {hi}, from all Wikipedias -- one era of a map of the most "
    "famous people of every era"
)
EMBED_SHORT_BATCH_ITEMS = 256  # keyphrases and names: 256 base64 vectors = ~5.6 MB per response
WINDOW_NAMES_DIR = DATA_DIR / "raw" / "window_names"  # one JSON per named window, so a rerun resumes


def window_names_parquet(variant: str = MAP_EMBEDDING) -> Path:
    """[window, layer, track, size, name]: Toponymy's name for each track in each window."""
    return DATA_DIR / f"window_names_{variant}.parquet"

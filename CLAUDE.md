# people-through-time-map — project notes

A datamap of the most famous people of **every era**, from **all** Wikipedias,
with a birth-year slider. Each slider position is its own AlignedUMAP layout of
the people born in that window; as the slider moves, people glide, appear and
disappear, and so do the region labels. Direct descendant of
living-people-map (`~/repos/living-people-map`), whose stages, gotchas and
pageview tooling this reuses; also a sibling of movie-madness-map, steam-atlas
and jeopardy-wikipedia-map. Follow the family conventions: numbered pipeline
stages, `pipeline/config.py` as the single source of knobs, no CLI args, uv +
ruff line-length 120, parquet for tables / npz for vectors, atomic writes,
`data/` gitignored and expensive.

Status (2026-11-09): live at https://stevenfazzio.com/people-through-time/
(public repo stevenfazzio/people-through-time, GitHub Pages from `docs/` on
`main`; see "Publishing"). Frontend v1 is a custom deck.gl page in `docs/`
(stages 11b-13; see "Frontend" below). Earlier status (2026-11-08): roster design settled (views × age, switched from
log-time stratification the same day). Stages 00–08 run; the bot filter
(access-ratio clip) is settled and lives in stage 03; birth years (with
fallbacks and the round-century fix) come from stage 04; stage 05 picks the
31k roster and its slider bins; stage 06 fetches and translates the leads;
stage 07 embeds them twice (plain and role-instructed Qwen3); the map uses
role, with gender projected out by stage 07b; stage 08 lays out all 32
windows with one AlignedUMAP; stage 09 clusters each window into three label
layers and links the clusters into tracks; stage 10 names every window's
clusters with Toponymy and Claude Sonnet 5.5; stage 11 gives each track one
name for its whole path; stage 11b gives each coarse track a short name for the
timeline chart; stage 12 exports what the page in `docs/` loads. `scoping/` holds the one-off analysis behind
the decisions below; `curated/` holds hand-made inputs (versioned, unlike
`data/`).

## Pipeline

Run in order with `uv run python pipeline/NN_*.py`; tests with `uv run pytest`.

| Stage | What | Output |
|---|---|---|
| 00 | Roster: Wikidata humans with ≥1 Wikipedia article (QLever; raw CSV cached in `data/raw/`) | `data/roster.parquet` |
| 01 | Every roster human's Wikipedia titles (`wb_items_per_site` dump) | `data/sitelinks.parquet` |
| 01b | enwiki redirect map (`page` + `redirect` dumps) | `data/redirects.parquet` |
| 02 | Monthly views per (person, language, access method), 12 dumps | `data/pageviews/pageviews_YYYYMM.parquet` |
| 03 | Per-person monthly totals and median after the bot filter, raw median, access shares, top language | `data/people_views.parquet` |
| 04 | Wikidata dates (birth, death, baptism, floruit, work start), gender, label, description for people above the floor (WDQS, per-chunk cache in `data/raw/wikidata_facts/`) | `data/people_facts.parquet` |
| 05 | Fame score, top `MAP_N` roster after `curated/birth_year_overrides.csv` (traditional chronologies, Wikidata errors), slider bin, enwiki and top-language titles | `data/map_people.parquet` |
| 06 | English lead per roster person (enwiki intro, else the top edition's intro translated by Claude), enwiki short description, thumbnail (caches in `data/raw/leads/`, `data/raw/translations/`) | `data/leads.parquet` |
| 07 | Qwen3-Embedding-8B vectors of the leads via OpenRouter (pinned to Nebius), one file per instruction variant (cache: `data/raw/embedding_cache.sqlite`) | `data/embeddings_<variant>.npz` |
| 07b | Gender erasure: one global centroid direction projected out of the role vectors (`ERASE_GENDER`) | `data/embeddings_role_nogender.npz` (+ `direction`) |
| 08 | One AlignedUMAP over the 32 slider windows (`config.MAP_EMBEDDING`) | `data/layout_role_nogender.parquet` (window, qid, x, y) |
| 09 | Label tracks: each window's layout clustered at three fixed sizes (`config.LABEL_TARGETS`), clusters linked across windows (with a lineage check against relays), tracks cut where their content drifts, majority-parent tree | `data/tracks_role_nogender.parquet` (window, qid, track_0..2; 0 finest, -1 unlabelled), `data/track_tree_role_nogender.parquet` |
| 10 | Per-window names: Toponymy over stage 09's clusters as a pre-fitted clusterer, Claude via `pipeline/namer.py` (cache `data/raw/llm_cache.sqlite`; one JSON per window in `data/raw/window_names/`, names keyed by their cluster's members, so reruns resume and relinking keeps them) | `data/window_names_role_nogender.parquet` (window, layer, track, size, name) |
| 11 | One name per track from its path of per-window names (Claude, structured self-checking reply), then renaming tracks of a layer that share a name in some window | `data/track_names_role_nogender.parquet` (layer, track, name, windows, gender shares, path, the reply's earliest / latest / candidate) |
| 11b | Short name (≤ 3 words, ≤ 18 characters) per coarse track, for the timeline chart's ribbons (Claude, same model and cache) | `data/track_short_names_role_nogender.parquet` (layer, track, short_name) |
| 12 | Frontend export: windows normalized, positions quantized, label centroids and sizes, ribbon order, colours, drift-cut predecessors, hovercard text | `docs/data/{core.json, points.bin, details.json}` (committed: Pages serves them) |
| 13 | Link card: headless system Chrome screenshots the page (`uv run --extra publish`) | `docs/social-preview.png` |

Stage 02 run 2026-11-08: 160 min for 12 dumps with 2 workers; 4.60–4.82M
people with views per month; desktop share 23–26% every month. Stage 03
(156 s, 9.8 GB peak): 793,215 people at or above the 300 median-views floor
raw, 792,471 after the bot filter. Stage 04 fetch (2026-11-08): 793 WDQS
queries of 1,000 QIDs, ~5 s each, 75 min including a restart (a body cut off
mid-stream; now retried) and 9 transient 429/502s; a rebuild from the cache
takes ~4 min. 94.0% have a P569, 1.0% a fallback, 4.9% (38,873) no birth year.
Stage 05 (12 s, 2026-11-08, after the birth-year corrections): bar 1,740,672
views/mo × years; 36 bins from 10,546 BC–651 BC (803 people) down to 3–4
years wide in 1940–1990, 736–1,002 per bin; 32 windows of 4,144–4,423,
137,574 AlignedUMAP points; 288 roster people (0.9%) have no enwiki article,
most of them Japanese-led.

Stage 06 (2026-11-08): 1,534 enwiki requests of 20 titles in 16 min at
~96/min with 2 workers, no 429s or maxlag. An enwiki page is used only if it
is the person's own article: 72 roster sitelinks are redirects, mostly renames
and joint articles (Cyril and Methodius, Askold and Dir, the Two Qiaos — kept,
so such pairs share a lead and a map position), but 9 redirect to another
human's article (Heather McCartney → Paul McCartney, Ariston of Athens →
Plato, Princess Sophie → Marie Antoinette, five Three Kingdoms figures →
their commanders), whose intro, short description and thumbnail would all be
the wrong person's. Those, Sulayman ibn Khalid (enwiki page gone) and Romuald
(empty TextExtracts intro) fall to their top edition. Result: 30,700 enwiki
leads (median 802 chars, p10 246), 297 translated (median ~356 chars; sources
median 122 chars, CJK expanding ×3–5 in English), 3 Wikidata-description
fallbacks (Romuald, Hyunri, Heather McCartney). Translation: Claude Opus 5.5
at low effort with `fallbacks: "default"`, ~$1.76 in all, no refusals; output
read clean (no preambles). Thumbnail 85%, enwiki short description 98.6%. A
rebuild from the caches takes ~10 s.

Stage 07 (2026-11-08): ~15 min per variant with 4 workers, 32 transient
429/529s from Nebius (529 "worker overloaded" crashed the first run until
every 5xx was retried); $0.05 (plain) + $0.09 (role, the instruction adds
tokens). Same text twice on Nebius: cosine 0.9999; Nebius vs DeepInfra 0.997.

Stage 08 (2026-11-08, first run on the unerased role vectors): 4.8 min,
2.5 GB peak, 137,579 points. A shared
person moves a median 0.055 layout radii between neighbouring windows (p90
0.151, p99 0.425; benchmark 0.049), fastest in windows 9–20 (~1260–1930);
0.68% of moves exceed 0.5 radii, mostly people between two groups. 15-NN
fidelity 0.364 (benchmark 0.288). Window centres stay put; spread narrows in
the middle centuries (x std 5.8 → 3.3 → 3.7). 2-d same-gender neighbour
share (for the erasure question): 0.96–0.98 in windows 0–10 (women 14–20%,
chance ~0.74), 0.87–0.93 from window 12 on (women 18–39%, chance 0.52–0.70);
as a share of the room above chance, ~0.85 in every era — women cluster
nearly completely throughout. Animated preview:
`scoping/layout_preview.py` → `data/scoping/layout_<embedding>_by_{role,gender}.html`.
Stage 07b + 08 rerun on the erased vectors (2026-11-08): held-out gender
probe AUC 1.000 → 0.722; 2-d same-gender share 0.924 → 0.736 averaged over
windows (chance 0.663), i.e. the excess over chance 0.78 → 0.22 — most left
in the earliest windows (0.31–0.37: consorts and queens, whose text is about
being someone's wife). Motion median 0.055 (p99 0.373), fidelity 0.357:
unchanged. Logs: `logs/08_aligned_umap.role.log` (before), `.log` (after).
After the birth-year corrections (current outputs): probe 0.708, 4.7 min,
motion median 0.058 (p99 0.370), fidelity 0.357.

Stage 09 (2026-11-08): 2 s. Min cluster sizes 39 / 76 / 158 give 39.5 /
20.2 / 9.7 labels per window; 270 / 110 / 50 tracks; 23–26% of a window
unlabelled; a labelled person's label switches in 14–18% of window steps
(42–49% of those are merges, the rest boundary crossings); layers nest
strictly within every window. Tree: 93% of fine tracks have a mid parent (4%
skip to coarse, 3% root), 97% of mid tracks a coarse parent; the median
track sits 100% under its parent. Same numbers as A+ in
`scoping/results/label_tracks.txt`. Log: `logs/09_label_tracks.log`.

Stage 09 rerun with the lineage check (2026-11-09; decision below): 2 s.
Links dropped and rematched 9 / 3 / 3 (fine / mid / coarse; the coarse three
are the relays); 270 / 111 / 48 tracks; a labelled person's label switches
in 18.2 / 14.5 / 14.9% of steps (was 17.7 / 14.3 / 14.4: relays hid real
switches); lowest carrier lineage share left 33 / 30 / 29%. Stage 10 kept all
2,218 names with no Claude calls; stage 11 cost $0.14 over both rule versions;
11b and 12 rerun (17 colours for 48 coarse tracks). Pre-check outputs and
logs: `data/before_lineage/`.

Stage 09 rerun with drift cuts (2026-11-09; decision below): 6 s (loading
the 4096-d vectors). Drift bars 0.580 / 0.467 / 0.315 (fine / mid / coarse:
each layer's median window-centred cosine to the nearest other region); cuts
28 / 17 / 10 in 24 / 17 / 9 tracks, 6 of the 10 coarse ones moved a window
or two by the boundary refinement; 298 / 128 / 58 tracks; label switches
20.1 / 16.4 / 16.5% of steps (a cut counts as one). Stage 10 kept every
name (no calls); stage 11 $0.63 and 11b $0.06 over the unrefined and
refined runs; stage 12: still 17 colours for 58 coarse tracks. Pre-cut
outputs: `data/before_drift/`.

Stage 10 (2026-11-09): 50.5 min, $17.33 (Sonnet 5.5, effort medium; 1,771
Claude calls, 8.2M input / 90k output tokens; 69 more from the cache of a
first attempt that crashed writing window 0's JSON). 2,218 names (1,263 /
646 / 309 fine / mid / coarse), none empty; median 7 / 6 / 5 words.
Keyphrase embedding: 1.0M tokens, $0.01, base64 responses in batches of 256
(`EMBED_SHORT_BATCH_ITEMS`); 75 transient Nebius 429s, all retried;
windows after the first reuse most phrases from the cache, so a window took
44–138 s. The appended instructions (`NAME_INSTRUCTIONS`) kept every name in
Title Case, but 8 names still misgender their cluster, nearly all through
the idiom "Leading Men" over clusters 36–46% women (Meryl Streep, Demi
Moore, Julia Roberts; tracks f186, f211, m66 in windows 18–28), plus "Male
Character Actors" (91% men) and "Notorious Female Criminals"; stage 11 gets
each track's gender mix to fix them. Paths per track:
`scoping/track_paths.py` → `data/scoping/track_paths.html` (c30 runs
"European Classical Composers and Performers" → "Classical Composers and
Hollywood Stars" → "Golden Age Hollywood Stars" → ... → "American Film and
TV Actors"; c0 runs from "Ancient Indian Rulers and Buddhists" through
"Islamic Caliphs and Sultans" and "Mughal Era South Asian Rulers" to "Heads
of State and Government"). Log: `logs/10_name_windows.log`.

Stage 11 (2026-11-09): 1.9 min, $1.65 (424 Sonnet 5.5 calls, 550k input /
55k output tokens). 430 track names; 2 pairs of mid tracks shared a name in
some window and were renamed apart in one round; no gendered or mis-cased
names left; median 8 / 5 / 4 words fine / mid / coarse (the tiers ask 4–8,
2–5, 1–4). Prompt per track: layer and length tier, windows and birth
years, the per-window names with window median birth year and size, the
best-known people overall and in the first and last windows, and the share
of women and men. Asked for the name only, Sonnet ignored small early
windows whatever the wording or effort (c30, composers then actors, stayed
"American Film and TV Actors") and invented a false umbrella for a drifting
track ("Infamous Sexual Criminals" over serial killers and adult-film
performers); a structured reply that states the first and last windows,
proposes a candidate, checks it against both ends and then gives the final
name fixed both (c30 "Classical Composers and Screen Actors", c27 "Political
and Business Elites" after its candidate failed the last windows). Cost of
honesty: 89% of names join parts with "and" or commas, and mixed tracks read
oddly ("Thinkers and Nazi Leaders", "Novelists and Murderers", "Mesopotamian
Kings and Companions of Muhammad") -- geometry signals, per the naming
decision. Review page: `scoping/track_paths.py` →
`data/scoping/track_paths.html` (final name over each path). Log:
`logs/11_name_tracks.log`.

Stage 11b (2026-11-09): 60 Sonnet 5.5 calls, $0.10. Full names already within
the limit are kept as they are (Western Painters, Star Athletes); 7 were too
long after the first ask, 2 still are after two re-ask rounds (c27 "Political
& Business" 20 characters, c34 "Novelists & Killers" 19) and are kept, since the
page decides what fits; no two tracks alive together share one. E.g. c30
"Composers & Actors", c17 "Popes & Monarchs", c4 "Kings & Companions". Log:
`logs/11b_short_names.log`.

Stage 12 (2026-11-09): 4 s. 17 colours for the 50 coarse tracks (1-window
buffer; 12 without); ribbon-order flow cost 12,944 → 3,376 people × tracks in
between; lead excerpts median 197 characters, 390 cut mid-sentence; 19 names
from a title. Files: core.json 1.41 MB (0.48 MB gzipped), points.bin 0.69 MB
(0.60), details.json 11.29 MB (3.79), page 0.05 MB: **13.44 MB raw, 4.89 MB
gzipped**, plus deck.gl 9.1.15 from unpkg (1.60 MB, 0.44 MB gzipped), the
Google fonts and thumbnails fetched on hover. Log: `logs/12_export_frontend.log`.
Serve `docs/` (it fetches its data files, so no `file://`):
`python3 -m http.server 8771 --bind 127.0.0.1 --directory docs`.

Planned next: see "Frontend, what's left".

## Working rules

- Smoke-test by lowering config knobs (fewer `PAGEVIEW_MONTHS`, smaller
  `MAP_N`), never by adding CLI flags.
- Never overwrite files in `data/` in place — `util.write_parquet_atomic` /
  `util.save_npz_atomic`.
- **QLever results are cached, not re-queried.** Stage 00 reuses
  `data/raw/roster_qlever.csv` when present (delete it to refresh). The first
  cache was seeded 2026-11-08 from `data/scoping/qlever/wp_counts.csv` — the
  identical query, run 2026-11-07 23:03 — because QLever was answering nginx
  429s again.
- Long runs: background them with output to `logs/` and check the log.
- Wikimedia etiquette: descriptive User-Agent with contact info
  (`config.USER_AGENT`), `maxlag=5` on Wikipedia action-API calls but not on
  Wikidata reads, modest sleeps. The `wikimedia-data` skill has the current
  rate limits and QLever behavior — read it before any Wikimedia fetch.

## Decisions (dated)

- **2026-11-07 — roster = Wikidata humans with ≥1 language-Wikipedia
  article.** 13,776,677 humans (`P31=Q5`); **4,848,996** have ≥1 Wikipedia
  article; 2,155,479 have an English one. Not an enwiki category roster: the
  category scoping pulled in non-person pages ("Deaths in 2026", "September 11
  attacks", "Murder of Brianna Ghey") and has no birth year for Jesus or
  Genghis Khan, both of which Wikidata dates (circa, with precision).
- **2026-11-07 — fame = median of 12 monthly *all-language* pageview totals**,
  the living-people-map metric (median resists news spikes). All-language,
  not enwiki: early eras on enwiki skew European, and the age weighting below
  makes early-era depth the point. Built from the monthly `pageview_complete`
  dumps (~6 GB each), credited via Wikidata sitelinks for every roster human —
  living-people-map's stage 01e did this for living people only, so the dumps
  must be re-fetched.
- **2026-11-08 — pageview window = 2025-11..2026-10**, the freshest 12
  complete months (the October dump was published 2026-11-01). Reusing
  living-people-map's 2025-07..2026-06 would only have bought comparability:
  every month had to be re-downloaded either way.
- **2026-11-07 — ranking by raw median views is rejected.** On the enwiki
  proxy (2.1M enwiki biographies by median enwiki views), the top 25k is 72.7%
  living, 9.1% born before 1900, and 73% identical to living-people-map's
  enwiki top 25k. Twenty equal-count windows over it put **all births before
  1828 into the first window** — history would be one slider position.
- **2026-11-08 — roster = top `MAP_N` (31k) by `median_views × age`**, age =
  2026 − birth year; supersedes the 2026-11-07 log-time stratification.
  Lindy reading: something famous for A years can expect ~A more, so today's
  rate × age ≈ estimated lifetime (past + future) attention, up to a constant.
  One formula yields a smooth per-era fame bar instead of hand-set ones. Proxy
  result (`scoping/results/roster_preview.txt`): 35.8% born before 1900, 17.9%
  before 1500, 44.6% living; lowest views admitted falls from 18,841/mo
  (1980+) through 6,954 (1900–49) and 851 (1000–1499) to the 300 floor (BCE).
  It naturally drops famous-by-parentage children (Princess Charlotte at
  78k/mo, Kim Ju Ae, Prince Louis, Prince Archie, Blue Ivy Carter are out).
  `AGE_OFFSET = 20` (years famous ≈ age − 20) is the closer-to-Lindy variant
  — 43.0% before 1900, harder on young moderns — kept as a knob, not used:
  plain age keeps the one-sentence story. `views × age^α` generalizes it, but
  only α = 1 has a story. Comparison: `scoping/results/fame_score.txt`.
  The superseded design (log-time buckets on `log(2060 − year)`, top 1,000 per
  bucket, 300 floor; `scoping/results/strata_logtime_rejected.txt`, whose
  script went with its knobs) gave history ~72% of the slider and 30,842
  people, but its 2060 had no principled basis, it needed three hand-set
  knobs, and window density ranged 1,958–5,000. Accepted cost of the switch:
  history's share of the slider falls from ~72% to ~33%.
- **2026-11-07 — floor 300 median views/month, applied before ranking.**
  Unfloored, `views × age` admits BCE people at 193/mo (~6/day), inside
  crawler noise. On the proxy the floor removes only 49 of 31k.
- **2026-11-08 — slider = single handle over precomputed windows of
  equal-count bins.** 36 bins over the roster's birth years with ~equal
  people (edges snapped to whole years, so a person's bin depends only on
  their birth year; `config.bin_edges()`), windows of 5 consecutive bins
  stepping one bin (80% overlap, so AlignedUMAP relations are dense). Proxy:
  bins span −2650..−86, −85..395, 400..810, 811..1073, … down to 2–3 years in
  the 1950s–80s; 12 of 36 bins lie wholly before 1900; 32 windows of
  4,086–4,470 people; ~137k AlignedUMAP points. A two-handle range slider is
  out: AlignedUMAP can only show slices it precomputed. The frontend
  interpolates positions between neighbouring windows.
- **2026-11-08 — AlignedUMAP is cheap and smooth enough; defaults kept.**
  Benchmark (`scoping/bench_aligned_umap.py`, results in
  `scoping/results/bench_aligned_umap.txt`) on living-people-map's 25k
  Cohere embeddings cut into this slider structure: 32 windows of
  3,091–3,845, 109,863 points, **7.9 min** single-threaded, 1.1 GB peak
  (the real ~137k-point job should take ~10 min). A shared person moves a
  median **0.049** layout radii between neighbouring windows (p90 0.112, p99
  0.291) vs 0.269 / 0.842 / 2.223 for independent UMAPs Procrustes-aligned
  afterwards — ~5× steadier. Cost: 15-NN fidelity to the 1024-d neighbours
  0.288 vs 0.304. umap-learn 0.5.12 defaults (`alignment_regularisation=1e-2`,
  `alignment_window_size=3`), family UMAP params (15 neighbours, min_dist
  0.05, cosine, seed 42). Revisit `alignment_regularisation` only if the
  frontend shows jumpy motion. Positions saved to
  `data/bench/aligned_living.npz` for a frontend prototype.
- **2026-11-07 — embed English text; translate when there's no English
  article.** English lead if the person has an enwiki article, else the lead
  of their most-viewed edition, LLM-translated. Rejected: native-language text
  with a multilingual embedder (language would likely become a map organizer,
  as gender did in living-people-map) and Wikidata-fact templates (cluster by
  template). Sizing (`scoping/results/non_english.txt`): of people with 10+
  editions only ~1% lack English, 0.1% of the top 31k by edition count; among
  5+ editions, 2.9–6.7% by death era, highest in antiquity. So a few hundred to
  ~2k translations, mostly in early windows. Real roster (stage 05): only 289
  (0.9%) lack an enwiki article — 188 Japanese-, 43 Chinese-, 22 Arabic-led;
  39 BCE, 128 born 1–1499. Stage 06 translates the intro only (`exintro`),
  as decided; Japanese intros are often one sentence, so translated leads run
  shorter than English ones (median 356 vs 802 chars). Check afterwards: kNN purity of an
  `is_translated` flag (same diagnostic as gender purity). Mark translated
  text in the hovercard; link to the source-language article.
- **2026-11-07 — bot filtering via the access-method split.** Keep desktop /
  mobile-web / mobile-app separate instead of summing. Ansel Adams drew a
  steady 1.0–1.4M "user" views every month of 2025-07..2026-06 (~2× Elizabeth
  II), Cleopatra 0.7–1.5M — probably crawlers, unverified. **The age
  weighting amplifies this for old pages**: a crawler adding 300 views/mo to a
  page for someone born in 500 BC adds as much score as 30k views/mo would for
  a 25-year-old. So bot filtering matters most in the earliest windows, and
  must run before scoring.
  **2026-11-08 finding — desktop share is audience, not bots**
  (`scoping/results/bot_features.txt`, `bot_access_ratio.txt`). Academic
  subjects run 45–70% desktop on every language (Aristotle, Descartes, Knuth),
  so a desktop-share filter would cut the earliest eras hardest. Ansel Adams
  (en 92% mobile-web) and Cleopatra are mobile-skewed, inside the range of
  real mobile-heavy topics (true crime, celebrities), so the access split
  can't flag them. A per-language desktop:mobile ratio vs that language's
  baseline does flag single-cell desktop skew (Tenskwatawa ruwiki 740k
  desktop / 7k mobile; Harriet Martineau en; Michelangelo fr; Bruce Lee en),
  and clipping it moves only 12–38 of 31k proxy-roster places.
  **Early eras** (`desktop_floor.txt`, `edition_floor.txt`): crawler bursts
  exist (Gungunum: ~2,600 desktop hits on each of 8 editions in 2025-11
  only), and the median absorbs them. Early pages' extra desktop share is
  proportional to traffic and dips in summer like student reading, and their
  desktop series share no crawler schedule (first principal component as
  strong in mobile). Every page carries a small desktop background (~15/mo on
  near-unread enwiki pages, ~3/mo on other editions); subtracting even
  5–10/mo per edition moves 244–381 of 31k proxy places, all marginal ancient
  royals (Ammuna, Neferu II), whose views are only 2–5% long-tail editions.
  **Settled 2026-11-08 — access-ratio clip in stage 03**
  (`config.MAX_DESKTOP_SKEW = 16`, `MAX_MOBILE_SKEW = 8`; no background
  subtraction). Per (person, language): r = log((D+1)/(M+1)) − log(the
  language's D/M over every roster human), measured on the cell's **median
  month**, clipped to [−log 8, +log 16] by shrinking the over-represented side
  by one fraction in every month. Median month because
  (`scoping/results/clip_granularity.txt`): with 12-month sums a one-month
  burst the median already ignores scales the clean months down too
  (Gungunum −12%, Harriet Martineau −36%, Chen Fan −23%); clipping each month
  separately is noisy on small cells (2,310 raw survivors fall below the floor
  vs 615; Martial −22%, Aa −21%). Run: 122k desktop-side and 3.8k mobile-side
  clips over 6.5M (person, language) cells remove 0.036% of floor survivors'
  views; 744 fall below the floor; the proxy roster moves 8–9 of 31k places
  (Aureolus of Aragon: frwiki 11.9k desktop / 515 mobile a year). Cut: steady
  single-edition crawlers — Fiann Paul (eswiki, 18.0k → 2.3k/mo), Craig Small
  (19.7k → 2.9k), Charles Stuart (abolitionist), mobile-side Nek and Neige
  (arzwiki); Michelangelo −13% (frwiki desktop); borderline Martial −10% (enwiki
  85% desktop, r 2.84). Untouched: bursts (Gungunum, Martineau, Tenskwatawa),
  Bruce Lee (enwiki r 2.53, inside the band), Ansel Adams and Cleopatra (their
  mobile traffic affects rank, not membership). `people_views.parquet` keeps
  `median_views_raw` and `clip_keep` beside the filtered `median_views`.
- **2026-11-08 — birth year = Wikidata P569, with fallbacks; the undated are
  unscoreable.** WDQS, not QLever (30 s cap, lockouts): chunked `VALUES`
  queries fit easily in WDQS's 60 s. Pick: preferred > normal, highest
  precision, lower median; round century/millennium values to their midpoint
  (gotcha below). Fallback when there is no usable P569, in order: baptism,
  floruit − 46, work-period start − 20, death − 71 (`BIRTH_FALLBACK_OFFSETS` =
  median gap to P569 among people with both, measured on this run). Births or
  deaths after 2026 count as missing; death before birth or a life over 122
  (both at year precision) → no birth year (`birth_source = "conflict"`, 76
  people: longevity claimants such as Li Ching-Yuen and Thomas Parr, Emperor
  Jimmu's legendary dates, typos like Marie Marvingt "born 2015"). Of 792k
  above the floor, 38,873 have no birth year, but only 11–58 of them would
  clear the roster bar at age 25–60 (`scoping/results/birth_coverage.txt`):
  mostly living Japanese idols, internet personalities and Indian actors with
  no P569 (Sophie Rain, Jūtarō Yamanaka), plus D. B. Cooper. Fallbacks place
  ~870 of the previewed 31k (764 by death year: Leonidas I, Edward the
  Confessor, the historical Jesus).
- **2026-11-08 — curated birth-year overrides; no age cap.** Rule (header of
  `curated/birth_year_overrides.csv`): override only a *historical* person
  whose Wikidata birth year comes from a traditional or religious chronology
  and sits centuries from the scholarly dating; legendary figures with no
  scholarly date (Ishmael, Shaodian, Gao Yao, Parashara) keep their
  traditional dates. Ranges become midpoints, as in stage 04. A scan of all
  792k floor survivors (Alvar/Nayanar names, Kali Yuga years, year-precise
  births before −2600) found five, all Tamil saints: Bhoothath and
  Thirumazhisai Alvar (−4202 → 650; Lochtefeld via enwiki *Alvars*: the first
  Alvars belong to the 7th century), Madhurakavi and Thondaradippodi Alvar
  (−3101 / −2813 → 750, alongside Nammalvar, Andal and Thiruppaan, whom
  Wikidata already dates to 750), Kannappa Nayanar (−3101 → 650; enwiki
  *Nayanars*: 6th–8th centuries). The four Alvars fall below the bar at
  their real dates; Kannappa stays, moving from the first bin to bin 6.
  Stage 05 warns when Wikidata's year changes under an override. **No age
  cap for prehistoric remains**: the 300 floor is already the only bar for
  anyone born more than ~5,800 years ago, a 5,000-year cap changes no roster
  place and 3,000 years swaps 66 marginal ancients. The roster's 29 remains
  (Ötzi, Cheddar Man, mostly Iron Age bog bodies) stay; the bog bodies span
  several windows and should form their own region.

- **2026-11-08 — embedding variant = role (instructed), gender erased (stage
  07b, see Gender erasure).** Comparison (
  `scoping/embed_variants.py` → `scoping/results/embed_variants.txt`, UMAP
  panels in `data/scoping/embed_variants.html`). Role instruction: "Identify
  what the person described in this Wikipedia article opening is known for:
  their role, their field, and the society they lived in". Within-window
  15-NN purity, plain → role: role category (keyword from the short
  description) 0.754 → 0.788 (chance 0.255; higher in every era); top
  language 0.836 → 0.835 (chance 0.564); **gender 0.908 → 0.945** (chance
  0.663); translated leads' neighbours translated 0.177 → 0.145 (chance 0.010
  — and 3.7× the rate of same-language untranslated people, so not just
  culture); birth-year locality equal. The variants share 63% of neighbours.
  Probes: plain leans to the person's family and circle (Napoleon → Carlo,
  Louis, Letizia Bonaparte; Leonardo → his mother, Lisa del Giocondo), role
  to their peers (Napoleon → his marshals; Leonardo → Raphael, Perugino;
  Genghis Khan → the later khans; Taylor Swift → country singers). Cost of
  role: stronger gender separation (Cleopatra → only Ptolemaic queens) —
  which stage 07b's erasure removes. Chosen for the peer grouping. Plain stays
  embedded for a side-by-side if needed.

- **2026-11-08 — junk Wikidata birth years are curated, and stage 06 warns
  about new ones.** Found in the layout preview: the age weighting turns a bad
  P569 into a huge score, and 14 roster people, all born in the 20th century,
  sat in early windows. Kinds: the day of the month entered as the year (Yurie
  Igoma "24" for February 24, 1999; Minami Hinata, Yu-ri Yoshida, Daniela
  Zaiushkina — only a birthday known); placeholder year 1 (Omar Metwally,
  Claudio Tapia, Xaviersobased); truncation (Jason Douglas 173, Amirhossein
  Mahmoodi 200); typos (Hwang In-yeop 1492, Durai Murugan 1834, Tommy
  Hilfiger 29, Chris McKenna 20); Hijri (Nawaf bin Saad 1391 AH = 1971).
  `curated/birth_year_overrides.csv` now takes plain data errors too, with
  the year from the person's own enwiki article, or an empty year when no
  source gives it (unscoreable). Rerun: 12 left the roster, 12 came in at the
  bottom (Niklas Luhmann, Creighton Abrams, Michael Chekhov, ...); Hilfiger
  (rank 12,220) and Hwang stay at their real ages. Stage 06 logs a warning for
  birth years the person's own article contradicts; it caught all 14, and the
  only one left is Veer Teja (folk hero: lead 1074–1103, Wikidata 1255,
  traditional dates — left alone). Wikidata descriptions are no automatic fix:
  their "(YYYY–YYYY)" is often a term of office (Obama 2009–2017).

- **2026-11-08 — label tracks: cluster each window on its own and link the
  clusters across windows; hierarchy by majority parent.** Compared with one
  3-d clustering of the stacked windows (`scoping/label_tracks.py` →
  `scoping/results/label_tracks.txt`; previews
  `data/scoping/label_tracks_{A,B}.html`, `label_tracks_frames.png`). Both
  with Toponymy's own primitives (leaf clusters, min_samples 5), matched on
  ~40 / 20 / 10 labels per window. A: per-window 2-d clusters (min size
  39 / 76 / 158) linked by mutual-best member overlap; A+ also bridges
  one-window gaps. B: all 137k (window, person) points in (x, y, λ·window),
  λ 0.02–0.16 (median 5-NN distance in a window ~0.08; a person moves a
  median 0.30 per step). B is steadier (a labelled person's label switches
  in 0.4–11% of steps vs 14–18%; label centroids move about half as far)
  but can't cover the map: 49–61% of each window unlabelled vs 23–26%, at
  most ~25 / 13 / 6 labels per window at A's size bars, and about half its
  coarse labels straddle separate pieces (A 18%); in window 16 the whole
  early-film region has no coarse label. Drift of a track's content from
  first to last window is similar (A+ 2–13% of tracks, B 3–17%). Half of
  A's switches are boundary crossings, half merges (two regions join for a
  window; the smaller track ends and its people take the larger's label).
  People per track per step as stacked areas: `scoping/track_sizes.py` →
  `data/scoping/track_sizes.html` (A: 1.6 tracks born and 1.7 ended per
  step, unlabelled 8–42% by window; B: 0.8 / 0.7, 44–87%; A's film & TV
  track runs windows 13–31 and peaks at 35% of window 23); size
  distributions: `scoping/cluster_sizes.py`.
  **Hierarchy.** Within a window the layers nest strictly (each is the same
  density tree cut at a larger size): a fine cluster lies wholly inside one
  mid cluster (84%) or wholly in mid's unlabelled space (16%), never split;
  mid in coarse 80 / 20. Across windows each layer is linked on its own, so
  46% of fine tracks change mid parent at least once and 49% of mid tracks
  change coarse parent — half of those because the parent track ended
  (merge flicker), half because the child moved. Chosen: each layer linked
  independently (a coarse flicker doesn't reset fine labels), then each
  child track's parent = the track at the next coarser layer holding most
  of its person-windows; a child never under a named parent there goes to
  the next layer up, else the root (Toponymy's convention). Measured: fine
  → mid for 93% of fine tracks, skip a layer 4%, root 3%; mid → coarse 97%,
  root 3%; the median child sits 100% under its chosen parent, 86–87% of
  person-windows overall, and only 4% of children are under it less than
  half the time. Rejected: top-down linking (a child continues only under
  the same parent) — strict tree, but every coarse break would reset all
  the tracks below it.

- **2026-11-08 — naming: Toponymy per window, then one name per track from
  its path.** Geometry decides the map's structure (stage 09's tracks: which
  regions exist and where a track starts and ends); the LLM only describes
  it. Toponymy names each window on its own, where the layers nest strictly
  and its tree, exemplars and disambiguation work as designed — and
  disambiguation runs among exactly the labels on screen together. A second
  LLM step turns each track's path of per-window names into one name for the
  whole path, generalizing where the content changes ("composers" → "film
  and TV" becomes something like "entertainment industry"), then checks that
  tracks sharing a window get distinct names. The LLM never cuts a track.
  Supersedes naming whole tracks with people filed under their most common
  track plus a hand-built tree. Drift cuts stay off: the prototype
  (`scoping/track_segments.py` → `scoping/results/track_segments.txt`,
  `data/scoping/track_segments.html`) cut 10 of 430 tracks on embedding
  drift, but whether that's needed shows only in the consolidated names; a
  path that only fits a very vague name is a signal to fix the geometry
  (linking or a geometric cut), not to let the LLM split it.
  **One-window test** (window 16, born 1831–1924; `scoping/name_window.py`
  → `scoping/results/name_window_opus.txt`, `data/scoping/name_window_16_opus.json`;
  2026-11-09): stage 09's clusters handed to Toponymy 0.5.4 as a pre-fitted
  clusterer (41 / 21 / 9 clusters), Opus 5.5 at medium effort, detail levels
  0.4–0.8. 58 Claude calls, 287k input / 2.6k output tokens (~4,950 in and
  45 out per call: eight full article openings per prompt dominate) = $1.20;
  0 empty names. 26.7 min, ~22 of them embedding Toponymy's keyphrase
  vocabulary (its default cap, 50,000 phrases per window, in batches of 32;
  43 transient Nebius 429s, all retried; $0.002). Names are specific and
  read well ("Late Romantic and Early Modern Classical Composers", "Soviet
  Bolshevik Leaders and Eastern Bloc Communist Politicians"); median 8 / 6 /
  5 words fine / mid / coarse. A cluster unchanged from the layer below keeps
  its name (12 of 21 mid, 2 of 9 coarse), so some coarse labels are long
  ("French Impressionist, Post-Impressionist and Modernist Painters"). Mixed
  clusters get list names ("Record-Holders, Supercentenarians, Sideshow
  Performers, Martial Artists, and Journalists"; c27 "American Tycoons,
  Physicists, Presidents"). c30 here is "Film, Stage and Music Performers"
  (Chaplin, Brando, Sinatra, Tchaikovsky). At this rate all 32 windows cost
  ~$38 (every second window ~$19), plus the per-track step.
  **Sonnet 5.5 on the same window** (2026-11-09, switched for cost;
  `scoping/results/name_window_sonnet.txt`; side by side in
  `scoping/compare_namers.py` → `data/scoping/namer_comparison.html`):
  same 58 calls and prompts, $0.60, 0 empty names; all 32 windows ~$19.
  Similar length (median 7 vs 8 words fine). Less faithful in places: a
  gender the cluster doesn't have ("Classic Hollywood American Male Actors"
  over Judy Garland and Katharine Hepburn, on a gender-erased map), a
  narrower name than the members ("Classic Hollywood Actors" for c30, which
  holds Sinatra and Tchaikovsky; "American Sports Hall of Famers" for record
  holders and supercentenarians), embellishments ("... in Paris",
  "Nobel-era"), and mixed title/sentence case. **Kept Sonnet 5.5**
  (2026-11-09) for the cost, knowing these gaps; normalize the case when
  names are written. The namer now opens a client
  per request: one kept across Toponymy's event loops threw harmless "Event
  loop is closed" tracebacks.

- **2026-11-09 — Frontend: a custom deck.gl page whose slider is a
  ThemeRiver of the coarse tracks.** Discussed and agreed with Steven before
  building; the timeline-as-slider was his addition (after Priestley's *Chart
  of Biography*, streamgraphs / ThemeRiver and Sparks's *Histomap*).
  - **Renderer**: plain JS on deck.gl 9.1.15 (the version datamapplot 0.7.3
    loads, pinned from unpkg), no build step: `docs/index.html`, `app.js`,
    `style.css`, hand-written and versioned. datamapplot + `custom_js` was
    rejected: its point layer, label layer, search and hover picking all
    assume one fixed point set, so only the title panel would have survived.
  - **Slider**: s over the 32 windows, continuous while dragging -- a person
    in windows w and w + 1 glides between their two positions, one in only one
    fades in or out in place; labels the same with their track's centroid --
    and it eases (420 ms) to the nearest window on release, since only windows
    are real layouts and clusters. Arrow keys step a window, play runs at 0.75
    windows/s, pausing settles. Opens on window 16 (born 1831-1924).
  - **Window normalization** (stage 12, `NORMALIZE_WINDOWS`): each window is
    centred and scaled to RMS radius 1. AlignedUMAP's windows breathe (RMS
    radius 6.7 at window 0, 3.95 at 14, 5.9 at 31; window 12 → 13 shrinks
    11%), which at the rim was bigger than a typical person's real move
    (median 0.058 radii). A similarity transform per window: nobody's place in
    their window changes, and a fixed camera stays framed.
  - **Timeline / slider**: SVG under the map (27vh, 190-270 px). x = the 36
    equal-count bins with year ticks at their true positions -- uneven in
    years, as Histomap's axis is -- and the window on the map is a 5-bin band
    with the rest shaded; drag it (by its handle or anywhere on it) or click
    anywhere. One ribbon per coarse
    track, as thick as its people in each window (window w's count at its
    centre bin w + 2.5; the half-window midpoint is 0 where a track starts or
    ends, so ribbons taper; first and last windows extend to the edges), drawn
    as a uniform cubic B-spline of stacked control points (smooth, never below
    zero, stacked boundaries coincide). 2 px gaps; people outside every coarse
    track are left out. Ribbon order: one fixed order (no
    crossings), from the tracks' mean height on the map, then single-track
    moves while they lower the people × tracks stacked between the two ends of
    each coarse-track flow between windows (merges, splits). Ribbon label:
    the full name where it fits horizontally inside the ribbon at 13-9 px,
    else stage 11b's short name, else none (tooltip on hover). Hovering a
    ribbon highlights its people on the map (others at 10% alpha); hovering a
    person highlights their ribbon.
  - **The river fills the chart** (2026-11-10, Steven asked whether the
    total said anything). It was centred like ThemeRiver and scaled to the
    largest labelled total, so its height was coarse coverage: windows hold
    ~equal people by design, so it could never show how many famous people
    were born then (the uneven axis does), only the share in a coarse region,
    58-92% by window -- a clustering artifact that read as "fewer famous
    people born 1866-1952" (the river's narrowest, 62% of full height), and
    jumped with size-threshold crossings (81% → 60% from window 17 to 18).
    Now each knot's ribbons fill the height: a ribbon is its region's share
    of the window's people in coarse regions. Measured on a scratch copy
    (ribbon labels, today / grey remainder / filled): 19 / 19 / 28 at
    1440×900, 10 / 7 / 17 at 1024×768. Cost: a ribbon swells when a
    neighbour leaves the coarse layer, but coverage alone changes a ribbon by
    a median 7% per window step (p90 23%, max 34% at 17→18) against a median
    28% (p75 59%) real change in a track's people; 23 of 241 track steps
    change direction, all near-flat (worst: European Royalty 372 → 313 drawn
    flat at 11%). Comparing one region across distant windows is off by up to
    1.6× (coverage 58% vs 92%). Rejected: the window's whole 100% with the
    uncovered share as a grey band (exact ribbons, but fewer labels and a big
    grey wedge under 1866-1952). Kept honest by words: the hint says "share
    of the window; people in no large region (grey on the map) are left out",
    the About text gives the 8-42%, and the ribbon tooltip keeps counts.
  - **Band handle** (2026-11-10, Steven: dragging the band also highlighted a
    region). Grabbing the band means pressing on a ribbon that hover has just
    highlighted, and ribbon hovers were ignored during the drag, so that
    region stayed highlighted (the map dimmed) for the whole drag. Now a tab
    on the band's top edge (46 × 12 px, grip lines, grab cursor) in a strip
    above the ribbons, its hit area the band's width; the cursor reaches it
    from the map without crossing a ribbon. Pressing the band clears the
    highlight, and after release hovers wait until the pointer moves 3 px:
    releasing pointer capture makes the browser enter the ribbon under a
    still pointer, which lit a region at the moment of letting go. The strip
    costs 2 ribbon labels at 1440×900 (26).
  - **Birth-year readout** (2026-11-10, Steven: the range was a small line
    under the map, and a reader could take the slider for the years people
    were famous -- someone on 1925-1952 asking where the WWII leaders are,
    who are in windows 14-19, born 1874-1891). The range is big
    on the map, "Born 1925 – 1952" in the title serif at 30% white behind the
    points (Gapminder's year; deck.gl's canvas is transparent), lower left,
    with the count and "aged 40 in 1965 – 1992" under it. The aged-40 line
    turns birth years into roughly the years these people were active, and
    shows only for windows of at most 150 years (`AGED_SPAN_MAX`, 1773-1914
    on): over centuries-wide windows 40 years is noise. It says "aged" so the
    tense still works once the years run past 2026. The band's edges carry
    the window's first and last years on the axis (bold, white; the ticks
    under them hide), and the subtitle says the slider is birth years, not
    fame years, with the example put as where to look ("you'll find World
    War II's leaders around 1880, not 1940"; stated as a birth-decade fact
    it read as a non-sequitur). The small header
    readout and its "window N of 32" are gone. Considered and dropped: three
    famous names per window (the top three by score repeat across windows
    and are awkward picks: Hitler tops windows 16-19, Ansel Adams -- the
    crawler suspect -- is second in 17-19, Epstein third in 1945-1964); a
    lifespan shadow on the timeline (the 1925-1952 people were alive in
    WWII, as children).
  - **Colour**: points by coarse track (grey `#5b6472` outside every coarse
    track; blended while a person changes track). Interval colouring with a
    1-window buffer, so no two coarse tracks alive within a window of each
    other share a colour and colour never implies continuity: 17 glasbey
    colours (CVD-safe, lightness 55-85 for the dark background). Ribbons share
    them, so the chart is the legend.
  - **Labels on the map**: all three layers, pixel sizes 12 / 14 / 17 (fine /
    mid / coarse), priority coarse > mid > fine, then region size, as in
    datamapplot: the overview shows mostly coarse labels and finer ones appear
    as zooming spreads them. Collisions are resolved on the CPU each frame
    (project the ~70-90 candidates, greedy by priority, padded boxes [10, 6]
    px) with a 180 ms crossfade. The title panel counts as a placed box, a
    label whose anchor is off screen is skipped, and one whose text runs past
    an edge is nudged back inside (added after the live page, in a 760 px
    pane, hid the top coarse label under the panel). deck.gl's CollisionFilterExtension was tried
    first, as datamapplot uses it, and let partly overlapping labels through:
    it tests each label only at its anchor against a picking-colour collision
    map, and even padded background boxes at test scale 2.5 didn't stop it
    here (not fully diagnosed).
  - **Hovercard** (living-people-map's layout, dark): coarse-region pill in
    its colour, name, short description, Born / Died / Views rows, thumbnail,
    lead excerpt (whole sentences of the first paragraph up to 280 characters,
    parentheticals dropped), footer. Born is "c. X" at decade precision or
    coarser and "c. X (est.)" when stage 04 inferred it from death, floruit or
    work start. Translated leads say "Translated by Claude from the Japanese
    Wikipedia"; click opens the article the lead came from (the source edition
    for translated leads, which also fixes the 9 whose enwiki title redirects
    to someone else; Wikidata for the 3 description leads).
  - **Search**: names (and short descriptions once details.json is in) of all
    31k, every token matched; tiers -- tokens start words of the name, are in
    the name, are in the description -- each in fame order. Typing dims
    non-matches on the map; picking a result moves the slider to the person's
    middle window, flies there and pins their card.
  - **Data**: core.json + points.bin first (positions as uint16 per (person,
    window), person by person in rank order; names, windows, labels, ribbon
    order, palette), details.json fetched after the first paint (the 3.8 MB
    gzipped bulk). Kept external, not inlined: one inline file would make
    every visitor wait for all 13 MB.

- **2026-11-09 — Publishing.** Public repo `stevenfazzio/people-through-time`,
  GitHub Pages (legacy build) from `docs/` on `main`, served at
  https://stevenfazzio.com/people-through-time/ through the user site's custom
  domain, as the sibling maps are. `docs/data/` is committed (13.4 MB): CI
  can't rebuild it, since `data/` stays local. Head carries OG / Twitter card
  tags; the card is stage 13's screenshot of window 16 at the page's own
  framing with the title panel hidden (zooming in cropped the top and bottom
  labels in the 1200 x 630 frame). The site's favicon reaches the page from
  the domain root. Listed on the stevenfazzio.com projects index
  (`src/content/projects/people-through-time.yaml` in stevenfazzio.github.io,
  thumbnail from its `scripts/capture-thumbnails.mjs`).

- **2026-11-09 — lineage check against relays (stage 09).** Found from three
  odd coarse names Steven flagged: "Thinkers and Nazi Leaders" (holding Neil
  Armstrong and Rosa Parks), "Novelists and Murderers", "Classical Composers
  and Screen Actors". The first two were **relays**: linking neighbouring
  windows alone lets a track's id pass to another group in two steps. c33
  (thinkers): in window 18, 109 people of c31 "Dictators and Wartime Leaders"
  crossed into its cluster; in 19 they regrouped with the rest of c31 into
  the Nazi cluster, and the 101 wearing c33's label outvoted c31's 85, so the
  Nazi cluster continued c33 and c31 ended (169 of its people were in c31 in
  window 17, 1 in c33). c34 (novelists): three tracks merged in window 20
  (novelists 134, religious thinkers 127, politicians and scientists 111:
  "American Writers, Politicians, and Criminals", Trump, JFK, Bundy, Manson),
  and the murderers split off in 21 with the novelists' id, though 5 of their
  97 carriers had been novelists. Also c19 "European Rulers, Saints, and
  Writers" (saints relayed in at window 8). **Rule**: a cluster continues
  track t only if its carriers -- the people in both it and t's cluster of
  the window before, labelled two windows earlier -- were at least 20% in t
  then (`TRACK_LINEAGE_MIN`, when there are at least 20); otherwise the link
  is dropped and mutual-best rerun without it, so the cluster can continue
  the track it does descend from. Carrier shares on the old tracks: coarse 0
  / 1 / 5% (the relays) then 29%+, mid 0 / 0 / 0% then 30%+, fine 0-14% then
  33%+. The first version tested all the cluster's people labelled two
  windows back (also a clean gap, relays 0-5% vs 22%+) but misfired once: the
  window-5 Byzantine emperors' older members were saints, so it moved them
  from the Roman track to the saints' -- yet the 90 people carrying the Roman
  label in were unlabelled or new two windows back, none of them crossers.
  Rejected: "t must be the largest ancestor" (rejects 14 coarse continuations,
  near-even merges among them); looking three windows back (catches slower
  handovers -- c0's Indian rulers to caliphs -- but windows three apart share
  2 of 5 bins, so a track's own people ageing out shades into drift);
  semantic tiebreaks from name or member-centroid embeddings (people give the
  same answer with a clean gap; names are noisy and depend on their
  neighbours; keeps the LLM out of structure). After: the Nazi cluster
  continues "Dictators and Wartime Leaders" (13-19), "European Royal
  Dynasties" (5-13) stands apart from "Christian Saints and European Writers"
  (4-17), the novelists end as "Anglophone Writers and American Politicians"
  (15-20) -- but the thinkers track now reaches the murderers ("Thinkers and
  Murderers", 14-24): absorbed into the window-20 hub, it is bridged back in
  21, where MLK, Hawking, Malcolm X and Benedict XVI share a cluster with
  Bundy, Manson and Gacy, new to the window; by 24 the murderers dominate.
  That is drift by arrival, like c30, not a relay -- fixed by the drift cut
  (next decision).

- **2026-11-09 — drift cuts: a track is one region worth one name (stage
  09).** Drifting tracks -- composers then actors (c30), thinkers then
  murderers, saints then novelists -- are real continuations by people whose
  population turns over as the window moves, so one name had to stretch over
  both ends (the window-16 thinkers, Freud and Rosa Parks, read "Thinkers and
  Murderers"). **Per-segment names rejected** (Steven): keeping a track's
  colour and ribbon but changing its name partway breaks what colour means --
  a ribbon claims a cohesive unit worthy of a single label. So such a track
  ends, and a new one (new colour, new ribbon, own name) takes over. **Test**:
  content = unit mean of the members' stage 07b vectors with each window's
  mean removed; a track is cut at the first of two windows in a row further
  from its segment's first window than a region typically is from its
  *nearest* neighbour in the same window (median of that cosine per layer,
  `TRACK_DRIFT_QUANTILE`; coarse 0.31); then each cut moves so the windows
  beside it go with the segment whose mean they resemble more. Measured on
  the way (scratch scripts, not kept): the prototype's raw bar
  (`scoping/track_segments.py`, cosine 0.732 = median between *any* two
  regions) cut 6 coarse tracks but not Thinkers -> Murderers (its end still
  0.79 to its start: the era drift all tracks share masks content); after
  removing window means that bar becomes unreachable (two regions in a window
  sit at -0.13, while even the window-31 actors keep 0.12 to the window-13
  composers); jumps between consecutive windows have no gap (0.37 up, led by
  early rulers whose windows span centuries; composers -> actors is a slide
  over 13-17, not a jump). Not gapped like the lineage check: at the p25 bar
  (0.21) Thinkers isn't cut; near p75 (0.5) 12 coarse tracks are, c0 into
  four. Without the
  refinement the mixed window 21 (MLK and Hawking with Bundy and Manson)
  stayed with the thinkers, whose name then still said "Murderers". Coarse
  result (windows): Indian Buddhism and Dynasties 0-3 -> Islamic and South
  Asian Rulers 5-15 -> Heads of State and Government 16-29; Ancient Near
  Eastern Rulers 0 -> Muhammad and His Companions 1-5; Roman Emperors and
  Rulers 0-4 -> Byzantine and Italian Rulers 5-7; Popes and Iberian Rulers
  3-4 -> European Royalty 6-12; European Saints, Scholars, and Writers 4-13
  -> Novelists and Poets 14-17; Russian Tsars and Napoleon 10-11 -> European
  Royal Families 12-18; American History Figures 10-15 -> Power Elites 16-31;
  Classical Composers and Hollywood Stars 13-16 -> American Film and TV
  Actors 17-31; Thinkers and Spiritual Leaders 14-19 -> Murderers and Public
  Figures 21-24. **Page**: where the new track starts the window after the
  old one ends (7 of the 10 coarse cuts) the ribbon hands over -- at the
  midpoint each carries half the band, so the river changes colour without
  pinching (measured: composers -> actors 34.5 px, 26 at the midpoint, 13 px,
  the real 871 -> 232 people); a cut across a window the region is absent
  from tapers as any track end. Points recolour and the label crossfades, as
  at any track change. Stage 09's tree records `cut_from`; stage 12 exports
  it as `tracks.from`.

## Proposed, not yet settled

- **Label tracks, what's left** (see the 2026-11-08 decision). Merge
  flicker: judged acceptable in the animated preview on 2026-11-08
  (`scoping/track_preview.py` → `data/scoping/track_preview.html`, which
  lists each step's appearing and disappearing coarse labels), so no
  hysteresis for now; revisit if the real frontend shows it. The idea, if
  needed: keep a track alive through a short merge and give it back its id
  when it splits off again (half of all label switches; also half of the
  parent changes). Naming is settled (2026-11-08 decision: per window,
  then per track); stage 09's majority-parent tree stays for the frontend's
  label hierarchy, but Toponymy no longer needs it. Label position = the track's centroid per window,
  interpolated between windows; labels fade in and out as tracks begin and
  end. **Cross-layer links** (measured 2026-11-09, before the lineage check):
  32 of 43 coarse-track ends are the region dropping below the coarse size
  and carrying on as one mid cluster that descends from it (for a median 3
  more windows), and 37 of 39 coarse starts came mostly from one mid cluster
  -- so a coarse "end" is usually a size-threshold crossing, which on the page
  recolours and relabels a region that stays put. Considered: a coarse track
  survives one layer down while its mid cluster lives (the short-merge idea
  above, with the mid layer as evidence the region still exists). Cost: up to
  4 more coarse labels per window (9.7 → 13.7), more colours, and it inherits
  mid-layer drift (of 3 regions that shrank and regrew, only c29 Scientists
  → c32 Industrialists reads right). Parked until the page shows the
  crossings jar. General cross-layer links rejected: a track belongs to one
  layer (name tier, label size and priority, tree, the river's
  one-coarse-track-per-person split).
- **Frontend is custom** -- settled 2026-11-09, see the decision "Frontend".
- **Frontend, what's left** (2026-11-09): a write-up post on stevenfazzio.com
  and the page's "← read the write-up" link; a phone layout (the
  timeline needs a different shape at 375 px); whether labels crossfading
  during a drag read as flicker; the palette's tone (bright on dark -- review
  by eye); search over more than names and short descriptions.

## Open questions

- **English stubs vs rich native articles** (medieval Chinese / Japanese /
  Islamic-world figures): translate the richer article instead? First
  measurement (stage 06 log): of 7,676 roster people whose most-viewed edition
  isn't English, 1,724 have an enwiki lead under 300 chars (ja 577, zh 254,
  ru 177, ar 146) — e.g. "Kyōzō Nagatsuka is a Japanese actor.", "Servius
  Sulpicius Galba was a consul of Rome in 144 BC." Not yet compared with the
  native leads' length; English stubs are common for moderns too, so check
  whether thin text actually misplaces people once the map exists.
- **Gender erasure** (tested 2026-11-08, `scoping/gender_erasure.py`,
  `gender_erasure_window.py`; panels in `data/scoping/gender_erasure.html`).
  Centroid (project out the female − male mean direction) and LEACE are
  indistinguishable again, as in living-people-map. Fit on the 30,925
  male/female people, applied to all, role variant: within-window 15-NN gender
  purity 0.945 → 0.802, 2-d (UMAP, windows 1/12/27) 0.933 → 0.752 (chance
  0.67); role, top language and birth-year locality unchanged; 85% of
  neighbours kept. Plain + erasure lands at 0.783 / 0.729 with less role
  structure (role kNN 0.751 vs 0.787). Probes de-gender sensibly (Cleopatra →
  Ptolemies of both sexes; Frida Kahlo → + Picasso, Dalí; Marie Curie → +
  Rutherford, Langevin). A *global* eraser leaves gender linearly decodable
  within each window (probe AUC 0.98–0.995; the direction differs by era),
  but per-window erasers, which kill that (0.50), move the 2-d purity only
  0.752 → 0.735; re-erasing after renormalizing (3 rounds) 0.741. The
  residue is the text itself ("actress", "queen", pronouns) — nonlinear,
  out of reach of linear erasure. First decided against (2026-11-08: wait for
  the map); **reversed the same day after the AlignedUMAP preview**, where
  women clustered almost completely in every window. Now stage 07b: one
  global centroid projection (simplest to explain; per-window adds little).
  The direction is saved in `embeddings_role_nogender.npz` and must also be
  projected out of Toponymy's keyphrase and name embeddings, so labels share
  the points' space. `concept-erasure` stays only for the scoping scripts.
- **`MAP_N` and `AGE_OFFSET`**: 31k and 0 are what stage 05 builds with,
  still provisional until the map can be looked at. Real data gives history
  more room than the enwiki proxy did (19 of 36 bins before 1903), so nothing
  argues for `AGE_OFFSET = 20`. First look (2026-11-08, `edition_floor.txt`): with
  all-language views and proxy birth years (enwiki category, else Wikidata
  death − 50), the top 31k is 7.8% BCE and 32% born before 1500, vs 17.9%
  before 1500 on the enwiki proxy. 177k floor survivors had no proxy date.
  **With stage 04's birth years** (`birth_coverage.txt`): 8.4% BCE, 33.3%
  before 1500, 52.0% before 1900, 29.9% born 1950+; the bar is ~1.74M
  views × years, so BCE admits down to ~380 views/mo — the floor is the binding
  constraint there.
- **Score as a display variable.** Cleopatra is #1 by score (1.19M views/mo
  × 2,094 years), ahead of Alexander and Jesus; her traffic is the suspected,
  unverified crawler case above. Harmless for membership. Since 2026-11-09
  log(score) sizes the map's markers (1.4-5.5 px); on a log scale she is
  barely bigger than Alexander, but her monthly series is still unchecked.
- **Missing labels** (resolved 2026-11-09 in stage 12): 19 roster people have
  no en/mul Wikidata label; the page shows the enwiki title, else the
  top-language title (慈雲院 (豊臣秀長室), whose translated lead says "Jiun-in").
  The hovercard's description is the enwiki short description, else Wikidata's.
- **Repo name** -- settled 2026-11-09: `people-through-time` (the site's naming
  policy wants 2-3 words; the URL path is the repo name). The local folder is
  still `~/repos/people-through-time-map`.

## Scoping data and gotchas (2026-11-07)

- `scoping/*.py` are one-off scripts; `scoping/results/*.txt` are their
  outputs. Their inputs live in `data/scoping/` (QLever CSVs ~490 MB, parsed
  category parquets ~170 MB). The 6.4 GB enwiki dumps behind
  `parse_cats.py` were not kept; its parquet caches make reruns dump-free.
  `report.py` reads living-people-map's redirect map and enwiki pageview
  parquets in place, read-only.
- **QLever public instance caps queries at 30 s** (403 if you ask for more)
  and a timeout returns 429 "Operation timed out". One timeout plus quick
  retries locked this machine out for ~50 min. GROUP BY on the item IRI over
  every human's sitelinks fit (17 s); every P569 statement with precision for
  all 13.8M humans did not — partition birth dates (or use a dump).
- `wikibase:wikiGroup` is on the site IRI (`?site wikibase:wikiGroup
  "wikipedia"`), not the sitelink node. abstract.wikipedia.org sitelinks
  exist on humans (group `"abstract"`).
- **Wikidata birth dates carry several statements and precisions**: Plato's
  preferred P569 is −428 at *decade* precision beside normal-rank day values
  for −428 and −426; Julius Caesar has four. Stage 04 picks preferred > normal
  (drop deprecated), then highest precision, then the lower median year.
  **A round century-precision year is a boundary, not a birth year**
  (`scoping/results/century_precision.txt`): CE 1600 at precision 7 means "16th
  century" and is its *last* year (Malyuta Skuratov died 1573, "born 1600";
  Kamehameha I "born 1800"); as stated, 28.7% of round CE century births come
  after the person's death. BCE −900 is "9th century BC" and its *earliest*
  year (median gap to death 92 vs ~57 for year-precision ancients). Values
  ending in 01 are the other boundary. Non-round years are estimates
  (Pythagoras −582 at century precision; his year-precision values run
  −569..−581). Stage 04 moves round values (ending 00/01, millennia likewise)
  to the midpoint and keeps the rest. Oddity: 787 century-precision births are
  exactly 1953. **WDQS returns BCE years in astronomical
  numbering** (Julius Caesar −0099 = 100 BC; Confucius −0550 = 551 BC), so
  age = 2026 − year needs no year-zero fix. QLever's `YEAR()` returns −428
  for `-0428-…`.
- enwiki birth categories as a cross-check: 89.1% of enwiki biographies have
  one, 97% at year precision; 375 conflicting-year pages, 21 born after death,
  35 lifespans > 122 — Wikidata will have its own junk (see living-people-map's
  2026-08-04 birth-year note: filter where the parquet row is built, null
  never clamp).
- **macOS `grep -F -f` with ~10k patterns is unusably slow** (24 min and not
  done on a 2.5 GB gz); slicing each SQL row from both ends in Python parsed
  all 200M+ categorylinks rows in under 90 s.

## Frontend gotchas (2026-11-09)

- **A hidden Browser pane throttles the page**: `requestAnimationFrame` ran 0
  times a second while the pane was hidden and ~1-9 with it shown but the
  window not in front, so animations stall and screenshots show stale
  frames. The page exposes `window.__ptt` (state, `setS`, `goTo`,
  `renderNow()`, which draws without waiting for a frame) for checking states
  from the console.
- **The pane may load the page at zero size**: the timeline and the initial
  fit wait for a real size (ResizeObserver) instead of drawing at 0 px.
- **`http.server` + the pane's cache**: after editing `app.js` the browser
  kept running the old copy (heuristic caching; `?v=` on the page URL doesn't
  reach the script). Re-fetch with `fetch(url, {cache: "reload"})` from the
  console, or serve with a handler that sends `Cache-Control: no-store`.
- **Pane screenshots are ~800 px wide** whatever the viewport, too coarse for
  a ribbon's shape: measure it instead, e.g. `path.isPointInFill(new
  DOMPoint(x, y))` down a column of the timeline SVG (`__ptt.core()` gives
  the exported data; ribbons are `path.ribbon`, filled with their palette
  colour).

## Inherited Wikipedia gotchas (from living-people-map — they will bite again)

- Join pageviews on **exact canonical titles** (underscores→spaces) after
  redirect resolution; never case-fold ("TeQuila" absorbed "Tequila" the
  drink). Resolve every dump title through a redirect map first — mid-window
  renames split a person's views across titles. Non-English redirects were
  never resolved in living-people-map (documented undercount); per-language
  redirect maps would fix it.
- The pageview dump's page_id column is unreliable — never join on it. Titles
  with double quotes are CSV-quoted with backslash escapes. Rows are per access
  method (sum them for the metric — but keep the split, above).
- `np.savez` appends `.npz` to names lacking it — atomic temp files must end
  in `.tmp.npz`.
- If datamapplot renders anything: label layers go **finest-first**.

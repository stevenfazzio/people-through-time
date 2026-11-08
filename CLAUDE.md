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

Status (2026-11-08): roster design settled (views × age, switched from
log-time stratification the same day). Stages 00–03 run; the bot filter
(access-ratio clip) is settled and lives in stage 03.
`scoping/` holds the one-off analysis behind the decisions below.

## Pipeline

Run in order with `uv run python pipeline/NN_*.py`; tests with `uv run pytest`.

| Stage | What | Output |
|---|---|---|
| 00 | Roster: Wikidata humans with ≥1 Wikipedia article (QLever; raw CSV cached in `data/raw/`) | `data/roster.parquet` |
| 01 | Every roster human's Wikipedia titles (`wb_items_per_site` dump) | `data/sitelinks.parquet` |
| 01b | enwiki redirect map (`page` + `redirect` dumps) | `data/redirects.parquet` |
| 02 | Monthly views per (person, language, access method), 12 dumps | `data/pageviews/pageviews_YYYYMM.parquet` |
| 03 | Per-person monthly totals and median after the bot filter, raw median, access shares, top language | `data/people_views.parquet` |

Stage 02 run 2026-11-08: 160 min for 12 dumps with 2 workers; 4.60–4.82M
people with views per month; desktop share 23–26% every month. Stage 03
(156 s, 9.8 GB peak): 793,215 people at or above the 300 median-views floor
raw, 792,471 after the bot filter.

Planned next: birth dates for people
above the floor (WDQS, chunks of ~1,000 QIDs) → score and roster → leads and
translation → embeddings → AlignedUMAP → label tracks → frontend.

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
  ~2k translations, mostly in early windows. Check afterwards: kNN purity of an
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

## Proposed, not yet settled

- **Labels across windows**: cluster each window on its own 2-d layout (the
  datamap rule — cluster in the plot space), link clusters in neighbouring
  windows by member overlap into *tracks*, and name each track once with
  Toponymy, handed the tracks as a pre-fitted clusterer. Label position =
  track centroid per window, interpolated between windows; labels fade in and
  out as tracks begin and end; on a split, the larger piece keeps the name.
  Start with one or two label layers.
- **Frontend is custom.** datamapplot can't animate positions: in 0.7.3's
  `datamap.js` the only deck.gl `transitions` are on `getFillColor`
  (colormap switches), and the histogram only highlights a selection. Either
  a custom deck.gl page or a datamapplot page whose point and label layers
  are replaced via `custom_js` (keeps hovercards, search, title panel —
  untested). Likely the largest engineering piece.

## Open questions

- **English stubs vs rich native articles** (medieval Chinese / Japanese /
  Islamic-world figures): translate the richer article instead? Measure first.
- **Gender erasure**: pre-modern windows are overwhelmingly male; whether the
  centroid erasure is worth carrying over.
- **`MAP_N` and `AGE_OFFSET`**: 31k and 0 are provisional — revisit once
  all-language views exist (expect more early, non-Western people than the
  enwiki proxy shows). First look (2026-11-08, `edition_floor.txt`): with
  all-language views and proxy birth years (enwiki category, else Wikidata
  death − 50), the top 31k is 7.8% BCE and 32% born before 1500, vs 17.9%
  before 1500 on the enwiki proxy. 177k floor survivors had no proxy date.
- **Repo name** is a working name; no GitHub remote yet.

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
  for −428 and −426; Julius Caesar has four. Pick preferred > normal (drop deprecated),
  then highest precision; keep precision so century-dated people land by
  midpoint (the earliest bins span centuries). QLever's
  `YEAR()` returns −428 for `-0428-…`.
- enwiki birth categories as a cross-check: 89.1% of enwiki biographies have
  one, 97% at year precision; 375 conflicting-year pages, 21 born after death,
  35 lifespans > 122 — Wikidata will have its own junk (see living-people-map's
  2026-08-04 birth-year note: filter where the parquet row is built, null
  never clamp).
- **macOS `grep -F -f` with ~10k patterns is unusably slow** (24 min and not
  done on a 2.5 GB gz); slicing each SQL row from both ends in Python parsed
  all 200M+ categorylinks rows in under 90 s.

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

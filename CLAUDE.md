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
log-time stratification the same day). Stages 00–08 run; the bot filter
(access-ratio clip) is settled and lives in stage 03; birth years (with
fallbacks and the round-century fix) come from stage 04; stage 05 picks the
31k roster and its slider bins; stage 06 fetches and translates the leads;
stage 07 embeds them twice (plain and role-instructed Qwen3); the map uses
role, with gender projected out by stage 07b; stage 08 lays out all 32
windows with one AlignedUMAP; stage 09 clusters each window into three label
layers and links the clusters into tracks. `scoping/` holds the one-off analysis behind
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
| 09 | Label tracks: each window's layout clustered at three fixed sizes (`config.LABEL_TARGETS`), clusters linked across windows, majority-parent tree | `data/tracks_role_nogender.parquet` (window, qid, track_0..2; 0 finest, -1 unlabelled), `data/track_tree_role_nogender.parquet` |

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

Planned next: merge flicker in stage 09's linking (see "Label tracks,
what's left"), then naming with Toponymy → frontend.

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

## Proposed, not yet settled

- **Label tracks, what's left** (see the 2026-11-08 decision). Merge
  flicker: keep a track alive through a short merge and give it back its id
  when it splits off again (half of all label switches; also half of the
  parent changes). Per-segment names where a track's content drifts (a coarse
  track went from Mozart, Beethoven and Bach to film actors). (Done in
  stage 09: fixed min cluster sizes across windows rather than Toponymy's
  per-run quantile rule, which would give each window different layers; the
  majority-parent tree.) Toponymy handoff: one row per person under
  their majority track in each layer, plus our own `cluster_tree_` from the
  majority parents — Toponymy's `build_cluster_tree` attaches a child to the
  first parent (by cluster number) holding any of its points, which assumes
  strict nesting. Label position = the track's centroid per window,
  interpolated between windows; labels fade in and out as tracks begin and
  end.
- **Frontend is custom.** datamapplot can't animate positions: in 0.7.3's
  `datamap.js` the only deck.gl `transitions` are on `getFillColor`
  (colormap switches), and the histogram only highlights a selection. Either
  a custom deck.gl page or a datamapplot page whose point and label layers
  are replaced via `custom_js` (keeps hovercards, search, title panel —
  untested). Likely the largest engineering piece.

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
  unverified crawler case above. Harmless for membership, but if score ever
  sizes markers or orders labels, check her monthly series first.
- **Missing labels**: 19 roster people have no en/mul Wikidata label and 104
  no description; the frontend should fall back to the article title.
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

# People Through Time

A data map of the 31,000 most famous people of every era, from all Wikipedia
language editions, placed by what their articles say. The slider under the map
is itself a chart: drag it through birth years and the map re-forms around the
people born in that window, with people gliding between positions and region
labels moving, appearing and fading.

**Live:** [stevenfazzio.com/people-through-time](https://stevenfazzio.com/people-through-time/)

Sibling of [living-people-map](https://github.com/stevenfazzio/living-people-map).

## How "famous" works across eras

Ranking everyone by pageviews makes a map of people alive today (about 73% of
the top 25,000). So people are ranked by **median monthly views across all
Wikipedias × years since birth**: by the Lindy effect, something that has
stayed famous for A years can expect about A more, so the product estimates
lifetime attention, past and future. The slider then splits the roster into
36 birth-year bins holding equal numbers of people — centuries wide in
antiquity, a few years wide for the 20th century — and each slider position
shows five consecutive bins (about 4,300 people).

## How the map is made

- Each person's Wikipedia lead (in English, or translated by Claude when there
  is no English article) is embedded with Qwen3-Embedding-8B, asked for what
  the person is known for; a single gender direction is projected out.
- AlignedUMAP lays out all 32 slider windows at once, so a person who appears
  in neighbouring windows stays in nearly the same place.
- Each window is clustered at three zoom levels, clusters are linked across
  windows into tracks, and Claude names each track once from the names its
  clusters got in each window (Toponymy).
- The page (`docs/`) is plain JavaScript on deck.gl. The slider is a
  ThemeRiver of the broadest regions: each ribbon is as thick as its number of
  people in each window.

## Running it

The pipeline is a sequence of numbered stages in `pipeline/`, configured only
through `pipeline/config.py`; `CLAUDE.md` records every decision and run.

```bash
uv run python pipeline/NN_<stage>.py      # in order, 00 .. 12
uv run --extra publish python pipeline/13_social_preview.py
uv run pytest
python3 -m http.server 8771 --bind 127.0.0.1 --directory docs
```

The early stages download Wikimedia dumps and pageview files (tens of GB) and
call paid APIs (OpenRouter for embeddings, Anthropic for translation and
naming); `data/` is not in the repo. `docs/` holds the finished page and its
data.

Text and images in the page come from Wikipedia (CC BY-SA) and Wikidata.

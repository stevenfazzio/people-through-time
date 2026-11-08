# people-through-time-map

A semantic map of the most famous people of every era, built from all
Wikipedia language editions, with a birth-year slider: drag it and the map
re-forms around the people born in that window, with region labels moving,
appearing and disappearing alongside them.

Sibling of [living-people-map](https://github.com/stevenfazzio/living-people-map).

**Status:** design stage. The roster design and the analysis behind it are in
`CLAUDE.md` and `scoping/`; no pipeline stages yet.

## How "famous" works across eras

Ranking everyone by pageviews makes a map of people alive today (about 73% of
the top 25,000). So people are ranked by **median monthly views across all
Wikipedias × years since birth**: by the Lindy effect, something that has
stayed famous for A years can expect about A more, so the product estimates
lifetime attention, past and future. The slider then splits the roster into
birth-year bins holding equal numbers of people — centuries wide in
antiquity, a few years wide for the 20th century.

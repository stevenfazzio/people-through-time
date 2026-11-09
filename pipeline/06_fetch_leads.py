"""Stage 06: the text each roster person is embedded by -- their Wikipedia lead, in English.

Decided 2026-11-07: the English lead when the person has an enwiki article,
otherwise the lead of their most-viewed edition (stage 03's top_lang),
translated to English by Claude. An enwiki page counts only if it exists, has
an intro, and is the person's own article: a sitelink that redirects to another
human's article (Heather McCartney -> Paul McCartney) is skipped, since its
intro, description and thumbnail describe the wrong person.

Fetch: action API prop=extracts (exintro, explaintext) | pageimages |
description, config.LEAD_BATCH titles per POST, maxlag=5, redirects followed,
config.LEAD_WORKERS concurrent requests paced to one start per
config.LEAD_MIN_INTERVAL s (Wikimedia's ~200/min budget). Each site's records
are appended to data/raw/leads/<site>.jsonl, one per requested title, so a rerun
fetches only titles never seen; delete a site's file to refresh it.

Translate: config.TRANSLATE_MODEL at config.TRANSLATE_EFFORT, with server-side
refusal fallbacks. One JSON per QID in data/raw/translations/, reused while the
source text and PROMPT_VERSION are unchanged. Failures are logged and retried on
the next run; the stage writes no output while any translation is missing.

Output: data/leads.parquet, one row per roster person, rank order:
  qid, lead (English text, stripped), lead_source ("enwiki" | "translated"),
  lead_site, lead_title (resolved, with spaces), lead_original (source text when
  translated), n_chars, pageid, enwiki_description (enwiki's short description),
  thumb_url. A page with no extract gets "label: short description" instead
  (lead_source "description"); only a person with neither keeps a null lead.
"""

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import pandas as pd

import config
from util import log, make_session, site_for_wiki_code, wiki_query, write_parquet_atomic

PROMPT_VERSION = 1
SYSTEM_PROMPT = """You translate the opening section of a Wikipedia article about a person into English.

Return only the translation, as plain text, keeping the paragraph breaks. Translate faithfully: do not add, explain, \
summarize or leave out anything, and do not comment on the text. Write names the way English-language sources usually \
write them, and transliterate when there is no usual English form. Keep dates and numbers as they are."""
SHORT_LEAD = 40  # chars; a lead this short is a stub or a parse failure


class Pacer:
    """At most one request start per `interval` seconds, shared across threads."""

    def __init__(self, interval: float):
        self.interval, self.lock, self.next_start = interval, threading.Lock(), 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.interval
        time.sleep(max(0.0, start - now))


_local = threading.local()


def thread_session():
    if not hasattr(_local, "session"):
        _local.session = make_session()
    return _local.session


def fetch_batch(session, api_url: str, titles: list[str], pacer: Pacer | None = None) -> list[dict]:
    """One record per requested (underscored) title: resolved title, pageid, extract, description, thumbnail."""
    params = {
        "prop": "extracts|pageimages|description",
        "exintro": "1",
        "explaintext": "1",
        "exlimit": "max",
        "piprop": "thumbnail",
        "pithumbsize": str(config.THUMB_SIZE),
        "pilimit": "max",
        "redirects": "1",
        "titles": "|".join(t.replace("_", " ") for t in titles),
    }
    res = wiki_query(session, api_url, params, before_request=pacer.wait if pacer else None)
    normalized = {n["from"]: n["to"] for n in res["normalized"]}
    redirects = {r["from"]: r["to"] for r in res["redirects"]}
    by_title = {p["title"]: p for p in res["pages"]}
    records = []
    for title in titles:
        want = title.replace("_", " ")
        norm = normalized.get(want, want)
        final = redirects.get(norm, norm)
        page = by_title.get(final, {})
        records.append(
            {
                "title": title,
                "resolved_title": page.get("title", final),
                "pageid": page.get("pageid"),
                "extract": page.get("extract"),
                "description": page.get("description"),
                "thumb_url": (page.get("thumbnail") or {}).get("source"),
            }
        )
    return records


def load_jsonl(path) -> dict[str, dict]:
    """title -> record; a line cut off by a crash is skipped (its title is fetched again)."""
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                log(f"WARNING: {path.name} line {n} unreadable (cut off?); its title will be fetched again")
                continue
            done[rec["title"]] = rec
    return done


def fetch_site(site: str, lang: str, titles: list[str], pacer: Pacer) -> dict[str, dict]:
    """Records for every title on one Wikipedia, fetching only those not cached in <site>.jsonl."""
    config.LEADS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.LEADS_RAW_DIR / f"{site}.jsonl"
    done = load_jsonl(path)
    todo = sorted(set(titles) - done.keys())
    log(f"{site}: {len(titles):,} titles, {len(titles) - len(todo):,} cached, {len(todo):,} to fetch")
    if not todo:
        return done
    api_url = config.WIKI_API_URL.format(lang=lang)
    batches = [todo[i : i + config.LEAD_BATCH] for i in range(0, len(todo), config.LEAD_BATCH)]
    t0 = time.monotonic()
    pool = ThreadPoolExecutor(config.LEAD_WORKERS)
    try:
        futures = [pool.submit(lambda b: fetch_batch(thread_session(), api_url, b, pacer), b) for b in batches]
        with open(path, "a", encoding="utf-8") as out:
            for n, fut in enumerate(as_completed(futures), 1):
                for rec in fut.result():
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done[rec["title"]] = rec
                out.flush()
                if n % 100 == 0 or n == len(batches):
                    rate = n / (time.monotonic() - t0)
                    log(f"  {site}: {n:,}/{len(batches):,} batches, {rate * 60:.0f}/min, "
                        f"ETA {(len(batches) - n) / rate / 60:.1f} min")  # fmt: skip
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    return done


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def translation_path(qid: str):
    return config.TRANSLATIONS_DIR / f"{qid}.json"


def cached_translation(qid: str, text: str) -> str | None:
    path = translation_path(qid)
    if not path.exists():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("source_sha256") != source_hash(text) or rec.get("prompt_version") != PROMPT_VERSION:
        return None
    return rec.get("text") or None


def translate(client, qid: str, lang: str, label: str | None, text: str) -> str:
    """Translate one lead (cached per QID). Raises on refusal, truncation or an empty answer."""
    hit = cached_translation(qid, text)
    if hit is not None:
        return hit
    user = f"Language code: {lang}\n"
    if isinstance(label, str) and label.strip():  # 19 roster people have no en/mul label (pd.NA)
        user += f"Person (English name on Wikidata): {label}\n"
    user += f"\n<article_opening>\n{text}\n</article_opening>"
    resp = client.beta.messages.create(
        model=config.TRANSLATE_MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        output_config={"effort": config.TRANSLATE_EFFORT},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": user}],
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason {resp.stop_reason} {getattr(resp, 'stop_details', None)}")
    out = "\n".join(b.text for b in resp.content if b.type == "text").strip()
    if not out:
        raise RuntimeError("empty translation")
    rec = {
        "qid": qid,
        "lang": lang,
        "source_sha256": source_hash(text),
        "prompt_version": PROMPT_VERSION,
        "model_requested": config.TRANSLATE_MODEL,
        "model": resp.model,
        "text": out,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
    config.TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = translation_path(qid).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, translation_path(qid))
    return out


def translate_all(jobs: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """jobs [qid, lang, label, text] -> ({qid: translation}, {qid: error})."""
    done, failed = {}, {}
    todo = []
    for job in jobs.itertuples(index=False):
        hit = cached_translation(job.qid, job.text)
        if hit is None:
            todo.append(job)
        else:
            done[job.qid] = hit
    log(f"Translations: {len(jobs):,} needed, {len(done):,} cached, {len(todo):,} to request")
    if not todo:
        return done, failed
    client = anthropic.Anthropic(max_retries=4)
    with ThreadPoolExecutor(config.TRANSLATE_WORKERS) as pool:
        futures = {pool.submit(translate, client, j.qid, j.lang, j.label, j.text): j.qid for j in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            qid = futures[fut]
            try:
                done[qid] = fut.result()
            except Exception as exc:  # one bad item must not stop the rest
                failed[qid] = f"{type(exc).__name__}: {exc}"
                log(f"WARNING: translation of {qid} failed: {failed[qid]}")
            if n % 25 == 0 or n == len(todo):
                log(f"  translated {n}/{len(todo)} ({len(failed)} failed)")
    return done, failed


def owner_lookup(pairs: set[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """(site, title with spaces) -> the QID whose sitelink it is, for the given pages (humans only)."""
    if not pairs:
        return {}
    sites = sorted({site for site, _ in pairs})
    sl = pd.read_parquet(config.SITELINKS_PARQUET, filters=[("site", "in", sites)])
    keys = list(zip(sl["site"], sl["title"].str.replace("_", " ")))
    return {k: q for k, q in zip(keys, sl["qid"]) if k in pairs}


def usable(rec: dict | None, qid: str, site: str, owners: dict) -> bool:
    """The page exists, has an intro, and is this person's own article -- not a redirect to a relative's."""
    if not rec or rec.get("pageid") is None or not (rec.get("extract") or "").strip():
        return False
    owner = owners.get((site, rec.get("resolved_title")))
    return owner is None or owner == qid


def redirected(records: dict[str, dict[str, dict]]) -> set[tuple[str, str]]:
    """(site, resolved title) for every record whose title resolved to a different page."""
    return {
        (site, r["resolved_title"])
        for site, recs in records.items()
        for title, r in recs.items()
        if r.get("resolved_title") and r["resolved_title"] != title.replace("_", " ")
    }


def choose_sources(people: pd.DataFrame, records: dict[str, dict[str, dict]], owners: dict) -> pd.DataFrame:
    """[qid, site, title, reason]: the page whose intro becomes the lead, or site None.

    The English article when it is usable (see usable()); otherwise the
    most-viewed edition's, if that isn't English and is usable; otherwise none,
    and fill_missing() falls back to the Wikidata description. 2026-11-08: nine
    enwiki sitelinks redirect to another person's article (Heather McCartney ->
    Paul McCartney, Ariston of Athens -> Plato), whose intro, short description
    and thumbnail would all describe the wrong person.
    """
    rows = []
    for p in people.itertuples(index=False):
        en = records.get("enwiki", {}).get(p.enwiki_title) if isinstance(p.enwiki_title, str) else None
        if en is not None and usable(en, p.qid, "enwiki", owners):
            rows.append((p.qid, "enwiki", p.enwiki_title, "enwiki"))
            continue
        reason = "no enwiki article" if en is None else "enwiki page unusable"
        if isinstance(p.top_lang, str) and p.top_lang != "en" and isinstance(p.top_title, str):
            site = site_for_wiki_code(p.top_lang)
            if usable(records[site][p.top_title], p.qid, site, owners):
                rows.append((p.qid, site, p.top_title, reason))
                continue
        rows.append((p.qid, None, None, reason))
    return pd.DataFrame(rows, columns=["qid", "site", "title", "reason"])


def assemble(choice: pd.DataFrame, records: dict[str, dict[str, dict]], translations: dict[str, str]) -> pd.DataFrame:
    """One row per person from the chosen pages and translations (see the module docstring for columns)."""
    rows = []
    for c in choice.itertuples(index=False):
        if not isinstance(c.site, str):
            rows.append({"qid": c.qid})
            continue
        rec = records[c.site][c.title]
        extract = (rec.get("extract") or "").strip() or None
        translated = c.site != "enwiki"
        rows.append(
            {
                "qid": c.qid,
                "lead": translations.get(c.qid) if translated else extract,
                "lead_source": "translated" if translated else "enwiki",
                "lead_site": c.site,
                "lead_title": rec.get("resolved_title"),
                "lead_original": extract if translated else None,
                "pageid": rec.get("pageid"),
                "enwiki_description": None if translated else rec.get("description"),
                "thumb_url": rec.get("thumb_url"),
            }
        )
    cols = ["qid", "lead", "lead_source", "lead_site", "lead_title", "lead_original", "pageid", "enwiki_description",
            "thumb_url"]  # fmt: skip
    out = pd.DataFrame(rows, columns=cols).astype(object)
    out["n_chars"] = out["lead"].str.len().astype("Int32")
    out["pageid"] = out["pageid"].astype("Int64")
    return out


def fill_missing(leads: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """A person whose page has no extract (deleted page, empty intro) gets "label: short description" instead.

    The enwiki short description first, else Wikidata's; lead_source "description".
    Three of 31k on 2026-11-08: Romuald and Hyunri (TextExtracts returns an empty
    intro), Sulayman ibn Khalid (enwiki page gone since the sitelinks dump).
    """
    out = leads.copy()
    p = people.set_index("qid")
    for i in out.index[out["lead"].isna()]:
        qid = out.at[i, "qid"]
        label, desc = p.at[qid, "label"], out.at[i, "enwiki_description"]
        if not isinstance(desc, str) or not desc.strip():
            desc = p.at[qid, "description"]
        if isinstance(label, str) and isinstance(desc, str) and desc.strip():
            out.at[i, "lead"] = f"{label}: {desc.strip()}"
            out.at[i, "lead_source"] = "description"
    out["n_chars"] = out["lead"].str.len().astype("Int32")
    return out


def main() -> None:
    people = pd.read_parquet(
        config.MAP_PEOPLE_PARQUET, columns=["rank", "qid", "label", "description", "birth_year_est", "enwiki_title",
                                            "top_lang", "top_title"]
    )  # fmt: skip
    no_en = people["enwiki_title"].isna()
    if people.loc[no_en, "top_title"].isna().any() or (people.loc[no_en, "top_lang"] == "en").any():
        raise SystemExit("people with no enwiki title need a non-English top-language title -- rerun stage 05")
    log(f"{len(people):,} roster people, {int(no_en.sum()):,} with no enwiki article")

    pacer = Pacer(config.LEAD_MIN_INTERVAL)
    records = {"enwiki": fetch_site("enwiki", "en", people["enwiki_title"].dropna().tolist(), pacer)}
    owners = owner_lookup(redirected(records))
    en_ok = [
        isinstance(t, str) and usable(records["enwiki"].get(t), q, "enwiki", owners)
        for q, t in zip(people["qid"], people["enwiki_title"])
    ]
    other = people[~pd.Series(en_ok, index=people.index) & people["top_lang"].ne("en") & people["top_title"].notna()]
    for lang, grp in other.groupby("top_lang"):
        records[site_for_wiki_code(lang)] = fetch_site(site_for_wiki_code(lang), lang, grp["top_title"].tolist(), pacer)
    owners = owner_lookup(redirected(records))
    choice = choose_sources(people, records, owners)
    moved = choice[choice["reason"] == "enwiki page unusable"].merge(people[["qid", "label", "enwiki_title"]])
    kind = choice["site"].map(lambda s: s if s in ("enwiki", None) else "other edition").fillna("none")
    log(f"Lead sources: {kind.value_counts().to_dict()}; {len(moved)} enwiki pages unusable "
        "(missing, empty intro, or another person's article):")  # fmt: skip
    table = moved.assign(now=moved["site"].fillna("description"))[["label", "enwiki_title", "now"]]
    log(table.to_string(index=False))

    leads = assemble(choice, records, {})
    jobs = leads[(leads["lead_source"] == "translated") & leads["lead_original"].notna()]
    jobs = jobs.merge(people[["qid", "label", "top_lang"]], on="qid")
    jobs = jobs.rename(columns={"top_lang": "lang", "lead_original": "text"})[["qid", "lang", "label", "text"]]
    translations, failed = translate_all(jobs)
    if failed:
        raise SystemExit(f"{len(failed)} translations failed; rerun to retry them (cached ones are kept)")
    leads = assemble(choice, records, translations)

    no_page = leads["pageid"].isna()
    no_lead = leads["lead"].isna()
    leads = fill_missing(leads, people)
    log(f"lead_source: {leads['lead_source'].value_counts().to_dict()}; still no lead: {leads['lead'].isna().sum()}")
    short = leads["n_chars"].lt(SHORT_LEAD).fillna(False)
    log(f"No page: {no_page.sum()}; no lead: {no_lead.sum()}; lead under {SHORT_LEAD} chars: {short.sum()}")
    bad = leads[no_page | no_lead | short].merge(people[["qid", "label"]], on="qid")
    if len(bad):
        table = bad[["qid", "label", "lead_site", "lead_title", "lead"]].to_string(index=False)
        log(f"WARNING: people without a usable lead:\n{table}")
    log(f"Lead length (chars): median {leads['n_chars'].median():.0f}, p10 {leads['n_chars'].quantile(0.1):.0f}, "
        f"p90 {leads['n_chars'].quantile(0.9):.0f}, max {leads['n_chars'].max():,}")  # fmt: skip
    log(f"thumbnail {leads['thumb_url'].notna().mean():.1%}, "
        f"enwiki description {leads['enwiki_description'].notna().mean():.1%}")  # fmt: skip

    # For the open "English stubs vs rich native articles" question: short English
    # leads of people read mostly in another language.
    en = leads.merge(people[["qid", "label", "top_lang", "birth_year_est"]], on="qid")
    en = en[(en["lead_source"] == "enwiki") & (en["top_lang"] != "en")]
    stub = en[en["n_chars"] < 300]
    log(f"Enwiki leads under 300 chars for people whose top language isn't English: {len(stub):,} of {len(en):,}; "
        f"by language {stub['top_lang'].value_counts().head(8).to_dict()}")  # fmt: skip
    if len(stub):
        log(stub.nsmallest(8, "n_chars")[["label", "birth_year_est", "top_lang", "lead"]].to_string(index=False))

    write_parquet_atomic(
        leads[
            [
                "qid",
                "lead",
                "lead_source",
                "lead_site",
                "lead_title",
                "lead_original",
                "n_chars",
                "pageid",
                "enwiki_description",
                "thumb_url",
            ]
        ],  # fmt: skip
        config.LEADS_PARQUET,
    )
    log(f"Wrote {config.LEADS_PARQUET}")


if __name__ == "__main__":
    main()

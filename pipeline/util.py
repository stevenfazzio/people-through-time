"""Shared helpers: HTTP session, QLever, downloads, atomic writes, SQL and pageview dump parsing.

Ported from living-people-map's util.py (2026-11-08), plus a QLever client and a
shared curl downloader.
"""

import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import requests

import config


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT
    return session


def qlever_csv(query: str, timeout: float = 600) -> str:
    """POST a SPARQL query to QLever and return the CSV body (decoded as UTF-8).

    The public instance caps queries at 30 s and answers a timeout with HTTP 429
    + JSON "Operation timed out"; repeated quick requests after that get nginx
    429s for ~50 min (2026-11-07). So: one attempt per query shape, and long
    waits (minutes, not seconds) on an nginx 429.
    """
    session = make_session()
    for attempt in range(5):
        t0 = time.time()
        resp = session.post(
            config.QLEVER_URL,
            data={"query": config.QLEVER_PREFIXES + query},
            headers={"Accept": "text/csv"},
            timeout=timeout,
        )
        if resp.status_code == 429 and b"nginx" in resp.content:
            wait = 300 * (attempt + 1)
            print(f"QLever nginx 429; waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        body = resp.content.decode("utf-8")  # QLever's CSV declares no charset
        if resp.status_code != 200 or body.lstrip().startswith("{"):
            try:
                err = json.loads(body).get("exception", body[:500])
            except json.JSONDecodeError:
                err = body[:500]
            raise RuntimeError(f"QLever HTTP {resp.status_code}: {err}")
        print(f"QLever: {time.time() - t0:.0f}s, {len(body) / 1e6:.1f} MB", flush=True)
        return body
    raise RuntimeError("QLever: still rate-limited after 5 attempts")


class WdqsTimeout(RuntimeError):
    """WDQS hit its 60 s query limit; the same query will time out again, so split it instead."""


def wdqs_select(session: requests.Session, query: str, timeout: float = 120, tries: int = 6) -> list[dict[str, str]]:
    """POST a SELECT to WDQS; rows as {variable: value}. Retries 429/5xx and network errors, not timeouts.

    No maxlag: on wikidata.org it also counts query-service update lag and can
    refuse every read for long stretches (wikimedia-data skill).
    """
    for attempt in range(tries):
        wait = 30 * (attempt + 1)
        try:
            resp = session.post(
                config.WDQS_URL,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=timeout,
            )
        except requests.RequestException as exc:  # incl. ChunkedEncodingError: a body cut off mid-stream
            log(f"WDQS {type(exc).__name__}; retry {attempt + 1} in {wait}s")
            time.sleep(wait)
            continue
        text = resp.content.decode("utf-8", errors="replace")
        if "TimeoutException" in text:
            raise WdqsTimeout(f"WDQS timed out (HTTP {resp.status_code})")
        if resp.status_code in (429, 500, 502, 503, 504):
            retry_after = resp.headers.get("Retry-After", "")
            wait = int(retry_after) if retry_after.isdigit() else wait
            log(f"WDQS HTTP {resp.status_code}; retry {attempt + 1} in {wait}s")
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"WDQS HTTP {resp.status_code}: {text[:300]}")
        return [{k: v["value"] for k, v in b.items()} for b in json.loads(text)["results"]["bindings"]]
    raise RuntimeError(f"WDQS: still failing after {tries} attempts")


def _wiki_post(session: requests.Session, api_url: str, params: dict, timeout: float, tries: int) -> dict:
    for attempt in range(tries):
        wait = 10 * (attempt + 1)
        try:
            resp = session.post(api_url, data=params, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After", "")
                wait = int(retry_after) if retry_after.isdigit() else wait
                log(f"{api_url} HTTP {resp.status_code}; retry {attempt + 1} in {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"{api_url} HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:  # network errors, bodies cut off mid-stream
            log(f"{api_url} {type(exc).__name__}; retry {attempt + 1} in {wait}s")
            time.sleep(wait)
            continue
        error = data.get("error")
        if error and error.get("code") == "maxlag":
            retry_after = resp.headers.get("Retry-After", "")
            wait = int(retry_after) if retry_after.isdigit() else 5
            log(f"{api_url} maxlag ({error.get('info', '')}); retry {attempt + 1} in {wait}s")
            time.sleep(wait)
            continue
        if error:
            raise RuntimeError(f"{api_url} API error: {error}")
        return data
    raise RuntimeError(f"{api_url}: still failing after {tries} attempts")


def wiki_query(session: requests.Session, api_url: str, params: dict, before_request=None, timeout: float = 60,
               tries: int = 6) -> dict:  # fmt: skip
    """One action=query on a Wikipedia, following `continue`: {"pages": [...], "normalized": [...], "redirects": [...]}.

    POST, because batches of non-Latin titles overflow the GET URL limit (HTTP 414).
    maxlag=5 (fine on Wikipedia, unlike wikidata.org). Pages are merged across
    continuations, since each prop may arrive in a different one. `before_request`,
    if given, is called before every request (a shared rate limiter).
    """
    base = {"action": "query", "format": "json", "formatversion": "2", "maxlag": "5", **params}
    pages: dict = {}
    out: dict = {"normalized": [], "redirects": []}
    cont: dict = {}
    while True:
        if before_request is not None:
            before_request()
        data = _wiki_post(session, api_url, {**base, **cont}, timeout, tries)
        query = data.get("query", {})
        for page in query.get("pages", []):
            pages.setdefault(page.get("pageid") or page["title"], {}).update(page)
        out["normalized"] += query.get("normalized", [])
        out["redirects"] += query.get("redirects", [])
        if "continue" not in data:
            break
        cont = data["continue"]
    out["pages"] = list(pages.values())
    return out


def download(url: str, dest: Path) -> Path:
    """Resumable curl download with the project User-Agent. Skips if dest exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"{dest.name}: already downloaded", flush=True)
        return dest
    part = dest.with_name(dest.name + ".part")
    print(f"Downloading {url}", flush=True)
    t0 = time.time()
    subprocess.run(
        [
            "curl", "-fsSL", "--retry", "5", "--retry-delay", "15", "-C", "-",
            "--speed-limit", "10000", "--speed-time", "120",
            "-A", config.USER_AGENT, "-o", str(part), url,
        ],
        check=True,
    )  # fmt: skip
    os.replace(part, dest)
    print(
        f"Downloaded {dest.name}: {dest.stat().st_size / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min", flush=True
    )
    return dest


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write to a temp file in the same directory, verify row count, atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp_path, index=False)
        n_verify = len(pd.read_parquet(tmp_path, columns=[df.columns[0]]))
        assert n_verify == len(df), f"row count mismatch writing {path}: {n_verify} vs {len(df)}"
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def save_npz_atomic(path: Path, **arrays) -> None:
    """np.savez_compressed to a temp file in the same directory, verify, atomic rename.

    The temp name MUST end in .npz: np.savez silently appends .npz to any other
    name, writing to a stray file while the rename moves an empty one into place.
    """
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        verify = np.load(tmp, allow_pickle=False)
        assert sorted(verify.files) == sorted(arrays), f"npz key mismatch writing {path}"
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def sql_row_lines(path: Path) -> Iterator[bytes]:
    """Raw row lines of a MediaWiki .sql.gz dump (current dumps put one row per line).

    Decompresses with an external gzip (much faster than Python's gzip module)
    and yields bytes, so callers can prefilter by slicing before any regex or
    decode. Dumps used to pack a whole INSERT onto one line; this yields only
    lines starting with "(", so assert on a nonzero row count afterwards.
    """
    proc = subprocess.Popen(["gzip", "-dc", str(path)], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    for line in proc.stdout:
        if line[:1] == b"(":
            yield line
    if proc.wait() != 0:
        raise RuntimeError(f"gzip -dc {path.name} exited {proc.returncode}")


def sql_dump_row_lines(path: Path) -> Iterator[str]:
    """Text lines that can hold rows, either dump layout (slow path; tests and small files)."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(("INSERT INTO", "(")):
                yield line


def require_rows(path: Path, n_rows: int) -> None:
    """A multi-GB dump that parses to nothing means the format moved, not that it's empty."""
    if n_rows == 0:
        raise RuntimeError(f"{path.name}: no rows parsed -- has the dump format changed?")


def parse_dump_title(raw: str) -> str:
    """Undo the CSV-style quoting pageview_complete applies to titles that
    contain double quotes (e.g. "Weird Al" Yankovic)."""
    if len(raw) >= 2 and raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return raw


# pageview dump wiki codes use hyphens; Wikidata site ids use underscores,
# plus a few legacy ids that don't follow the rule.
SPECIAL_SITE_IDS = {"be-tarask": "be_x_oldwiki", "gsw": "alswiki", "lzh": "zh_classicalwiki", "nan": "zh_min_nanwiki"}


def site_for_wiki_code(lang: str) -> str:
    """Pageview-dump language code ('zh-min-nan', 'en') -> Wikidata sitelink site id ('zh_min_nanwiki', 'enwiki')."""
    return SPECIAL_SITE_IDS.get(lang, lang.replace("-", "_") + "wiki")


def log(msg: str) -> None:
    print(msg, file=sys.stderr if msg.startswith("WARNING") else sys.stdout, flush=True)

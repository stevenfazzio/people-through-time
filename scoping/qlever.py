"""Tiny QLever client: POST a query, get CSV bytes decoded as UTF-8."""

import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402  (for the project's descriptive User-Agent)

URL = "https://qlever.dev/api/wikidata"
PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX schema: <http://schema.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


def run(query: str, timeout: float = 900) -> str:
    for attempt in range(6):
        t0 = time.time()
        r = _post(query, timeout)
        if r.status_code == 429 and b"nginx" in r.content:  # front-end rate limit, not a query timeout
            wait = 180 * (attempt + 1)
            print(f"  nginx 429; waiting {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait)
            continue
        break
    return _check(r, t0)


def _post(query, timeout):
    return requests.post(
        URL,
        data={"query": PREFIXES + query},
        headers={"Accept": "text/csv", "User-Agent": config.USER_AGENT},
        timeout=timeout,
    )


def _check(r, t0):
    body = r.content.decode("utf-8")
    if r.status_code != 200 or body.lstrip().startswith("{"):
        try:
            err = json.loads(body)
            raise RuntimeError(f"QLever error {r.status_code}: {err.get('exception', body[:500])}")
        except json.JSONDecodeError:
            raise RuntimeError(f"QLever HTTP {r.status_code}: {body[:500]}")
    print(f"  ({time.time() - t0:.0f}s, {len(body) / 1e6:.1f} MB)", file=sys.stderr)
    return body


if __name__ == "__main__":
    print(run(sys.stdin.read()))

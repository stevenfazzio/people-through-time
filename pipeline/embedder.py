"""Qwen3-Embedding-8B through OpenRouter, pinned to one provider, with a disk cache.

Ported from tv-tropes-map's embedder.py (2026-11-08). Implements Toponymy's
TextEmbedderProtocol (`encode(texts, show_progress_bar)`), so the same object
embeds the leads now and Toponymy's keyphrases and region names later. Every
vector is cached in SQLite, keyed by model, provider and the exact text sent
(instruction included), so no string is ever paid for twice.

Requests are pinned to config.EMBED_PROVIDER with fallbacks off: tv-tropes-map
found the same text embedded by two OpenRouter providers at cosine 0.996, not
1.0. A map built from a mix would carry that noise, and Toponymy's later
embeddings must come from the provider that embedded the documents.
"""

import hashlib
import os
import random
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

import config
from util import log

RETRY_STATUS = {408, 429}  # plus every 5xx: Nebius answers 529 "worker overloaded" under load (2026-11-08)


def with_instruction(instruction: str | None, text: str) -> str:
    """The query format from the Qwen3-Embedding model card."""
    return f"Instruct: {instruction}\nQuery:{text}" if instruction else text


class OpenRouterEmbedder:
    def __init__(self, instruction: str | None = None, provider: str = None, cache_path=None, session=None):
        self.instruction = instruction
        self.provider = provider or config.EMBED_PROVIDER
        self.tokens = 0
        self.cost_usd = 0.0
        self.session = session or requests.Session()
        if session is None:
            self.session.headers["Authorization"] = f"Bearer {os.environ['OPENROUTER_API_KEY']}"
        cache_path = cache_path or config.EMBED_CACHE
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    def _key(self, sent: str) -> str:
        return hashlib.sha256(f"{config.EMBED_MODEL}\x00{self.provider}\x00{sent}".encode()).hexdigest()

    @staticmethod
    def batches(texts: list[str]):
        """Batches closing at config.EMBED_BATCH_ITEMS texts or EMBED_BATCH_CHARS characters, whichever comes first."""
        batch, chars = [], 0
        for t in texts:
            if batch and (len(batch) == config.EMBED_BATCH_ITEMS or chars + len(t) > config.EMBED_BATCH_CHARS):
                yield batch
                batch, chars = [], 0
            batch.append(t)
            chars += len(t)
        if batch:
            yield batch

    def _embed_remote(self, texts: list[str], max_retries: int = 6) -> tuple[np.ndarray, dict]:
        payload = {
            "model": config.EMBED_MODEL,
            "input": texts,
            "encoding_format": "float",
            "provider": {"order": [self.provider], "allow_fallbacks": False},
        }
        for attempt in range(max_retries):
            wait = min(2**attempt * 5, 120) + random.uniform(0, 3)
            try:
                r = self.session.post(config.EMBED_API_URL, json=payload, timeout=300)
                if (r.status_code in RETRY_STATUS or r.status_code >= 500) and attempt < max_retries - 1:
                    log(f"  OpenRouter {r.status_code}: {r.text[:200]!r}; retrying in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                if not r.ok:  # raise_for_status() would drop OpenRouter's message
                    raise requests.HTTPError(f"{r.status_code} from OpenRouter: {r.text[:500]}", response=r)
                body = r.json()
            except (requests.Timeout, requests.ConnectionError, ValueError) as exc:
                if attempt == max_retries - 1:
                    raise
                log(f"  OpenRouter {type(exc).__name__}: {exc!r}; retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            if "data" not in body:  # an error reported inside a 200
                if attempt < max_retries - 1:
                    log(f"  OpenRouter: no data in response {str(body)[:200]!r}; retrying in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"no data in response: {str(body)[:500]}")
            served = str(body.get("provider", ""))
            if served.lower() != self.provider.lower():
                raise RuntimeError(f"served by {served!r}, not {self.provider!r}")
            data = sorted(body["data"], key=lambda d: d["index"])
            if len(data) != len(texts):
                raise RuntimeError(f"{len(data)} vectors for {len(texts)} texts")
            return np.asarray([d["embedding"] for d in data], dtype=np.float32), body.get("usage", {})
        raise RuntimeError("exhausted retries")

    def encode(self, texts, show_progress_bar=None, verbose=None, **kwargs) -> np.ndarray:
        # The API rejects an empty string, and Toponymy embeds one whenever naming
        # a region fails, so an empty text is sent as a single space.
        sent = [with_instruction(self.instruction, t) or " " for t in texts]
        keys = [self._key(s) for s in sent]
        cached = {}
        for i in range(0, len(keys), 900):  # stay under SQLite's variable limit
            chunk = keys[i : i + 900]
            rows = self.db.execute(f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(chunk))})", chunk)
            cached.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})

        todo = list(dict.fromkeys(s for s, k in zip(sent, keys) if k not in cached))  # each string once
        if todo:
            batches = list(self.batches(todo))
            loud = bool(show_progress_bar or verbose)
            if loud:
                log(f"  embedding {len(todo):,} new texts in {len(batches):,} batches via {self.provider}")
            t0 = time.monotonic()
            pool = ThreadPoolExecutor(config.EMBED_WORKERS)
            futures = {pool.submit(self._embed_remote, b): b for b in batches}
            try:
                # Network calls run in worker threads; the SQLite connection stays on
                # this thread, committed per batch, so a crash loses only batches in flight.
                for n, future in enumerate(as_completed(futures), 1):
                    batch = futures[future]
                    vecs, usage = future.result()
                    rows = [(self._key(s), v.tobytes()) for s, v in zip(batch, vecs)]
                    with self.db:
                        self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
                    cached.update({k: v for (k, _), v in zip(rows, vecs)})
                    self.tokens += usage.get("prompt_tokens", 0)
                    self.cost_usd += usage.get("cost", 0.0) or 0.0
                    if loud and (n % 100 == 0 or n == len(batches)):
                        rate = n / (time.monotonic() - t0)
                        log(f"  {n:,}/{len(batches):,} batches, ETA {(len(batches) - n) / rate / 60:.1f} min; "
                            f"{self.tokens:,} tokens, ${self.cost_usd:.4f}")  # fmt: skip
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
        return np.vstack([cached[k] for k in keys])

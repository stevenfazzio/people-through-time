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


def log(msg: str) -> None:
    print(msg, file=sys.stderr if msg.startswith("WARNING") else sys.stdout, flush=True)

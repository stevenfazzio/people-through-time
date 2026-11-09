import json

import numpy as np
import pandas as pd
import pytest

import util


def test_save_npz_atomic_writes_real_data(tmp_path):
    # np.savez appends .npz to any name lacking it; a .npz.tmp temp name once
    # wrote the data to a stray file and renamed an empty one into place.
    path = tmp_path / "x.npz"
    util.save_npz_atomic(path, a=np.arange(5))
    assert np.load(path)["a"].tolist() == [0, 1, 2, 3, 4]
    assert [p.name for p in tmp_path.iterdir()] == ["x.npz"]


def test_write_parquet_atomic_roundtrip(tmp_path):
    path = tmp_path / "t.parquet"
    util.write_parquet_atomic(pd.DataFrame({"a": [1, 2]}), path)
    assert pd.read_parquet(path)["a"].tolist() == [1, 2]


@pytest.mark.parametrize(("raw", "want"), [("Plain", "Plain"), ('"\\"Weird_Al\\"_Yankovic"', '"Weird_Al"_Yankovic')])
def test_parse_dump_title(raw, want):
    assert util.parse_dump_title(raw) == want


class FakeResponse:
    def __init__(self, status, text, headers=None):
        self.status_code, self.content, self.headers = status, text.encode(), headers or {}
        self.text = text

    def json(self):
        return json.loads(self.text)


class FakeSession:
    """Plays back a script of responses or exceptions, one per post()."""

    def __init__(self, script):
        self.script = list(script)

    def post(self, *args, **kwargs):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


OK = FakeResponse(200, '{"results": {"bindings": [{"x": {"type": "literal", "value": "1"}}]}}')


def test_wdqs_select_retries_cut_off_bodies_and_5xx_but_not_timeouts(monkeypatch):
    monkeypatch.setattr(util.time, "sleep", lambda s: None)
    cut = util.requests.exceptions.ChunkedEncodingError("Response ended prematurely")
    session = FakeSession([cut, FakeResponse(503, "busy", {"Retry-After": "5"}), OK])
    assert util.wdqs_select(session, "SELECT") == [{"x": "1"}]
    timeout = FakeResponse(500, "java.util.concurrent.TimeoutException")
    with pytest.raises(util.WdqsTimeout):
        util.wdqs_select(FakeSession([timeout, OK]), "SELECT")


class RecordingSession(FakeSession):
    def __init__(self, script):
        super().__init__(script)
        self.sent = []

    def post(self, url, data=None, **kwargs):
        self.sent.append(dict(data))
        return super().post(url, data=data, **kwargs)


def test_wiki_query_waits_out_maxlag_and_merges_continuations(monkeypatch):
    monkeypatch.setattr(util.time, "sleep", lambda s: None)
    maxlag = FakeResponse(200, '{"error": {"code": "maxlag", "info": "lagged"}}', {"Retry-After": "5"})
    first = FakeResponse(
        200,
        json.dumps({
            "continue": {"excontinue": "1", "continue": "||"},
            "query": {"normalized": [{"from": "a", "to": "A"}],
                      "pages": [{"pageid": 1, "title": "A", "extract": "x"}, {"title": "Gone", "missing": True}]},
        }),
    )  # fmt: skip
    second = FakeResponse(200, json.dumps({"query": {"pages": [{"pageid": 1, "title": "A", "description": "d"}]}}))
    session = RecordingSession([maxlag, first, second])
    calls = []
    res = util.wiki_query(session, "https://x/api.php", {"titles": "a|Gone"}, before_request=lambda: calls.append(1))
    assert res["normalized"] == [{"from": "a", "to": "A"}]
    assert sorted(res["pages"], key=lambda p: p["title"]) == [
        {"pageid": 1, "title": "A", "extract": "x", "description": "d"},
        {"title": "Gone", "missing": True},
    ]
    assert len(calls) == 2  # once per logical request, not per retry
    assert session.sent[-1]["excontinue"] == "1" and session.sent[0]["maxlag"] == "5"
    with pytest.raises(RuntimeError, match="API error"):
        util.wiki_query(FakeSession([FakeResponse(200, '{"error": {"code": "badvalue"}}')]), "u", {})

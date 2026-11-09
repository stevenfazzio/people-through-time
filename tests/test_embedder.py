import json

import numpy as np
import pytest

import config
import embedder as emb


class Resp:
    def __init__(self, status, body):
        self.status_code, self.body = status, body
        self.ok = status < 400
        self.text = json.dumps(body)

    def json(self):
        return self.body


class FakeSession:
    """Answers each post() with one vector per input text: [len(text), index]; or plays back a script."""

    def __init__(self, provider="Nebius", script=None):
        self.provider, self.script, self.calls = provider, list(script or []), []

    def post(self, url, json=None, timeout=None):
        self.calls.append(json)
        if self.script:
            return self.script.pop(0)
        data = [{"index": i, "embedding": [float(len(t)), float(i)]} for i, t in enumerate(json["input"])]
        return Resp(200, {"provider": self.provider, "data": data[::-1], "usage": {"prompt_tokens": 7, "cost": 0.001}})


def make(tmp_path, **kw):
    session = kw.pop("session", None) or FakeSession()
    return emb.OpenRouterEmbedder(cache_path=tmp_path / "c.sqlite", session=session, **kw), session


def test_with_instruction_uses_the_model_card_format():
    assert emb.with_instruction(None, "text") == "text"
    assert emb.with_instruction("Do X", "text") == "Instruct: Do X\nQuery:text"


def test_encode_orders_dedupes_caches_and_pins_the_provider(tmp_path):
    e, session = make(tmp_path)
    out = e.encode(["aa", "b", "aa", ""])
    assert out[:, 0].tolist() == [2, 1, 2, 1]  # sorted back by index; empty text sent as " "
    assert session.calls[0]["input"] == ["aa", "b", " "]  # each string once
    assert session.calls[0]["provider"] == {"order": [config.EMBED_PROVIDER], "allow_fallbacks": False}
    e.encode(["b", "aa"])
    assert len(session.calls) == 1  # all cached
    e2, session2 = make(tmp_path, instruction="Do X")  # same cache, different text sent
    e2.encode(["b"])
    assert session2.calls[0]["input"] == ["Instruct: Do X\nQuery:b"]


def test_encode_refuses_another_provider(tmp_path):
    e, _ = make(tmp_path, session=FakeSession(provider="SiliconFlow"))
    with pytest.raises(RuntimeError, match="SiliconFlow"):
        e.encode(["x"])


def test_retries_5xx_and_errors_inside_a_200(tmp_path, monkeypatch):
    monkeypatch.setattr(emb.time, "sleep", lambda s: None)
    ok = Resp(200, {"provider": "nebius", "data": [{"index": 0, "embedding": [1.0, 2.0]}], "usage": {}})
    script = [Resp(529, {"error": "worker overloaded"}), Resp(200, {"error": {"message": "upstream"}}), ok]
    e, _ = make(tmp_path, session=FakeSession(script=script))
    assert e.encode(["x"]).tolist() == [[1.0, 2.0]]  # provider match is case-insensitive


def test_batches_close_on_items_or_chars(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BATCH_ITEMS", 3)
    monkeypatch.setattr(config, "EMBED_BATCH_CHARS", 10)
    batches = list(emb.OpenRouterEmbedder.batches(["a"] * 4 + ["x" * 9, "y" * 20]))
    assert [len(b) for b in batches] == [3, 2, 1]  # "a" + 9 x fills 10 chars exactly
    assert np.all([sum(map(len, b)) <= 10 or len(b) == 1 for b in batches])

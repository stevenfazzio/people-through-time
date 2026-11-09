import json
import time
from types import SimpleNamespace

import pandas as pd
import pytest
from conftest import load_stage

import config

leads = load_stage("06_fetch_leads.py")


def test_fetch_batch_resolves_normalized_redirected_and_missing_titles(monkeypatch):
    def fake_wiki_query(session, api_url, params, before_request=None):
        assert params["titles"] == "Plato|Old name|Gone|pythagoras"
        return {
            "normalized": [{"from": "pythagoras", "to": "Pythagoras"}],
            "redirects": [{"from": "Old name", "to": "New name"}],
            "pages": [
                {
                    "pageid": 1,
                    "title": "Plato",
                    "extract": "Plato was",
                    "description": "Greek",
                    "thumbnail": {"source": "u"},
                },
                {"pageid": 2, "title": "New name", "extract": "Renamed"},
                {"pageid": 3, "title": "Pythagoras", "extract": "Py"},
                {"title": "Gone", "missing": True},
            ],
        }

    monkeypatch.setattr(leads, "wiki_query", fake_wiki_query)
    recs = leads.fetch_batch(None, "u", ["Plato", "Old_name", "Gone", "pythagoras"])
    assert [r["title"] for r in recs] == ["Plato", "Old_name", "Gone", "pythagoras"]
    assert [r["pageid"] for r in recs] == [1, 2, None, 3]
    assert [r["resolved_title"] for r in recs] == ["Plato", "New name", "Gone", "Pythagoras"]
    assert recs[0]["thumb_url"] == "u" and recs[0]["description"] == "Greek"
    assert recs[2]["extract"] is None


def test_load_jsonl_skips_a_cut_off_line(tmp_path):
    path = tmp_path / "enwiki.jsonl"
    path.write_text(json.dumps({"title": "A", "extract": "a"}) + "\n" + '{"title": "B", "extr')
    assert list(leads.load_jsonl(path)) == ["A"]


class FakeMessages:
    def __init__(self, fail_on=None, stop_reason="end_turn"):
        self.calls, self.fail_on, self.stop_reason = [], fail_on, stop_reason

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = kwargs["messages"][0]["content"]
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("boom")
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=" Translated \n"),
            ],
            model=config.TRANSLATE_MODEL,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )


def fake_client(**kwargs):
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(**kwargs)))


def test_translate_caches_per_qid_and_invalidates_on_source_or_prompt_change(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRANSLATIONS_DIR", tmp_path)
    client = fake_client()
    assert leads.translate(client, "Q1", "ja", "Name", "原文") == "Translated"
    assert leads.translate(client, "Q1", "ja", "Name", "原文") == "Translated"
    sent = client.beta.messages.calls
    assert len(sent) == 1
    assert sent[0]["fallbacks"] == "default" and "Name" in sent[0]["messages"][0]["content"]
    assert json.loads((tmp_path / "Q1.json").read_text())["text"] == "Translated"
    leads.translate(client, "Q1", "ja", "Name", "新しい原文")  # the source changed
    monkeypatch.setattr(leads, "PROMPT_VERSION", leads.PROMPT_VERSION + 1)
    leads.translate(client, "Q1", "ja", "Name", "新しい原文")  # the prompt changed
    assert len(sent) == 3


def test_translate_raises_on_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRANSLATIONS_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="refusal"):
        leads.translate(fake_client(stop_reason="refusal"), "Q1", "ja", None, "x")
    assert not (tmp_path / "Q1.json").exists()


def test_translate_all_isolates_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRANSLATIONS_DIR", tmp_path)
    monkeypatch.setattr(leads.anthropic, "Anthropic", lambda **kw: fake_client(fail_on="bad"))
    jobs = pd.DataFrame({"qid": ["Q1", "Q2", "Q3"], "lang": ["ja", "zh", "ja"], "label": ["A", "B", None],
                         "text": ["good", "bad", "unnamed"]})  # fmt: skip
    jobs["label"] = jobs["label"].astype("string")  # a missing label is pd.NA, as in stage 05's output
    done, failed = leads.translate_all(jobs)
    assert done == {"Q1": "Translated", "Q3": "Translated"} and list(failed) == ["Q2"]


def rec(title, pageid=1, extract="Some text.", **kw):
    return {"resolved_title": title, "pageid": pageid, "extract": extract, **kw}


def test_choose_sources_skips_missing_empty_and_other_peoples_pages():
    people = pd.DataFrame(
        {
            "qid": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"],
            "enwiki_title": ["Plato", "Heather_McCartney", "Gone", None, "Romuald", "Askold"],
            "top_lang": ["en", "fr", "en", "ja", "ja", "ru"],
            "top_title": ["Plato", "Heather", "Gone", "原", "ロムアルド", "Аскольд"],
        }
    )
    records = {
        "enwiki": {
            "Plato": rec("Plato"),
            "Heather_McCartney": rec("Paul McCartney"),  # redirect to another human's article
            "Gone": rec("Gone", pageid=None, extract=None),
            "Romuald": rec("Romuald", extract=""),
            "Askold": rec("Askold and Dir"),  # joint article: no human owns it
        },
        "frwiki": {"Heather": rec("Heather")},
        "jawiki": {"原": rec("原"), "ロムアルド": rec("別人")},
    }
    owners = {("enwiki", "Paul McCartney"): "Q9", ("jawiki", "別人"): "Q8"}
    assert leads.redirected(records) == {("enwiki", "Paul McCartney"), ("enwiki", "Askold and Dir"), ("jawiki", "別人")}
    out = leads.choose_sources(people, records, owners)
    assert out["site"].tolist() == ["enwiki", "frwiki", None, "jawiki", None, "enwiki"]
    assert out["reason"].tolist()[:4] == ["enwiki", "enwiki page unusable", "enwiki page unusable", "no enwiki article"]


def test_assemble_takes_enwiki_or_translation():
    choice = pd.DataFrame(
        {"qid": ["Q1", "Q2", "Q3"], "site": ["enwiki", "jawiki", None], "title": ["Plato", "原", None]}
    )
    records = {
        "enwiki": {"Plato": rec("Plato", extract=" Plato was. ", description="Greek")},
        "jawiki": {"原": rec("原", pageid=9, extract="原文", description="日本の", thumb_url="t")},
    }
    out = leads.assemble(choice, records, {"Q2": "Translated"})
    assert out["lead"].tolist()[:2] == ["Plato was.", "Translated"] and pd.isna(out["lead"].iloc[2])
    assert out["lead_source"].tolist()[:2] == ["enwiki", "translated"]
    assert out["lead_original"].tolist()[1] == "原文" and pd.isna(out["lead_original"].iloc[0])
    assert out["enwiki_description"].tolist()[0] == "Greek" and pd.isna(out["enwiki_description"].iloc[1])
    assert out["n_chars"].tolist()[:2] == [10, 10]


def test_pacer_spaces_request_starts():
    pacer = leads.Pacer(0.05)
    t0 = time.monotonic()
    for _ in range(4):
        pacer.wait()
    assert time.monotonic() - t0 >= 0.14


def test_fill_missing_falls_back_to_a_description():
    people = pd.DataFrame(
        {"qid": ["Q1", "Q2", "Q3", "Q4"], "label": ["Romuald", "Hyunri", "Plato", None],
         "description": ["Italian saint", "actress", "Greek philosopher", "x"]}
    ).astype({"label": "string", "description": "string"})  # fmt: skip
    leads_in = pd.DataFrame(
        {"qid": ["Q1", "Q2", "Q3", "Q4"], "lead": [None, None, "Plato was.", None],
         "lead_source": ["enwiki", "translated", "enwiki", "enwiki"],
         "enwiki_description": ["Founder of the Camaldolese order", None, "Greek", None]}
    )  # fmt: skip
    out = leads.fill_missing(leads_in, people)
    assert out["lead"].tolist()[:3] == ["Romuald: Founder of the Camaldolese order", "Hyunri: actress", "Plato was."]
    assert out["lead_source"].tolist() == ["description", "description", "enwiki", "enwiki"]
    assert pd.isna(out["lead"].iloc[3])  # no label: stays null

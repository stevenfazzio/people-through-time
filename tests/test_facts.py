import pandas as pd
import pytest
from conftest import load_stage

facts = load_stage("04_fetch_facts.py")

E, ONT = facts.ENTITY, facts.ONTOLOGY


def date(qid, prop, year, prec, rank="NormalRank"):
    t = f"{'-' if year < 0 else ''}{abs(year):04d}-01-01T00:00:00Z"
    return {"item": E + qid, "kind": prop, "rank": ONT + rank, "t": t, "prec": str(prec)}


def raw_of(rows, qids):
    return facts.rows_frame(rows, qids).query("kind != @facts.REQUESTED")


def test_build_query_lists_every_date_property_and_qid():
    q = facts.build_query(["Q1", "Q42"])
    assert "VALUES ?item { wd:Q1 wd:Q42 }" in q
    for p in facts.DATE_PROPS:
        assert f'(p:{p} psv:{p} "{p}")' in q


def test_rows_frame_strips_prefixes_and_marks_requested():
    rows = [
        date("Q1", "P569", -99, 11, "PreferredRank"),
        {"item": E + "Q1", "kind": "P21", "val": E + "Q6581097", "vlabel": "male"},
        {"item": E + "Q1", "kind": "label", "val": "Gaius Julius Caesar", "lang": "en"},
    ]
    df = facts.rows_frame(rows, ["Q1", "Q2"])
    assert list(df.columns) == facts.RAW_COLUMNS
    assert df.loc[df["kind"] == "P569", "rank"].item() == "PreferredRank"
    assert df.loc[df["kind"] == "P21", "val"].item() == "Q6581097"
    assert sorted(df.loc[df["kind"] == facts.REQUESTED, "qid"]) == ["Q1", "Q2"]
    assert facts.rows_frame([], ["Q3"])["kind"].tolist() == [facts.REQUESTED]  # no rows at all


def test_pick_dates_rank_then_precision_then_lower_median():
    rows = [
        # Pythagoras-like: preferred century value beats precise normal ones
        date("Q1", "P569", -582, 7, "PreferredRank"), date("Q1", "P569", -570, 9), date("Q1", "P569", -581, 9),
        date("Q1", "P569", -450, 11, "DeprecatedRank"),
        # no preferred: highest precision, then lower median of the ties
        date("Q2", "P569", 1900, 9), date("Q2", "P569", 1903, 11), date("Q2", "P569", 1901, 11),
        date("Q2", "P569", 1902, 11), date("Q2", "P569", 1904, 11),
        date("Q3", "P570", 1950, 11), date("Q3", "P570", 1951, 11),
    ]  # fmt: skip
    picked = facts.pick_dates(raw_of(rows, ["Q1", "Q2", "Q3"])).set_index(["qid", "kind"])
    assert picked.loc[("Q1", "birth"), ["year", "prec", "n_values"]].tolist() == [-582, 7, 3]  # deprecated not counted
    assert picked.loc[("Q2", "birth"), ["year", "prec"]].tolist() == [1902, 11]  # median of 1901..1904 -> lower
    assert picked.loc[("Q3", "death"), "year"] == 1950


def test_build_fallbacks_junk_and_conflicts(monkeypatch):
    monkeypatch.setattr(facts.config, "BIRTH_FALLBACK_OFFSETS", {"baptism": 0, "floruit": 40, "death": 70})
    rows = [
        date("Q1", "P569", -99, 11), date("Q1", "P570", -43, 11),  # Caesar: P569
        date("Q2", "P1636", 1564, 11), date("Q2", "P570", 1616, 11),  # baptism before death fallback
        date("Q3", "P1317", -600, 9), date("Q3", "P570", -550, 7),  # floruit - 40
        date("Q4", "P570", 1800, 9),  # death - 70
        date("Q5", "P569", 2090, 11), date("Q5", "P570", 1990, 11),  # future birth = missing -> death fallback
        date("Q6", "P569", 1700, 9), date("Q6", "P570", 1650, 9),  # death first: conflict
        date("Q7", "P569", -500, 6), date("Q7", "P570", -542, 9),  # death first, but birth only to the millennium: kept
        date("Q8", "P569", 1500, 9), date("Q8", "P570", 1700, 9),  # 200-year life: conflict
        {"item": E + "Q1", "kind": "label", "val": "Caesar (mul)", "lang": "mul"},
        {"item": E + "Q1", "kind": "label", "val": "Julius Caesar", "lang": "en"},
        {"item": E + "Q1", "kind": "desc", "val": "Roman general"},
        {"item": E + "Q1", "kind": "P21", "val": E + "Q6581097", "vlabel": "male"},
        {"item": E + "Q2", "kind": "P21", "val": E + "Q123"},  # no English label: the QID
    ]  # fmt: skip
    qids = [f"Q{i}" for i in range(1, 10)]
    out = facts.build(qids, raw_of(rows, qids)).set_index("qid")
    assert list(out.index) == qids
    est, src = out["birth_year_est"], out["birth_source"]
    assert (est["Q1"], src["Q1"]) == (-99, "P569")
    assert (est["Q2"], src["Q2"]) == (1564, "baptism")
    assert (est["Q3"], src["Q3"]) == (-640, "floruit")
    assert (est["Q4"], src["Q4"]) == (1730, "death")
    assert pd.isna(out.loc["Q5", "birth_year"]) and (est["Q5"], src["Q5"]) == (1920, "death")
    assert pd.isna(est["Q6"]) and src["Q6"] == "conflict"
    assert (est["Q7"], src["Q7"]) == (-500, "P569")
    assert pd.isna(est["Q8"]) and src["Q8"] == "conflict"
    assert pd.isna(est["Q9"]) and pd.isna(src["Q9"])  # nothing at all
    assert out.loc["Q1", ["label", "label_lang", "description", "gender"]].tolist() == [
        "Julius Caesar", "en", "Roman general", "male",
    ]  # fmt: skip
    assert out.loc["Q2", "gender"] == "Q123"


def test_fetch_chunk_halves_on_timeout(monkeypatch):
    calls = []

    def fake_select(session, query):
        n = query.count("wd:Q")
        calls.append(n)
        if n > 2:
            raise facts.WdqsTimeout("too big")
        return []

    monkeypatch.setattr(facts, "wdqs_select", fake_select)
    df = facts.fetch_chunk(None, ["Q1", "Q2", "Q3", "Q4", "Q5"])
    assert sorted(df["qid"]) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert calls == [5, 2, 3, 1, 2]


def test_load_raw_takes_each_qid_from_its_latest_chunk(monkeypatch, tmp_path):
    monkeypatch.setattr(facts.config, "FACTS_RAW_DIR", tmp_path)
    old = facts.rows_frame([date("Q1", "P569", 1900, 9)], ["Q1"])
    new = facts.rows_frame([date("Q1", "P569", 1901, 9), date("Q2", "P569", 1950, 9)], ["Q1", "Q2"])
    old.to_parquet(tmp_path / "chunk_00000.parquet")
    new.to_parquet(tmp_path / "chunk_00001.parquet")
    raw = facts.load_raw(["Q1", "Q2"])
    assert sorted(raw["t"].str.slice(0, 4)) == ["1901", "1950"]
    assert facts.requested_qids() == {"Q1", "Q2"}
    with pytest.raises(RuntimeError, match="never fetched"):
        facts.load_raw(["Q1", "Q3"])


@pytest.mark.parametrize(
    ("year", "prec", "want"),
    [
        (1600, 7, 1550),  # "16th century" stored as its last year (Malyuta Skuratov, died 1573)
        (1501, 7, 1550),  # ... or its first
        (1800, 7, 1750),  # Kamehameha I
        (-900, 7, -850),  # "9th century BC" stored as its earliest year (Homer)
        (-401, 7, -450),
        (-582, 7, -582),  # non-round: the editors' estimate (Pythagoras)
        (1550, 7, 1550),
        (-4000, 6, -3500),  # "4th millennium BC"
        (-500, 6, -500),  # non-round at millennium precision (the Buddha)
        (1580, 8, 1580),  # decades are kept
        (1600, 9, 1600),
        (0, 7, -50),
    ],
)
def test_coarse_midpoint(year, prec, want):
    assert facts.coarse_midpoint(pd.Series([year]), pd.Series([prec])).item() == want

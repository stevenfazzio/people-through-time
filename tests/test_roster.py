import numpy as np
import pandas as pd
import pytest
from conftest import load_stage

import config

roster = load_stage("05_score_roster.py")


def people_frame(rows):
    """rows: (qid, median_views, birth_year, birth_year_est, birth_source)."""
    df = pd.DataFrame(rows, columns=["qid", "median_views", "birth_year", "birth_year_est", "birth_source"])
    df["birth_year"] = df["birth_year"].astype("Int32")
    df["birth_year_est"] = df["birth_year_est"].astype("Int32")
    df["birth_source"] = df["birth_source"].astype("string")
    return df


def write_csv(tmp_path, body: str):
    path = tmp_path / "ov.csv"
    path.write_text("# a comment line\nqid,label,wikidata_year,birth_year,source\n" + body)
    return path


def test_curated_overrides_file_loads():
    ov = roster.load_overrides(config.BIRTH_YEAR_OVERRIDES_CSV)
    assert len(ov) >= 1
    assert ov["qid"].is_unique and ov["source"].str.len().gt(20).all()
    assert ov["birth_year"].between(-3000, config.FAME_REFERENCE_YEAR).all()


@pytest.mark.parametrize(
    "body, match",
    [
        ("Q1,A,-4202,650,src\nQ1,A,-4202,650,src\n", "twice"),
        ("3537100,A,-4202,650,src\n", "malformed"),
        ("Q1,A,-4202,650,\n", "source"),
        ("Q1,A,-4202,,src\n", "birth_year"),
        ("Q1,A,-4202,2100,src\n", "birth_year"),
    ],
)
def test_load_overrides_rejects_bad_rows(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        roster.load_overrides(write_csv(tmp_path, body))


def test_apply_overrides(tmp_path, capsys):
    people = people_frame(
        [("Q1", 800.0, -4202, -4202, "P569"), ("Q2", 900.0, -2813, -2813, "P569"), ("Q3", 500.0, None, 1900, "death")]
    )
    ov = roster.load_overrides(
        write_csv(tmp_path, "Q2,Two,-3101,750,src two\nQ1,One,-4202,650,src one\nQ9,Nine,,10,x y\n")
    )
    out = roster.apply_overrides(people, ov)
    assert out["birth_year_est"].tolist() == [650, 750, 1900]
    assert out["birth_source"].tolist() == ["override", "override", "death"]
    assert out["birth_year"].tolist()[:2] == [-4202, -2813]  # Wikidata's year is kept
    assert people["birth_year_est"].tolist() == [-4202, -2813, 1900]  # input untouched
    captured = capsys.readouterr()
    assert "Q9 (Nine) not above the views floor" in captured.out
    assert "WARNING: override Q2" in captured.err and "Q1" not in captured.err  # Q2's Wikidata year changed


def test_select_roster_floor_undated_and_ties():
    year = config.FAME_REFERENCE_YEAR - 100
    people = people_frame(
        [
            ("Q5", 1000.0, year, year, "P569"),  # score 100k
            ("Q3", 2000.0, year + 50, year + 50, "P569"),  # same score, more views: ranks first
            ("Q4", 1000.0, year, year, "P569"),  # same score and views as Q5: QID breaks the tie
            ("Q1", 10_000.0, None, None, None),  # undated
            ("Q2", config.MIN_MEDIAN_VIEWS - 1, -9000, -9000, "P569"),  # below the floor
            ("Q6", 400.0, year, year, "P569"),
        ]
    )
    out = roster.select_roster(people, 3)
    assert out["qid"].tolist() == ["Q3", "Q4", "Q5"]
    assert out["rank"].tolist() == [1, 2, 3]
    assert out["score"].tolist() == [100_000, 100_000, 100_000]


def test_birth_bins_and_windows():
    edges = np.array([-np.inf, 0, 1500, 1900, np.inf])
    bins = roster.birth_bins([-500, 0, 1499, 1500, 1899, 1900, 2020], edges)
    assert bins.tolist() == [0, 1, 1, 2, 2, 3, 3]
    n_bins = config.WINDOW_BINS + 2
    sizes = roster.window_sizes(np.repeat(np.arange(n_bins), 10), n_bins)
    assert sizes.tolist() == [10 * config.WINDOW_BINS] * 3


def test_article_titles_uses_the_top_language_site():
    df = pd.DataFrame({"qid": ["Q1", "Q2", "Q3"], "top_lang": ["ja", "nan", None]})
    sitelinks = pd.DataFrame(
        {
            "qid": ["Q1", "Q1", "Q2", "Q3"],
            "site": ["jawiki", "enwiki", "zh_min_nanwiki", "dewiki"],
            "title": ["ja_title", "En_title", "Nan_title", "De_title"],
        }
    )
    out = roster.article_titles(df, sitelinks)
    assert out["enwiki_title"].iloc[0] == "En_title" and out["enwiki_title"].iloc[1:].isna().all()
    assert out["top_title"].tolist()[:2] == ["ja_title", "Nan_title"]
    assert pd.isna(out["top_title"].iloc[2])

import json

import numpy as np
from conftest import load_stage

import util

names = load_stage("10_name_windows.py")


def test_local_labels_renumber_tracks_and_keep_unlabelled():
    lab, tracks = names.local_labels(np.array([7, -1, 3, 7, 12, -1]))
    assert tracks.tolist() == [3, 7, 12]
    assert lab.tolist() == [1, -1, 0, 1, 2, -1]


def test_not_title_case():
    assert not names.not_title_case("Late Romantic and Early Modern Classical Composers")
    assert not names.not_title_case("Princesses and Queens Consort of the European Royal Houses")
    assert not names.not_title_case("Mid-20th-Century English Stage and Film Actors")
    assert not names.not_title_case("Grand Viziers and Governors under Mehmed II")
    assert not names.not_title_case("Italian Cinema Actors, Commedia all'Italiana")
    assert not names.not_title_case("World Cup-Winning Footballers, Ballon d'Or Era")
    assert names.not_title_case("Soviet and Eastern Bloc communist party leaders")
    assert names.not_title_case("composers of opera")


def test_review_flags_empty_gendered_case_and_clashes_within_a_window_only():
    entries = [
        {"window": 3, "layer": 0, "track": 1, "name": "Classic Hollywood Male Actors"},
        {"window": 3, "layer": 0, "track": 2, "name": ""},
        {"window": 3, "layer": 0, "track": 4, "name": "Painters"},
        {"window": 3, "layer": 0, "track": 5, "name": "painters"},
        {"window": 4, "layer": 0, "track": 4, "name": "Painters"},  # same name, other window: fine
    ]
    notes = names.review(entries)
    assert any("gendered" in n for n in notes)
    assert any("EMPTY" in n for n in notes)
    assert any("not title case: painters" in n for n in notes)
    assert sum("share the name" in n for n in notes) == 1


def test_write_json_atomic_round_trips(tmp_path):
    path = tmp_path / "sub" / "w.json"
    util.write_json_atomic([{"name": "Ötzi"}], path)
    assert json.loads(path.read_text(encoding="utf-8")) == [{"name": "Ötzi"}]
    assert not list(path.parent.glob("*.tmp"))

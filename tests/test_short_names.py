import json

from conftest import load_stage

sn = load_stage("11b_short_names.py")


def test_too_long_counts_characters_and_words_but_not_ampersands():
    assert not sn.too_long("Rock & Pop Music")
    assert sn.too_long("Political & Business Elites")  # 27 characters
    assert sn.too_long("A B C D")


def test_clashes_need_a_shared_window():
    short = {(2, 1): "Scientists", (2, 2): "scientists!", (2, 3): "Scientists", (2, 4): "Poets"}
    alive = {(2, 1): {3, 4}, (2, 2): {4, 5}, (2, 3): {9}, (2, 4): {4}}
    assert sn.clashes(short, alive) == [[(2, 1), (2, 2)]]


def test_parse_group_matches_region_ids():
    text = json.dumps(
        {"names": [{"region": "Region c4", "short_name": '"Kings"'}, {"region": "c9", "short_name": "X"}]}
    )
    assert sn.parse_group(text, [(2, 4)]) == {(2, 4): "Kings"}

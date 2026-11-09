import json

import pandas as pd
from conftest import load_stage

tracks = load_stage("11_name_tracks.py")


def facts(layer=2, track=30, female=0.4):
    return {
        "layer": layer, "track": track, "path": [(13, 1805, 181, "Composers"), (14, 1830, 338, "Hollywood Stars")],
        "first": 13, "last": 14, "first_best": ["Mozart"], "last_best": ["Chaplin"], "born": (1839, 1922),
        "best_known": ["Mozart", "Chaplin"], "n_people": 500,
        "female_share": female, "male_share": 1 - female,
    }  # fmt: skip


def test_track_prompt_carries_path_tier_and_gender_mix():
    p = tracks.track_prompt(facts())
    assert "w13 · 1805 · 181: Composers" in p and "w14 · 1830 · 338: Hollywood Stars" in p
    assert "1 to 4 word" in p  # coarse layer tier
    assert "40% are women and 60% are men" in p
    assert "Mozart; Chaplin" in p


def test_collisions_group_tracks_sharing_a_name_in_some_window():
    names = pd.DataFrame({
        "window": [0, 0, 0, 1, 1, 2, 2],
        "layer": [2, 2, 1, 2, 2, 2, 2],
        "track": [1, 2, 7, 2, 3, 4, 5],
    })  # fmt: skip
    final = {
        (2, 1): "Rulers",
        (2, 2): "rulers!",
        (1, 7): "Rulers",
        (2, 3): "Rulers",
        (2, 4): "Poets",
        (2, 5): "Painters",
    }
    groups = tracks.collisions(final, names)
    # 1 and 2 share window 0; 2 and 3 share window 1 -> one group; layer 1's "Rulers" is on another layer
    assert groups == [[(2, 1), (2, 2), (2, 3)]]


def test_parse_renames_keeps_only_the_ids_asked_about():
    text = json.dumps({"names": [{"region": "c1", "name": "Ancient Rulers"},
                                 {"region": "Region c2", "name": '"Medieval Kings".'},
                                 {"region": "c9", "name": "Not Asked"}]})  # fmt: skip
    got = tracks.parse_renames(text, [(2, 1), (2, 2)])
    assert got == {(2, 1): "Ancient Rulers", (2, 2): "Medieval Kings"}

import numpy as np
import pandas as pd
from conftest import load_stage

ex = load_stage("12_export_frontend.py")


def test_strip_parentheticals_nested_and_brackets():
    text = "Cleopatra VII Thea Philopator (Koine Greek: Κλεοπάτρα (lit. 'x'); 70/69 BC – 30 BC) was Queen [1] , ok."
    assert ex.strip_parentheticals(text) == "Cleopatra VII Thea Philopator was Queen, ok."


def test_excerpt_keeps_whole_sentences_and_skips_abbreviations():
    lead = (
        "John Smith (born 1900) was a U.S. Army general. He served with St. Clair in the war. "
        "He was born in Ohio.\nSecond paragraph."
    )
    assert (
        ex.excerpt(lead, 300)
        == "John Smith was a U.S. Army general. He served with St. Clair in the war. He was born in Ohio."
    )
    assert ex.excerpt(lead, 75) == "John Smith was a U.S. Army general. He served with St. Clair in the war."
    assert ex.excerpt(lead, 40) == "John Smith was a U.S. Army general."


def test_excerpt_cuts_a_long_first_sentence_at_a_word():
    lead = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu."
    out = ex.excerpt(lead, 30)
    assert out.endswith("…") and len(out) <= 31 and " " not in out[-2:]
    assert ex.excerpt("", 30) == "" and ex.excerpt(None, 30) == ""


def test_initials_are_not_sentence_ends():
    lead = "J. R. R. Tolkien was an English writer. He wrote books."
    assert ex.excerpt(lead, 45) == "J. R. R. Tolkien was an English writer."


def test_format_year_astronomical():
    assert ex.format_year(-355) == "356 BC"
    assert ex.format_year(0) == "1 BC"
    assert ex.format_year(-2050) == "2051 BC"
    assert ex.format_year(-10545) == "10,546 BC"
    assert ex.format_year(451) == "AD 451"
    assert ex.format_year(1879) == "1879"


def test_born_and_died_labels():
    assert ex.born_label(-355, 11, "P569") == "356 BC"
    assert ex.born_label(-428, 8, "P569") == "c. 429 BC"
    assert ex.born_label(1450, pd.NA, "death") == "c. 1450 (est.)"
    assert ex.born_label(1450, pd.NA, "override") == "1450"
    assert ex.died_label(pd.NA, pd.NA) == ""
    assert ex.died_label(1600, 7) == "c. 1600"


def test_display_name_falls_back_to_titles():
    assert ex.display_name("Plato", "Plato", "Plato") == "Plato"
    assert ex.display_name(pd.NA, "Some_Person", None) == "Some Person"
    assert ex.display_name(None, None, "慈雲院_(豊臣秀長室)") == "慈雲院 (豊臣秀長室)"


def test_article_ref_and_url():
    assert ex.article_ref("enwiki", "Alexander the Great", "Q8409") == "en:Alexander_the_Great"
    assert ex.article_ref("zh_yuewiki", "X Y", "Q1") == "zh-yue:X_Y"
    assert ex.article_ref(None, None, "Q379691") == "wd:Q379691"
    assert ex.article_url("ja:項燕") == "https://ja.wikipedia.org/wiki/%E9%A0%85%E7%87%95"
    assert ex.article_url("wd:Q1") == "https://www.wikidata.org/wiki/Q1"


def test_thumb_code_round_trip():
    url = "https://thumb.wikimedia.org/wikipedia/commons/thumb/3/3e/K.jpg/330px-K.jpg?utm_source=en.wikipedia.org"
    code = ex.thumb_code(url)
    assert code == "T/commons/thumb/3/3e/K.jpg/330px-K.jpg"
    assert ex.THUMB_HOSTS["T/"] + code[2:] == url.split("?")[0]
    assert ex.thumb_code(None) == "" and ex.thumb_code("https://example.org/a.png") == "https://example.org/a.png"


def test_translated_from():
    assert ex.translated_from("translated", "jawiki") == "Japanese"
    assert ex.translated_from("enwiki", "enwiki") == ""
    assert ex.translated_from("translated", "xxwiki") == "xx"


def test_normalize_windows_centres_and_scales_each_window():
    rng = np.random.default_rng(0)
    lay = pd.DataFrame({
        "window": np.repeat([0, 1], 50),
        "qid": [f"Q{i}" for i in range(100)],
        "x": np.r_[rng.normal(3, 5, 50), rng.normal(-1, 2, 50)],
        "y": np.r_[rng.normal(-2, 5, 50), rng.normal(4, 2, 50)],
    })  # fmt: skip
    out = ex.normalize_windows(lay)
    for _, g in out.groupby("window"):
        assert abs(g["x"].mean()) < 1e-5 and abs(g["y"].mean()) < 1e-5
        assert abs(np.sqrt((g["x"] ** 2 + g["y"] ** 2).mean()) - 1) < 1e-5


def test_quantize_round_trip():
    v = np.linspace(-2.5, 3.1, 1000)
    q = ex.quantize(v, -2.6, 3.2)
    assert q.dtype == np.dtype("<u2")
    assert np.abs(ex.dequantize(q, -2.6, 3.2) - v).max() < 5.8 / 65535


def test_person_windows_requires_consecutive_windows():
    lay = pd.DataFrame({"qid": ["A", "A", "B", "B", "B"], "window": [3, 4, 0, 1, 2]})
    w0, nw = ex.person_windows(lay, pd.Series(["B", "A"]))
    assert w0.tolist() == [0, 3] and nw.tolist() == [3, 2]
    bad = pd.DataFrame({"qid": ["A", "A"], "window": [1, 3]})
    try:
        ex.person_windows(bad, pd.Series(["A"]))
    except SystemExit:
        pass
    else:
        raise AssertionError("gap not caught")


def test_interval_colors_respect_the_buffer_and_use_the_minimum():
    first = np.array([0, 0, 3, 5, 6, 10])
    last = np.array([2, 4, 4, 8, 6, 12])
    for buffer in (0, 1):
        color = ex.interval_colors(first, last, buffer)
        lo, hi = first - buffer, last + buffer
        for i in range(len(first)):
            for j in range(i + 1, len(first)):
                if lo[i] <= hi[j] and lo[j] <= hi[i]:
                    assert color[i] != color[j]
        assert color.max() + 1 == ex.max_overlap(first, last, buffer)


def test_ribbon_order_brings_flow_partners_together():
    # Tracks 0..3 all alive at step 0; 0 flows into 3, but the start order puts 1 and 2 between them.
    alive = np.ones((1, 4), bool)
    flows = pd.DataFrame({"a": [0], "b": [3], "step": [0], "people": [40]})
    start = [0, 1, 2, 3]
    order = ex.ribbon_order(start, flows, alive)
    pos = ex.positions(order, 4)
    assert abs(pos[0] - pos[3]) == 1
    assert ex.order_cost(pos, flows["a"].to_numpy(), flows["b"].to_numpy(), flows["people"].to_numpy(float),
                         alive[flows["step"].to_numpy()]) == 0  # fmt: skip


def test_coarse_flows_counts_moves_between_tracks():
    tracks = pd.DataFrame({
        "window": [0, 1, 0, 1, 0, 1],
        "qid": ["A", "A", "B", "B", "C", "C"],
        "track_2": [0, 1, 0, 0, 2, -1],
    })  # fmt: skip
    f = ex.coarse_flows(tracks)
    assert f[["a", "b", "step", "people"]].values.tolist() == [[0, 1, 0, 1]]

from conftest import load_stage

fetch_pageviews = load_stage("02_fetch_pageviews.py")

LOOKUP = {
    b"enwiki Douglas_Adams": 42,
    b"enwiki The_Hitchhiker": 42,  # e.g. a folded redirect source
    b'enwiki "Weird_Al"_Yankovic': 99,
    b"dewiki Douglas_Adams": 42,
    b"be_x_oldwiki Some_Title": 7,
}

LINES = [
    b"en.wikipedia Douglas_Adams 8091 desktop 100 A100\n",
    b"en.wikipedia Douglas_Adams null mobile-web 50 A50\n",
    b"en.wikipedia The_Hitchhiker 1 desktop 5 A5\n",  # same person, same lang, same access -> summed
    b'en.wikipedia "\\"Weird_Al\\"_Yankovic" 2 mobile-app 7 A7\n',  # CSV-quoted title
    b"de.wikipedia Douglas_Adams 3 desktop 11 A11\n",
    b"be-tarask.wikipedia Some_Title 4 desktop 3 A3\n",  # legacy site id
    b"en.wikipedia Not_A_Person 5 desktop 1000 A1000\n",  # not in lookup
    b"short line\n",
]


def test_aggregate_lines_keeps_lang_and_access():
    df, n_rows = fetch_pageviews.aggregate_lines(iter(LINES), LOOKUP, langs=[])
    assert n_rows == 7
    got = {(r.qid_int, r.lang, r.access): r.views for r in df.itertuples()}
    assert got == {
        (42, "en", "desktop"): 105,
        (42, "en", "mobile-web"): 50,
        (42, "de", "desktop"): 11,
        (99, "en", "mobile-app"): 7,
        (7, "be-tarask", "desktop"): 3,
    }


def test_large_qids_survive_key_packing():
    qid = 139_000_000  # beyond current Wikidata max, still must round-trip
    df, _ = fetch_pageviews.aggregate_lines(
        iter([b"zh-min-nan.wikipedia X 1 mobile-app 4 A4\n"]), {b"zh_min_nanwiki X": qid}, langs=[]
    )
    assert df.to_dict("records") == [{"qid_int": qid, "lang": "zh-min-nan", "access": "mobile-app", "views": 4}]

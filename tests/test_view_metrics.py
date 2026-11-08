import numpy as np
import pandas as pd
import pytest
from conftest import load_stage

view_metrics = load_stage("03_view_metrics.py")


def month(rows):
    return pd.DataFrame(rows, columns=["qid_int", "lang", "access", "views"])


ROSTER = pd.DataFrame({"qid": ["Q42", "Q7", "Q1000"], "n_wikipedias": [50, 3, 1]})
MONTHS = {
    "2025-11": month([
        (42, "en", "desktop", 60), (42, "en", "mobile-web", 40), (42, "de", "desktop", 10), (7, "fr", "mobile-app", 5),
    ]),
    "2025-12": month([(42, "en", "mobile-web", 200), (7, "fr", "desktop", 5)]),
    "2026-01": month([(1000, "en", "desktop", 1)]),
}  # fmt: skip


def test_accumulate_monthly_totals_and_access_shares():
    out = view_metrics.accumulate(ROSTER, MONTHS).set_index("qid")
    assert out.loc["Q42", ["views_202511", "views_202512", "views_202601"]].tolist() == [110, 200, 0]
    assert out.loc["Q42", "median_views"] == 110
    assert out.loc["Q42", "n_months"] == 2 and out.loc["Q42", "median_present"] == 155
    assert out.loc["Q42", "desktop_share"] == pytest.approx(70 / 310)
    assert out.loc["Q7", "app_share"] == pytest.approx(0.5)
    assert out.loc["Q1000", "median_views"] == 0  # absent months count as zero
    assert list(out.index) == ["Q42", "Q7", "Q1000"]  # roster order preserved


def test_accumulate_rejects_unknown_qids():
    with pytest.raises(ValueError, match="not in the roster"):
        view_metrics.accumulate(ROSTER, {"2025-11": month([(999, "en", "desktop", 1)])})


def test_accumulate_applies_cell_keep_to_every_month_but_not_access_shares():
    keep = pd.Series([0.5], index=pd.MultiIndex.from_tuples([(42, "en", True)], names=view_metrics.CELL_INDEX))
    out = view_metrics.accumulate(ROSTER, MONTHS, keep).set_index("qid")
    assert out.loc["Q42", ["views_202511", "views_202512"]].tolist() == [80, 200]  # 60 en desktop -> 30
    assert out.loc["Q42", "desktop_share"] == pytest.approx(70 / 310)  # raw
    assert out.loc["Q1000", "views_202601"] == 1  # other people's en desktop untouched


def test_access_cells_baseline_over_everyone_cells_for_kept_only():
    log_base, cells = view_metrics.access_cells(MONTHS, np.array([42]))
    assert log_base["en"] == pytest.approx(np.log(61 / 240))  # Q1000's desktop view counts in the baseline
    assert log_base["fr"] == pytest.approx(0.0)
    assert log_base["de"] == np.inf  # no mobile views at all
    cells = cells.set_index(["qid_int", "lang"])
    assert cells.loc[(42, "en"), ["D", "M"]].tolist() == [60, 240]
    assert cells.loc[(42, "en"), ["D_med", "M_med"]].tolist() == [0, 40]  # months [60, 0, 0] and [40, 200, 0]
    assert cells.loc[(42, "de"), ["D", "M", "D_med", "M_med"]].tolist() == [10, 0, 0, 0]
    assert set(cells.index.get_level_values("qid_int")) == {42}


def test_one_month_burst_left_to_the_median_steady_crawler_clipped():
    def rows(j):
        return [
            (5, "en", "desktop", 10_000 if j == 0 else 10), (5, "en", "mobile-web", 30),  # one-month burst
            (6, "en", "desktop", 10_000), (6, "en", "mobile-web", 30),  # steady crawler
            (7, "en", "desktop", 100_000), (7, "en", "mobile-web", 300_000),  # sets the en baseline
        ]  # fmt: skip

    months = {f"2026-{j + 1:02d}": month(rows(j)) for j in range(5)}
    log_base, cells = view_metrics.access_cells(months, np.array([5, 6]))
    out = view_metrics.clip_cells(cells, log_base).set_index("qid_int")
    assert out.loc[5, ["keep_D", "keep_M"]].tolist() == [1.0, 1.0]
    assert out.loc[6, "keep_D"] < 0.05 and out.loc[6, "keep_M"] == 1.0  # ~31 x 0.37 x 16 of 10,000 kept


def test_clip_cells_shrinks_only_the_over_represented_side():
    log_base = pd.Series({"xx": np.log(0.5), "yy": np.inf})
    cells = pd.DataFrame({
        "qid_int": [1, 2, 3, 4, 5],
        "lang": ["xx", "xx", "xx", "yy", "xx"],
        "D_med": [100, 10_000, 9, 10_000, 0],
        "M_med": [100, 99, 10_000, 0, 5],
    })  # fmt: skip
    out = view_metrics.clip_cells(cells, log_base).set_index("qid_int")
    assert out.loc[1, ["keep_D", "keep_M"]].tolist() == [1.0, 1.0]  # r = log 2, inside the band
    d = 10_000 * out.loc[2, "keep_D"]  # desktop crawler: r lands exactly on +log MAX_DESKTOP_SKEW
    assert np.log((d + 1) / 100) - np.log(0.5) == pytest.approx(np.log(view_metrics.config.MAX_DESKTOP_SKEW))
    assert out.loc[2, "keep_M"] == 1.0
    m = 10_000 * out.loc[3, "keep_M"]  # mobile crawler: r lands exactly on -log MAX_MOBILE_SKEW
    assert np.log(10 / (m + 1)) - np.log(0.5) == pytest.approx(-np.log(view_metrics.config.MAX_MOBILE_SKEW))
    assert out.loc[3, "keep_D"] == 1.0
    assert out.loc[4, ["keep_D", "keep_M"]].tolist() == [1.0, 1.0]  # no finite baseline: left alone
    assert out.loc[5, ["keep_D", "keep_M"]].tolist() == [1.0, 1.0]  # 5 mobile, 0 desktop: small counts survive


def test_cell_keep_series_lists_clipped_sides_only():
    clipped = pd.DataFrame({
        "qid_int": [1, 2, 3], "lang": ["en", "en", "fr"], "D": [0, 0, 0], "M": [0, 0, 0],
        "keep_D": [1.0, 0.25, 1.0], "keep_M": [1.0, 1.0, 0.5],
    })  # fmt: skip
    s = view_metrics.cell_keep_series(clipped)
    assert s.to_dict() == {(2, "en", True): 0.25, (3, "fr", False): 0.5}


def test_language_metrics():
    per_lang = pd.DataFrame({"qid_int": [42, 42, 42, 7], "lang": ["en", "de", "fr", "fr"], "views": [300, 10, 0, 5]})
    res = view_metrics.language_metrics(per_lang).set_index("qid_int")
    assert res.loc[42, "top_lang"] == "en"
    assert res.loc[42, "top_lang_share"] == pytest.approx(300 / 310)
    assert res.loc[42, "en_share"] == pytest.approx(300 / 310)
    assert res.loc[42, "n_langs"] == 2  # a fully clipped language doesn't count
    assert res.loc[7, "en_share"] == 0
    assert np.isfinite(res["top_lang_share"]).all()

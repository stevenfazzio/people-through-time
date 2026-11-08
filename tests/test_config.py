import numpy as np

import config


def test_fame_score_is_views_times_age():
    assert config.fame_score(1000, config.FAME_REFERENCE_YEAR - 50) == 1000 * (50 - config.AGE_OFFSET)
    assert config.fame_score(10, config.FAME_REFERENCE_YEAR + 3) == 10  # age floored at 1


def test_bin_edges_equal_count_and_whole_years():
    rng = np.random.default_rng(0)
    years = np.r_[rng.integers(-500, 1800, 3000), rng.integers(1900, 2010, 30000)].astype(float)
    edges = config.bin_edges(years)
    assert np.isneginf(edges[0]) and np.isposinf(edges[-1])
    inner = edges[1:-1]
    assert np.all(inner == np.floor(inner)) and np.all(np.diff(edges) > 0)
    counts = np.histogram(years, bins=np.r_[-1e9, inner, 1e9])[0]
    assert len(counts) == config.N_BINS
    assert counts.max() / counts.min() < 1.6  # ~equal, modulo whole-year snapping

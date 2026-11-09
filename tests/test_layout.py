import numpy as np
from conftest import load_stage

import config

layout = load_stage("08_aligned_umap.py")


def test_windows_and_relations():
    bins = np.repeat(np.arange(config.WINDOW_BINS + 2), 2)  # two people per bin, rank order
    members = layout.window_members(bins, n_windows=3)
    assert [len(m) for m in members] == [2 * config.WINDOW_BINS] * 3
    assert members[1][0] == 2 and members[2][-1] == len(bins) - 1
    rel = layout.relations(members)
    assert len(rel) == 2 and len(rel[0]) == 2 * (config.WINDOW_BINS - 1)
    for (a, b), r in zip(zip(members, members[1:]), rel):  # same person on both sides
        assert all(a[i] == b[j] for i, j in r.items())


def test_displacement_is_scaled_by_rms_radius():
    a = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])  # RMS radius 1
    b = a + np.array([0.5, 0.0])  # everyone moves 0.5; RMS radius about the mean is still 1
    d = layout.displacement([a, b], [np.arange(4), np.arange(4)])
    assert np.allclose(d, 0.5)


def test_knn_fidelity_and_same_share():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(50, 2))
    assert layout.knn_fidelity(np.c_[pts, np.ones(50) * 10], pts, k=5) > 0.5
    vals = np.array(["a"] * 24 + ["b"] * 26, dtype=object)
    nn = np.array([[i ^ 1] for i in range(50)])  # neighbours are pairs (0,1), (2,3)..., never across a/b
    share, chance = layout.same_share(vals, nn)
    assert share == 1.0 and abs(chance - (24 * 23 + 26 * 25) / 50 / 49) < 1e-9


def test_binary_gender_handles_missing_values():
    import pandas as pd

    g = layout.binary_gender(pd.Series(["male", pd.NA, "female", "trans woman"], dtype="string"))
    assert g.tolist() == ["male", None, "female", None]
    assert (g == "female").tolist() == [False, False, True, False]

import numpy as np
import pytest
from conftest import load_stage

tracks = load_stage("09_label_tracks.py")


def test_window_slices_need_window_order():
    sl = tracks.window_slices(np.array([0, 0, 1, 1, 1, 2]))
    assert [(s.start, s.stop) for s in sl] == [(0, 2), (2, 5), (5, 6)]
    with pytest.raises(SystemExit):
        tracks.window_slices(np.array([0, 1, 0]))


def test_leaf_labels_find_two_blobs():
    rng = np.random.default_rng(0)
    xy = np.r_[rng.normal(0, 0.1, (60, 2)), rng.normal(10, 0.1, (60, 2))]
    lab = tracks.leaf_labels(tracks.linkage(xy), 40, len(xy))  # 40 > 60 / 2: a blob can't split in two
    a, b = lab[:60][lab[:60] >= 0], lab[60:][lab[60:] >= 0]
    assert lab.max() + 1 == 2
    assert len(set(a)) == 1 and len(set(b)) == 1 and a[0] != b[0]
    assert (lab >= 0).mean() > 0.8


def test_closest_breaks_ties_to_the_smaller_size():
    assert tracks.closest({10: 40.0, 20: 21.0, 30: 19.0}, 20) == 20
    assert tracks.closest({10: 40.0, 20: 21.0, 30: 19.0}, 39) == 10


def test_match_follows_people_not_label_numbers():
    pids = np.arange(10)
    a = np.r_[np.zeros(5, int), np.ones(5, int)]
    b = np.r_[np.ones(5, int), np.zeros(5, int)]
    assert tracks.match(a, pids, b, pids) == {1: 0, 0: 1}
    assert tracks.match(a, pids, b, pids, link_min=1.1) == {}
    assert tracks.match(a, pids, b, pids, cand_a={1}) == {0: 1}


def test_match_needs_a_mutual_best():
    # a0 = people 0-9, a1 = 10-29; in b, cluster 0 holds all of a1 plus three of a0, the rest of a0 unclustered
    pid_a = np.arange(30)
    lab_a = np.r_[np.zeros(10, int), np.ones(20, int)]
    lab_b = np.r_[np.zeros(3, int), -np.ones(7, int), np.zeros(20, int)]
    assert tracks.match(lab_a, pid_a, lab_b, pid_a) == {0: 1}  # a1 continues as b0; a0 ends (a merge)


def test_link_tracks_continue_and_bridge_a_one_window_gap():
    pids = [np.arange(20)] * 4
    local = [
        np.r_[np.zeros(10, int), np.ones(10, int)],
        np.r_[np.ones(10, int), np.zeros(10, int)],  # label numbers swapped: tracks follow the people
        np.r_[np.zeros(10, int), -np.ones(10, int)],  # second group unclustered for one window
        np.r_[np.zeros(10, int), np.ones(10, int)],
    ]
    t = tracks.link_tracks(local, pids, bridge=True)
    first, second = t[0][0], t[0][10]
    assert first != second
    assert (t[1][:10] == first).all() and (t[1][10:] == second).all()
    assert (t[2][:10] == first).all() and (t[2][10:] == -1).all()
    assert (t[3][10:] == second).all()  # bridged back to its old track
    t2 = tracks.link_tracks(local, pids, bridge=False)
    assert t2[3][10] not in {t2[0][0], t2[0][10]}  # without bridging, a new track


def test_split_share():
    win = np.zeros(6, int)
    child = np.array([0, 0, 1, 1, -1, -1])
    assert tracks.split_share(win, child, np.array([5, 5, -1, -1, 5, 7])) == 0.0
    assert tracks.split_share(win, child, np.array([5, 6, 7, 7, 5, 7])) == 0.5


def test_parent_tree_majority_skip_and_root():
    rows = (
        [[0, 0, 5]] * 6  # fine 0: mostly under mid 0 ...
        + [[0, 1, -1]] * 2  # ... sometimes under mid 1
        + [[1, -1, 5]] * 4  # fine 1: never under a named mid cluster, under coarse 5 -> skips a layer
        + [[2, -1, -1]] * 3  # fine 2: never under anything -> root
    )
    tree = tracks.parent_tree(np.array(rows)).set_index(["layer", "track"])
    root = (3, 0)
    assert tuple(tree.loc[(0, 0), ["parent_layer", "parent_track"]]) == (1, 0)
    assert tree.loc[(0, 0), "parent_share"] == pytest.approx(0.75)
    assert tuple(tree.loc[(0, 1), ["parent_layer", "parent_track"]]) == (2, 5)
    assert tuple(tree.loc[(0, 2), ["parent_layer", "parent_track"]]) == root
    assert np.isnan(tree.loc[(0, 2), "parent_share"])
    assert tuple(tree.loc[(1, 0), ["parent_layer", "parent_track"]]) == (2, 5)
    assert tuple(tree.loc[(1, 1), ["parent_layer", "parent_track"]]) == root  # its rows have no coarse cluster
    assert tuple(tree.loc[(2, 5), ["parent_layer", "parent_track"]]) == root


def test_track_stats():
    s = tracks.track_stats(np.array([0, 0, 1, 2]), np.array([0, 1, 0, 0]), np.array([[3], [3], [3], [3]]))
    r = s.iloc[0]
    assert (r.first_window, r.last_window, r.n_windows, r.n_people, r.peak_size) == (0, 2, 3, 2, 2)

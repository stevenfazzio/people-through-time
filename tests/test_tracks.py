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


def test_lineage_check_stops_a_relay_and_rematches():
    # A = people 0-19, B = 20-49. Window 1: twelve of B (20-31) cross into A's cluster; A keeps its id (20 vs 12)
    # and B its own (18 vs 12). Window 2: B's 40-49 have left; A's people split in two; the crossers regroup
    # with B's rest (32-39) -- and, having worn A's label, would carry A's id with them.
    pids = [np.arange(50), np.arange(50), np.arange(40)]
    local = [
        np.r_[np.zeros(20, int), np.ones(30, int)],
        np.r_[np.zeros(32, int), np.ones(18, int)],
        np.r_[np.zeros(10, int), np.ones(10, int), np.full(20, 2)],
    ]
    a, b = 0, 20
    t = tracks.link_tracks(local, pids, lineage_min=0.0)
    assert t[2][20] == t[0][a]  # the relay: the B group now holds A's id ...
    assert t[2][0] not in {t[0][a], t[0][b]}  # ... and A's own people get a new track
    drops = []
    t = tracks.link_tracks(local, pids, lineage_min=0.1, lineage_people=5, dropped=drops)
    assert t[2][20] == t[0][b] and t[2][39] == t[0][b]  # the B group continues B
    assert t[2][0] == t[0][a]  # A continues in one of its own pieces
    assert t[2][10] not in {t[0][a], t[0][b]}  # the other piece is new
    assert len(drops) == 1
    d = drops[0]  # the carriers are the twelve crossers, all from B
    assert (d["window"], d["track"], d["n"], d["share"], d["top_track"]) == (2, t[0][a], 12, 0.0, t[0][b])
    assert d["became"] == t[0][b] and not d["new"] and d["track_continues"]
    sl = [slice(0, 50), slice(50, 100), slice(100, 140)]
    shares = tracks.lineage_shares(np.concatenate(pids), sl, np.concatenate(t), min_people=5)
    assert len(shares) and shares.min() >= 0.1


def test_lineage_check_ignores_older_members_who_dont_carry_the_link():
    # Track A = people 0-19, S = 20-39. Window 1: newcomers 40-59 join A's cluster. Window 2: A's first people
    # have left; S's people 20-34 join A's newcomers. The cluster's older members are all from S, but the
    # people carrying A's label into it are A's newcomers, so it continues A (Roman -> Byzantine rulers).
    pids = [np.arange(40), np.arange(60), np.arange(20, 60)]
    local = [
        np.r_[np.zeros(20, int), np.ones(20, int)],
        np.r_[np.zeros(20, int), np.ones(20, int), np.zeros(20, int)],
        np.r_[np.zeros(15, int), np.ones(5, int), np.zeros(20, int)],
    ]
    drops = []
    t = tracks.link_tracks(local, pids, lineage_min=0.2, lineage_people=5, dropped=drops)
    at = [dict(zip(p.tolist(), x.tolist())) for p, x in zip(pids, t)]
    assert at[2][40] == at[2][20] == at[0][0]
    assert drops == []


def test_lineage_check_skips_few_carriers_or_new_tracks():
    lab, pid = np.array([0, 0, 0, 0]), np.arange(4)
    prev = (np.array([0, 0, 0, 1]), np.arange(4), np.array([7, 9]))  # w-1: clusters 0 (track 7), 1 (track 9)
    old = (np.arange(4), np.array([7, 8, 8, 5]))  # w-2 tracks per person
    lin = tracks.Lineage(lab, pid, prev, old, min_share=0.5, min_people=2)
    assert not lin.ok(0, 0)  # carriers 0-2: one of three was in track 7
    assert lin.dropped[0]["top_track"] == 8
    assert tracks.Lineage(lab, pid, prev, old, min_share=0.3, min_people=2).ok(0, 0)
    assert lin.ok(1, 0)  # track 9 didn't exist at w-2: nothing to check
    assert tracks.Lineage(lab, pid, prev, old, min_share=0.5, min_people=4).ok(0, 0)  # too few carriers
    assert tracks.Lineage(lab, pid).ok(0, 0)  # no earlier windows


def _unit(deg):
    return np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])


def test_segment_starts_need_two_windows_in_a_row_and_reanchor():
    bar = np.cos(np.radians(45))
    assert tracks.segment_starts([_unit(a) for a in (0, 10, 20, 60, 70, 80)], bar) == [
        (3, pytest.approx(np.cos(np.radians(60))))
    ]
    assert tracks.segment_starts([_unit(a) for a in (0, 60, 10, 20)], bar) == []  # one window out: noise
    assert tracks.segment_starts([_unit(a) for a in (0, 10, 60)], bar) == []  # the last window alone can't cut
    starts = tracks.segment_starts([_unit(a) for a in (0, 50, 60, 110, 120)], bar)
    assert [i for i, _ in starts] == [1, 3]  # anchored at 50 after the first cut, so 110 is cut again


def test_neighbour_similarity_takes_the_nearest_other_track():
    cents = {(0, 0): _unit(0), (1, 0): _unit(30), (2, 0): _unit(100), (0, 1): _unit(0)}  # window 1: alone
    near = sorted(tracks.neighbour_similarity(cents).round(3).tolist())
    assert near == sorted([round(np.cos(np.radians(30)), 3)] * 2 + [round(np.cos(np.radians(70)), 3)])


def test_refine_bounds_puts_a_turn_window_with_the_side_it_resembles():
    left = [_unit(a) for a in (0, 5, 10, 70, 80, 85)]  # 70 found far from 0 only at index 4, but is like 80-85
    assert tracks.refine_bounds(left, [4]) == [3]
    right = [_unit(a) for a in (0, 5, 10, 15, 80, 85)]
    assert tracks.refine_bounds(right, [3]) == [4]  # 15 belongs with 0-10
    assert tracks.refine_bounds([_unit(a) for a in (0, 80)], [1]) == [1]  # each side keeps a window


def test_drift_cut_renumbers_the_rest_of_a_track():
    sl = [slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8)]
    track = np.array([5, 7, 5, 7, 5, 7, 5, 7])
    cents = {(5, 0): _unit(0), (5, 1): _unit(10), (5, 2): _unit(80), (5, 3): _unit(85)}
    cents |= {(7, w): _unit(180) for w in range(4)}
    out, cuts = tracks.drift_cut(sl, track, cents, bar=0.5)
    assert out.tolist() == [5, 7, 5, 7, 8, 7, 8, 7]
    cos80 = pytest.approx(np.cos(np.radians(80)))
    assert cuts == [{"track": 5, "from": 5, "new": 8, "window": 2, "found": 2, "cos": cos80}]


def test_track_centroids_remove_each_windows_mean():
    vectors = np.array([[1.0, 0.0], [1.0, 2.0], [3.0, 0.0], [3.0, 2.0]])
    rows = np.array([0, 1, 2, 3])
    t = np.array([[0], [1], [0], [1]])  # one window of four points
    c = tracks.track_centroids(vectors, rows, [slice(0, 4)], t)[0]
    # window mean (2, 1): track 0 -> mean of (-1, -1), (1, -1) = (0, -1); track 1 -> (0, 1)
    assert c[(0, 0)] == pytest.approx([0.0, -1.0]) and c[(1, 0)] == pytest.approx([0.0, 1.0])


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

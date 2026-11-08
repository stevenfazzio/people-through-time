import numpy as np
import pandas as pd
import pytest

import util


def test_save_npz_atomic_writes_real_data(tmp_path):
    # np.savez appends .npz to any name lacking it; a .npz.tmp temp name once
    # wrote the data to a stray file and renamed an empty one into place.
    path = tmp_path / "x.npz"
    util.save_npz_atomic(path, a=np.arange(5))
    assert np.load(path)["a"].tolist() == [0, 1, 2, 3, 4]
    assert [p.name for p in tmp_path.iterdir()] == ["x.npz"]


def test_write_parquet_atomic_roundtrip(tmp_path):
    path = tmp_path / "t.parquet"
    util.write_parquet_atomic(pd.DataFrame({"a": [1, 2]}), path)
    assert pd.read_parquet(path)["a"].tolist() == [1, 2]


@pytest.mark.parametrize(("raw", "want"), [("Plain", "Plain"), ('"\\"Weird_Al\\"_Yankovic"', '"Weird_Al"_Yankovic')])
def test_parse_dump_title(raw, want):
    assert util.parse_dump_title(raw) == want

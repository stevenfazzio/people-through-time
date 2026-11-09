import numpy as np
from conftest import load_stage

erase = load_stage("07b_erase_gender.py")


def test_erasure_removes_the_direction_and_renormalizes():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 8)).astype(np.float32)
    female = np.arange(200) < 60
    x[female, 0] += 2.0  # a gender signal along axis 0
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    d = erase.gender_direction(x, female, ~female)
    assert np.isclose(np.linalg.norm(d), 1.0) and abs(d[0]) > 0.9
    out = erase.erase(x, d)
    assert np.allclose(out @ d, 0, atol=1e-6)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    assert np.allclose(erase.erase(out, d), out, atol=1e-6)  # idempotent


def test_same_gender_share():
    x = np.array([[1, 0], [0.99, 0.1], [0, 1], [0.1, 0.99]], dtype=np.float32)
    sex = np.array(["female", "female", "male", ""])
    assert erase.same_gender_share(x, sex, k=1) == 1.0  # the unknown person is skipped

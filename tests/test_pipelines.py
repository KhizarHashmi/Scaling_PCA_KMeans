"""Checks on the preprocessing recipes, the rotation property and the gap rule."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipelines import RAW, RAW_PCA, RECIPES, Z, Z_PCA, make_preprocessor, rotation_check, transform_pair  # noqa: E402
from selection import choose_by_gap  # noqa: E402
import pandas as pd  # noqa: E402


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 4)) * np.array([1000.0, 1.0, 0.1, 0.01])
    b = rng.normal(size=(100, 4)) * np.array([1000.0, 1.0, 0.1, 0.01]) + 5
    return a, b


def test_scaler_uses_fit_part_only(toy):
    a, b = toy
    A, B, prep = transform_pair(Z, a, b)
    assert np.allclose(A.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(A.std(axis=0), 1, atol=1e-10)
    assert not np.allclose(B.mean(axis=0), 0, atol=1e-2)
    assert np.allclose(prep.named_steps["scale"].mean_, a.mean(axis=0))


def test_raw_changes_nothing(toy):
    a, b = toy
    A, B, _ = transform_pair(RAW, a, b)
    assert np.array_equal(A, a) and np.array_equal(B, b)


def test_pca_keeps_requested_variance(toy):
    a, b = toy
    for recipe in (RAW_PCA, Z_PCA):
        prep = make_preprocessor(recipe).fit(a)
        assert prep.named_steps["pca"].explained_variance_ratio_.sum() >= 0.95


def test_four_recipes_are_the_2x2():
    assert {(r.scaler, r.pca is not None) for r in RECIPES} == {
        ("none", False), ("none", True), ("zscore", False), ("zscore", True)}


def test_rotation_leaves_objective_alone(toy):
    a, _ = toy
    Ja, Jb, la, lb = rotation_check(a, k=3)
    assert abs(Ja - Jb) / Ja < 1e-9
    assert adjusted_rand_score(la, lb) == 1.0


def test_gap_rule():
    curve = pd.DataFrame({"k": [2, 3, 4], "gap": [1.0, 2.0, 2.0], "gap_se": [0.1, 0.1, 0.1]})
    assert choose_by_gap(curve) == (3, True)
    rising = pd.DataFrame({"k": [2, 3, 4], "gap": [1.0, 2.0, 3.0], "gap_se": [0.1, 0.1, 0.1]})
    assert choose_by_gap(rising) == (4, False)

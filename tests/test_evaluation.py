"""Check on the shuffled-data null used for the silhouette."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation import shuffle_columns  # noqa: E402


def test_shuffle_keeps_each_column_but_breaks_links():
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    X = np.column_stack([x, x + rng.normal(scale=0.01, size=2000)])
    S = shuffle_columns(X, 3)
    assert np.array_equal(np.sort(S[:, 0]), np.sort(X[:, 0]))
    assert abs(np.corrcoef(S.T)[0, 1]) < 0.1 < np.corrcoef(X.T)[0, 1]

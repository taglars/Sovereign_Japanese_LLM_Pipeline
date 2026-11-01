"""The statistics, checked against values derived by hand."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jv.analyze import (kendall_tau_b, krippendorff_alpha, pearson, rank,  # noqa: E402
                        spearman, within_one)

# The standard three-observer, twelve-unit worked example.
A = [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None]
B = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3]
C = [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None]
UNITS = [[v for v in col if v is not None] for col in zip(A, B, C)]


def test_alpha_interval():
    """alpha = 1 - Do/De for the example above."""
    assert math.isclose(krippendorff_alpha(UNITS), 1 - (10 / 28) / (1958 / 756),
                        rel_tol=1e-9)


def test_alpha_bounds():
    assert krippendorff_alpha([[3, 3], [5, 5], [7, 7]]) == 1.0
    # Every unit identical in value: no variance anywhere, treated as total agreement
    # rather than 0/0.
    assert krippendorff_alpha([[4, 4], [4, 4]]) == 1.0
    # Systematic disagreement is worse than chance.
    assert krippendorff_alpha([[1, 9], [9, 1], [1, 9], [9, 1]]) < 0
    assert math.isnan(krippendorff_alpha([[1], [2]]))       # nothing pairable


def test_spearman():
    assert math.isclose(spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]), 1.0)
    assert math.isclose(spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), -1.0)
    # Monotone but not linear: Spearman sees 1.0 where Pearson does not.
    assert math.isclose(spearman([1, 2, 3, 4], [1, 4, 9, 16]), 1.0)
    assert pearson([1, 2, 3, 4], [1, 4, 9, 16]) < 1.0
    assert math.isnan(spearman([1], [1]))


def test_rank_ties_are_averaged():
    assert rank([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]


def test_kendall():
    assert kendall_tau_b([4, 3, 2, 1], [4, 3, 2, 1]) == 1.0
    assert kendall_tau_b([4, 3, 2, 1], [1, 2, 3, 4]) == -1.0


def test_within_one():
    assert within_one([5, 5, 5], [5, 6, 4]) == 1.0
    assert within_one([5, 5], [5, 8]) == 0.5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("analyze: all checks pass")

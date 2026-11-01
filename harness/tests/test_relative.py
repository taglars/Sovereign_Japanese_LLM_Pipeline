"""Relative agreement: scale-free by construction."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jv import relative as R                                       # noqa: E402

S = [f"S{i:02d}" for i in range(1, 9)]
A = {s: float(i) for i, s in enumerate(S)}


def test_level_and_spread_do_not_matter():
    harsh = {s: 0.5 * v - 3 for s, v in A.items()}       # same order, other scale
    r = R.pairwise(A, harsh, S)
    assert r["rate"] == 1.0 and r["decided"] == 28


def test_reversed_order_is_total_disagreement():
    assert R.pairwise(A, {s: -v for s, v in A.items()}, S)["rate"] == 0.0


def test_ties_are_not_disagreement():
    flat = {s: 7.0 for s in S}
    flat["S01"] = 6.0
    r = R.pairwise(A, flat, S)
    assert r["tie_b"] == 21 and r["decided"] == 7 and r["rate"] == 1.0


def test_kendall_w_bounds():
    assert abs(R.kendall_w([A, A, A], S) - 1.0) < 1e-12
    assert abs(R.kendall_w([A, {s: -v for s, v in A.items()}], S)) < 1e-12


def test_standardised_alpha_ignores_offsets():
    shifted = {s: v + 4 for s, v in A.items()}
    z = [R.standardise(x, S) for x in (A, shifted)]
    assert R.alpha([A, shifted], S) < 0.6 and abs(R.alpha(z, S) - 1.0) < 1e-9


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except Exception as exc:                                # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {exc!r}")
    print("relative: " + ("all checks pass" if not failures else f"{failures} FAILED"))
    raise SystemExit(1 if failures else 0)

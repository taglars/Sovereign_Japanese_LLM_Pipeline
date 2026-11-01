"""Experiment 3 analysis, on data with a known answer."""

from __future__ import annotations

import csv
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jv import exp3                                                # noqa: E402
from jv.rubric import SCORED_KEYS                                  # noqa: E402
from jv.store import Record, Store                                 # noqa: E402


def _fixture(tmp: Path, effect: dict, n: int = 60, repeats: int = 3):
    """Fake judge store: JA scores shifted by `effect[aspect]`, plus noise."""
    rng = random.Random(7)
    key_rows, store = [], Store(tmp / "fake-judge.db")
    for c in range(n):
        base = rng.uniform(3, 7)
        for cond in ("en", "ja"):
            sid = f"G{c:03d}{cond}"
            key_rows.append({"script_id": sid, "customer": f"C{c:03d}",
                             "condition": cond, "family": ("light", "heavy")[c % 2]})
            for lang in ("en", "ja"):
                for rep in range(repeats):
                    scores = {a: max(1, min(10, round(base + (effect.get(a, 0) if cond == "ja"
                                                             else 0) + rng.gauss(0, 0.8))))
                              for a in SCORED_KEYS}
                    store.write(Record(script_id=sid, judge="fake-judge", rubric_lang=lang,
                                       repeat=rep, scores=scores))
    with (tmp / "key.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["script_id", "customer", "condition", "family"])
        w.writeheader()
        w.writerows(key_rows)
    return [tmp / "fake-judge.db"], tmp / "key.csv"


def test_finds_a_real_effect_and_not_a_null_one():
    with tempfile.TemporaryDirectory() as tmp:
        stores, key = _fixture(Path(tmp), {"keigo_score": 1.5})
        result = exp3.analyse(stores, key)
        rows = {(r["lang"], r["aspect"]): r for r in result["overall"]}
        keigo, overall = rows[("en", "keigo_score")], rows[("en", "overall_score")]
        assert keigo["n"] == 60 and 1.0 < keigo["diff"] < 2.0
        assert keigo["p_holm"] < 1e-6 and keigo["ci"][0] > 0
        assert abs(overall["diff"]) < 0.4 and overall["p_holm"] > 0.05
        assert set(result["by_family"]) == {"light", "heavy"}
        assert "keigo_score" in exp3.render(result)


def test_wilcoxon_and_sign_test_against_known_values():
    # all 20 positive: sign test p = 2 / 2**20
    assert abs(exp3.sign_test(20, 0) - 2 / 2 ** 20) < 1e-12
    # symmetric differences carry no signal
    assert exp3.wilcoxon([1, -1, 2, -2, 3, -3, 4, -4]) > 0.9
    assert exp3.wilcoxon([float(i) for i in range(1, 31)]) < 1e-5


def test_ols_recovers_a_known_line():
    from jv.checks import ols
    xs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    a, b = ols(xs, [0.3 - 0.2 * x for x in xs])
    assert abs(a - 0.3) < 1e-12 and abs(b + 0.2) < 1e-12


def test_holm_is_monotone_and_capped():
    adj = exp3.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adj["a"] == 0.03 and adj["c"] == 0.06 and adj["b"] == 0.06 and adj["d"] is None


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except Exception as exc:                                # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {exc!r}"[:300])
    print("exp3: " + ("all checks pass" if not failures else f"{failures} FAILED"))
    raise SystemExit(1 if failures else 0)

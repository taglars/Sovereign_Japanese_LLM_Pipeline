"""Experiment 3, analysed: does instruction language change script quality?"""

from __future__ import annotations

import csv
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .rubric import SCORED_KEYS
from .store import Store

ROOT = Path(__file__).resolve().parents[1]
EXP3 = ROOT / "out" / "exp3"


def load_key(path: Path = EXP3 / "key.csv") -> Dict[str, Dict[str, str]]:
    with Path(path).open(encoding="utf-8") as handle:
        return {r["script_id"]: r for r in csv.DictReader(handle)}


def per_customer(stores: Sequence[Path], key: Dict[str, Dict[str, str]]) -> Dict:
    """(judge, lang) -> aspect -> customer -> condition -> mean over repeats."""
    acc: Dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    families: Dict[str, str] = {}
    for path in stores:
        if not Path(path).exists():
            continue
        for r in Store(path).read(only_ok=True):
            k = key.get(r.script_id)
            if not k:
                continue
            families[k["customer"]] = k["family"]
            for aspect, value in r.scores.items():
                acc[(r.judge, r.rubric_lang)][aspect][k["customer"]][k["condition"]].append(value)
    out: Dict = {}
    for cell, aspects in acc.items():
        out[cell] = {a: {c: {cond: st.mean(v) for cond, v in conds.items()}
                         for c, conds in custs.items()}
                     for a, custs in aspects.items()}
    return {"cells": out, "families": families}


# --------------------------------------------------------------------------
# statistics, stdlib only


def wilcoxon(diffs: Sequence[float]) -> Optional[float]:
    """Two-sided signed-rank p, normal approximation with tie correction. Adequate at n =
    100; zeros dropped (Wilcoxon's convention).
    """
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 6:
        return None
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i, ties = 0, 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    if var <= 0:
        return None
    z = (w_plus - mean) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2))


def sign_test(up: int, down: int) -> Optional[float]:
    n, k = up + down, min(up, down)
    if not n:
        return None
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap_ci(diffs: Sequence[float], *, reps: int = 10000,
                 seed: int = 20260924) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def holm(pvalues: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    items = sorted(((p, k) for k, p in pvalues.items() if p is not None))
    m, out, running = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return {k: out.get(k) for k in pvalues}


# --------------------------------------------------------------------------
# the analysis


def paired(cells: Dict, families: Dict[str, str],
           family: Optional[str] = None) -> List[Dict]:
    rows = []
    for (judge, lang), aspects in sorted(cells.items()):
        for aspect in SCORED_KEYS:
            custs = aspects.get(aspect, {})
            pairs = [(v["en"], v["ja"]) for c, v in sorted(custs.items())
                     if "en" in v and "ja" in v and (family is None or families.get(c) == family)]
            if len(pairs) < 6:
                continue
            diffs = [ja - en for en, ja in pairs]
            sd = st.stdev(diffs) if len(diffs) > 1 else 0.0
            lo, hi = bootstrap_ci(diffs)
            up, down = sum(x > 0 for x in diffs), sum(x < 0 for x in diffs)
            rows.append({"judge": judge, "lang": lang, "aspect": aspect, "n": len(pairs),
                         "en": st.mean(e for e, _ in pairs), "ja": st.mean(j for _, j in pairs),
                         "diff": st.mean(diffs), "ci": (lo, hi),
                         "dz": st.mean(diffs) / sd if sd else 0.0,
                         "ja_higher": up, "ja_lower": down,
                         "p_wilcoxon": wilcoxon(diffs), "p_sign": sign_test(up, down)})
    adjusted = holm({f"{r['judge']}|{r['lang']}|{r['aspect']}": r["p_wilcoxon"] for r in rows})
    for r in rows:
        r["p_holm"] = adjusted[f"{r['judge']}|{r['lang']}|{r['aspect']}"]
    return rows


def analyse(stores: Optional[Sequence[Path]] = None,
            key_path: Path = EXP3 / "key.csv") -> Dict:
    stores = stores or sorted(p for p in EXP3.glob("*.db") if p.name != "generation.db")
    data = per_customer(stores, load_key(key_path))
    fams = sorted(set(data["families"].values()))
    return {"overall": paired(data["cells"], data["families"]),
            "by_family": {f: paired(data["cells"], data["families"], f) for f in fams},
            "stores": [str(s) for s in stores]}


def render(result: Dict) -> str:
    def fmt_p(p):
        return "  -  " if p is None else (f"{p:.3f}" if p >= 0.001 else f"{p:.0e}")
    lines = ["EXPERIMENT 3  instruction language at generation",
             "JA minus EN, paired by customer (n = customers with both scripts judged).",
             "Positive: the script generated from Japanese instructions scored higher.", ""]
    current = None
    for r in result["overall"]:
        cell = (r["judge"], r["lang"])
        if cell != current:
            current = cell
            lines += ["", f"{r['judge']} / {r['lang']} rubric",
                      f"  {'aspect':22} {'n':>4} {'EN':>6} {'JA':>6} {'JA-EN':>7} "
                      f"{'95% CI':>15} {'dz':>6} {'up/down':>8} {'p':>6} {'p(Holm)':>8}"]
        lo, hi = r["ci"]
        lines.append(f"  {r['aspect']:22} {r['n']:4d} {r['en']:6.2f} {r['ja']:6.2f} "
                     f"{r['diff']:+7.2f} {f'[{lo:+.2f}, {hi:+.2f}]':>15} {r['dz']:+6.2f} "
                     f"{r['ja_higher']:>3}/{r['ja_lower']:<4} {fmt_p(r['p_wilcoxon']):>6} "
                     f"{fmt_p(r['p_holm']):>8}")
    lines += ["", "By truck family, overall_score only (JA-EN, p Wilcoxon):"]
    for fam, rows in result["by_family"].items():
        for r in rows:
            if r["aspect"] == "overall_score":
                lines.append(f"  {fam:8} {r['judge']:20} {r['lang']}  {r['diff']:+.2f}  "
                             f"n={r['n']}  p={fmt_p(r['p_wilcoxon'])}")
    return "\n".join(lines)

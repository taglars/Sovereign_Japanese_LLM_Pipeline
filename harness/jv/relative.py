"""Relative agreement: does a rater prefer the same scripts as another?"""

from __future__ import annotations

import random
import statistics as st
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .analyze import cells, krippendorff_alpha, load_reference
from .human import load_all
from .store import Store

ROOT = Path(__file__).resolve().parents[1]
TIE = 1e-9

JUDGE_ORDER = [("gpt-5.6-sol", "en"), ("deepseek-v4-flash", "en"),
               ("deepseek-v4-flash", "ja"), ("glm-4.7-flash", "en"),
               ("glm-4.7-flash", "ja")]
JUDGE_NAME = {"gpt-5.6-sol": "GPT-5.6", "deepseek-v4-flash": "DeepSeek",
              "glm-4.7-flash": "GLM-4.7"}


def label(key: Tuple[str, str]) -> str:
    judge, lang = key
    if judge.startswith("human:"):
        return f"{judge.split(':', 1)[1]} (human)"
    return f"{JUDGE_NAME.get(judge, judge)} / {lang.upper()}"


def collect(db: Path = ROOT / "out" / "judgements.db",
            aspect: Optional[str] = None) -> Dict:
    """Per-rater, per-script scores, humans first, then the judges."""
    records = list(Store(db).read(only_ok=True))
    records += load_reference(ROOT / "data" / "reference_scores.csv")
    arms = {r.script_id: r.arm for r in records if r.arm}
    by_key = cells(records + load_all(arms))
    humans = sorted(k for k in by_key if k[0].startswith("human:"))
    keys = humans + [k for k in JUDGE_ORDER if k in by_key]

    def score(cell, sid):
        if aspect is None:
            return cell.overall.get(sid)
        return cell.per_script.get(sid, {}).get(aspect)

    scores = {k: {sid: score(by_key[k], sid) for sid in by_key[k].per_script
                  if score(by_key[k], sid) is not None} for k in keys}
    common = sorted(set.intersection(*(set(s) for s in scores.values())))
    return {"keys": keys, "humans": humans, "scores": scores, "scripts": common,
            "arms": arms, "aspect": aspect or "component_mean"}


# --------------------------------------------------------------------------
# measures


def _sign(x: float) -> int:
    return 0 if abs(x) < TIE else (1 if x > 0 else -1)


def pairwise(a: Dict[str, float], b: Dict[str, float],
             scripts: Sequence[str]) -> Dict:
    agree = disagree = tie_a = tie_b = 0
    for s, t in combinations(scripts, 2):
        pa, pb = _sign(a[s] - a[t]), _sign(b[s] - b[t])
        tie_a += pa == 0
        tie_b += pb == 0
        if pa and pb:
            agree += pa == pb
            disagree += pa != pb
    decided = agree + disagree
    return {"agree": agree, "disagree": disagree, "decided": decided,
            "pairs": len(scripts) * (len(scripts) - 1) // 2,
            "rate": agree / decided if decided else None,
            "tie_a": tie_a, "tie_b": tie_b}


def bootstrap_rate(a: Dict[str, float], b: Dict[str, float],
                   scripts: Sequence[str], *, reps: int = 2000,
                   seed: int = 20260924) -> Tuple[Optional[float], Optional[float]]:
    """95% interval for the agreement rate, scripts resampled with replacement. Pairs of a
    script with its own copy carry no preference and are skipped.
    """
    rng = random.Random(seed)
    n, rates = len(scripts), []
    for _ in range(reps):
        draw = [rng.choice(scripts) for _ in range(n)]
        agree = decided = 0
        for i, j in combinations(range(n), 2):
            s, t = draw[i], draw[j]
            if s == t:
                continue
            pa, pb = _sign(a[s] - a[t]), _sign(b[s] - b[t])
            if pa and pb:
                decided += 1
                agree += pa == pb
        if decided:
            rates.append(agree / decided)
    if not rates:
        return None, None
    rates.sort()
    return rates[int(0.025 * len(rates))], rates[int(0.975 * len(rates)) - 1]


def _ranks(values: Sequence[float]) -> Tuple[List[float], float]:
    """Average ranks and the tie correction sum(t^3 - t)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, ties, i = [0.0] * len(values), 0.0, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(values[order[j + 1]] - values[order[i]]) < TIE:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    return ranks, ties


def kendall_w(raters: Sequence[Dict[str, float]], scripts: Sequence[str]) -> Optional[float]:
    m, n = len(raters), len(scripts)
    if m < 2 or n < 2:
        return None
    totals, tie_sum = [0.0] * n, 0.0
    for r in raters:
        ranks, ties = _ranks([r[s] for s in scripts])
        tie_sum += ties
        totals = [x + y for x, y in zip(totals, ranks)]
    mean = sum(totals) / n
    s = sum((x - mean) ** 2 for x in totals)
    denom = m * m * (n ** 3 - n) - m * tie_sum
    return 12 * s / denom if denom else None


def standardise(r: Dict[str, float], scripts: Sequence[str]) -> Dict[str, float]:
    values = [r[s] for s in scripts]
    mu, sd = st.mean(values), st.pstdev(values)
    return {s: (r[s] - mu) / sd if sd else 0.0 for s in scripts}


def alpha(raters: Sequence[Dict[str, float]], scripts: Sequence[str]) -> float:
    return krippendorff_alpha([[r[s] for r in raters] for s in scripts])


# --------------------------------------------------------------------------
# the analysis


def analyse(aspect: Optional[str] = None) -> Dict:
    data = collect(aspect=aspect)
    keys, scripts, scores = data["keys"], data["scripts"], data["scores"]
    matrix = {}
    for a, b in combinations(keys, 2):
        res = pairwise(scores[a], scores[b], scripts)
        res["ci"] = bootstrap_rate(scores[a], scores[b], scripts)
        matrix[(a, b)] = matrix[(b, a)] = res
    z = {k: standardise(scores[k], scripts) for k in keys}
    groups = {"all raters": keys, "humans": data["humans"],
              "AI judges": [k for k in keys if k not in data["humans"]]}
    summary = {}
    for name, members in groups.items():
        if len(members) < 2:
            continue
        summary[name] = {
            "kendall_w": kendall_w([scores[k] for k in members], scripts),
            "alpha_raw": alpha([scores[k] for k in members], scripts),
            "alpha_standardised": alpha([z[k] for k in members], scripts)}
    n_pairs = len(scripts) * (len(scripts) - 1) // 2
    ties = {k: sum(_sign(scores[k][s] - scores[k][t]) == 0
                   for s, t in combinations(scripts, 2)) / n_pairs for k in keys}
    return {**data, "matrix": matrix, "z": z, "groups": summary, "ties": ties}


def render(result: Dict) -> str:
    keys, m = result["keys"], result["matrix"]
    n = len(result["scripts"])
    lines = [f"RELATIVE AGREEMENT  which script is better, scale differences removed",
             f"score: {result['aspect']}, {n} scripts, {n * (n - 1) // 2} script pairs",
             "pairwise preference agreement: share of pairs both raters order the same way",
             "(50% = chance), with a 95% bootstrap interval over scripts", ""]
    width = max(len(label(k)) for k in keys)
    for a in keys:
        lines.append(f"  {label(a):{width}}  ties {result['ties'][a]:4.0%}")
        for b in keys:
            if a == b:
                continue
            r = m[(a, b)]
            if r["rate"] is None:
                continue
            lo, hi = r["ci"]
            lines.append(f"      vs {label(b):{width}}  {r['rate']:5.0%}  "
                         f"[{lo:.0%}, {hi:.0%}]  ({r['agree']}/{r['decided']} decided pairs)")
    lines += ["", "group concordance"]
    for name, g in result["groups"].items():
        lines.append(f"  {name:11} Kendall W {g['kendall_w']:.2f}   alpha raw "
                     f"{g['alpha_raw']:+.2f}   alpha standardised {g['alpha_standardised']:+.2f}")
    return "\n".join(lines)

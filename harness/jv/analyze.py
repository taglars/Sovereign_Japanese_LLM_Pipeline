"""Agreement statistics: self-agreement, agreement across judges, and rubric language."""

from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .rubric import SCORED_KEYS, component_mean
from .store import Record, Store

ASPECTS: Tuple[str, ...] = SCORED_KEYS
NAN = float("nan")


# --------------------------------------------------------------------------
# statistics


def rank(values: Sequence[float]) -> List[float]:
    """Fractional ranks, ties averaged."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return NAN
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den = (sum((x - mean_a) ** 2 for x in a)
           * sum((y - mean_b) ** 2 for y in b)) ** 0.5
    return num / den if den else NAN


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return NAN
    return pearson(rank(a), rank(b))


def kendall_tau_b(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank agreement, tie-corrected. Used on the 4-arm ordering."""
    n = len(a)
    if n != len(b) or n < 2:
        return NAN
    concordant = discordant = tied_a = tied_b = 0
    for i, j in combinations(range(n), 2):
        da, db = a[i] - a[j], b[i] - b[j]
        if da == 0 and db == 0:
            tied_a += 1
            tied_b += 1
            continue
        if da == 0:
            tied_a += 1
            continue
        if db == 0:
            tied_b += 1
            continue
        if (da > 0) == (db > 0):
            concordant += 1
        else:
            discordant += 1
    n0 = n * (n - 1) / 2
    den = ((n0 - tied_a) * (n0 - tied_b)) ** 0.5
    return (concordant - discordant) / den if den else NAN


def krippendorff_alpha(units: Iterable[Sequence[float]]) -> float:
    """Interval-metric alpha over units that may have different coder counts."""
    kept = [list(u) for u in units if len(u) >= 2]
    if not kept:
        return NAN
    values = [v for u in kept for v in u]
    n = len(values)
    if n < 2:
        return NAN

    observed = 0.0
    for unit in kept:
        m = len(unit)
        # Ordered pairs by position, i != j: equal values from two coders still agree.
        pairs = sum((unit[i] - unit[j]) ** 2
                    for i in range(m) for j in range(m) if i != j)
        observed += pairs / (m - 1)
    observed /= n

    expected = sum((values[i] - values[j]) ** 2
                   for i in range(n) for j in range(n) if i != j)
    expected /= n * (n - 1)
    if expected == 0:
        # Every value identical everywhere. Agreement is total but alpha is 0/0.
        return 1.0
    return 1 - observed / expected


def within_one(a: Sequence[float], b: Sequence[float]) -> float:
    """Share of paired values no more than one scale point apart."""
    if not a:
        return NAN
    return sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / len(a)


# --------------------------------------------------------------------------
# shaping the records


@dataclass
class Cell:
    """One judge under one rubric language, collapsed to per-script means."""

    judge: str
    rubric_lang: str
    #: script_id -> aspect -> mean across repeats
    per_script: Dict[str, Dict[str, float]]
    #: script_id -> mean of the five components
    overall: Dict[str, float]
    arm_of: Dict[str, str]
    repeats: Dict[str, int]

    @property
    def label(self) -> str:
        return f"{self.judge}/{self.rubric_lang}"


def load_reference(path: str | Path) -> List[Record]:
    """The reference (GPT-5.6) judge's raw per-repeat scores, exported to CSV."""
    path = Path(path)
    if not path.exists():
        return []
    out: List[Record] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            scores = {}
            for aspect in ASPECTS:
                raw = (row.get(aspect) or "").strip()
                if raw:
                    try:
                        scores[aspect] = int(float(raw))
                    except ValueError:
                        pass
            if not scores:
                continue
            out.append(Record(
                script_id=(row.get("script_id") or "").strip(),
                judge=(row.get("judge") or "gpt-5.6-sol").strip(),
                rubric_lang=(row.get("rubric_lang") or "en").strip(),
                repeat=int(float(row.get("repeat") or 0)),
                arm=(row.get("arm") or "").strip(),
                vin=(row.get("vin") or "").strip(),
                scores=scores,
                # Same statistic as for the other judges.
                component_mean=component_mean(scores)))
    return out


def cells(records: Sequence[Record]) -> Dict[Tuple[str, str], Cell]:
    """Collapse repeats to a per-script mean, one Cell per (judge, language)."""
    grouped: Dict[Tuple[str, str], Dict[str, List[Record]]] = defaultdict(
        lambda: defaultdict(list))
    for record in records:
        if record.scores:
            grouped[(record.judge, record.rubric_lang)][record.script_id].append(record)

    out: Dict[Tuple[str, str], Cell] = {}
    for (judge, lang), by_script in grouped.items():
        per_script: Dict[str, Dict[str, float]] = {}
        overall: Dict[str, float] = {}
        arm_of: Dict[str, str] = {}
        repeats: Dict[str, int] = {}
        for script_id, rows in by_script.items():
            aspect_means = {}
            for aspect in ASPECTS:
                values = [r.scores[aspect] for r in rows if aspect in r.scores]
                if values:
                    aspect_means[aspect] = sum(values) / len(values)
            per_script[script_id] = aspect_means
            components = [aspect_means[a] for a in ASPECTS
                          if a != "overall_score" and a in aspect_means]
            if components:
                overall[script_id] = sum(components) / len(components)
            arm_of[script_id] = next((r.arm for r in rows if r.arm), "")
            repeats[script_id] = len(rows)
        out[(judge, lang)] = Cell(judge, lang, per_script, overall, arm_of, repeats)
    return out


def common_scripts(*cs: Cell) -> List[str]:
    """Scripts every given cell scored, in a stable order."""
    if not cs:
        return []
    shared = set(cs[0].overall)
    for cell in cs[1:]:
        shared &= set(cell.overall)
    return sorted(shared)


def arm_means(cell: Cell, aspect: Optional[str] = None) -> Dict[str, float]:
    """Mean per arm."""
    buckets: Dict[str, List[float]] = defaultdict(list)
    for script_id, arm in cell.arm_of.items():
        if not arm:
            continue
        if aspect is None:
            if script_id in cell.overall:
                buckets[arm].append(cell.overall[script_id])
        elif aspect in cell.per_script.get(script_id, {}):
            buckets[arm].append(cell.per_script[script_id][aspect])
    return {arm: sum(v) / len(v) for arm, v in sorted(buckets.items()) if v}


# --------------------------------------------------------------------------
# self-agreement, the ceiling on everything else


def self_agreement(records: Sequence[Record]) -> Dict[Tuple[str, str], Dict]:
    """How well each judge reproduces itself across repeats."""
    grouped: Dict[Tuple[str, str], Dict[str, List[Record]]] = defaultdict(
        lambda: defaultdict(list))
    for record in records:
        if record.scores:
            grouped[(record.judge, record.rubric_lang)][record.script_id].append(record)

    out: Dict[Tuple[str, str], Dict] = {}
    for key, by_script in grouped.items():
        pairs = [rows for rows in by_script.values() if len(rows) >= 2]
        if not pairs:
            continue
        identical = 0
        gaps: List[float] = []
        per_aspect: Dict[str, List[float]] = defaultdict(list)
        for rows in pairs:
            first, second = rows[0], rows[1]
            same = all(first.scores.get(a) == second.scores.get(a) for a in ASPECTS
                       if a in first.scores and a in second.scores)
            identical += int(same)
            for aspect in ASPECTS:
                if aspect in first.scores and aspect in second.scores:
                    per_aspect[aspect].append(
                        abs(first.scores[aspect] - second.scores[aspect]))
            if first.component_mean is not None and second.component_mean is not None:
                gaps.append(abs(first.component_mean - second.component_mean))
        out[key] = {
            "scripts": len(pairs),
            "identical": identical,
            "identical_share": identical / len(pairs),
            "mean_abs_gap": st.mean(gaps) if gaps else NAN,
            "alpha": krippendorff_alpha(
                [[r.component_mean for r in rows if r.component_mean is not None]
                 for rows in pairs]),
            "per_aspect_gap": {a: st.mean(v) for a, v in per_aspect.items()},
        }
    return out


# --------------------------------------------------------------------------
# agreement across judges


def cross_judge(cells_by_key: Dict[Tuple[str, str], Cell],
                lang: str = "en") -> Dict:
    """Every pair of judges under one rubric language."""
    chosen = [c for (judge, l), c in sorted(cells_by_key.items()) if l == lang]
    pairs = []
    for left, right in combinations(chosen, 2):
        shared = common_scripts(left, right)
        if len(shared) < 3:
            continue
        row = {
            "left": left.judge, "right": right.judge, "n": len(shared),
            "overall_spearman": spearman([left.overall[s] for s in shared],
                                         [right.overall[s] for s in shared]),
            "overall_pearson": pearson([left.overall[s] for s in shared],
                                       [right.overall[s] for s in shared]),
            "within_one": within_one([left.overall[s] for s in shared],
                                     [right.overall[s] for s in shared]),
            "bias": st.mean([left.overall[s] - right.overall[s] for s in shared]),
            "per_aspect": {},
            "arm_tau": NAN,
        }
        for aspect in ASPECTS:
            a = [left.per_script[s].get(aspect) for s in shared]
            b = [right.per_script[s].get(aspect) for s in shared]
            usable = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            if len(usable) >= 3:
                row["per_aspect"][aspect] = spearman([x for x, _ in usable],
                                                     [y for _, y in usable])
        left_arms, right_arms = arm_means(left), arm_means(right)
        shared_arms = sorted(set(left_arms) & set(right_arms))
        if len(shared_arms) >= 3:
            row["arm_tau"] = kendall_tau_b([left_arms[a] for a in shared_arms],
                                           [right_arms[a] for a in shared_arms])
            row["arm_order_left"] = sorted(shared_arms,
                                           key=lambda a: -left_arms[a])
            row["arm_order_right"] = sorted(shared_arms,
                                            key=lambda a: -right_arms[a])
        pairs.append(row)

    # Alpha over all judges at once, per script, on the component mean.
    units = []
    for script_id in common_scripts(*chosen) if chosen else []:
        units.append([c.overall[script_id] for c in chosen])
    return {"lang": lang,
            "judges": [c.judge for c in chosen],
            "pairs": pairs,
            "alpha": krippendorff_alpha(units) if units else NAN,
            "arm_means": {c.judge: arm_means(c) for c in chosen}}


# --------------------------------------------------------------------------
# rubric language


def rubric_language(cells_by_key: Dict[Tuple[str, str], Cell]) -> List[Dict]:
    """Within each judge, English rubric against Japanese rubric."""
    judges = sorted({judge for judge, _ in cells_by_key})
    out = []
    for judge in judges:
        en = cells_by_key.get((judge, "en"))
        ja = cells_by_key.get((judge, "ja"))
        if not en or not ja:
            continue
        shared = common_scripts(en, ja)
        if len(shared) < 3:
            continue
        row = {
            "judge": judge, "n": len(shared),
            "mean_en": st.mean([en.overall[s] for s in shared]),
            "mean_ja": st.mean([ja.overall[s] for s in shared]),
            "spearman": spearman([en.overall[s] for s in shared],
                                 [ja.overall[s] for s in shared]),
            "within_one": within_one([en.overall[s] for s in shared],
                                     [ja.overall[s] for s in shared]),
            "per_aspect": {},
            "arm_tau": NAN,
        }
        row["shift"] = row["mean_ja"] - row["mean_en"]
        for aspect in ASPECTS:
            a = [en.per_script[s].get(aspect) for s in shared]
            b = [ja.per_script[s].get(aspect) for s in shared]
            usable = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            if len(usable) >= 3:
                row["per_aspect"][aspect] = {
                    "en": st.mean([x for x, _ in usable]),
                    "ja": st.mean([y for _, y in usable]),
                    "shift": st.mean([y - x for x, y in usable]),
                    "spearman": spearman([x for x, _ in usable],
                                         [y for _, y in usable]),
                }
        en_arms, ja_arms = arm_means(en), arm_means(ja)
        shared_arms = sorted(set(en_arms) & set(ja_arms))
        if len(shared_arms) >= 3:
            row["arm_tau"] = kendall_tau_b([en_arms[a] for a in shared_arms],
                                           [ja_arms[a] for a in shared_arms])
            row["arm_order_en"] = sorted(shared_arms, key=lambda a: -en_arms[a])
            row["arm_order_ja"] = sorted(shared_arms, key=lambda a: -ja_arms[a])
            row["arm_means_en"] = en_arms
            row["arm_means_ja"] = ja_arms
        out.append(row)
    return out


def human_agreement(human_records: Sequence[Record],
                    by_key: Dict[Tuple[str, str], Cell]) -> Dict:
    """Every rater against every judge cell, plus raters against each other."""
    from .human import range_usage

    raters = cells(human_records)
    rows = []
    for (rater, _), hcell in sorted(raters.items()):
        for (judge, lang), jcell in sorted(by_key.items()):
            shared = common_scripts(hcell, jcell)
            if len(shared) < 3:
                continue
            h_arms, j_arms = arm_means(hcell), arm_means(jcell)
            common = sorted(set(h_arms) & set(j_arms))
            rows.append({
                "rater": rater, "judge": judge, "lang": lang, "n": len(shared),
                "rho": spearman([hcell.overall[s] for s in shared],
                                [jcell.overall[s] for s in shared]),
                "arm_tau": (kendall_tau_b([h_arms[a] for a in common],
                                          [j_arms[a] for a in common])
                            if len(common) >= 3 else NAN),
                "bias": st.mean(hcell.overall[s] - jcell.overall[s] for s in shared),
            })
    inter = []
    names = sorted(raters)
    for a, b in combinations(names, 2):
        shared = common_scripts(raters[a], raters[b])
        if len(shared) >= 3:
            inter.append({"left": a[0], "right": b[0], "n": len(shared),
                          "rho": spearman([raters[a].overall[s] for s in shared],
                                          [raters[b].overall[s] for s in shared])})
    alpha = NAN
    if len(names) >= 2:
        units = [[raters[k].overall[s] for k in names if s in raters[k].overall]
                 for s in sorted({s for k in names for s in raters[k].overall})]
        alpha = krippendorff_alpha(units)
    return {"raters": [k[0] for k in names], "vs_judges": rows,
            "inter_rater": inter, "inter_rater_alpha": alpha,
            "range": range_usage(list(human_records)),
            "arm_means": {k[0]: arm_means(c) for k, c in raters.items()}}


def analyse(db: str | Path, reference_csv: Optional[str | Path] = None,
            human_dir: Optional[str | Path] = None) -> Dict:
    """Everything, from the store plus the reference scores and any human raters."""
    records = list(Store(db).read(only_ok=True))
    if reference_csv:
        records += load_reference(reference_csv)
    by_key = cells(records)

    humans: List[Record] = []
    if human_dir is not None:
        from .human import load_all
        arms = {r.script_id: r.arm for r in records if r.arm}
        humans = load_all(arms, Path(human_dir))

    return {
        "records": len(records),
        "self": self_agreement(records),
        "cells": by_key,
        "e1_en": cross_judge(by_key, "en"),
        "e1_ja": cross_judge(by_key, "ja"),
        "e2": rubric_language(by_key),
        "human": human_agreement(humans, by_key) if humans else None,
    }

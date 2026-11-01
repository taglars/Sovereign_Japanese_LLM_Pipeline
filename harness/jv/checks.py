"""Robustness checks on Experiment 3, from data already collected. No new runs."""

from __future__ import annotations

import csv
import random
import statistics as st
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .analyze import cells, krippendorff_alpha, load_reference, self_agreement, spearman
from .exp3 import load_key
from .relative import label, pairwise, standardise
from .rubric import SCORED_KEYS
from .store import Store

ROOT = Path(__file__).resolve().parents[1]
EXP3 = ROOT / "out" / "exp3"
JUDGES = ["deepseek-v4-flash", "glm-4.7-flash"]
CELLS = [(j, l) for j in JUDGES for l in ("en", "ja")]
MORPH = {"keigo_defects_per_100_sentences": "keigo errors / 100 sent.",
         "keigo_register_consistency": "register consistency",
         "keigo_referent_honorifics_per_100_sentences": "honorifics to customer / 100 sent.",
         "keigo_keigo_density": "keigo density"}


def _score(cell, sid: str, aspect: Optional[str]) -> Optional[float]:
    return cell.overall.get(sid) if aspect is None else \
        cell.per_script.get(sid, {}).get(aspect)


def load() -> Dict:
    records = []
    for j in JUDGES:
        path = EXP3 / f"{j}.db"
        if path.exists():
            records += list(Store(path).read(only_ok=True))
    key = load_key()
    metrics = {}
    with (EXP3 / "metrics.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metrics[(row["customer"], row["condition"])] = row
    by_key = cells(records)
    scripts = sorted(set.intersection(*(set(by_key[k].per_script) for k in CELLS
                                        if k in by_key)))
    per_script_metrics = {sid: metrics.get((key[sid]["customer"], key[sid]["condition"]), {})
                          for sid in scripts if sid in key}
    legacy_records = list(Store(ROOT / "out" / "judgements.db").read(only_ok=True))
    legacy_records += load_reference(ROOT / "data" / "reference_scores.csv")
    return {"records": records, "key": key, "cells": by_key, "scripts": scripts,
            "metrics": per_script_metrics, "legacy": cells(legacy_records),
            "legacy_records": legacy_records}


# --------------------------------------------------------------------------
# A  agreement at scale


def _agree(ca, cb, scripts, aspect=None) -> Dict:
    a = {s: _score(ca, s, aspect) for s in scripts}
    b = {s: _score(cb, s, aspect) for s in scripts}
    keep = [s for s in scripts if a[s] is not None and b[s] is not None]
    pw = pairwise(a, b, keep)
    return {"n": len(keep), "rho": spearman([a[s] for s in keep], [b[s] for s in keep]),
            "pairwise": pw["rate"],
            "alpha_std": krippendorff_alpha([[x, y] for x, y in zip(
                standardise(a, keep).values(), standardise(b, keep).values())])}


def agreement(data: Dict) -> Dict:
    out = {"pairs": [], "per_aspect": {}}
    legacy_scripts = sorted(set.intersection(*(set(data["legacy"][k].per_script)
                                               for k in CELLS if k in data["legacy"])))
    for ka, kb in combinations(CELLS, 2):
        row = {"a": ka, "b": kb, "exp3": _agree(data["cells"][ka], data["cells"][kb],
                                                 data["scripts"])}
        if ka in data["legacy"] and kb in data["legacy"]:
            row["legacy12"] = _agree(data["legacy"][ka], data["legacy"][kb], legacy_scripts)
        out["pairs"].append(row)
        out["per_aspect"][(ka, kb)] = {asp: _agree(data["cells"][ka], data["cells"][kb],
                                                   data["scripts"], asp)["rho"]
                                       for asp in SCORED_KEYS}
    return out


# --------------------------------------------------------------------------
# B  length as a control


def ols(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Intercept and slope of y = a + b x."""
    mx, my = st.mean(x), st.mean(y)
    sxx = sum((v - mx) ** 2 for v in x)
    b = sum((u - mx) * (v - my) for u, v in zip(x, y)) / sxx if sxx else 0.0
    return my - b * mx, b


def length_control(data: Dict, *, reps: int = 2000, seed: int = 20260924) -> List[Dict]:
    """Regress the JA - EN score difference on the JA - EN length difference (1,000 characters).

    Per customer; the intercept is the effect at equal length. Bootstrap over customers.
    """
    key, metrics = data["key"], data["metrics"]
    by_customer: Dict[str, Dict[str, str]] = {}
    for sid in data["scripts"]:
        by_customer.setdefault(key[sid]["customer"], {})[key[sid]["condition"]] = sid
    pairs = {c: v for c, v in by_customer.items() if "en" in v and "ja" in v}
    rows = []
    for cell_key in CELLS:
        cell = data["cells"][cell_key]
        for aspect in [None] + list(SCORED_KEYS):
            xs, ys = [], []
            for c, v in sorted(pairs.items()):
                se, sj = _score(cell, v["en"], aspect), _score(cell, v["ja"], aspect)
                le, lj = metrics[v["en"]].get("chars"), metrics[v["ja"]].get("chars")
                if None in (se, sj) or not le or not lj:
                    continue
                xs.append((float(lj) - float(le)) / 1000)
                ys.append(sj - se)
            a, b = ols(xs, ys)
            rng = random.Random(seed)
            boot_a, boot_b = [], []
            idx = list(range(len(xs)))
            for _ in range(reps):
                draw = [rng.choice(idx) for _ in idx]
                ba, bb = ols([xs[i] for i in draw], [ys[i] for i in draw])
                boot_a.append(ba)
                boot_b.append(bb)
            boot_a.sort()
            boot_b.sort()
            lo, hi = int(0.025 * reps), int(0.975 * reps) - 1
            # Within condition: does a longer script simply score higher?
            within = {}
            for cond in ("en", "ja"):
                sids = [v[cond] for v in pairs.values()]
                sc = [_score(cell, s, aspect) for s in sids]
                ln = [float(metrics[s]["chars"]) for s in sids]
                keep = [(p, q) for p, q in zip(sc, ln) if p is not None]
                within[cond] = spearman([p for p, _ in keep], [q for _, q in keep])
            rows.append({"cell": cell_key, "aspect": aspect or "component_mean",
                         "n": len(xs), "raw": st.mean(ys), "adjusted": a,
                         "adjusted_ci": (boot_a[lo], boot_a[hi]), "per_1000_chars": b,
                         "slope_ci": (boot_b[lo], boot_b[hi]),
                         "mean_len_diff": st.mean(xs), "rho_len_within": within})
    return rows


# --------------------------------------------------------------------------
# C  keigo validity


def legacy_morph() -> Dict[str, Dict[str, float]]:
    """Morphological keigo measures for the 12 legacy scripts (fugashi)."""
    from .corpus import load as load_corpus
    from .exp3 import ROOT as _R  # noqa: F401  (same repo root)
    import sys
    vendor = str(ROOT / "gen" / "vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from eval.honorific_scoring.keigo import HonorificScorer
    scorer = HonorificScorer()
    out = {}
    for script in load_corpus(ROOT / "data" / "corpus.json"):
        text = "\n".join(body for _, body in script.sections)
        d = scorer.score(text, is_html=False).as_dict()
        out[script.script_id] = {f"keigo_{k}": d.get(k) for k in (
            "defects_per_100_sentences", "register_consistency",
            "referent_honorifics_per_100_sentences", "keigo_density")}
    return out


def keigo_validity(data: Dict) -> Dict:
    def corr(cell, scripts, measures):
        res = {}
        for m in MORPH:
            pairs = [(_score(cell, s, "keigo_score"), measures.get(s, {}).get(m))
                     for s in scripts]
            pairs = [(p, float(q)) for p, q in pairs if p is not None and q not in (None, "")]
            res[m] = spearman([p for p, _ in pairs], [q for _, q in pairs]) \
                if len(pairs) > 5 else None
        return res

    exp3 = {k: corr(data["cells"][k], data["scripts"], data["metrics"]) for k in CELLS}
    legacy = {}
    try:
        morph = legacy_morph()
        from .human import load_all
        arms = {r.script_id: r.arm for r in data["legacy_records"] if r.arm}
        legacy_cells = cells(data["legacy_records"] + load_all(arms))
        for k, cell in legacy_cells.items():
            legacy[k] = corr(cell, sorted(morph), morph)
    except ImportError as exc:                                  # fugashi missing
        legacy = {"error": str(exc)}
    return {"exp3": exp3, "legacy12": legacy}


# --------------------------------------------------------------------------
# D  self-agreement


def self_consistency(data: Dict) -> Dict:
    exp3 = self_agreement(data["records"])
    legacy = self_agreement([r for r in data["legacy_records"]
                             if (r.judge, r.rubric_lang) in CELLS])
    return {"exp3": exp3, "legacy12": legacy}


# --------------------------------------------------------------------------


def analyse() -> Dict:
    data = load()
    return {"n_scripts": len(data["scripts"]), "agreement": agreement(data),
            "length": length_control(data), "keigo": keigo_validity(data),
            "self": self_consistency(data)}


def render(r: Dict) -> str:
    def f(v, fmt="{:+.2f}"):
        return "  -  " if v is None else fmt.format(v)
    lines = ["EXPERIMENT 3 ROBUSTNESS CHECKS", ""]
    lines += [f"A  JUDGE AGREEMENT on {r['n_scripts']} Exp 3 scripts (with customer data) "
              "vs the 12 legacy scripts",
              "   score = component mean; pairwise = share of script pairs ordered alike (50% chance)",
              f"   {'pair':38} {'rho 200':>8} {'pairs 200':>10} {'a_std 200':>10} "
              f"{'rho 12':>8} {'pairs 12':>9}"]
    for p in r["agreement"]["pairs"]:
        e, l = p["exp3"], p.get("legacy12", {})
        lines.append(f"   {label(p['a']) + ' vs ' + label(p['b']):38} {f(e['rho']):>8} "
                     f"{f(e['pairwise'], '{:.0%}'):>10} {f(e['alpha_std']):>10} "
                     f"{f(l.get('rho')):>8} {f(l.get('pairwise'), '{:.0%}'):>9}")
    lines += ["", "   per aspect, Spearman on 200 scripts:",
              "   " + " " * 38 + "".join(f"{a.split('_')[0][:10]:>11}" for a in SCORED_KEYS)]
    for (ka, kb), asp in r["agreement"]["per_aspect"].items():
        lines.append(f"   {label(ka) + ' vs ' + label(kb):38}" +
                     "".join(f"{f(asp[a]):>11}" for a in SCORED_KEYS))

    lines += ["", "B  LENGTH AS A CONTROL  (JA minus EN per customer, regressed on the length "
              "difference in 1,000 characters)",
              "   adjusted = effect at equal length; slope = points per 1,000 extra characters",
              f"   {'judge / rubric':20} {'aspect':22} {'raw':>7} {'adjusted':>9} "
              f"{'95% CI':>17} {'slope':>7} {'95% CI':>17} {'rho(len) EN/JA':>15}"]
    for row in r["length"]:
        alo, ahi = row["adjusted_ci"]
        blo, bhi = row["slope_ci"]
        w = row["rho_len_within"]
        lines.append(f"   {label(row['cell']):20} {row['aspect']:22} {row['raw']:+7.2f} "
                     f"{row['adjusted']:+9.2f} {f'[{alo:+.2f}, {ahi:+.2f}]':>17} "
                     f"{row['per_1000_chars']:+7.2f} {f'[{blo:+.2f}, {bhi:+.2f}]':>17} "
                     f"{f(w['en'])}/{f(w['ja'])}")

    lines += ["", "C  KEIGO VALIDITY  (Spearman: judge keigo score vs morphological measure)",
              "   expected sign: errors negative, consistency positive",
              "   " + " " * 22 + "".join(f"{v[:16]:>18}" for v in MORPH.values())]
    for scope in ("exp3", "legacy12"):
        block = r["keigo"][scope]
        if "error" in block:
            lines.append(f"   {scope}: skipped ({block['error'][:60]})")
            continue
        lines.append(f"   {('200 Exp 3 scripts' if scope == 'exp3' else '12 legacy scripts')}:")
        for k, res in block.items():
            lines.append(f"   {label(k):22}" + "".join(f"{f(res[m]):>18}" for m in MORPH))

    lines += ["", "D  SELF-AGREEMENT over repeats (Exp 3: 3 repeats, legacy: 10)",
              f"   {'judge / rubric':20} {'alpha 200':>10} {'identical':>10} {'mean gap':>9} "
              f"{'alpha 12':>9}"]
    for k in CELLS:
        e = r["self"]["exp3"].get(k, {})
        l = r["self"]["legacy12"].get(k, {})
        lines.append(f"   {label(k):20} {f(e.get('alpha'), '{:.2f}'):>10} "
                     f"{f(e.get('identical_share'), '{:.0%}'):>10} "
                     f"{f(e.get('mean_abs_gap'), '{:.2f}'):>9} {f(l.get('alpha'), '{:.2f}'):>9}")
    return "\n".join(lines)

"""Macros for the experiment results. Imported by macros.py."""

from __future__ import annotations

import csv
import json
import sqlite3
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from constants import PAYG_USD, SPOT_USD

THESIS = Path(__file__).resolve().parents[1]
DATA = THESIS / "data"
JV = THESIS.parent / "harness"
if str(JV) not in sys.path:
    sys.path.insert(0, str(JV))

from jv import codebook as cb  # noqa: E402
from jv.exp3 import bootstrap_ci, holm, sign_test, wilcoxon  # noqa: E402

ARM = {"batch": "Batch", "qwen_only": "QwenThree", "qwen36_bf16": "QwenSix",
       "qwen38_bf16": "QwenEight"}
FAMILY = {"light": "Light", "electric": "Electric", "heavy": "Heavy"}
JUDGE = {"gpt-5.6-sol": "Gpt", "deepseek-v4-flash": "DeepSeek", "glm-4.7-flash": "Glm"}
LANG = {"en": "En", "ja": "Ja"}
ASPECT = {"keigo_score": "Keigo", "fluency_score": "Fluency",
          "personalisation_score": "Personalisation", "readiness_score": "Readiness",
          "persuasion_score": "Persuasion", "overall_score": "Overall"}
# notebook 17 part E column names
NB_ASPECT = {"mean": "Mean", "fluency": "Fluency", "keigo": "Keigo", "overall": "Overall",
             "personalisation": "Personalisation", "persuasion": "Persuasion",
             "readiness": "Readiness"}


def _rows(name: str) -> list[dict]:
    with open(DATA / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x: float, nd: int = 2) -> str:
    """A signed decimal for math mode: -0.15, 0.37."""
    return f"{x:.{nd}f}"


def _pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}"


def _p(p) -> str:
    """'p = 0.013' or 'p < 0.001', for use inside $...$."""
    if p is None:
        return "p = \\text{n/a}"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _num(text: str) -> str:
    """The number in a status-deck cell such as '~10' or '46 tok/s'."""
    out = "".join(ch for ch in text if ch.isdigit() or ch in ".")
    return out


# --------------------------------------------------------------------------
# the baseline


def baseline() -> dict[str, str]:
    cost = {r["approach"]: r for r in _rows("baseline_cost.csv")}
    b, leg = cost["batch"], cost["legacy"]
    b_in, b_out = int(b["input_tokens_per_vehicle"]), int(b["output_tokens_per_vehicle"])
    usd = float(b["usd_per_vehicle"])
    vehicles = int(b["vehicles"])
    discount = 0.5  # legacy_cost.json pricing_illustrative; asserted in extract.py's note
    stages = _rows("baseline_stages.csv")
    by_stage = Counter()
    cached = inp = 0
    for r in stages:
        by_stage[r["stage"]] += float(r["usd_per_vehicle"])
        cached += int(r["cached_tokens_per_vehicle"])
        inp += int(r["input_tokens_per_vehicle"])
    total = sum(by_stage.values())
    assert abs(total - usd) < 0.0005, f"stages {total:.4f} != batch {usd:.4f}"
    prof = {(r["corpus"], r["family"]): r for r in _rows("baseline_markup.csv")}
    fams = {r["family"]: float(r["requests_per_vehicle"]) for r in _rows("baseline_families.csv")}
    la, ba = prof[("legacy", "ALL")], prof[("batch", "ALL")]
    return {
        "BatchInputTokensPerVehicle": f"{b_in:,}",
        "BatchOutputTokensPerVehicle": f"{b_out:,}",
        "BatchInputSharePct": _pct(b_in / (b_in + b_out)),
        "LegacyInputTokensPerVehicle": f"{int(leg['input_tokens_per_vehicle']):,}",
        "LegacyOutputTokensPerVehicle": f"{int(leg['output_tokens_per_vehicle']):,}",
        "BatchCostDraftPct": _pct(by_stage["A"] / total),
        "BatchCostReviewPct": _pct(by_stage["B"] / total),
        "BatchCostTranslatePct": _pct(by_stage["C"] / total),
        "BatchCostDiscountPct": _pct(by_stage["D"] / total),
        "BatchCachedPct": _pct(cached / inp, 1),
        "BatchTotalUSD": f"{usd * vehicles:,.0f}",
        "BatchUndiscountedUSD": f"{usd / discount:.2f}",
        "LegacyProfileScripts": f"{int(la['scripts']):,}",
        "LegacyProfileSections": f"{int(la['sections']):,}",
        "LegacyProfileInvalidPct": _pct(int(la["invalid_sections"]) / int(la["sections"]), 1),
        "BatchProfileScripts": f"{int(ba['scripts']):,}",
        "BatchProfileSections": f"{int(ba['sections']):,}",
        "BatchProfileInvalidPct": _pct(int(ba["invalid_sections"]) / int(ba["sections"])),
        "BatchProfileProblems": f"{int(ba['all_problems']):,}",
        "BatchProfileNonMarkdown": str(int(ba["all_problems"]) - int(ba["markdown_problems"])),
        "BatchDiscountInputTokens": f"{sum(int(r['input_tokens_per_vehicle']) for r in stages if r['stage'] == 'D'):,}",
        "BatchDiscountOutputTokens": f"{sum(int(r['output_tokens_per_vehicle']) for r in stages if r['stage'] == 'D'):,}",
        "FamilyLightCalls": _f(fams["light"], 0),
        "FamilyElectricCalls": _f(fams["electric"], 0),
        "FamilyHeavyCalls": _f(fams["heavy"], 0),
        "LegacyProfileKeigoDensity": _f(float(la["keigo_density"])),
        "BatchProfileKeigoDensity": _f(float(ba["keigo_density"])),
    }


# --------------------------------------------------------------------------
# Experiment 1


def experiment1() -> dict[str, str]:
    out: dict[str, str] = {}
    # serving, from the status review
    serving = {r["measure"]: r for r in _rows("exp1_serving.csv")}
    s = lambda key: next(v for k, v in serving.items() if k.startswith(key))  # noqa: E731
    out.update({
        "ServingTokensBefore": _num(s("Throughput")["before"]),
        "ServingTokensAfter": _num(s("Throughput")["after"]),
        "ServingSectionBefore": _num(s("Time per section")["before"]),
        "ServingSectionAfter": _num(s("Time per section")["after"]),
        "ServingGpuHoursBefore": _num(s("GPU-hours")["before"]),
        "ServingGpuHoursAfter": _num(s("GPU-hours")["after"]),
        "ServingStructuralBefore": _num(s("Output failing")["before"]),
        "ServingStructuralAfter": _num(s("Output failing")["after"]),
        "ServingColdStart": _num(s("Cold start")["after"]),
        "ServingNoPoliteReasoning": _num(s("Sentences without")["before"]),
        "ServingNoPoliteBaseline": _num(s("Sentences without")["after"]),
    })
    # coverage per run
    for r in _rows("exp1_coverage.csv"):
        name = "Split" if r["arm"] == "split" else ARM[r["arm"]]
        out[f"{name}TasksDone"] = f"{int(r['done']):,}"
        out[f"{name}TasksQuarantined"] = r["poison"]
        out[f"{name}TasksStranded"] = r["pending"]
        out[f"{name}Tasks"] = f"{int(r['tasks']):,}"
    # the split design
    split = {r["arm"]: r for r in _rows("exp1_split.csv")}
    out["SplitMean"] = _f(float(split["split"]["mean"]))
    out["SplitBatchMean"] = _f(float(split["batch"]["mean"]))
    out["SplitJudgements"] = str(sum(int(r["judgements"]) for r in split.values()))
    mk = {r["arm"]: r for r in _rows("exp1_split_markup.csv")}
    out["SplitBadMarkupPct"] = _pct(int(mk["split"]["bad_sections"]) / int(mk["split"]["sections"]))
    out["SplitBatchBadMarkupPct"] = _pct(int(mk["batch"]["bad_sections"]) / int(mk["batch"]["sections"]))
    out["SplitSections"] = mk["split"]["sections"]
    kinds = dict((k.rsplit(" ", 1)[0], int(k.rsplit(" ", 1)[1]))
                 for k in mk["split"]["kinds"].split(", "))
    out["SplitProblemKinds"] = str(sum(kinds.values()))
    out["SplitMarkdownKinds"] = str(sum(n for k, n in kinds.items() if k.startswith("markdown")))
    # judge-free, per arm and per family
    for r in _rows("exp1_judge_free.csv"):
        arm, fam = ARM[r["arm"]], r["family"]
        suffix = "" if fam == "all" else FAMILY[fam]
        if fam == "all":
            if arm not in ("Batch", "QwenEight"):  # defined in macros.py
                out[f"{arm}BadMarkupPct"] = _pct(float(r["bad_markup"]))
            out[f"{arm}Consistency"] = f"{float(r['register_consistency']):.3f}"
            out[f"{arm}PlainRate"] = _f(float(r["plain_per_100"]))
        else:
            out[f"{arm}{suffix}BadMarkupPct"] = _pct(float(r["bad_markup"]))
    # judged, arm means per aspect
    for r in _rows("exp1_judged.csv"):
        out[f"{ARM[r['arm']]}Judged{NB_ASPECT[r['aspect']]}"] = _f(float(r["score"]))
    for r in _rows("exp1_keigo_check.csv"):
        out[f"{ARM[r['arm']]}KeigoDensity"] = _f(float(r["keigo_density"]))
    # cost
    base = float({r["approach"]: r for r in _rows("baseline_cost.csv")}["batch"]["usd_per_vehicle"])
    vehicles = int({r["approach"]: r for r in _rows("baseline_cost.csv")}["batch"]["vehicles"])
    comp = {r["arm"]: r for r in _rows("exp1_compute.csv")}
    for arm, r in comp.items():
        nh = float(r["node_hours_per_vehicle"])
        out[f"{ARM[arm]}NodeMinutes"] = f"{nh * 60:.1f}"
        # \QwenEightUSDPerVehicle and \BreakEvenUSD are defined in macros.py
        if arm != "qwen38_bf16":
            out[f"{ARM[arm]}USDPerVehicle"] = f"{nh * SPOT_USD:.3f}"
            out[f"{ARM[arm]}BreakEvenUSD"] = f"{base / nh:.2f}"
        out[f"{ARM[arm]}Concurrency"] = r["concurrency"]
    bf16 = st.mean(float(comp[a]["node_hours_per_vehicle"]) for a in ("qwen36_bf16", "qwen38_bf16"))
    out["BfNodeTimeRatio"] = f"{bf16 / float(comp['qwen_only']['node_hours_per_vehicle']):.1f}"
    out["QwenEightTotalUSD"] = f"{float(comp['qwen38_bf16']['node_hours_per_vehicle']) * SPOT_USD * vehicles:,.0f}"
    light = {r["arm"]: r for r in _rows("exp1_compute_light.csv")}
    s_nh, q_nh = (float(light[a]["node_hours_per_vehicle"]) for a in ("split", "qwen_only"))
    out["SplitNodeMinutesLight"] = f"{s_nh * 60:.2f}"
    out["QwenThreeNodeMinutesLight"] = f"{q_nh * 60:.2f}"
    out["SplitNodeSavingPct"] = _pct(1 - s_nh / q_nh)
    out["SplitUSDPerVehicle"] = f"{s_nh * SPOT_USD:.3f}"
    out["LightVehiclesSolo"] = light["qwen_only"]["vehicles"]
    fam_cost = {r["family"]: float(r["usd_per_vehicle"]) for r in _rows("baseline_family_cost.csv")}
    out["BatchLightUSDPerVehicle"] = f"{fam_cost['light']:.2f}"
    out["BatchElectricUSDPerVehicle"] = f"{fam_cost['electric']:.2f}"
    out["BatchHeavyUSDPerVehicle"] = f"{fam_cost['heavy']:.2f}"
    out["SplitBreakEvenUSD"] = f"{fam_cost['light'] / s_nh:.2f}"
    out["SplitMoeTaskPct"] = _pct(float(light["split"]["moe_task_share"]))
    out["QwenThreeLightUSDPerVehicle"] = f"{q_nh * SPOT_USD:.3f}"
    out["QwenThreeLightBreakEvenUSD"] = f"{fam_cost['light'] / q_nh:.2f}"
    return out


# --------------------------------------------------------------------------
# Experiment 2


def experiment2() -> dict[str, str]:
    rep = json.loads((JV / "out" / "report.json").read_text())
    out: dict[str, str] = {}
    alphas, ident = [], []
    for key, v in rep["self"].items():
        judge, lang = key.split("|")
        out[f"{JUDGE[judge]}{LANG[lang]}SelfAlpha"] = _f(v["alpha"])
        alphas.append(v["alpha"])
        ident.append(v["identical_share"])
    out["SelfAlphaMin"], out["SelfAlphaMax"] = _f(min(alphas)), _f(max(alphas))
    out["SelfIdenticalMinPct"], out["SelfIdenticalMaxPct"] = _pct(min(ident)), _pct(max(ident))
    for p in rep["e1_en"]["pairs"]:
        a, b = JUDGE[p["left"]], JUDGE[p["right"]]
        out[f"{a}{b}RhoEn"] = _f(p["overall_spearman"])
        out[f"{a}{b}WithinOnePct"] = _pct(p["within_one"])
        out[f"{a}{b}BiasEn"] = _f(p["bias"], 1)
        out[f"{b}{a}BiasEn"] = _f(-p["bias"], 1)
        out[f"{a}{b}TauEn"] = _f(p["arm_tau"])
    out["AlphaJudgesEn"] = _f(rep["e1_en"]["alpha"])
    flu = [p["per_aspect"]["fluency_score"] for p in rep["e1_en"]["pairs"]]
    kei = [p["per_aspect"]["keigo_score"] for p in rep["e1_en"]["pairs"]]
    out["EnFluencyRhoMin"], out["EnFluencyRhoMax"] = _f(min(flu)), _f(max(flu))
    out["EnKeigoRhoMin"], out["EnKeigoRhoMax"] = _f(min(kei)), _f(max(kei))
    ja = rep["e1_ja"]["pairs"][0]
    out["JudgesRhoJa"] = _f(ja["overall_spearman"])
    out["JudgesWithinOneJaPct"] = _pct(ja["within_one"])
    out["AlphaJudgesJa"] = _f(rep["e1_ja"]["alpha"])
    for v in rep["e2"]:
        j = JUDGE[v["judge"]]
        out[f"{j}LangRho"] = _f(v["spearman"])
        out[f"{j}LangShift"] = _f(v["mean_ja"] - v["mean_en"])
        out[f"{j}KeigoLangShift"] = _f(v["per_aspect"]["keigo_score"]["shift"])
        out[f"{j}FluencyLangShift"] = _f(v["per_aspect"]["fluency_score"]["shift"])
    # relative agreement among the judges only
    from jv import relative
    rel = relative.analyse()
    judges = [k for k in rel["keys"] if not k[0].startswith("human:")]
    name = lambda k: f"{JUDGE[k[0]]}{LANG[k[1]]}"  # noqa: E731
    for i, a in enumerate(judges):
        for b in judges[i + 1:]:
            out[f"Rel{name(a)}{name(b)}Pct"] = _pct(rel["matrix"][(a, b)]["rate"])
    g = rel["groups"]["AI judges"]
    out["RelJudgesW"] = _f(g["kendall_w"])
    out["RelJudgesAlphaStd"] = _f(g["alpha_standardised"])
    out["RelJudgesAlphaRaw"] = _f(g["alpha_raw"])
    return out


# --------------------------------------------------------------------------
# Experiment 3


def _paired(values: dict, key: dict) -> list[tuple[float, float]]:
    """(EN, JA) per customer from {script_id: value}."""
    by: dict = {}
    for sid, v in values.items():
        if sid in key and v is not None:
            by.setdefault(key[sid]["customer"], {})[key[sid]["condition"]] = v
    return [(c["en"], c["ja"]) for c in by.values() if "en" in c and "ja" in c]


def exp3_judge_free_rows() -> list[dict]:
    """Paired tests on the judge-free measures; also the source of the table."""
    with open(JV / "out" / "exp3" / "metrics.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by: dict = {}
    for r in rows:
        by.setdefault(r["customer"], {})[r["condition"]] = r
    pairs = [v for v in by.values() if "en" in v and "ja" in v]
    measures = [("chars", "length"), ("html_ok_share", "markup"), ("latin_share", "latin"),
                ("keigo_referent_honorifics_per_100_sentences", "honorifics"),
                ("keigo_defects_per_100_sentences", "errors"),
                ("keigo_register_consistency", "consistency")]
    out = []
    for col, label in measures:
        use = [(float(p["en"][col]), float(p["ja"][col])) for p in pairs
               if p["en"][col] != "" and p["ja"][col] != ""]
        diffs = [j - e for e, j in use]
        up, down = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        lo, hi = bootstrap_ci(diffs)
        out.append({"measure": label, "n": len(use), "en": st.mean(e for e, _ in use),
                    "ja": st.mean(j for _, j in use), "diff": st.mean(diffs),
                    "ci": (lo, hi), "up": up, "down": down,
                    "p_wilcoxon": wilcoxon(diffs), "p_sign": sign_test(up, down),
                    "rel": st.mean((j - e) / e for e, j in use if e) if label == "length" else None,
                    "over2_en": sum(e > 0.02 for e, _ in use) if label == "latin" else None,
                    "over2_ja": sum(j > 0.02 for _, j in use) if label == "latin" else None,
                    # share of the summed difference from the five largest drops, and ties
                    "top5_share": sum(sorted(diffs)[:5]) / sum(diffs) if sum(diffs) else None,
                    "zeros": sum(d == 0 for d in diffs)})
    return out


def exp3_script_score() -> dict:
    """Experiment 3, script score (five-component mean): Japanese minus English
    instructions per customer, for each judge and rubric.
    """
    from jv import checks
    data = checks.load()
    rows, tests = {}, {}
    for key, cell in data["cells"].items():
        diffs = [j - e for e, j in _paired(cell.overall, data["key"])]
        lo, hi = bootstrap_ci(diffs)
        rows[key] = {"diff": st.mean(diffs), "lo": lo, "hi": hi}
        tests[key] = wilcoxon(diffs)
    for key, p in holm(tests).items():
        rows[key]["p_holm"] = p
    return rows


def experiment3() -> dict[str, str]:
    out: dict[str, str] = {}
    jf = {r["measure"]: r for r in exp3_judge_free_rows()}
    out["LengthIncreasePct"] = _pct(jf["length"]["rel"])
    out["LongerCustomers"] = str(jf["length"]["up"])
    out["ExpThreeMarkupEnPct"] = _pct(jf["markup"]["en"], 1)
    out["ExpThreeMarkupJaPct"] = _pct(jf["markup"]["ja"], 1)
    out["ExpThreeMarkupUp"], out["ExpThreeMarkupDown"] = str(jf["markup"]["up"]), str(jf["markup"]["down"])
    out["ExpThreeMarkupWilcoxonP"] = _p(jf["markup"]["p_wilcoxon"])
    out["ExpThreeMarkupSignP"] = _p(jf["markup"]["p_sign"])
    out["LatinOverEn"], out["LatinOverJa"] = str(jf["latin"]["over2_en"]), str(jf["latin"]["over2_ja"])
    out["HonorificsEn"], out["HonorificsJa"] = _f(jf["honorifics"]["en"]), _f(jf["honorifics"]["ja"])
    out["HonorificsP"] = _p(jf["honorifics"]["p_wilcoxon"])
    out["KeigoErrorsP"] = _p(jf["errors"]["p_wilcoxon"])
    out["ConsistencyP"] = _p(jf["consistency"]["p_wilcoxon"])
    out["KeigoErrorsDiff"] = _f(jf["errors"]["diff"])
    out["LatinTopFivePct"] = _pct(jf["latin"]["top5_share"])
    out["ErrorsTopFivePct"] = _pct(jf["errors"]["top5_share"])
    out["ErrorsZeroPairs"] = str(jf["errors"]["zeros"])
    out["ConsistencyDiff"] = _f(jf["consistency"]["diff"], 3)
    # judged, per aspect (the harness's report)
    rep = json.loads((JV / "out" / "exp3" / "report.json").read_text())
    for r in rep["overall"]:
        cell = f"ExpThree{JUDGE[r['judge']]}{LANG[r['lang']]}{ASPECT[r['aspect']]}"
        out[f"{cell}Diff"] = _f(r["diff"])
        out[f"{cell}Holm"] = _p(r["p_holm"])
    # the script score (five-component mean), paired, Holm across the four cells
    for (judge, lang), r in exp3_script_score().items():
        name = f"ExpThree{JUDGE[judge]}{LANG[lang]}Score"
        out[f"{name}Diff"] = _f(r["diff"])
        out[f"{name}Lo"], out[f"{name}Hi"] = _f(r["lo"]), _f(r["hi"])
        out[f"{name}Holm"] = _p(r["p_holm"])
    # robustness checks
    chk = json.loads((JV / "out" / "exp3" / "checks.json").read_text())
    for r in chk["length"]:
        judge, lang = r["cell"]
        cell = f"Len{JUDGE[judge]}{LANG[lang]}"
        if r["aspect"] == "component_mean":
            out[f"{cell}Raw"] = _f(r["raw"])
            out[f"{cell}Adjusted"] = _f(r["adjusted"])
            out[f"{cell}Lo"], out[f"{cell}Hi"] = _f(r["adjusted_ci"][0]), _f(r["adjusted_ci"][1])
            out[f"{cell}Slope"] = _f(r["per_1000_chars"])
            out["LengthDiffThousand"] = _f(r["mean_len_diff"], 1)
        elif (judge, lang, r["aspect"]) == ("deepseek-v4-flash", "en", "fluency_score"):
            out["LenDeepSeekEnFluencySlope"] = _f(r["per_1000_chars"])
            out["LenDeepSeekEnFluencySlopeLo"] = _f(r["slope_ci"][0])
            out["LenDeepSeekEnFluencySlopeHi"] = _f(r["slope_ci"][1])
        elif (judge, lang, r["aspect"]) == ("glm-4.7-flash", "ja", "overall_score"):
            out["LenGlmJaOverallAdjusted"] = _f(r["adjusted"])
            out["LenGlmJaOverallLo"] = _f(r["adjusted_ci"][0])
            out["LenGlmJaOverallHi"] = _f(r["adjusted_ci"][1])
    pw = [p["exp3"]["pairwise"] for p in chk["agreement"]["pairs"]]
    out["ScalePairwiseMinPct"], out["ScalePairwiseMaxPct"] = _pct(min(pw)), _pct(max(pw))
    for p in chk["agreement"]["pairs"]:
        a = f"{JUDGE[p['a'][0]]}{LANG[p['a'][1]]}"
        b = f"{JUDGE[p['b'][0]]}{LANG[p['b'][1]]}"
        out[f"Scale{a}{b}Pct"] = _pct(p["exp3"]["pairwise"])
        out[f"Scale{a}{b}LegacyPct"] = _pct(p["legacy12"]["pairwise"])
    selfs = [v["alpha"] for v in chk["self"]["exp3"].values()]
    out["ScaleSelfAlphaMin"], out["ScaleSelfAlphaMax"] = _f(min(selfs)), _f(max(selfs))
    k3 = [abs(v) for cell in chk["keigo"]["exp3"].values() for v in cell.values()]
    out["KeigoValidityMaxRho"] = _f(max(k3))
    out["GptKeigoErrorsRho"] = _f(chk["keigo"]["legacy12"]["gpt-5.6-sol|en"]["keigo_defects_per_100_sentences"])
    return out


# --------------------------------------------------------------------------
# Experiment 4


def experiment4() -> dict[str, str]:
    out: dict[str, str] = {}
    # score behaviour on the 12 scripts (judges only)
    from jv.figures import collect
    fig = collect()
    for key in fig["keys"]:
        if key[0].startswith("human:"):
            continue
        name = f"{JUDGE[key[0]]}{LANG[key[1]]}"
        scores = [s for v in fig["raw"][key].values() for s in v]
        counts = Counter(scores)
        mode, n = counts.most_common(1)[0]
        out[f"{name}ScaleMean"] = _f(st.mean(scores), 1)
        out[f"{name}ScaleMode"] = str(mode)
        out[f"{name}ScaleModePct"] = _pct(n / len(scores))
        out[f"{name}ScaleDistinct"] = str(len(counts))
        first = sum(1 for asp, arms in fig["grid"][key].items()
                    if max(arms, key=arms.get) == "batch")
        out[f"{name}BaselineFirst"] = str(first)
    # the codebook
    rep = json.loads((JV / "out" / "codebook" / "report.json").read_text())
    for kind, block in rep["judge"].items():
        out[f"Codebook{kind.capitalize()}V"] = _f(block["test"]["cramers_v"])
        for judge, shares in block["shares"].items():
            for cat in ("S_PERSONAL", "S_FIGURES", "S_PRODUCT", "S_PERSUASION", "S_REGISTER",
                        "W_ACCURACY", "W_VERBOSE", "W_FLUENCY", "W_GENERIC", "W_KEIGO",
                        "G_OBJECTION", "G_PRICE", "G_PRODUCT", "G_NODATA", "G_NEXTSTEP"):
                if cat[0] != kind[0].upper():  # only the categories of the point's own kind
                    continue
                label = "".join(w.capitalize() for w in cat.split("_")[1:])
                out[f"{JUDGE[judge]}{kind.capitalize()}{label}Pct"] = _pct(shares.get(cat, 0))
    for kind, block in rep["exp3_condition"].items():
        if kind != "weakness":
            continue
        for group, shares in block["shares"].items():
            judge, cond = group.split("|")
            for cat in ("W_VERBOSE", "W_KEIGO"):
                label = cat.split("_")[1].capitalize()
                out[f"{JUDGE[judge]}{kind.capitalize()}{label}{LANG[cond]}ScriptsPct"] = \
                    _pct(shares.get(cat, 0))
    # G_NODATA by corpus, and keigo in free feedback, from the labelled items
    items = cb.attach(cb.read_items(JV / "data" / "codebook" / "items.jsonl"),
                      cb.LabelStore(JV / "out" / "codebook" / "labels.db").all())
    legacy = lambda i: len(i["script_id"]) <= 3 and i["script_id"].startswith("S")  # noqa: E731
    for judge in ("deepseek-v4-flash", "glm-4.7-flash"):
        for corpus, test in (("Blind", legacy), ("Synth", lambda i: not legacy(i))):
            gaps = [i for i in items if i["judge"] == judge and i["kind"] == "gap" and test(i)]
            out[f"{JUDGE[judge]}NoData{corpus}Pct"] = _pct(
                sum(i["category"] == "G_NODATA" for i in gaps) / len(gaps))
        fb = [i for i in items if i["judge"] == judge and i["source"] == "feedback"]
        out[f"{JUDGE[judge]}FeedbackKeigoCount"] = str(sum(i["category"] == "W_KEIGO" for i in fb))
        out[f"{JUDGE[judge]}FeedbackPoints"] = f"{len(fb):,}"
    # the rubric's own keigo-error lists, entries per judgement
    stores = [JV / "out" / "judgements.db"] + [JV / "out" / "exp3" / f"{j}.db"
                                              for j in ("deepseek-v4-flash", "glm-4.7-flash")]
    entries: Counter = Counter()
    judgements: Counter = Counter()
    for path in stores:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        for judge, evidence in db.execute("SELECT judge, evidence FROM judgements WHERE ok = 1"):
            if judge == "gpt-5.6-sol":
                continue
            judgements[judge] += 1
            entries[judge] += len(json.loads(evidence or "{}").get("keigo_errors") or [])
        db.close()
    for judge in ("deepseek-v4-flash", "glm-4.7-flash"):
        out[f"{JUDGE[judge]}RubricKeigoEntries"] = f"{entries[judge]:,}"
        out[f"{JUDGE[judge]}RubricKeigoPerJudgement"] = _f(entries[judge] / judgements[judge], 1)
    return out


# --------------------------------------------------------------------------
# cost against Azure's node prices


EXP1_NODES = 2  # nodes per pool in Experiment 1


def _cold_start(q8: float, spot: float) -> dict[str, str]:
    """A node's start-up as a fixed cost, against the node time of one script."""
    import re
    row = next(r for r in _rows("exp1_serving.csv") if r["measure"].startswith("Cold start"))
    cold_h = float(re.search(r"[\d.]+", row["after"]).group()) / 3600
    n = int(_rows("exp1_compute.csv")[0]["vehicles"])
    return {
        "ColdStartUSD": f"{cold_h * spot:.2f}",
        "ColdStartScripts": f"{cold_h / q8:.1f}",
        "ColdStartSmallRunPct": _pct(EXP1_NODES * cold_h / (n * q8)),
    }


def discussion() -> dict[str, str]:
    spot, payg = SPOT_USD, PAYG_USD
    comp = {r["arm"]: float(r["node_hours_per_vehicle"]) for r in _rows("exp1_compute.csv")}
    base = float({r["approach"]: r for r in _rows("baseline_cost.csv")}["batch"]["usd_per_vehicle"])
    q8, q3 = comp["qwen38_bf16"], comp["qwen_only"]
    ctx = _rows("exp1_context.csv")[0]
    judged = {r["arm"]: float(r["score"]) for r in _rows("exp1_judged.csv") if r["aspect"] == "mean"}
    vehicles = int({r["approach"]: r for r in _rows("baseline_cost.csv")}["batch"]["vehicles"])
    # Batch mode writes the discount argument per item in two languages, Approach 3
    # once per vehicle: the like-for-like saving leaves batch mode's stage D out.
    stages = _rows("baseline_stages.csv")
    batch_calls = {s: sum(float(r["calls_per_vehicle"]) for r in stages if r["stage"] == s) for s in "AD"}
    discount_usd = sum(float(r["usd_per_vehicle"]) for r in stages if r["stage"] == "D")
    own = {r["stage"]: float(r["tasks_per_vehicle"]) for r in _rows("exp1_stage_tasks.csv")
           if r["arm"] == "qwen38_bf16"}
    tasks = {r["arm"]: float(r["tasks_per_vehicle"]) for r in _rows("exp1_compute.csv")}
    assert len(set(tasks.values())) == 1, tasks
    assert own["discount"] == 1, own  # one discount task per vehicle
    return {
        "ExpOneTasksPerVehicle": f"{tasks['qwen38_bf16']:.1f}",
        "ExpOneDraftCalls": f"{own['draft']:.1f}",
        "BatchDraftCalls": f"{batch_calls['A']:.1f}",
        "BatchDiscountCalls": f"{batch_calls['D']:.1f}",
        "BatchNoDiscountUSD": f"{base - discount_usd:.2f}",
        "QwenEightLikeSavingPct": _pct(1 - q8 * spot / (base - discount_usd)),
        "SpotPriceJapanEast": f"{spot:.2f}",
        "PaygPriceJapanEast": f"{payg:.2f}",
        "PaygSpotRatio": f"{payg / spot:.1f}",
        "QwenEightUSDAtPayg": f"{q8 * payg:.2f}",
        "QwenEightPaygSavingPct": _pct(1 - q8 * payg / base),
        "CostRatioQwenEight": f"{base / (q8 * SPOT_USD):.1f}",
        "QwenEightSpotSavingPct": _pct(1 - q8 * spot / base),
        "QwenThreeSpotSavingPct": _pct(1 - q3 * spot / base),
        "CostRatioQwenThree": f"{base / (q3 * SPOT_USD):.1f}",
        "LongestPromptChars": f"{int(ctx['longest_prompt_chars']):,}",
        "RunSavingUSD": f"{(base - q8 * SPOT_USD) * vehicles:,.0f}",
        **_cold_start(q8, spot),
        "QwenThreeGapToBatch": _f(judged["batch"] - judged["qwen_only"]),
    }


# --------------------------------------------------------------------------
# the human raters


#: Raters in the order their forms came back, as the files in harness/data/human/ sort.
RATER_ORDER = ["human:rater1", "human:rater2", "human:rater3", "human:rater4",
               "human:rater5"]
RATER_WORD = ["One", "Two", "Three", "Four", "Five"]


def rater_data() -> dict:
    """Script scores of raters and judges on the blind scripts, and the consensus."""
    import csv as _csv
    from jv.analyze import cells, load_reference, spearman
    from jv.human import load_all
    from jv.relative import standardise
    from jv.store import Store
    with open(JV / "data" / "blind_key.csv", newline="", encoding="utf-8") as fh:
        arms = {r["script_id"]: r["side"] for r in _csv.DictReader(fh)}
    hc = cells(load_all(arms))
    assert sorted(k[0] for k in hc) == sorted(RATER_ORDER), f"raters: {sorted(hc)}"
    raters = {n: hc[(n, "form")] for n in RATER_ORDER}
    scripts = sorted(raters[RATER_ORDER[0]].overall)
    z = {n: standardise(c.overall, scripts) for n, c in raters.items()}
    judges = cells(list(Store(JV / "out" / "judgements.db").read(only_ok=True))
                   + load_reference(JV / "data" / "reference_scores.csv"))

    def consensus(members, aspect=None):
        if aspect is None:
            zz = {n: z[n] for n in members}
        else:
            zz = {n: standardise({s: raters[n].per_script[s][aspect] for s in scripts}, scripts)
                  for n in members}
        return {s: st.mean(zz[n][s] for n in members) for s in scripts}

    return {"arms": arms, "raters": raters, "scripts": scripts, "z": z,
            "judges": judges, "consensus": consensus, "spearman": spearman}


def raters() -> dict[str, str]:
    from jv.analyze import krippendorff_alpha
    from jv.human import range_usage, load_all
    from jv.relative import kendall_w, analyse
    d = rater_data()
    R, S, z, sp, cons = d["raters"], d["scripts"], d["z"], d["spearman"], d["consensus"]
    out: dict[str, str] = {"RatersCount": str(len(R))}
    word = dict(zip(RATER_ORDER, RATER_WORD))
    usage = range_usage(load_all(d["arms"]))
    loo = {}
    for n in RATER_ORDER:
        w = word[n]
        u = usage[n]
        out[f"Rater{w}Mean"] = _f(st.mean(R[n].overall.values()), 1)
        out[f"Rater{w}Distinct"] = str(u["distinct"])
        out[f"Rater{w}Mode"] = str(u["modal"])
        out[f"Rater{w}ModePct"] = _pct(u["modal_share"])
        rest = cons([m for m in RATER_ORDER if m != n])
        loo[n] = sp([R[n].overall[s] for s in S], [rest[s] for s in S])
        out[f"Rater{w}Loo"] = _f(loo[n])
    for i, a in enumerate(RATER_ORDER):
        for b in RATER_ORDER[i + 1:]:
            out[f"Rater{word[a]}{word[b]}Rho"] = _f(sp([R[a].overall[s] for s in S],
                                                      [R[b].overall[s] for s in S]))
    vals = sorted(loo.values())
    out["RatersLooMin"], out["RatersLooMax"] = _f(vals[0]), _f(vals[-1])
    out["RatersLooMedian"] = _f(st.median(vals))
    # the same benchmark among Human Raters 2 to 5 only (sensitivity check)
    rest4 = RATER_ORDER[1:]
    loo4 = sorted(sp([R[n].overall[s] for s in S], [cons([m for m in rest4 if m != n])[s] for s in S])
                  for n in rest4)
    out["RatersLooMinWithoutOne"], out["RatersLooMaxWithoutOne"] = _f(loo4[0]), _f(loo4[-1])
    out["RatersLooMedianWithoutOne"] = _f(st.median(loo4))
    out["RatersAlphaRaw"] = _f(krippendorff_alpha([[R[n].overall[s] for n in RATER_ORDER] for s in S]))
    out["RatersAlphaStd"] = _f(krippendorff_alpha([[z[n][s] for n in RATER_ORDER] for s in S]))
    out["RatersW"] = _f(kendall_w([R[n].overall for n in RATER_ORDER], S))
    without_one = [n for n in RATER_ORDER if n != RATER_ORDER[0]]
    zz = {n: z[n] for n in without_one}
    out["RatersAlphaStdWithoutOne"] = _f(krippendorff_alpha([[zz[n][s] for n in without_one] for s in S]))
    # per aspect: how far the raters agree (standardised alpha)
    for asp, label in ASPECT.items():
        per = {n: {s: R[n].per_script[s][asp] for s in S} for n in RATER_ORDER}
        from jv.relative import standardise
        zs = {n: standardise(per[n], S) for n in RATER_ORDER}
        out[f"RatersAlpha{label}"] = _f(krippendorff_alpha([[zs[n][s] for n in RATER_ORDER] for s in S]))
    # readiness: the raters agree, the judges run against them
    ready = cons(RATER_ORDER, "readiness_score")
    rr = [sp([c.per_script[s]["readiness_score"] for s in S], [ready[s] for s in S])
          for c in d["judges"].values()]
    out["JudgesHumanReadinessRhoMin"], out["JudgesHumanReadinessRhoMax"] = _f(min(rr)), _f(max(rr))
    # judges against the consensus, on the same footing as a rater
    for (judge, lang), cell in sorted(d["judges"].items()):
        rhos = [sp([cell.overall[s] for s in S], [cons([m for m in RATER_ORDER if m != n])[s] for s in S])
                for n in RATER_ORDER]
        name = f"{JUDGE[judge]}{LANG[lang]}"
        out[f"{name}HumanRho"] = _f(st.mean(rhos))
        # without Human Rater 1, on the same footing: mean rho with the four three-rater
        # consensuses of Human Raters 2 to 5
        out[f"{name}HumanRhoWithoutOne"] = _f(st.mean(
            sp([cell.overall[s] for s in S], [cons([m for m in without_one if m != n])[s] for s in S])
            for n in without_one))
        for asp in ("keigo_score", "fluency_score"):
            c = cons(RATER_ORDER, asp)
            out[f"{name}Human{ASPECT[asp]}Rho"] = _f(sp([cell.per_script[s][asp] for s in S],
                                                        [c[s] for s in S]))
    # the consensus order of the systems, with and without Human Rater 1
    for tag, members in (("", RATER_ORDER), ("WithoutOne", without_one)):
        c = cons(members)
        by: dict = {}
        for s in S:
            by.setdefault(d["arms"][s], []).append(c[s])
        for arm, v in by.items():
            out[f"Consensus{ARM[arm]}{tag}"] = _f(st.mean(v))
    # each rater's top system (arm means of the raw script score)
    for n in RATER_ORDER:
        by: dict = {}
        for s in S:
            by.setdefault(d["arms"][s], []).append(R[n].overall[s])
        means = {a: st.mean(v) for a, v in by.items()}
        out[f"Rater{word[n]}TopArm"] = max(means, key=means.get)  # used by tables.py only
    # relative agreement: humans among themselves and with the judges
    rel = analyse()
    g = rel["groups"]["humans"]
    out["RelRatersW"], out["RelRatersAlphaStd"] = _f(g["kendall_w"]), _f(g["alpha_standardised"])
    hh = [rel["matrix"][(a, b)]["rate"] for i, a in enumerate(rel["keys"]) for b in rel["keys"][i + 1:]
          if a[0].startswith("human:") and b[0].startswith("human:")]
    out["RelRatersMinPct"], out["RelRatersMaxPct"] = _pct(min(hh)), _pct(max(hh))
    # keigo: human keigo scores against the morphological measures (check C)
    chk = json.loads((JV / "out" / "exp3" / "checks.json").read_text())["keigo"]["legacy12"]
    err = [chk[f"{n}|form"]["keigo_defects_per_100_sentences"] for n in RATER_ORDER]
    out["RatersKeigoErrorsRhoMin"], out["RatersKeigoErrorsRhoMax"] = _f(min(err)), _f(max(err))
    hon = [chk[f"{n}|form"]["keigo_referent_honorifics_per_100_sentences"] for n in RATER_ORDER]
    out["RatersKeigoHonorificsRhoMin"], out["RatersKeigoHonorificsRhoMax"] = _f(min(hon)), _f(max(hon))
    # how uncertain one rho on twelve scripts is: bootstrap over scripts, best judge
    import random
    rng = random.Random(20261022)
    full = cons(RATER_ORDER)
    ds = d["judges"][("deepseek-v4-flash", "en")].overall
    boot = []
    while len(boot) < 10_000:
        pick = [rng.choice(S) for _ in S]
        if len({ds[s] for s in pick}) > 1 and len({full[s] for s in pick}) > 1:
            boot.append(sp([ds[s] for s in pick], [full[s] for s in pick]))
    boot.sort()
    out["DeepSeekEnHumanRhoFull"] = _f(sp([ds[s] for s in S], [full[s] for s in S]))
    out["DeepSeekEnHumanRhoLo"], out["DeepSeekEnHumanRhoHi"] = _f(boot[249]), _f(boot[9749])
    # robustness: a script score from the three aspects that need no customer data
    # (personalisation and persuasion presuppose it, and the blind scripts came without)
    from jv.relative import standardise
    intrinsic = ("keigo_score", "fluency_score", "readiness_score")

    def score3(cell, s):
        return st.mean(cell.per_script[s][a] for a in intrinsic)
    z3 = {n: standardise({s: score3(R[n], s) for s in S}, S) for n in RATER_ORDER}

    def cons3(members):
        return {s: st.mean(z3[n][s] for n in members) for s in S}
    loo3 = [sp([score3(R[n], s) for s in S], [cons3([m for m in RATER_ORDER if m != n])[s] for s in S])
            for n in RATER_ORDER]
    out["RatersLooMedianIntrinsic"] = _f(st.median(loo3))
    ds_cell = d["judges"][("deepseek-v4-flash", "en")]
    out["DeepSeekEnHumanRhoIntrinsic"] = _f(st.mean(
        sp([score3(ds_cell, s) for s in S], [cons3([m for m in RATER_ORDER if m != n])[s] for s in S])
        for n in RATER_ORDER))
    c3 = cons3(RATER_ORDER)
    by3: dict = {}
    for s in S:
        by3.setdefault(d["arms"][s], []).append(c3[s])
    for arm, v in by3.items():
        out[f"Consensus{ARM[arm]}Intrinsic"] = _f(st.mean(v))
    return {k: v for k, v in out.items() if not k.endswith("TopArm")}


def results() -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (baseline(), experiment1(), experiment2(), experiment3(), experiment4(),
                 discussion(), raters()):
        clash = set(out) & set(part)
        assert not clash, f"duplicate macros: {sorted(clash)}"
        out.update(part)
    assert all(ch.isalpha() for k in out for ch in k), "macro names must be letters only"
    return out

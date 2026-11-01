"""Numbers quoted in the prose, as LaTeX macros, from thesis/data."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path

import yaml

from constants import SPOT_USD

THESIS = Path(__file__).resolve().parents[1]
DATA = THESIS / "data"
JV = THESIS.parent / "harness"


def _rows(name: str) -> list[dict]:
    with open(DATA / name, newline="") as fh:
        return list(csv.DictReader(fh))


def macros() -> dict[str, str]:
    base = {r["approach"]: r for r in _rows("baseline_cost.csv")}
    core = {
        # batch telemetry over the production population (legacy_cost.json)
        "BatchVehicles": f"{int(base['batch']['vehicles']):,}",
        "BatchCallsPerVehicle": f"{float(base['batch']['calls_per_vehicle']):.0f}",
        "BatchTokensPerVehicle": f"{int(base['batch']['input_tokens_per_vehicle']) + int(base['batch']['output_tokens_per_vehicle']):,}",
        # measured tokens at list prices with the batch discount; contracted prices unknown
        "BatchUSDPerVehicle": f"{float(base['batch']['usd_per_vehicle']):.2f}",
        # Approach 1, reconstructed from retained output: a lower bound
        "LegacyCallsPerVehicle": f"{float(base['legacy']['calls_per_vehicle']):.0f}",
        "LegacyUSDPerVehicle": f"{float(base['legacy']['usd_per_vehicle']):.2f}",
        **families(),
        **overview(),
        **experiment1(base),
        **experiment2(),
        **experiment3(),
        **experiment4(),
    }
    from results_macros import results
    extra = results()
    clash = set(core) & set(extra)
    assert not clash, f"results macros redefine {sorted(clash)}"
    return {**core, **extra}


def families() -> dict[str, str]:
    """Anonymised family names: light-duty, electric light-duty, heavy-duty."""
    name = {"light": "Light", "electric": "Electric", "heavy": "Heavy"}
    return {f"Family{name[r['family']]}Vehicles": f"{int(r['vehicles']):,}"
            for r in _rows("baseline_families.csv")}


def overview() -> dict[str, str]:
    """Sizes of the four experiments, for the overview table."""
    comp = _rows("exp1_compute.csv")
    vehicles = {int(r["vehicles"]) for r in comp}
    assert len(vehicles) == 1, "Experiment 1 arms ran over different vehicle counts"
    design = yaml.safe_load((JV / "config" / "experiment.yaml").read_text())
    legacy = json.loads((JV / "data" / "corpus.json").read_text())
    exp3 = json.loads((JV / "out" / "exp3" / "corpus.json").read_text())
    gen = sqlite3.connect(f"file:{JV / 'out' / 'exp3' / 'generation.db'}?mode=ro&immutable=1", uri=True)
    calls = gen.execute("select count(*) from calls").fetchone()[0]
    codebook = json.loads((JV / "out" / "codebook" / "report.json").read_text())
    return {
        "ExpOneVehicles": str(vehicles.pop()),
        "ExpOneJudgements": _rows("exp1_judged.csv")[0]["judgements"],
        "SplitVehicles": _rows("exp1_split.csv")[0]["vehicles"],
        "BlindScripts": str(len(legacy)),
        "ExpTwoRepeats": str(design["run"]["repeats"]),
        "ExpThreeCustomers": str(design["exp3"]["customers"]),
        "ExpThreeScripts": str(len(exp3)),
        "ExpThreeCalls": f"{calls:,}",
        "ExpThreeRepeats": str(design["exp3"]["judging"]["repeats"]),
        "FeedbackPoints": f"{codebook['n_items']:,}",
        "CoderSample": str(codebook["agreement_llm_coder"]["n"]),
        **data_sections(),
    }


def data_sections() -> dict[str, str]:
    """Corpus counts: pairs, the blind sample, the synthetic customers."""
    name = {"light": "Light", "electric": "Electric", "heavy": "Heavy"}
    sel = _rows("exp1_selection.csv")
    with open(JV / "data" / "blind_key.csv", newline="") as fh:
        blind = Counter(r["rec_type"] for r in csv.DictReader(fh))
    with open(JV / "out" / "exp3" / "key.csv", newline="") as fh:
        synth = Counter(fam for _, fam in {(r["customer"], r["family"]) for r in csv.DictReader(fh)})
    out = {"PairedVehicles": f"{sum(int(r['paired']) for r in sel):,}",
           "InputRows": f"{int(_rows('baseline_inputs.csv')[0]['input_rows']):,}"}
    for r in sel:
        out[f"Paired{name[r['family']]}"] = r["paired"]
        out[f"ExpOne{name[r['family']]}"] = r["selected"]
    for fam, label in name.items():
        out[f"Blind{label}"] = str(blind[fam])
        out[f"Synth{label}"] = str(synth[fam])
    return out


def experiment1(base: dict) -> dict[str, str]:
    comp = {r["arm"]: r for r in _rows("exp1_compute.csv")}
    judged = {r["arm"]: float(r["score"]) for r in _rows("exp1_judged.csv") if r["aspect"] == "mean"}
    split = {r["arm"]: float(r["mean"]) for r in _rows("exp1_split.csv")}
    markup = {r["arm"]: float(r["bad_markup"]) for r in _rows("exp1_judge_free.csv") if r["family"] == "all"}
    node_hours = float(comp["qwen38_bf16"]["node_hours_per_vehicle"])
    pairs = _rows("exp1_pairwise.csv")[0]
    return {
        # split run's pairwise comparison, cut short by the rate limit; not a result
        "SplitPairsCompleted": pairs["completed"],
        # request latency / concurrency: a lower bound on billed node time
        "QwenEightUSDPerVehicle": f"{node_hours * SPOT_USD:.3f}",
        "BreakEvenUSD": f"{float(base['batch']['usd_per_vehicle']) / node_hours:.2f}",
        # GPT-5.6 judge, mean of the five component aspects (overall_score excluded); the
        # split run is a separate light-duty sample
        "SoloGap": f"{judged['batch'] - judged['qwen38_bf16']:.2f}",
        "SplitGap": f"{split['batch'] - split['split']:.2f}",
        "BatchBadMarkupPct": f"{markup['batch'] * 100:.0f}",
        "QwenEightBadMarkupPct": f"{markup['qwen38_bf16'] * 100:.0f}",
    }


def experiment2() -> dict[str, str]:
    rep = json.loads((JV / "out" / "report.json").read_text())
    rhos = [p["overall_spearman"] for p in rep["e1_en"]["pairs"]]
    # Calls that never completed (cut off at max_tokens on every attempt), and the fewest
    # repeats any script kept in any judge x rubric cell
    db = sqlite3.connect(JV / "out" / "judgements.db")
    missing = db.execute("SELECT count(*) FROM judgements WHERE ok = 0").fetchone()[0]
    least = db.execute("SELECT min(n) FROM (SELECT sum(ok) AS n FROM judgements "
                       "GROUP BY judge, rubric_lang, script_id)").fetchone()[0]
    db.close()
    return {"JudgeRhoMin": f"{min(rhos):.2f}", "JudgeRhoMax": f"{max(rhos):.2f}",
            "ExpTwoMissing": str(missing), "ExpTwoMinRepeats": str(least)}


def experiment3() -> dict[str, str]:
    with open(JV / "out" / "exp3" / "metrics.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    col = "keigo_referent_honorifics_per_100_sentences"
    mean = lambda cond: sum(float(r[col]) for r in rows if r["condition"] == cond) / \
        sum(1 for r in rows if r["condition"] == cond)
    # instruction segments translated once into the Japanese table
    segments = json.loads((JV / "gen" / "translations" / "batch_prod_ja.json").read_text())
    # judge calls that never completed, both judges and rubrics
    missing = 0
    for judge in ("deepseek-v4-flash", "glm-4.7-flash"):
        db = sqlite3.connect(JV / "out" / "exp3" / f"{judge}.db")
        missing += db.execute("SELECT count(*) FROM judgements WHERE ok = 0").fetchone()[0]
        db.close()
    return {"HonorificRatio": f"{mean('ja') / mean('en'):.1f}",
            "TranslationSegments": str(len(segments)),
            "ExpThreeMissing": str(missing)}


def experiment4() -> dict[str, str]:
    with open(JV / "data" / "codebook" / "codebook.csv", newline="", encoding="utf-8-sig") as fh:
        categories = sum(1 for _ in csv.DictReader(fh))
    # unscored feedback calls: one per script, judge and language, both corpora
    calls = 0
    for db_path in sorted((JV / "out").glob("**/feedback/*.db")):
        db = sqlite3.connect(db_path)
        calls += db.execute("SELECT count(*) FROM feedback WHERE ok = 1").fetchone()[0]
        db.close()
    # share of the filed points that come from the rubric's own evidence lists
    with open(JV / "data" / "codebook" / "items.jsonl", encoding="utf-8") as fh:
        sources = Counter(json.loads(line)["source"].startswith("rubric:")
                          for line in fh if line.strip())
    # classifier against the independent LLM coder (jv codebook-report)
    coder = json.loads((JV / "out" / "codebook" / "report.json").read_text())["agreement_llm_coder"]
    return {"CodebookCategories": str(categories),
            "CoderKappa": f"{coder['kappa']:.2f}",
            "CoderKappaKind": f"{coder['kappa_kind_level']:.2f}",
            "CoderRawPct": f"{100 * coder['raw_agreement']:.0f}",
            "FeedbackCalls": str(calls),
            "FeedbackRubricPct": f"{100 * sources[True] / sum(sources.values()):.0f}"}


def main() -> None:
    lines = ["% Generated by scripts/macros.py from thesis/data. Do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros().items()]
    (THESIS / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

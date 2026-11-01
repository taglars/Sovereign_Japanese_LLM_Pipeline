"""Pull the Experiment 1 and baseline numbers into small CSVs in thesis/data."""

from __future__ import annotations

import csv
import html
import json
import re
import statistics as st
import zipfile
from pathlib import Path

import yaml

THESIS = Path(__file__).resolve().parents[1]
MT = THESIS.parent
# Withheld company inputs (see README), under anonymised names in this copy.
EXPORT = MT / "company" / "export"
PIPELINE = MT / "company" / "pipeline"
SOURCES = MT / "sources"
NB17 = SOURCES / "notebooks" / "17_model_solo (10).ipynb"
LEGACY_COST = SOURCES / "baseline_analysis" / "20260813_0543_legacy_cost.json"
BATCH_TELEMETRY = LEGACY_COST.with_name("20260813_0526_batch_telemetry.json")
INDUSTRY_PROFILE = LEGACY_COST.with_name("20260812_1045_industry_miss_rate.json")
BASELINE_QUALITY = LEGACY_COST.with_name("20260813_0543_baseline_quality.json")
NB13 = SOURCES / "notebooks" / "13_baseline_headtohead (8).ipynb"
STATUS_DECK = SOURCES / "status_review.pptx"
OUT = THESIS / "data"

SOLO_RUN = "solo-all-20260830-115815"
SPLIT_RUN = "h2h-light-20260828-022746"
# GPT-4o mini 2024-07-18, global, per Mtok (Azure pricing page, accessed 2026-10-15),
# before the batch discount. legacy_cost.json prices stage C at this rate.
MINI_PER_MTOK = (0.15, 0.60)
ARM_CONFIG = {"qwen_only": "routing_qwen_only.yaml",
              "qwen36_bf16": "routing_qwen36_bf16.yaml",
              "qwen38_bf16": "routing_qwen38_bf16.yaml"}
ASPECTS = ["mean", "fluency", "keigo", "overall", "personalisation", "persuasion",
           "readiness"]


def _write(name: str, rows: list[dict]) -> None:
    OUT.mkdir(exist_ok=True)
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  {name}: {len(rows)} rows")


def _notebook_text(path: Path) -> str:
    parts = []
    for cell in json.loads(path.read_text())["cells"]:
        for o in cell.get("outputs", []):
            parts.append("".join(o.get("text", ""))
                         or "".join(o.get("data", {}).get("text/plain", "")))
    return "\n".join(parts)


def exp1_judge_free(text: str) -> None:
    """Part D: pooled and per-family rows, one validator over all four arms."""
    src = f"{NB17.name}, part D"
    rows = []
    pooled = re.compile(r"^\s*(batch|qwen\w+)\s+(\d+)\s+(\d+)\s+(\d+)%\s+([\d.]+)\s+([\d.]+)\s*$")
    family = re.compile(r"^\s*(batch|qwen\w+)\s+(light|electric|heavy)\s+(\d+)\s+(\d+)%\s+([\d.]+)\s+([\d.]+)\s*$")
    for line in text.splitlines():
        if m := family.match(line):
            arm, fam, n, bad, cons, plain = m.groups()
        elif m := pooled.match(line):
            arm, _, n, bad, cons, plain = m.groups()
            fam = "all"
        else:
            continue
        rows.append({"arm": arm, "family": fam, "sections": int(n),
                     "bad_markup": int(bad) / 100, "register_consistency": float(cons),
                     "plain_per_100": float(plain), "source": src})
    assert len(rows) == 16, f"part D: expected 16 rows, got {len(rows)}"
    _write("exp1_judge_free.csv", rows)


def exp1_judged(text: str) -> None:
    """Part E: the GPT-5.6 judge's arm means. Per-script scores were not exported."""
    block = text.split("SCRIPT SCORES (1-10)")[1].split("READ THE SPLIT ARM")[0]
    count = int(re.search(r"(\d+) judgements read back", text).group(1))
    rows = []
    for line in block.splitlines():
        cells = line.split()
        if len(cells) == 8 and (cells[0] == "batch" or cells[0].startswith("qwen")):
            for aspect, value in zip(ASPECTS, cells[1:]):
                rows.append({"arm": cells[0], "aspect": aspect, "score": float(value),
                             "judge": "gpt-5.6-sol", "judgements": count,
                             "source": f"{NB17.name}, part E"})
    assert len(rows) == 28, f"part E: expected 28 rows, got {len(rows)}"
    _write("exp1_judged.csv", rows)


def exp1_keigo_check(text: str) -> None:
    """Part F: morphological keigo measures beside the judge's keigo score."""
    block = text.split("judge keigo")[1].split("THE TWO RANKINGS")[0]
    rows = []
    for line in block.splitlines():
        cells = line.split()
        if len(cells) == 5 and (cells[0] == "batch" or cells[0].startswith("qwen")):
            rows.append({"arm": cells[0], "keigo_density": float(cells[1]),
                         "register_consistency": float(cells[2]),
                         "plain_per_100": float(cells[3]), "judge_keigo": float(cells[4]),
                         "source": f"{NB17.name}, part F"})
    assert len(rows) == 4, f"part F: expected 4 rows, got {len(rows)}"
    _write("exp1_keigo_check.csv", rows)


def exp1_selection(text: str) -> None:
    """Part A: vehicles in both corpora per family, and how many were taken."""
    paired = dict(re.findall(r"^\s*(light|electric|heavy)\s+(\d+) vehicle\(s\) in both corpora",
                             text, re.M))
    selected = dict(re.findall(r"^\s*(light|electric|heavy)\s+(\d+) selected of \d+", text, re.M))
    assert set(paired) == set(selected) == {"light", "electric", "heavy"}, "part A not found"
    _write("exp1_selection.csv", [{"family": f, "paired": int(paired[f]),
                                   "selected": int(selected[f]),
                                   "source": f"{NB17.name}, part A"}
                                  for f in ("light", "electric", "heavy")])


def exp1_compute() -> None:
    """Node time per vehicle from the export's task table."""
    rows = []
    for arm, cfg in ARM_CONFIG.items():
        tasks = json.loads((EXPORT / f"{SOLO_RUN}-{arm}" / "raw" / "tasks.json").read_text())
        timings = json.loads((EXPORT / f"{SOLO_RUN}-{arm}" / "raw" / "vehicle_timings.json").read_text())
        done = [t for t in tasks if t.get("state") == "done"]
        config = yaml.safe_load((PIPELINE / "config" / cfg).read_text())
        models = config["models"]
        assert len(models) == 1, f"{cfg}: expected one model"
        spec = next(iter(models.values()))
        conc = int(spec["concurrency"])
        latency = st.mean(t["gpu_seconds"] for t in done)
        per_vehicle = len(tasks) / len(timings)
        rows.append({
            "arm": arm, "model": spec["hf_id"], "concurrency": conc,
            "vehicles": len(timings), "tasks": len(tasks), "tasks_done": len(done),
            "tasks_per_vehicle": round(per_vehicle, 3),
            "mean_latency_s": round(latency, 3),
            "node_hours_per_vehicle": round(per_vehicle * latency / conc / 3600, 6),
            "input_tokens_per_vehicle": round(sum(t["input_tokens"] for t in done) / len(done) * per_vehicle),
            "output_tokens_per_vehicle": round(sum(t["output_tokens"] for t in done) / len(done) * per_vehicle),
            "source": f"export {SOLO_RUN}-{arm}/raw/tasks.json; config/{cfg}",
        })
    _write("exp1_compute.csv", rows)


def exp1_split() -> None:
    """The split run: GPT-5.6 judge, both sides, light-duty only."""
    run = "h2h-light-20260828-022746"
    judgements = json.loads((EXPORT / run / "raw" / "judgements.json").read_text())
    vehicles = len(json.loads((EXPORT / run / "raw" / "vehicle_timings.json").read_text()))
    rows = []
    for side in ("batch", "ours"):
        means = [j["mean"] for j in judgements if j.get("side") == side and j.get("mean") is not None]
        rows.append({"side": side, "arm": "batch" if side == "batch" else "split",
                     "vehicles": vehicles,
                     "judgements": len(means), "mean": round(st.mean(means), 4),
                     "source": f"export {run}/raw/judgements.json"})
    _write("exp1_split.csv", rows)


def exp1_pairwise() -> None:
    """The split run's pairwise comparison: how many pairs the judge completed."""
    run = "h2h-light-20260828-022746"
    verdicts = json.loads((EXPORT / run / "raw" / "verdicts.json").read_text())
    done = [v for v in verdicts if not v.get("problems")]
    _write("exp1_pairwise.csv", [{
        "attempted": len(verdicts), "completed": len(done),
        "failed": len(verdicts) - len(done),
        "baseline_wins": sum(v.get("winner") == "batch" for v in done),
        "position_bias": sum(bool(v.get("position_bias")) for v in done),
        "source": f"export {run}/raw/verdicts.json"}])


def _node_hours(tasks: list[dict], vehicles: int, conc: dict[str, int]) -> float:
    """Node-hours per vehicle, summed over the models a run uses."""
    total = 0.0
    for model, c in conc.items():
        mine = [t for t in tasks if t.get("model_name") == model]
        done = [t for t in mine if t.get("state") == "done"]
        if done:
            total += len(mine) / vehicles * st.mean(t["gpu_seconds"] for t in done) / c / 3600
    return total


def exp1_stage_tasks() -> None:
    """Tasks per vehicle and stage for the single-model arms (drafts, reviews, ...)."""
    stage = {"intro": "draft", "tech": "draft", "service": "draft", "options": "draft",
             "review": "review", "translate": "translate", "discount": "discount"}
    rows = []
    for arm in ARM_CONFIG:
        tasks = json.loads((EXPORT / f"{SOLO_RUN}-{arm}" / "raw" / "tasks.json").read_text())
        vehicles = len(json.loads((EXPORT / f"{SOLO_RUN}-{arm}" / "raw" / "vehicle_timings.json").read_text()))
        for s in ("draft", "review", "translate", "discount"):
            n = sum(stage[t["task_type"]] == s for t in tasks)
            rows.append({"arm": arm, "stage": s, "tasks_per_vehicle": round(n / vehicles, 3),
                         "source": f"export {SOLO_RUN}-{arm}/raw/tasks.json"})
    _write("exp1_stage_tasks.csv", rows)

def exp1_compute_light() -> None:
    """Light-duty vehicles only: the split design against the single-model arms."""
    rows = []
    split = yaml.safe_load((PIPELINE / "config" / "routing_split.yaml").read_text())["models"]
    tasks = json.loads((EXPORT / SPLIT_RUN / "raw" / "tasks.json").read_text())
    vehicles = len(json.loads((EXPORT / SPLIT_RUN / "raw" / "vehicle_timings.json").read_text()))
    conc = {name: int(spec["concurrency"]) for name, spec in split.items()}
    rows.append({"arm": "split", "vehicles": vehicles,
                 "tasks_per_vehicle": round(len(tasks) / vehicles, 3),
                 "concurrency": "/".join(f"{n}:{c}" for n, c in conc.items()),
                 "node_hours_per_vehicle": round(_node_hours(tasks, vehicles, conc), 6),
                 "moe_task_share": round(sum(t.get("model_name") == "swallow-30b-a3b"
                                             for t in tasks) / len(tasks), 4),
                 "source": f"export {SPLIT_RUN}/raw/tasks.json; config/routing_split.yaml"})
    for arm, cfg in ARM_CONFIG.items():
        spec = next(iter(yaml.safe_load((PIPELINE / "config" / cfg).read_text())["models"].items()))
        tasks = [t for t in json.loads((EXPORT / f"{SOLO_RUN}-{arm}" / "raw" / "tasks.json").read_text())
                 if t.get("recommendation_type") == "light"]
        vehicles = len({tuple(t["custom_id"].split("|")[:3]) for t in tasks})
        conc = {spec[0]: int(spec[1]["concurrency"])}
        rows.append({"arm": arm, "vehicles": vehicles,
                     "tasks_per_vehicle": round(len(tasks) / vehicles, 3),
                     "concurrency": f"{spec[0]}:{spec[1]['concurrency']}",
                     "node_hours_per_vehicle": round(_node_hours(tasks, vehicles, conc), 6),
                     "moe_task_share": "",
                     "source": f"export {SOLO_RUN}-{arm}/raw/tasks.json (light); config/{cfg}"})
    _write("exp1_compute_light.csv", rows)


def exp1_coverage() -> None:
    """Task states per run: done, quarantined (poison), and stranded behind them."""
    rows = []
    for arm, run in [("split", SPLIT_RUN)] + [(a, f"{SOLO_RUN}-{a}") for a in ARM_CONFIG]:
        tasks = json.loads((EXPORT / run / "raw" / "tasks.json").read_text())
        states = {s: sum(t.get("state") == s for t in tasks) for s in ("done", "poison", "pending")}
        assert sum(states.values()) == len(tasks), f"{run}: unexpected task state"
        rows.append({"arm": arm, "tasks": len(tasks), **states,
                     "source": f"export {run}/raw/tasks.json"})
    _write("exp1_coverage.csv", rows)


def exp1_split_markup() -> None:
    """Notebook 13, part D: the one validator over the split run and the baseline."""
    text = _notebook_text(NB13)
    rows = []
    for side, bad, n, bad_x, n_x, kinds in re.findall(
            r"^\s*(ours|batch)\s+(\d+)/(\d+)\s+\d+%\s+(\d+)/(\d+)\s+\d+%\s+(.+)$", text, re.M):
        rows.append({"arm": "split" if side == "ours" else "batch",
                     "bad_sections": int(bad), "sections": int(n),
                     "bad_sections_excluding_intro": int(bad_x),
                     "sections_excluding_intro": int(n_x), "kinds": kinds.strip(),
                     "source": f"{NB13.name}, part D (formatting)"})
    assert len(rows) == 2, f"notebook 13 formatting: expected 2 rows, got {len(rows)}"
    _write("exp1_split_markup.csv", rows)


def exp1_serving() -> None:
    """What changed throughput and output validity, as the status review recorded it."""
    with zipfile.ZipFile(STATUS_DECK) as z:
        def slide(n: int) -> list[str]:
            xml = z.read(f"ppt/slides/slide{n}.xml").decode("utf-8")
            paras = re.findall(r"<a:p>.*?</a:p>", xml, re.S)
            texts = ["".join(re.findall(r"<a:t>([^<]*)</a:t>", p)).strip() for p in paras]
            return [html.unescape(t) for t in texts if t]
        table = slide(9)
        reasoning = slide(8)
    start = table.index("NOW") + 1
    end = next(i for i, t in enumerate(table) if t.startswith("Cold start is per node"))
    cells = table[start:end]
    assert len(cells) % 3 == 0, f"slide 9: {len(cells)} cells"
    rows = [{"measure": cells[i], "before": cells[i + 1], "after": cells[i + 2],
             "source": f"{STATUS_DECK.name}, slide 9"} for i in range(0, len(cells), 3)]
    pct = [t for t in reasoning if re.fullmatch(r"[\d.]+%", t)]
    assert len(pct) == 2, f"slide 8: expected two shares, got {pct}"
    rows.append({"measure": "Sentences without a politeness marker (reasoning on / GPT-4o baseline)",
                 "before": pct[0], "after": pct[1], "source": f"{STATUS_DECK.name}, slide 8"})
    _write("exp1_serving.csv", rows)


def baseline_stages() -> None:
    """Batch telemetry by stage and model: tokens per vehicle and their price."""
    vehicles = json.loads(BATCH_TELEMETRY.read_text())["totals"]["vehicles"]
    price = json.loads(LEGACY_COST.read_text())["pricing_illustrative"]
    rates = {"gpt-4o-mini-2": (MINI_PER_MTOK[0], MINI_PER_MTOK[1])}
    agg: dict = {}
    for path in sorted(BATCH_TELEMETRY.parent.glob(f"{BATCH_TELEMETRY.name[:13]}_telemetry_*.json")):
        for r in json.loads(path.read_text()):
            a = agg.setdefault((r["stage"], r["model_name"]), [0, 0, 0, 0, 0])
            a[0] += 1
            a[1] += int(r.get("input_tokens") or 0)
            a[2] += int(r.get("cached_input_tokens") or 0)
            a[3] += int(r.get("output_tokens") or 0)
            a[4] += int(r.get("request_total") or 0)
    rows = []
    for (stage, model), (jobs, tin, cached, tout, calls) in sorted(agg.items()):
        p_in, p_out = rates.get(model, (price["input_per_mtok"], price["output_per_mtok"]))
        d = price["batch_discount"]
        usd = ((tin - cached) * p_in * d + cached * p_in * d * 0.5 + tout * p_out * d) / 1e6
        rows.append({"stage": stage, "model": model, "jobs": jobs,
                     "input_tokens_per_vehicle": round(tin / vehicles),
                     "cached_tokens_per_vehicle": round(cached / vehicles),
                     "output_tokens_per_vehicle": round(tout / vehicles),
                     "calls_per_vehicle": round(calls / vehicles, 2),
                     "usd_per_vehicle": round(usd / vehicles, 6),
                     "source": f"{BATCH_TELEMETRY.name[:13]}_telemetry_*.json"})
    _write("baseline_stages.csv", rows)


def baseline_stage_calls() -> None:
    """Batch calls per vehicle by truck family and stage, from the job telemetry."""
    tel = json.loads(BATCH_TELEMETRY.read_text())["per_family"]
    rows = []
    for fam, info in tel.items():
        path = BATCH_TELEMETRY.with_name(f"{BATCH_TELEMETRY.name[:13]}_telemetry_{fam}.json")
        calls: dict = {}
        for r in json.loads(path.read_text()):
            calls[r["stage"]] = calls.get(r["stage"], 0) + int(r.get("request_total") or 0)
        for stage, n in sorted(calls.items()):
            rows.append({"family": fam, "stage": stage, "vehicles": info["vehicles"],
                         "calls_per_vehicle": round(n / info["vehicles"], 2),
                         "source": path.name})
    _write("baseline_stage_calls.csv", rows)


def baseline_family_cost() -> None:
    """Batch cost per vehicle per truck family, priced as in baseline_stages."""
    tel = json.loads(BATCH_TELEMETRY.read_text())["per_family"]
    price = json.loads(LEGACY_COST.read_text())["pricing_illustrative"]
    d = price["batch_discount"]
    rows = []
    for fam, info in tel.items():
        path = BATCH_TELEMETRY.with_name(f"{BATCH_TELEMETRY.name[:13]}_telemetry_{fam}.json")
        usd = 0.0
        for r in json.loads(path.read_text()):
            p_in, p_out = (MINI_PER_MTOK if r["model_name"] == "gpt-4o-mini-2"
                           else (price["input_per_mtok"], price["output_per_mtok"]))
            tin, cached = int(r.get("input_tokens") or 0), int(r.get("cached_input_tokens") or 0)
            usd += ((tin - cached) * p_in * d + cached * p_in * d * 0.5
                    + int(r.get("output_tokens") or 0) * p_out * d) / 1e6
        rows.append({"family": fam, "vehicles": info["vehicles"],
                     "usd_per_vehicle": round(usd / info["vehicles"], 6),
                     "source": path.name})
    _write("baseline_family_cost.csv", rows)


def baseline_markup() -> None:
    """Production output of both baselines under the pipeline's validator."""
    d = json.loads(BASELINE_QUALITY.read_text())
    rows = [{"corpus": r["corpus"], "family": r["family"], "scripts": r["vehicles"],
             "sections": r["sections"], "invalid_sections": r["html_invalid"],
             "register_consistency": round(r["register_consistency"], 4),
             "plain_per_100": round(r["plain_per_100_sent"], 3),
             "keigo_density": round(r["keigo_density"], 3),
             "markdown_problems": sum(n for k, n in _kinds(d, r).items() if k.startswith("markdown")),
             "all_problems": sum(_kinds(d, r).values()),
             "source": BASELINE_QUALITY.name} for r in d["rows"]]
    _write("baseline_markup.csv", rows)


def _kinds(d: dict, row: dict) -> dict:
    by_family = d["html_problems"].get(row["corpus"], {})
    if row["family"] == "ALL":
        total: dict = {}
        for fam in by_family.values():
            for k, n in fam.items():
                total[k] = total.get(k, 0) + n
        return total
    return by_family.get(row["family"], {})


def azure_vm_prices() -> None:
    """List prices of the A100 node type in Japan East, from the Azure Retail Prices API."""
    path = SOURCES / "azure_retail_prices" / "NC24ads_A100_v4_japaneast_2026-10-21.json"
    items = json.loads(path.read_text())["Items"]
    rows = [{"meter": i["meterName"], "type": i["type"], "usd_per_hour": i["retailPrice"],
             "effective": i["effectiveStartDate"][:10], "source": path.name}
            for i in items if "Linux" in i["productName"] and i["type"] == "Consumption"]
    _write("azure_vm_prices.csv", rows)


def exp1_context() -> None:
    """Part B of notebook 17: the longest prompt any arm had to accept."""
    m = re.search(r"longest prompt ([\d,]+) chars \(~([\d,]+) tokens", _notebook_text(NB17))
    assert m, "part B: longest prompt not found"
    _write("exp1_context.csv", [{"longest_prompt_chars": int(m.group(1).replace(",", "")),
                                 "longest_prompt_tokens": int(m.group(2).replace(",", "")),
                                 "source": f"{NB17.name}, part B"}])


def baseline_families() -> None:
    """Batch-mode population per truck family, from the production telemetry."""
    d = json.loads(BATCH_TELEMETRY.read_text())
    rows = [{"family": fam, "vehicles": v["vehicles"],
             "requests_per_vehicle": round(v["request_total"] / v["vehicles"], 2),
             "source": BATCH_TELEMETRY.name}
            for fam, v in d["per_family"].items()]
    _write("baseline_families.csv", rows)


def baseline_inputs() -> None:
    """Rows of the input table the baseline profiling read (industry frequencies)."""
    d = json.loads(INDUSTRY_PROFILE.read_text())
    _write("baseline_inputs.csv", [{"input_rows": d["consumption_rows"],
                                    "source": INDUSTRY_PROFILE.name}])


def baseline_cost() -> None:
    """Legacy and batch per vehicle: measured tokens at illustrative list prices."""
    d = json.loads(LEGACY_COST.read_text())
    price = d["pricing_illustrative"]
    note = (f"list ${price['input_per_mtok']}/${price['output_per_mtok']} per Mtok; "
            f"batch discount {price['batch_discount']}")
    rows = [
        {"approach": "legacy", "usd_per_vehicle": d["legacy"]["cost_per_vehicle"],
         "calls_per_vehicle": d["legacy"]["tasks_per_vehicle"], "vehicles": "",
         "input_tokens_per_vehicle": d["legacy"]["input_tokens_per_vehicle"],
         "output_tokens_per_vehicle": d["legacy"]["output_tokens_per_vehicle"],
         "basis": "lower bound: reconstructed input, retries never logged; " + note,
         "source": LEGACY_COST.name},
        {"approach": "batch", "usd_per_vehicle": d["batch_measured"]["cost_per_vehicle"],
         "calls_per_vehicle": d["batch_measured"]["requests_per_vehicle"],
         "vehicles": d["batch_measured"]["vehicles"],
         "input_tokens_per_vehicle": d["batch_measured"]["input_tokens_per_vehicle"],
         "output_tokens_per_vehicle": d["batch_measured"]["output_tokens_per_vehicle"],
         "basis": f"telemetry over {d['batch_measured']['vehicles']} vehicles, per-model prices; " + note,
         "source": LEGACY_COST.name},
    ]
    _write("baseline_cost.csv", rows)


def main() -> None:
    text = _notebook_text(NB17)
    exp1_selection(text)
    exp1_judge_free(text)
    exp1_judged(text)
    exp1_keigo_check(text)
    exp1_compute()
    exp1_split()
    exp1_pairwise()
    exp1_compute_light()
    exp1_stage_tasks()
    exp1_coverage()
    exp1_split_markup()
    exp1_serving()
    baseline_stages()
    baseline_stage_calls()
    baseline_family_cost()
    baseline_markup()
    azure_vm_prices()
    exp1_context()
    baseline_cost()
    baseline_families()
    baseline_inputs()


if __name__ == "__main__":
    main()

"""Command line, the same on a laptop and on the cluster."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import analyze, corpus
from .client import VLLMJudge
from .judge import run as run_judge
from .rubric import LANGS, SCORED_KEYS, render_rubric
from .store import Store

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "experiment.yaml"


def load_config(path: str | Path) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        raise SystemExit(
            "PyYAML is not installed. `pip install -r requirements.txt`, or pass "
            "the values on the command line.")


def _fmt(value: Optional[float], places: int = 3, width: int = 7) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-".rjust(width)
    return f"{value:{width}.{places}f}"


# --------------------------------------------------------------------------
# corpus


def export_reference(xlsx: str | Path, out: str | Path,
                     judge: str = "gpt-5.6-sol") -> int:
    """The reference (GPT-5.6) judge's per-repeat scores, from the workbook into a CSV."""
    try:
        import openpyxl
    except ImportError:
        raise SystemExit("openpyxl is needed to read the workbook: "
                         "pip install openpyxl")
    book = openpyxl.load_workbook(xlsx, data_only=True)
    if "judge_raw" not in book.sheetnames:
        raise SystemExit(f"{xlsx} has no judge_raw sheet; found "
                         f"{book.sheetnames}. The per-repeat rows live there.")
    sheet = book["judge_raw"]
    header = [c.value for c in sheet[1]]
    index = {name: i for i, name in enumerate(header)}
    for needed in ("script_id", "repeat"):
        if needed not in index:
            raise SystemExit(f"judge_raw has no {needed!r} column: {header}")

    rows: List[Dict[str, Any]] = []
    for raw in sheet.iter_rows(min_row=2, values_only=True):
        if raw[index["script_id"]] is None:
            continue
        row = {
            "script_id": str(raw[index["script_id"]]).strip(),
            "judge": judge,
            "rubric_lang": "en",
            "repeat": int(raw[index["repeat"]] or 0),
            "arm": str(raw[index["arm"]] or "") if "arm" in index else "",
            "vin": str(raw[index["vin"]] or "") if "vin" in index else "",
        }
        for aspect in SCORED_KEYS:
            row[aspect] = raw[index[aspect]] if aspect in index else ""
        rows.append(row)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["script_id", "judge", "rubric_lang", "repeat",
                                "arm", "vin", *SCORED_KEYS])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def cmd_corpus(args) -> int:
    config = load_config(args.config)
    spec = config["corpus"]
    scripts = corpus.build(ROOT / spec["rating_form"], ROOT / spec["blind_key"],
                           ROOT / spec["out"])
    print(f"corpus: {len(scripts)} script(s) -> {spec['out']}")
    per_arm: Dict[str, int] = {}
    for script in scripts:
        per_arm[script.arm] = per_arm.get(script.arm, 0) + 1
    for arm, count in sorted(per_arm.items()):
        print(f"  {arm:14} {count}")
    print(f"  {len({s.vin for s in scripts})} distinct vehicle(s), "
          f"{sum(s.chars for s in scripts):,} characters total")

    reference = config.get("reference") or {}
    xlsx = reference.get("scores")
    if xlsx and (ROOT / xlsx).exists():
        out = ROOT / "data" / "reference_scores.csv"
        n = export_reference(ROOT / xlsx, out, reference.get("name", "gpt-5.6-sol"))
        print(f"reference: {n} row(s) -> {out.relative_to(ROOT)}")
    else:
        print(f"reference: {xlsx} not found, skipping")
    return 0


def cmd_prompt(args) -> int:
    """Print a prompt exactly as it goes over the wire."""
    from .rubric import build_messages
    scripts = corpus.load(ROOT / args.corpus)
    script = next((s for s in scripts if s.script_id == args.script_id), None)
    if script is None:
        raise SystemExit(f"no script {args.script_id!r}; have "
                         f"{', '.join(s.script_id for s in scripts)}")
    messages = build_messages(args.lang, script=script.render(), facts="",
                              label=script.script_id)
    for message in messages:
        print(f"===== {message['role'].upper()} =====")
        print(message["content"])
        print()
    return 0


def cmd_rubric(args) -> int:
    print(render_rubric(args.lang))
    return 0


# --------------------------------------------------------------------------
# run


def cmd_run(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    run_spec = config["run"]

    judges = {j["name"]: j for j in config["judges"]}
    if args.judge not in judges:
        raise SystemExit(f"unknown judge {args.judge!r}; config has "
                         f"{', '.join(judges)}")
    spec = judges[args.judge]

    scripts = corpus.load(ROOT / (args.corpus or config["corpus"]["out"]))
    facts_for = None
    if args.facts:
        facts = json.loads((ROOT / args.facts).read_text(encoding="utf-8"))
        missing_facts = [s.script_id for s in scripts if s.script_id not in facts]
        if missing_facts:
            raise SystemExit(f"--facts has no entry for {len(missing_facts)} "
                             f"script(s), e.g. {missing_facts[0]}")
        facts_for = lambda script: facts[script.script_id]      # noqa: E731
    rubric_langs = (args.rubric_langs.split(",") if args.rubric_langs
                    else run_spec["rubric_langs"])
    repeats = int(args.repeats or run_spec["repeats"])
    store = Store(args.db)
    if args.dry_run:
        # One number, cells still to fill.
        from .store import missing
        left = len(list(missing(store, scripts, [spec["name"]], rubric_langs, repeats,
                                temperature=float(run_spec.get("temperature", 0.0)))))
        if args.feedback_db:
            from .feedback import FeedbackStore
            fb_langs = (args.feedback_langs.split(",") if args.feedback_langs
                        else rubric_langs)
            filled = FeedbackStore(args.feedback_db).filled()
            left += sum(1 for s in scripts for lang in fb_langs
                        if (s.script_id, spec["name"], lang, 0) not in filled)
        print(left)
        return 0
    judge = VLLMJudge(args.base_url, model=args.model or spec["hf_id"],
                      name=spec["name"],
                      timeout=float(run_spec.get("request_timeout_seconds", 600)),
                      schema_mode=args.schema_mode)

    print(f"judge   {judge.name}  ({judge.model})")
    print(f"server  {judge.base_url}")
    print(f"corpus  {len(scripts)} script(s)")
    print(f"design  {rubric_langs} x {repeats} repeat(s)"
          f"{', with customer facts' if facts_for else ''}")
    judge.wait_until_ready(timeout=args.ready_timeout, on_line=print)

    written = run_judge(
        judge, scripts, store,
        rubric_langs=rubric_langs,
        repeats=repeats,
        facts_for=facts_for,
        parallelism=int(args.parallelism or run_spec.get("parallelism", 4)),
        max_tokens=int(args.max_tokens or run_spec.get("max_tokens", 8000)),
        temperature=float(run_spec.get("temperature", 0.0)),
        seed=run_spec.get("seed"),
        on_line=print)
    print(f"\nwrote {written} judgement(s) to {args.db}")

    # Feedback pass: same loaded model, separate store.
    if args.feedback_db:
        from .feedback import FeedbackStore, run as run_feedback
        fb_langs = (args.feedback_langs.split(",") if args.feedback_langs
                    else rubric_langs)
        run_feedback(judge, scripts, FeedbackStore(args.feedback_db),
                     langs=fb_langs, repeats=1, facts_for=facts_for,
                     parallelism=int(args.parallelism or run_spec.get("parallelism", 4)),
                     temperature=float(run_spec.get("temperature", 0.0)),
                     seed=run_spec.get("seed"), on_line=print)
    return 0


def cmd_merge(args) -> int:
    store = Store(args.db)
    total = 0
    for other in args.others:
        added = store.merge(other)
        total += added
        print(f"  {other}: +{added} new row(s)")
    print(f"merged into {args.db}, {total} new row(s)")
    return 0


def cmd_status(args) -> int:
    store = Store(args.db)
    rows = store.summary()
    if not rows:
        print(f"{args.db}: empty")
        return 0
    print(f"{args.db}")
    print(f"  {'judge':22} {'lang':5} {'n':>4} {'ok':>4} {'in':>10} {'out':>9} "
          f"{'s/call':>7}")
    for row in rows:
        print(f"  {row['judge']:22} {row['rubric_lang']:5} {row['n']:4d} "
              f"{row['ok'] or 0:4d} {row['input_tokens']:10,} "
              f"{row['output_tokens']:9,} {row['latency'] or 0:7.1f}")
    bad = [r for r in store.read(only_ok=False) if not r.ok]
    if bad:
        print(f"\n  {len(bad)} row(s) with problems:")
        seen: Dict[str, int] = {}
        for record in bad:
            reason = (record.problems or ["?"])[0][:90]
            seen[reason] = seen.get(reason, 0) + 1
        for reason, count in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"    x{count:<3d} {reason}")
    return 0


# --------------------------------------------------------------------------
# report


def _json_safe(value: Any) -> Any:
    """JSON object keys must be strings, and several of ours are tuples."""
    if isinstance(value, dict):
        return {("|".join(str(p) for p in k) if isinstance(k, tuple) else str(k)):
                _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _report(result: Dict[str, Any], out=sys.stdout) -> None:
    def line(text: str = "") -> None:
        print(text, file=out)

    line("=" * 78)
    line("JUDGE VALIDATION")
    line("=" * 78)
    line(f"{result['records']} judgement(s) in the analysis")

    line()
    line("-" * 78)
    line("SELF-AGREEMENT  the ceiling on everything below")
    line("-" * 78)
    line(f"  {'judge / rubric':30} {'scripts':>7} {'identical':>10} "
         f"{'mean gap':>9} {'alpha':>7}")
    for (judge, lang), row in sorted(result["self"].items()):
        line(f"  {judge + ' / ' + lang:30} {row['scripts']:7d} "
             f"{row['identical_share']:9.0%} {row['mean_abs_gap']:9.3f} "
             f"{_fmt(row['alpha'])}")
    line()
    line("  A judge cannot agree with another more than it agrees with itself.")

    for key, title in (("e1_en", "E1  CROSS-JUDGE AGREEMENT, english rubric"),
                       ("e1_ja", "E1  CROSS-JUDGE AGREEMENT, japanese rubric")):
        block = result[key]
        if not block["pairs"]:
            continue
        line()
        line("-" * 78)
        line(title)
        line("-" * 78)
        line(f"  judges: {', '.join(block['judges'])}")
        line(f"  Krippendorff alpha across all of them: {_fmt(block['alpha'])}")
        line()
        line(f"  {'pair':44} {'n':>3} {'rho':>7} {'w/in 1':>7} {'bias':>7} "
             f"{'arm tau':>8}")
        for pair in block["pairs"]:
            name = f"{pair['left']} vs {pair['right']}"
            line(f"  {name:44} {pair['n']:3d} {_fmt(pair['overall_spearman'])} "
                 f"{pair['within_one']:7.0%} {_fmt(pair['bias'])} "
                 f"{_fmt(pair['arm_tau'], 3, 8)}")
        line()
        line("  per aspect (Spearman on per-script means)")
        aspects = [a for a in SCORED_KEYS]
        line(f"  {'pair':32} " + " ".join(f"{a.replace('_score',''):>10}"[:10]
                                          for a in aspects))
        for pair in block["pairs"]:
            name = f"{pair['left'][:14]} vs {pair['right'][:14]}"
            cells = " ".join(_fmt(pair["per_aspect"].get(a), 2, 10) for a in aspects)
            line(f"  {name:32} {cells}")
        line()
        line("  arm means, as each judge ranks them")
        for judge, means in sorted(block["arm_means"].items()):
            order = " > ".join(f"{a} {v:.2f}" for a, v in
                               sorted(means.items(), key=lambda kv: -kv[1]))
            line(f"    {judge:24} {order}")

    if result["e2"]:
        line()
        line("-" * 78)
        line("E2  RUBRIC LANGUAGE  english against japanese, same judge")
        line("-" * 78)
        line(f"  {'judge':24} {'n':>3} {'mean EN':>8} {'mean JA':>8} {'shift':>7} "
             f"{'rho':>7} {'arm tau':>8}")
        for row in result["e2"]:
            line(f"  {row['judge']:24} {row['n']:3d} {row['mean_en']:8.3f} "
                 f"{row['mean_ja']:8.3f} {row['shift']:+7.3f} "
                 f"{_fmt(row['spearman'])} {_fmt(row['arm_tau'], 3, 8)}")
        line()
        line("  shift = japanese minus english. Positive means the japanese")
        line("  rubric scores the same script higher.")
        for row in result["e2"]:
            line()
            line(f"  {row['judge']}")
            line(f"    {'aspect':24} {'EN':>7} {'JA':>7} {'shift':>7} {'rho':>7}")
            for aspect in SCORED_KEYS:
                stats = row["per_aspect"].get(aspect)
                if not stats:
                    continue
                line(f"    {aspect:24} {stats['en']:7.2f} {stats['ja']:7.2f} "
                     f"{stats['shift']:+7.2f} {_fmt(stats['spearman'])}")
            if "arm_order_en" in row:
                line(f"    arm order EN: {' > '.join(row['arm_order_en'])}")
                line(f"    arm order JA: {' > '.join(row['arm_order_ja'])}")
                if row["arm_order_en"] != row["arm_order_ja"]:
                    line("    ORDER CHANGES with the rubric language.")

    human = result.get("human")
    if human:
        line()
        line("-" * 78)
        line("HUMAN RATERS  against every judge")
        line("-" * 78)
        for rater, u in human["range"].items():
            line(f"  {rater}: used {u['distinct']} values ({u['low']}-{u['high']}), "
                 f"{u['modal_share']:.0%} were {u['modal']}")
        line()
        line(f"  {'rater':22} {'judge / rubric':28} {'n':>3} {'rho':>7} "
             f"{'arm tau':>8} {'bias':>7}")
        for row in human["vs_judges"]:
            line(f"  {row['rater'][:22]:22} {row['judge'] + ' / ' + row['lang']:28} "
                 f"{row['n']:3d} {_fmt(row['rho'])} {_fmt(row['arm_tau'], 3, 8)} "
                 f"{_fmt(row['bias'])}")
        line()
        for rater, means in human["arm_means"].items():
            order = " > ".join(f"{a} {v:.2f}" for a, v in
                               sorted(means.items(), key=lambda kv: -kv[1]))
            line(f"  {rater[:22]:22} {order}")
        if human["inter_rater"]:
            line()
            line(f"  inter-rater alpha across {len(human['raters'])} raters: "
                 f"{_fmt(human['inter_rater_alpha'])}")
            for row in human["inter_rater"]:
                line(f"    {row['left']} vs {row['right']}: rho {_fmt(row['rho'])}")
        else:
            line()
            line("  one rater only: inter-rater agreement, the ceiling on everything")
            line("  above, cannot be computed yet.")

    line()
    line("=" * 78)


def cmd_report(args) -> int:
    reference = ROOT / "data" / "reference_scores.csv"
    result = analyze.analyse(args.db, reference if reference.exists() else None,
                             human_dir=ROOT / "data" / "human")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        serialisable = {k: _json_safe(v) for k, v in result.items() if k != "cells"}
        Path(args.json).write_text(
            json.dumps(serialisable, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
        print(f"wrote {args.json}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out).open("w", encoding="utf-8") as handle:
            _report(result, handle)
        print(f"wrote {args.out}")
    if not args.out or args.also_print:
        _report(result)
    return 0


def cmd_figures(args) -> int:
    try:
        from . import figures
    except ImportError as exc:
        raise SystemExit(f"figures need matplotlib: pip install matplotlib ({exc})")
    for path in figures.render_all(Path(args.out)):
        print(f"wrote {path}")
    return 0


# --------------------------------------------------------------------------
# Experiment 4b: categorised feedback


CODEBOOK_DIR = ROOT / "data" / "codebook"


def _judge_names(config) -> List[str]:
    return [j["name"] for j in config["judges"]]


def cmd_codebook_items(args) -> int:
    """Collect every feedback point into data/codebook/ (synced to the cluster)."""
    from . import codebook as cb
    config = load_config(args.config)
    judges = _judge_names(config)
    # Named paths, not out/exp3/*.db: that glob would open generation.db as a judgement
    # store.
    items = cb.collect(
        [str(ROOT / "out" / "judgements.db")]
        + [str(ROOT / "out" / "exp3" / f"{j}.db") for j in judges],
        [str(ROOT / "out" / "feedback" / "*.db"),
         str(ROOT / "out" / "exp3" / "feedback" / "*.db")],
        judges=judges)
    path = cb.write_items(items, CODEBOOK_DIR / "items.jsonl")
    print(f"wrote {len(items)} item(s) to {path}")
    for (judge, source), n in sorted(Counter((i["judge"], i["source"]) for i in items).items()):
        print(f"  {judge:22} {source:26} {n:6d}")
    sheet = CODEBOOK_DIR / "hand_coding.csv"
    coded = 0
    if sheet.exists():
        with sheet.open(encoding="utf-8-sig") as handle:
            coded = sum(1 for r in csv.DictReader(handle) if (r.get("category") or "").strip())
    if args.sample and (not sheet.exists() or coded == 0):
        # An untouched sheet is redrawn, so it covers everything collected so far,
        # including a judge whose feedback arrived later.
        cb.write_hand_sheet(cb.hand_sample(items, n=args.sample), sheet)
        print(f"wrote the hand-coding sheet {sheet} and codebook.csv beside it")
    elif sheet.exists():
        # Never overwrite coding that has started: it may hold hours of work.
        print(f"kept the existing {sheet} ({coded} item(s) already coded)")
    return 0


def cmd_classify(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from . import codebook as cb
    items = cb.read_items(Path(args.items))
    client = VLLMJudge(args.base_url, model=args.model, name="classifier",
                       timeout=600.0, schema_mode=args.schema_mode,
                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    client.wait_until_ready(timeout=args.ready_timeout, on_line=print)
    return cb.classify(client, items, cb.LabelStore(args.db),
                       parallelism=args.parallelism)


def cmd_codebook_report(args) -> int:
    from . import codebook as cb
    items = cb.attach(cb.read_items(Path(args.items)), cb.LabelStore(args.labels).all())
    labelled = [i for i in items if i["category"]]
    print(f"{len(labelled)} of {len(items)} item(s) labelled")
    report: Dict[str, Any] = {"n_items": len(items), "n_labelled": len(labelled)}
    hand = Path(args.hand)
    if hand.exists():
        report["agreement"] = cb.agreement(hand, items)
        a = report["agreement"]
        if a["n"]:
            print(f"human coder, {a['n']} items: kappa {a['kappa']:.3f} "
                  f"(kind level {a['kappa_kind_level']:.3f}), raw {a['raw_agreement']:.2f}")
    # A second, independent LLM coder (blind to the classifier). Reported as such, never
    # as human validation.
    second = Path(args.second_coder)
    if second.exists():
        report["agreement_llm_coder"] = cb.agreement(second, items)
        a = report["agreement_llm_coder"]
        if a["n"]:
            print(f"independent LLM coder ({second.name}), {a['n']} items: kappa "
                  f"{a['kappa']:.3f} (kind level {a['kappa_kind_level']:.3f}), "
                  f"raw {a['raw_agreement']:.2f}")
    views = {"judge": lambda i: i["judge"],
             "judge_lang": lambda i: f"{i['judge']}|{i['lang']}",
             "source": lambda i: f"{i['judge']}|{i['source']}"}
    for name, by in views.items():
        report[name] = cb.profile(labelled, by)
    # Exp 3 link: on the generated scripts, does a judge criticise scripts from Japanese
    # instructions differently?
    key_path = ROOT / "out" / "exp3" / "key.csv"
    if key_path.exists():
        with key_path.open(encoding="utf-8") as handle:
            cond = {r["script_id"]: r["condition"] for r in csv.DictReader(handle)}
        exp3_items = [i for i in labelled if i["script_id"] in cond]
        if exp3_items:
            report["exp3_condition"] = cb.profile(
                exp3_items, lambda i: f"{i['judge']}|{cond[i['script_id']]}")
            print("\nExp 3 scripts, by generation condition (instruction language):")
            for kind, block in report["exp3_condition"].items():
                for judge in sorted({g.split("|")[0] for g in block["shares"]}):
                    en = block["shares"].get(f"{judge}|en", {})
                    ja = block["shares"].get(f"{judge}|ja", {})
                    sub = {g: c for g, c in block["counts"].items() if g.startswith(judge + "|")}
                    test = cb.chi_square({g: Counter(c) for g, c in sub.items()})
                    if test.get("p") is None:
                        continue
                    moved = sorted(set(en) | set(ja), key=lambda c: -abs(ja.get(c, 0) - en.get(c, 0)))[:3]
                    print(f"  {kind:9} {judge:20} chi2 p {test['p']:.2g}  biggest shifts JA vs EN: "
                          + ", ".join(f"{c} {en.get(c, 0):.0%}->{ja.get(c, 0):.0%}" for c in moved))
    for kind, block in report["judge"].items():
        test = block["test"]
        if test.get("chi2") is None:
            continue
        print(f"\n{kind}: chi2 {test['chi2']:.1f}, df {test['df']}, p {test['p']:.2g}, "
              f"Cramer's V {test['cramers_v']:.2f}, n {test['n']}")
        groups = sorted(block["shares"])
        cats = sorted({c for g in groups for c in block["shares"][g]})
        print("  " + " " * 14 + "".join(f"{g[:18]:>20}" for g in groups))
        for c in cats:
            print(f"  {c:14}" + "".join(
                f"{block['shares'][g].get(c, 0):19.1%} " for g in groups))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    try:
        from . import figures
        for path in figures.feedback(report, ROOT / "out" / "figures"):
            print(f"wrote {path}")
    except ImportError:
        print("(figure skipped: matplotlib not installed)")
    return 0


def cmd_exp3_report(args) -> int:
    from . import exp3
    result = exp3.analyse()
    if not result["overall"]:
        print("no Exp 3 judgements yet (out/exp3/<judge>.db); run make gen-fetch")
        return 1
    text = exp3.render(result)
    out = ROOT / "out" / "exp3"
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    (out / "report.json").write_text(json.dumps(_json_safe(result), ensure_ascii=False,
                                                indent=1), encoding="utf-8")
    print(text)
    print(f"\nwrote {out / 'report.txt'} and report.json")
    try:
        from . import figures
        for path in figures.exp3_effects(result, ROOT / "out" / "figures"):
            print(f"wrote {path}")
    except ImportError:
        print("(figure skipped: matplotlib not installed)")
    return 0


def cmd_relative(args) -> int:
    from . import relative
    result = relative.analyse(aspect=args.aspect or None)
    text = relative.render(result)
    suffix = f"_{args.aspect}" if args.aspect else ""
    out = ROOT / "out" / f"relative{suffix}.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nwrote {out}")
    try:
        from . import figures
        for path in figures.relative_agreement(result, ROOT / "out" / "figures"):
            print(f"wrote {path}")
    except ImportError:
        print("(figure skipped: matplotlib not installed)")
    return 0


def cmd_checks(args) -> int:
    from . import checks
    result = checks.analyse()
    text = checks.render(result)
    out = ROOT / "out" / "exp3"
    (out / "checks.txt").write_text(text + "\n", encoding="utf-8")
    (out / "checks.json").write_text(json.dumps(_json_safe(result), ensure_ascii=False,
                                                indent=1), encoding="utf-8")
    print(text)
    print(f"\nwrote {out / 'checks.txt'} and checks.json")
    try:
        from . import figures
        for path in figures.exp3_checks(result, ROOT / "out" / "figures"):
            print(f"wrote {path}")
    except ImportError:
        print("(figure skipped: matplotlib not installed)")
    return 0


def cmd_rating_workbook(args) -> int:
    from .rating_workbook import build
    print(f"wrote {build(Path(args.out))}")
    return 0


# --------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="jv", description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("corpus", help="build the corpus and reference scores")
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("rubric", help="print the rubric")
    p.add_argument("--lang", choices=LANGS, default="en")
    p.set_defaults(func=cmd_rubric)

    p = sub.add_parser("prompt", help="print one full prompt, as sent")
    p.add_argument("script_id")
    p.add_argument("--lang", choices=LANGS, default="en")
    p.add_argument("--corpus", default="data/corpus.json")
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("run", help="fill one judge's cells")
    p.add_argument("--judge", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="")
    p.add_argument("--db", default="out/judgements.db")
    p.add_argument("--corpus", default="")
    # Experiment 3: generated scripts come with the customer facts the generator saw, and
    # a design of their own.
    p.add_argument("--facts", default="",
                   help="JSON {script_id: facts text} handed to the judge")
    p.add_argument("--repeats", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=0,
                   help="answer ceiling; default run.max_tokens from the config")
    p.add_argument("--rubric-langs", default="", help="e.g. en,ja")
    p.add_argument("--feedback-db", default="",
                   help="also run the unscored feedback pass into this store")
    p.add_argument("--feedback-langs", default="", help="default: the rubric langs")
    p.add_argument("--dry-run", action="store_true",
                   help="print how many cells are still empty, call nothing")
    p.add_argument("--parallelism", type=int, default=0)
    p.add_argument("--ready-timeout", type=float, default=3600.0)
    p.add_argument("--schema-mode", default="json_schema",
                   choices=["json_schema", "guided_json", "json_object", "none"])
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("merge", help="fold stores together")
    p.add_argument("--db", default="out/judgements.db")
    p.add_argument("others", nargs="+")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("status", help="what is in a store")
    p.add_argument("--db", default="out/judgements.db")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("report", help="the analysis")
    p.add_argument("--db", default="out/judgements.db")
    p.add_argument("--out", default="")
    p.add_argument("--json", default="")
    p.add_argument("--also-print", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("figures", help="detailed-rating figures (Experiment 4a)")
    p.add_argument("--out", default=str(ROOT / "out" / "figures"))
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("relative", help="which script is better: agreement with "
                       "every rater's scale removed")
    p.add_argument("--aspect", default="", help="one aspect instead of the component mean")
    p.set_defaults(func=cmd_relative)

    p = sub.add_parser("checks", help="Exp 3 robustness: agreement at scale, "
                       "length control, keigo validity, self-agreement")
    p.set_defaults(func=cmd_checks)

    p = sub.add_parser("exp3-report", help="Exp 3: paired EN vs JA analysis")
    p.set_defaults(func=cmd_exp3_report)

    p = sub.add_parser("rating-workbook",
                       help="the human rating form as an Excel workbook")
    p.add_argument("--out", default=str(ROOT / "data" / "rating_form.xlsx"))
    p.set_defaults(func=cmd_rating_workbook)

    p = sub.add_parser("codebook-items", help="collect feedback points (Exp 4b)")
    p.add_argument("--sample", type=int, default=150,
                   help="hand-coding sample size; written once, never overwritten")
    p.set_defaults(func=cmd_codebook_items)

    p = sub.add_parser("classify", help="file feedback points into the codebook")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--items", default=str(CODEBOOK_DIR / "items.jsonl"))
    p.add_argument("--db", default=str(ROOT / "out" / "codebook" / "labels.db"))
    p.add_argument("--parallelism", type=int, default=32)
    p.add_argument("--ready-timeout", type=float, default=1800.0)
    p.add_argument("--schema-mode", default="json_schema",
                   choices=["json_schema", "guided_json", "json_object", "none"])
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("codebook-report", help="category profiles and agreement")
    p.add_argument("--items", default=str(CODEBOOK_DIR / "items.jsonl"))
    p.add_argument("--labels", default=str(ROOT / "out" / "codebook" / "labels.db"))
    p.add_argument("--hand", default=str(CODEBOOK_DIR / "hand_coding.csv"))
    p.add_argument("--second-coder", default=str(CODEBOOK_DIR / "llm_coding_claude.csv"))
    p.add_argument("--out", default=str(ROOT / "out" / "codebook" / "report.json"))
    p.set_defaults(func=cmd_codebook_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

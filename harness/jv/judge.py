"""Running the design: every script, by every judge, under every rubric language."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Sequence

from .client import JudgeError, JudgePermanent, JudgeUnavailable, VLLMJudge
from .corpus import Script
from .rubric import (JSON_SCHEMA, RUBRIC_VERSION, build_messages, component_mean,
                     validate)
from .store import Record, Store, missing

logger = logging.getLogger(__name__)

#: Recorded on every judgement made without a customer record.
NO_CONTEXT = ("NO CONTEXT: no customer record survived the export, so "
              "personalisation_score and persuasion_score are judgements about "
              "prose rather than about this customer. Agreement between judges is "
              "still meaningful; the absolute values are not.")


def score_one(judge: VLLMJudge, script: Script, *, rubric_lang: str, repeat: int,
              facts: str = "", max_tokens: int = 8000, temperature: float = 0.0,
              seed: Optional[int] = None) -> Record:
    """One script, one judge, one rubric language. Never raises for a bad answer."""
    record = Record(script_id=script.script_id, judge=judge.name,
                    rubric_lang=rubric_lang, repeat=repeat, arm=script.arm,
                    vin=script.vin, family=script.family,
                    model_id=judge.model, rubric_version=RUBRIC_VERSION,
                    temperature=temperature)

    rendered = script.render()
    if not rendered.strip():
        record.problems = ["the script was empty"]
        return record

    messages = build_messages(rubric_lang, script=rendered, facts=facts,
                              label=script.script_id)
    # Vary the seed per repeat, or every repeat is the same sample and the repeat-
    # agreement number measures nothing.
    call_seed = None if seed is None else seed + repeat

    try:
        answer = judge.ask(messages, JSON_SCHEMA, max_tokens=max_tokens,
                           temperature=temperature, seed=call_seed)
        payload = answer.json()
    except JudgeUnavailable:
        raise
    except JudgeError as exc:
        record.problems = [f"judge call failed: {str(exc)[:300]}"]
        return record

    record.schema_mode = answer.schema_mode
    record.input_tokens = answer.input_tokens
    record.output_tokens = answer.output_tokens
    record.latency_seconds = answer.latency_seconds

    if not isinstance(payload, dict):
        record.problems = [f"judge returned {type(payload).__name__}, not an object"]
        return record

    record.notes = str(payload.pop("notes", "") or "")
    payload.pop("reasoning", None)
    clean, problems = validate(payload)

    record.scores = {k: v for k, v in clean.items() if k.endswith("_score")}
    record.component_mean = component_mean(record.scores)
    record.has_cta = clean.get("has_cta")
    record.evidence = {k: clean.get(k, []) for k in
                       ("keigo_errors", "generic_spans", "missing_for_call",
                        "objections", "non_japanese_sections")}
    record.rubric_version = clean.get("rubric_version", RUBRIC_VERSION)
    record.problems = problems
    # A caveat, not a defect.
    record.context_note = NO_CONTEXT if not facts.strip() else ""
    return record


def run(judge: VLLMJudge, scripts: Sequence[Script], store: Store, *,
        rubric_langs: Sequence[str] = ("en", "ja"), repeats: int = 3,
        facts_for: Optional[Callable[[Script], str]] = None,
        parallelism: int = 4, max_tokens: int = 8000, temperature: float = 0.0,
        seed: Optional[int] = None, on_line: Callable[[str], None] = print) -> int:
    """Fill every empty cell for this judge. Returns how many were written."""
    todo = [t for t in missing(store, scripts, [judge.name], rubric_langs,
                               repeats, temperature=temperature)]
    if not todo:
        on_line(f"  {judge.name}: nothing to do, all cells filled")
        return 0

    total = len(scripts) * len(rubric_langs) * repeats
    on_line(f"  {judge.name}: {len(todo)} of {total} cell(s) to fill, "
            f"temperature {temperature}, parallelism {parallelism}")

    written = 0
    failed = 0
    lock = threading.Lock()

    def work(item):
        script, _, lang, repeat = item
        facts = facts_for(script) if facts_for else ""
        return score_one(judge, script, rubric_lang=lang, repeat=repeat,
                         facts=facts, max_tokens=max_tokens,
                         temperature=temperature, seed=seed)

    with ThreadPoolExecutor(max_workers=max(1, parallelism)) as pool:
        futures = {pool.submit(work, item): item for item in todo}
        for future in as_completed(futures):
            script, _, lang, repeat = futures[future]
            try:
                record = future.result()
            except JudgeUnavailable as exc:
                # The server is gone.
                on_line(f"  ABORT: {exc}")
                for other in futures:
                    other.cancel()
                raise
            except Exception as exc:                            # noqa: BLE001
                logger.exception("unexpected failure scoring %s", script.script_id)
                record = Record(script_id=script.script_id, judge=judge.name,
                                rubric_lang=lang, repeat=repeat, arm=script.arm,
                                vin=script.vin, family=script.family,
                                problems=[f"harness error: {exc}"])
            with lock:
                store.write(record)
                written += 1
                if not record.ok:
                    failed += 1
                    on_line(f"    {record.script_id} {lang} r{repeat}: "
                            f"{record.problems[0][:110]}")
                if written % 10 == 0:
                    on_line(f"    {written}/{len(todo)} "
                            f"({failed} with problems)")

    on_line(f"  {judge.name}: wrote {written}, {failed} carrying a problem, "
            f"{judge.input_tokens:,} in / {judge.output_tokens:,} out tokens")
    return written

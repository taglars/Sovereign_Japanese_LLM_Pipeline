"""Experiment 3: every synthetic customer's script, under both conditions."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sqlite3
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from . import templates as T
from .customers import Customer
from pipeline.common.prompts.registry import Fidelity, PromptSet, get as get_prompts  # noqa: E402
from pipeline.common.tasks import RecommendationType  # noqa: E402
from pipeline.common.vehicle import (ContextLookup, build_kwargs_from_row,  # noqa: E402
                                     extract_industry, normalise_key,
                                     safe_parse_customer_info)

from jv.client import JudgeError, JudgeUnavailable, VLLMJudge

logger = logging.getLogger(__name__)

CONDITIONS = ("en", "ja")

#: Section label prefixes, as the 12 judged scripts carry them.
LABEL_PREFIX = {"tech": "technical_specs", "options": "options",
                "service": "service_package_financing"}


# --------------------------------------------------------------------------
# sampling


@dataclass(frozen=True)
class Sampling:
    """Sampling of the Qwen3.8 arm (config/routing_qwen38_bf16.yaml), unchanged."""

    temperature: float = 0.34
    top_p: float = 0.9
    top_k: int = -1
    max_tokens: int = 8192
    enable_thinking: bool = False
    #: Sent in the system slot, which production left empty, so user prompts stay
    #: byte-identical.
    system_prompt: str = "/no_think"
    seed: int = 20260924

    def extra_body(self) -> Dict:
        return {"top_p": self.top_p, "top_k": self.top_k,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}}

    @classmethod
    def from_config(cls, raw: Mapping) -> "Sampling":
        fields = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        return cls(**fields)


def _seed(base: int, customer: str, stage: str) -> int:
    """Same seed for the same call in both conditions. Stable across reruns."""
    return (base + zlib.crc32(f"{customer}|{stage}".encode())) % (2 ** 31)


# --------------------------------------------------------------------------
# store


SCHEMA = """
CREATE TABLE IF NOT EXISTS contexts (
    kind TEXT, family TEXT, key TEXT,
    text TEXT, ok INTEGER, error TEXT,
    input_tokens INTEGER, output_tokens INTEGER, latency REAL, ts TEXT,
    PRIMARY KEY (kind, family, key));
CREATE TABLE IF NOT EXISTS calls (
    customer TEXT, condition TEXT, stage TEXT,
    task TEXT, section TEXT, prompt_sha TEXT, prompt_chars INTEGER,
    text TEXT, ok INTEGER, error TEXT,
    input_tokens INTEGER, output_tokens INTEGER, latency REAL, ts TEXT,
    PRIMARY KEY (customer, condition, stage));
"""


class GenStore:
    """SQLite, one connection per thread, WAL so readers never block writers."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=60)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def done(self, customer: str, condition: str, stage: str,
             prompt: Optional[str] = None) -> Optional[str]:
        """The stored answer, or None."""
        row = self._conn().execute(
            "SELECT text, prompt_sha FROM calls WHERE customer=? AND condition=? "
            "AND stage=? AND ok=1", (customer, condition, stage)).fetchone()
        if not row:
            return None
        if prompt is not None and row[1] != hashlib.sha256(prompt.encode()).hexdigest():
            return None
        return row[0]

    def put(self, customer: str, condition: str, stage: str, task: str,
            section: str, prompt: str, text: str, ok: bool, error: str = "",
            input_tokens: int = 0, output_tokens: int = 0,
            latency: float = 0.0) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO calls VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (customer, condition, stage, task, section,
                 hashlib.sha256(prompt.encode()).hexdigest(), len(prompt),
                 text, int(ok), error, input_tokens, output_tokens, latency,
                 time.strftime("%Y-%m-%dT%H:%M:%S")))

    def context(self, kind: str, family: str, key: str) -> Optional[str]:
        row = self._conn().execute(
            "SELECT text FROM contexts WHERE kind=? AND family=? AND key=? AND ok=1",
            (kind, family, key)).fetchone()
        return row[0] if row else None

    def put_context(self, kind: str, family: str, key: str, text: str, ok: bool,
                    error: str = "", input_tokens: int = 0, output_tokens: int = 0,
                    latency: float = 0.0) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO contexts VALUES (?,?,?,?,?,?,?,?,?,?)",
                (kind, family, key, text, int(ok), error, input_tokens,
                 output_tokens, latency, time.strftime("%Y-%m-%dT%H:%M:%S")))

    def rows(self, sql: str, args: Tuple = ()) -> List[Tuple]:
        return self._conn().execute(sql, args).fetchall()

    def summary(self) -> List[Tuple]:
        return self.rows(
            "SELECT condition, task, COUNT(*), SUM(ok), SUM(input_tokens), "
            "SUM(output_tokens), AVG(latency) FROM calls "
            "GROUP BY condition, task ORDER BY condition, task")


# --------------------------------------------------------------------------
# the model


class Generator:
    """VLLMJudge with no schema, the arm's sampling, and a system slot."""

    def __init__(self, base_url: str, model: str, sampling: Sampling, *,
                 timeout: float = 900.0) -> None:
        self.sampling = sampling
        self.client = VLLMJudge(base_url, model, name="generator", timeout=timeout,
                                schema_mode="none",
                                extra_body=sampling.extra_body())

    def __call__(self, prompt: str, seed: Optional[int]):
        messages = []
        if self.sampling.system_prompt:
            messages.append({"role": "system", "content": self.sampling.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.client.ask(messages, {}, max_tokens=self.sampling.max_tokens,
                               temperature=self.sampling.temperature, seed=seed)


# --------------------------------------------------------------------------
# stage 0: industry and prefecture context


def contexts_needed(customers: Iterable[Customer]) -> List[Tuple[str, str, str]]:
    """(kind, family, key) for every context some customer's prompt will read."""
    need = set()
    for c in customers:
        info = safe_parse_customer_info(c.row["customer_info"])
        industry = extract_industry(info)
        if industry:
            need.add(("industry", c.family, industry))
        need.add(("prefecture", c.family, c.row["prefecture"]))
    return sorted(need)


def context_prompt(prompts: PromptSet, kind: str, family: str, key: str) -> str:
    rec = RecommendationType(family)
    fn = prompts.industry_context(rec) if kind == "industry" else prompts.pref_context(rec)
    return fn(key)


def run_contexts(gen: Generator, store: GenStore, prompts: PromptSet,
                 customers: List[Customer], *, workers: int = 32,
                 on_line: Callable[[str], None] = print) -> int:
    todo = [k for k in contexts_needed(customers) if store.context(*k) is None]
    on_line(f"context: {len(todo)} to generate")
    failed = 0

    def work(item):
        kind, family, key = item
        prompt = context_prompt(prompts, kind, family, key)
        try:
            answer = gen(prompt, _seed(gen.sampling.seed, f"ctx:{kind}", f"{family}|{key}"))
        except JudgeUnavailable:
            raise
        except JudgeError as exc:
            store.put_context(kind, family, key, "", False, str(exc)[:500])
            return False
        store.put_context(kind, family, key, answer.text, True, "",
                          answer.input_tokens, answer.output_tokens,
                          answer.latency_seconds)
        return True

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for ok in pool.map(work, todo):
            failed += 0 if ok else 1
    on_line(f"context: done, {failed} failed")
    return failed


def lookup_from(store: GenStore) -> ContextLookup:
    """The production ContextLookup, built from what stage 0 wrote."""
    industry, prefecture = [], []
    for kind, family, key, text in store.rows(
            "SELECT kind, family, key, text FROM contexts WHERE ok=1"):
        if kind == "industry":
            industry.append({"truck_type": family, "industry": key,
                             "generated_industry_context": text})
        else:
            prefecture.append({"truck_type": family, "prefecture": key,
                               "generated_prefecture_context": text})
    return ContextLookup.from_records(industry, prefecture, strict=True)


# --------------------------------------------------------------------------
# stages 1-3: one vehicle, one condition


def _stage(spec: T.PromptSpec) -> str:
    return f"{spec.task}/{spec.section}" if spec.section else spec.task


def run_vehicle(gen: Optional[Generator], store: GenStore, prompts: PromptSet,
                table: Optional[Mapping[str, str]], customer: Customer,
                condition: str, lookup: ContextLookup) -> Tuple[int, int]:
    """Every stage for one (customer, condition). Returns (written, failed)."""
    kwargs = build_kwargs_from_row(customer.row, context=lookup)
    if condition == "ja":
        kwargs = T.ja_kwargs(kwargs, customer.lease, table)

    finals: Dict[Tuple[str, str], str] = {}     # (task, section) -> latest text
    written = failed = 0
    for spec in T.vehicle_graph(prompts, customer.row):
        stage = _stage(spec)
        extra: Dict[str, str] = {}
        if spec.task == "review":
            task_value, section = spec.section.split(":", 1)
            draft = finals.get((task_value, section))
            if draft is None:               # construct failed; nothing to review
                continue
            extra["draft"] = draft
        if spec.task == "discount":
            service = {s: t for (task, s), t in finals.items() if task == "service"}
            point, sales_arg = T.discount_inputs(customer.row, spec, service)
            extra.update(discount_point=point, sales_arg=sales_arg)

        prompt = (T.render(prompts, spec, kwargs, **extra) if condition == "en"
                  else T.render_ja(prompts, spec, kwargs, table, **extra))
        text = store.done(customer.id, condition, stage, prompt)
        if text is None and gen is None:
            return written, failed + 1
        if text is None:
            try:
                answer = gen(prompt, _seed(gen.sampling.seed, customer.id, stage))
            except JudgeUnavailable:
                raise
            except JudgeError as exc:
                store.put(customer.id, condition, stage, spec.task, spec.section,
                          prompt, "", False, str(exc)[:500])
                failed += 1
                continue
            text = answer.text
            store.put(customer.id, condition, stage, spec.task, spec.section,
                      prompt, text, True, "", answer.input_tokens,
                      answer.output_tokens, answer.latency_seconds)
            written += 1

        if spec.task == "review":
            task_value, section = spec.section.split(":", 1)
            finals[(task_value, section)] = text
        elif spec.task in ("intro", "tech", "options", "service"):
            finals[(spec.task, spec.section)] = text
    return written, failed


def pending(store: GenStore, customers: List[Customer], *,
            table: Optional[Mapping[str, str]], conditions=CONDITIONS) -> int:
    """(customer, condition) chains with at least one call still to make."""
    prompts = get_prompts(Fidelity.BATCH_PROD)
    if any(store.context(*k) is None for k in contexts_needed(customers)):
        return len(customers) * len(conditions)
    lookup = lookup_from(store)
    return sum(1 for c in customers for cond in conditions
               if run_vehicle(None, store, prompts, table, c, cond, lookup)[1])


def run_all(gen: Generator, store: GenStore, customers: List[Customer], *,
            table: Optional[Mapping[str, str]], conditions=CONDITIONS,
            workers: int = 96, on_line: Callable[[str], None] = print) -> int:
    """Context first, then every (customer, condition) chain in parallel."""
    prompts = get_prompts(Fidelity.BATCH_PROD)
    if run_contexts(gen, store, prompts, customers, on_line=on_line):
        on_line("context generation had failures; rerun before generating, "
                "or the affected prompts would silently lose their context")
        return 1
    lookup = lookup_from(store)

    jobs = [(c, cond) for c in customers for cond in conditions]
    on_line(f"generation: {len(jobs)} vehicle chain(s), {workers} in flight")
    total_written = total_failed = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(run_vehicle, gen, store, prompts, table, c, cond,
                               lookup): (c.id, cond) for c, cond in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            cid, cond = futures[future]
            try:
                written, failed = future.result()
            except JudgeUnavailable as exc:
                on_line(f"ABORT: {exc}")
                for other in futures:
                    other.cancel()
                return 2
            total_written += written
            total_failed += failed
            if n % 10 == 0 or n == len(jobs):
                rate = total_written / max(1.0, time.monotonic() - started) * 60
                on_line(f"  {n}/{len(jobs)} chains, {total_written} calls, "
                        f"{total_failed} failed, {rate:.0f} calls/min")
    return 1 if total_failed else 0


# --------------------------------------------------------------------------
# export for the judges


def blind_id(customer: str, condition: str, salt: str = "exp3") -> str:
    """Neutral script id. The judge sees it as a label, so it must not say which condition
    wrote the script.
    """
    return "G" + hashlib.sha256(f"{salt}|{customer}|{condition}".encode()).hexdigest()[:7]


def assemble(store: GenStore, prompts: PromptSet, customer: Customer,
             condition: str) -> Optional[List[Tuple[str, str]]]:
    """(label, markup) per section, final text, None if anything is missing."""
    from jv.corpus import strip_markup
    sections: List[Tuple[str, str]] = []
    for spec in T.vehicle_graph(prompts, customer.row):
        if spec.task in ("review", "discount"):
            continue
        construct = store.done(customer.id, condition, _stage(spec))
        review = store.done(customer.id, condition,
                            _stage(T.PromptSpec("review", f"{spec.task}:{spec.section}",
                                                spec.family)))
        reviewed = prompts.review(spec.family, T.TaskType(spec.task), spec.section) \
            if spec.task != "intro" else None
        text = review if reviewed is not None else construct
        if text is None:
            return None
        label = "intro" if spec.task == "intro" else \
            f"{LABEL_PREFIX[spec.task]} / {spec.section}"
        sections.append((label, strip_markup(text)))
    return sections


def export(store: GenStore, customers: List[Customer], out_dir: str | Path,
           conditions=CONDITIONS) -> Dict[str, int]:
    """corpus.json for `jv run --corpus`, facts.json, and the unblinding key."""
    prompts = get_prompts(Fidelity.BATCH_PROD)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus, facts, key = [], {}, []
    incomplete = 0
    for c in customers:
        built = {cond: assemble(store, prompts, c, cond) for cond in conditions}
        if any(v is None for v in built.values()):
            incomplete += 1
            continue
        for cond, sections in built.items():
            sid = blind_id(c.id, cond)
            # arm is recorded for analysis and never reaches the prompt
            # (tests/test_endtoend.py::test_the_arm_never_reaches_the_prompt).
            corpus.append({"script_id": sid, "arm": f"exp3_{cond}", "vin": c.id,
                           "family": c.family, "sections": sections,
                           "chars": sum(len(t) for _, t in sections)})
            facts[sid] = facts_text(c)
            key.append({"script_id": sid, "customer": c.id, "condition": cond,
                        "family": c.family})
    # Shuffle deterministically so file order does not reveal pairing.
    corpus.sort(key=lambda r: r["script_id"])
    (out_dir / "corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
    with (out_dir / "key.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["script_id", "customer",
                                                    "condition", "family"])
        writer.writeheader()
        writer.writerows(sorted(key, key=lambda r: r["script_id"]))
    return {"scripts": len(corpus), "incomplete_customers": incomplete}


def facts_text(customer: Customer) -> str:
    """What the generator knew about this customer, for the judge."""
    row = customer.row
    parts = [
        f"Customer info: {row['customer_info']}",
        f"Prefecture: {row['prefecture']}",
        f"Current vehicle: {row['model_code_current']} {row['current_model_info']}",
        f"Proposed vehicle: {row['model_code_recommended']} {row['proposed_model_info']}",
        f"Customer financial history: {row['finance_history']}",
    ]
    if row.get("tco"):
        parts.append(f"Total cost of ownership: {row['tco']}")
    if row.get("recommended_options_text"):
        parts.append(f"Recommended options: {row['recommended_options_text'].strip()}")
    if row.get("prefecture_subsidy_details"):
        parts.append(f"Prefecture subsidies: {row['prefecture_subsidy_details']}")
    return "\n".join(parts)

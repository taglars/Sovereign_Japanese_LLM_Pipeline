"""Where judgements land. SQLite, because the run has to survive the walltime."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS judgements (
    script_id    TEXT NOT NULL,
    judge        TEXT NOT NULL,
    rubric_lang  TEXT NOT NULL,
    repeat       INTEGER NOT NULL,
    arm          TEXT NOT NULL DEFAULT '',
    vin          TEXT NOT NULL DEFAULT '',
    family       TEXT NOT NULL DEFAULT '',
    scores       TEXT NOT NULL DEFAULT '{}',
    component_mean REAL,
    has_cta      INTEGER,
    evidence     TEXT NOT NULL DEFAULT '{}',
    notes        TEXT NOT NULL DEFAULT '',
    problems     TEXT NOT NULL DEFAULT '[]',
    context_note TEXT NOT NULL DEFAULT '',
    temperature  REAL NOT NULL DEFAULT 0,
    ok           INTEGER NOT NULL DEFAULT 0,
    schema_mode  TEXT NOT NULL DEFAULT '',
    model_id     TEXT NOT NULL DEFAULT '',
    rubric_version TEXT NOT NULL DEFAULT '',
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    latency_seconds REAL NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (script_id, judge, rubric_lang, repeat)
);
CREATE INDEX IF NOT EXISTS judgements_cell ON judgements (judge, rubric_lang);
"""


@dataclass
class Record:
    """One judgement, as it goes in and comes back out."""

    script_id: str
    judge: str
    rubric_lang: str
    repeat: int
    arm: str = ""
    vin: str = ""
    family: str = ""
    scores: Dict[str, int] = field(default_factory=dict)
    component_mean: Optional[float] = None
    has_cta: Optional[bool] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    problems: List[str] = field(default_factory=list)
    #: A caveat about the judgement, NOT a defect in it.
    context_note: str = ""
    #: Sampling temperature this judgement was produced at.
    temperature: float = 0.0
    schema_mode: str = ""
    model_id: str = ""
    rubric_version: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.scores) and not self.problems

    @property
    def cell(self) -> Tuple[str, str, str, int]:
        return (self.script_id, self.judge, self.rubric_lang, self.repeat)


class Store:
    """Every write is its own transaction. A killed job loses at most one call."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            # Older stores have no temperature column; their runs were at temperature 0.
            have = {r[1] for r in conn.execute("PRAGMA table_info(judgements)")}
            if "temperature" not in have:
                conn.execute("ALTER TABLE judgements ADD COLUMN "
                             "temperature REAL NOT NULL DEFAULT 0")
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Concurrent readers while the job writes, so results can be pulled back and
        # looked at without stopping the run.
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def write(self, record: Record) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO judgements
                   (script_id, judge, rubric_lang, repeat, arm, vin, family,
                    scores, component_mean, has_cta, evidence, notes, problems,
                    context_note, temperature, ok, schema_mode, model_id,
                    rubric_version, input_tokens, output_tokens,
                    latency_seconds)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (record.script_id, record.judge, record.rubric_lang, record.repeat,
                 record.arm, record.vin, record.family,
                 json.dumps(record.scores, ensure_ascii=False),
                 record.component_mean,
                 None if record.has_cta is None else int(record.has_cta),
                 json.dumps(record.evidence, ensure_ascii=False),
                 record.notes,
                 json.dumps(record.problems, ensure_ascii=False),
                 record.context_note, float(record.temperature),
                 int(record.ok), record.schema_mode, record.model_id,
                 record.rubric_version, record.input_tokens, record.output_tokens,
                 record.latency_seconds))
            conn.commit()

    def done(self, *, only_ok: bool = True,
             temperature: Optional[float] = None) -> Set[Tuple[str, str, str, int]]:
        """Cells already filled, so a resumed run can skip them."""
        query = "SELECT script_id, judge, rubric_lang, repeat FROM judgements WHERE 1=1"
        args: List[Any] = []
        if only_ok:
            query += " AND ok = 1"
        if temperature is not None:
            query += " AND ABS(temperature - ?) < 1e-9"
            args.append(float(temperature))
        with closing(self._connect()) as conn:
            return {(r[0], r[1], r[2], r[3]) for r in conn.execute(query, args)}

    def read(self, *, judge: Optional[str] = None,
             rubric_lang: Optional[str] = None,
             only_ok: bool = True) -> List[Record]:
        query = "SELECT * FROM judgements WHERE 1=1"
        args: List[Any] = []
        if judge:
            query += " AND judge = ?"
            args.append(judge)
        if rubric_lang:
            query += " AND rubric_lang = ?"
            args.append(rubric_lang)
        if only_ok:
            query += " AND ok = 1"
        query += " ORDER BY judge, rubric_lang, script_id, repeat"
        with closing(self._connect()) as conn:
            return [self._row(r) for r in conn.execute(query, args)]

    @staticmethod
    def _row(row: sqlite3.Row) -> Record:
        return Record(
            script_id=row["script_id"], judge=row["judge"],
            rubric_lang=row["rubric_lang"], repeat=row["repeat"],
            arm=row["arm"], vin=row["vin"], family=row["family"],
            scores=json.loads(row["scores"]),
            component_mean=row["component_mean"],
            has_cta=None if row["has_cta"] is None else bool(row["has_cta"]),
            evidence=json.loads(row["evidence"]),
            notes=row["notes"], problems=json.loads(row["problems"]),
            context_note=row["context_note"],
            temperature=row["temperature"],
            schema_mode=row["schema_mode"], model_id=row["model_id"],
            rubric_version=row["rubric_version"],
            input_tokens=row["input_tokens"], output_tokens=row["output_tokens"],
            latency_seconds=row["latency_seconds"])

    def summary(self) -> List[Dict[str, Any]]:
        """One line per design cell, for the progress print."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """SELECT judge, rubric_lang,
                          COUNT(*) AS n,
                          SUM(ok) AS ok,
                          SUM(input_tokens) AS input_tokens,
                          SUM(output_tokens) AS output_tokens,
                          AVG(latency_seconds) AS latency
                     FROM judgements GROUP BY judge, rubric_lang
                     ORDER BY judge, rubric_lang""").fetchall()
        return [dict(r) for r in rows]

    def merge(self, other: str | Path) -> int:
        """Fold another store into this one. Two judges, two jobs, two files."""
        other = Path(other)
        if not other.exists():
            raise FileNotFoundError(other)
        with closing(self._connect()) as conn:
            conn.execute("ATTACH DATABASE ? AS incoming", (str(other),))
            try:
                mine = [r[1] for r in conn.execute("PRAGMA table_info(judgements)")]
                theirs = [r[1] for r in
                          conn.execute("PRAGMA incoming.table_info(judgements)")]
                shared = [c for c in mine if c in theirs]
                if not shared:
                    raise ValueError(f"{other} has no columns in common")
                names = ", ".join(shared)
                before = conn.execute(
                    "SELECT COUNT(*) FROM judgements").fetchone()[0]
                conn.execute(f"INSERT OR REPLACE INTO judgements ({names}) "
                             f"SELECT {names} FROM incoming.judgements")
                after = conn.execute(
                    "SELECT COUNT(*) FROM judgements").fetchone()[0]
                conn.commit()
                dropped = [c for c in theirs if c not in mine]
                if dropped:
                    logging.getLogger(__name__).warning(
                        "%s has columns this store lacks, ignored: %s",
                        other.name, ", ".join(dropped))
            finally:
                conn.execute("DETACH DATABASE incoming")
        return after - before


def missing(store: Store, scripts: Iterable, judges: Iterable[str],
            langs: Iterable[str], repeats: int,
            temperature: Optional[float] = None) -> List[Tuple[Any, str, str, int]]:
    """The cells of the design that are still empty, in a stable order."""
    have = store.done(only_ok=True, temperature=temperature)
    todo = []
    for judge in judges:
        for lang in langs:
            for repeat in range(repeats):
                for script in scripts:
                    key = (script.script_id, judge, lang, repeat)
                    if key not in have:
                        todo.append((script, judge, lang, repeat))
    return todo

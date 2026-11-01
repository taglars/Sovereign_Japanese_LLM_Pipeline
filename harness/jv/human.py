"""Returned human ratings, read as if the raters were judges."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from .rubric import SCORED_KEYS, component_mean
from .store import Record

HUMAN_DIR = Path(__file__).resolve().parents[1] / "data" / "human"


def _rows_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _rows_xlsx(path: Path) -> List[Dict[str, str]]:
    import openpyxl
    sheet = openpyxl.load_workbook(path, data_only=True).active
    header = [str(c.value or "").strip() for c in sheet[1]]
    out = []
    for raw in sheet.iter_rows(min_row=2, values_only=True):
        out.append({header[i]: raw[i] for i in range(len(header)) if header[i]})
    return out


def read_file(path: str | Path, arms: Dict[str, str]) -> List[Record]:
    """One returned ratings file -> Records. Rows without a script id are skipped."""
    path = Path(path)
    # Decide by content, not by the last extension: Excel happily saves a workbook under a
    # .csv name, and the reverse.
    with path.open("rb") as handle:
        is_zip = handle.read(2) == b"PK"
    if is_zip:
        # The Excel version of the form (jv/rating_workbook.py) has one sheet per script;
        # a workbook saved from the CSV export has one flat sheet.
        from .rating_workbook import is_rating_workbook, read as read_workbook
        rows = read_workbook(path) if is_rating_workbook(path) else _rows_xlsx(path)
    else:
        rows = _rows_csv(path)

    records: List[Record] = []
    for row in rows:
        sid = str(row.get("script_id") or "").strip()
        if not sid:
            continue
        # An empty name field falls back to the file name, minus the form's "ratings_"
        # prefix: ratings_rater2.csv -> "rater2".
        fallback = path.stem.split(".")[0]
        fallback = fallback[len("ratings_"):] if fallback.startswith("ratings_") else fallback
        rater = str(row.get("rater") or fallback).strip() or fallback
        scores = {}
        for aspect in SCORED_KEYS:
            value = row.get(aspect)
            if value is None or str(value).strip() == "":
                continue
            try:
                scores[aspect] = int(float(value))
            except ValueError:
                continue
        if not scores:
            continue
        records.append(Record(
            script_id=sid, judge=f"human:{rater}", rubric_lang="form", repeat=0,
            arm=arms.get(sid, ""), scores=scores,
            component_mean=component_mean(scores),
            notes=str(row.get("comments") or "")))
    return records


def load_all(arms: Dict[str, str], directory: Path = HUMAN_DIR) -> List[Record]:
    """Every rater in data/human/."""
    if not directory.exists():
        return []
    out: List[Record] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in (".csv", ".xlsx") and not path.name.startswith("~"):
            out.extend(read_file(path, arms))
    return out


def range_usage(records: List[Record]) -> Dict[str, Dict[str, float]]:
    """How much of the 1-10 scale each rater actually used."""
    by: Dict[str, List[int]] = {}
    for r in records:
        by.setdefault(r.judge, []).extend(r.scores.values())
    out = {}
    for judge, values in by.items():
        modal = max(set(values), key=values.count)
        out[judge] = {"n": len(values), "distinct": len(set(values)),
                      "low": min(values), "high": max(values), "modal": modal,
                      "modal_share": values.count(modal) / len(values)}
    return out

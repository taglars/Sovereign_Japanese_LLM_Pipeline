"""Experiment 4b: sort what the judges say into categories, then compare judges."""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import random
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .client import JudgeError, JudgeUnavailable, VLLMJudge

ROOT = Path(__file__).resolve().parents[1]
CODEBOOK_VERSION = "cb-1"


@dataclass(frozen=True)
class Category:
    id: str
    kind: str          # strength | weakness | gap
    label: str
    definition: str


CODEBOOK: Tuple[Category, ...] = (
    # strengths
    Category("S_PERSONAL", "strength", "Personalised",
             "Uses this customer's own situation: industry, region, fleet, mileage, "
             "finance history, named company."),
    Category("S_FIGURES", "strength", "Concrete figures",
             "Specific numbers, calculations, savings, specifications that make the "
             "argument checkable."),
    Category("S_PRODUCT", "strength", "Product knowledge",
             "Explains features, packages or services clearly and correctly."),
    Category("S_STRUCTURE", "strength", "Structure and readability",
             "Well organised, easy to scan, sensible order, appropriate length."),
    Category("S_REGISTER", "strength", "Appropriate register",
             "Polite, natural business Japanese; honorifics used well."),
    Category("S_PERSUASION", "strength", "Persuasive framing",
             "Benefit-led, anticipates needs, builds a convincing case."),
    # weaknesses
    Category("W_KEIGO", "weakness", "Honorific errors",
             "Wrong or inconsistent keigo, plain form mixed in, double honorifics, "
             "wrong referent elevated."),
    Category("W_FLUENCY", "weakness", "Unnatural Japanese",
             "Awkward, translated-sounding or ungrammatical Japanese, odd word choice."),
    Category("W_GENERIC", "weakness", "Generic content",
             "Boilerplate that would fit any customer; claimed personalisation that "
             "is not specific."),
    Category("W_ACCURACY", "weakness", "Accuracy or consistency",
             "Wrong or inconsistent facts or numbers, unsupported or exaggerated "
             "claims, internal contradictions."),
    Category("W_FORMAT", "weakness", "Format or language leakage",
             "Broken markup, leftover code fences, English or other non-Japanese "
             "text, labels or artefacts in the body."),
    Category("W_VERBOSE", "weakness", "Verbose or repetitive",
             "Too long, repeats the same point, filler."),
    Category("W_TONE", "weakness", "Wrong tone or audience",
             "Too pushy or too casual, addresses the wrong person, reads as "
             "instructions to the rep rather than a pitch."),
    # gaps
    Category("G_PRICE", "gap", "Price, cost or financing detail",
             "Missing price, quote, total cost, lease terms, payment or discount."),
    Category("G_NEXTSTEP", "gap", "Next step",
             "Missing call to action, proposal of a meeting, test drive, timeline."),
    Category("G_OBJECTION", "gap", "Objection handling",
             "A concern the customer would raise (cost, charging, reliability, "
             "change) that the script does not answer, or missing comparison with "
             "alternatives or competitors."),
    Category("G_PRODUCT", "gap", "Product or evidence detail",
             "Missing specifications, proof, references, case studies, warranty or "
             "service specifics."),
    Category("G_NODATA", "gap", "Customer data unavailable (confound)",
             "The point is that the script lacks information about THIS customer "
             "(their needs, operations, current fleet) that the script could not "
             "have had. Reflects the setup, not the script."),
    Category("OTHER", "other", "Other", "Fits none of the above."),
)
IDS = [c.id for c in CODEBOOK]
BY_ID = {c.id: c for c in CODEBOOK}


# --------------------------------------------------------------------------
# items


def _item_id(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def from_judgements(db: str | Path, *, keys=("missing_for_call", "objections"),
                    judges: Optional[Sequence[str]] = None) -> List[Dict]:
    """Free-text evidence from a scored store, one item per list entry."""
    from .store import Store
    out = []
    for r in Store(db).read(only_ok=True):
        if judges and r.judge not in judges:
            continue
        for key in keys:
            for i, entry in enumerate(r.evidence.get(key) or []):
                text = entry if isinstance(entry, str) else \
                    (entry.get("item") or entry.get("text") or json.dumps(entry, ensure_ascii=False))
                if not str(text).strip():
                    continue
                out.append({"item_id": _item_id("rb", r.judge, r.rubric_lang, r.script_id,
                                                 r.repeat, key, i),
                            "source": f"rubric:{key}", "kind": "gap",
                            "judge": r.judge, "lang": r.rubric_lang,
                            "script_id": r.script_id, "arm": r.arm,
                            "text": str(text).strip(), "span": ""})
    return out


def from_feedback(db: str | Path) -> List[Dict]:
    from .feedback import FeedbackStore, items
    out = []
    for item in items(FeedbackStore(db).rows()):
        item["kind"] = {"strengths": "strength", "weaknesses": "weakness",
                        "gaps": "gap"}[item["kind"]]
        item["item_id"] = _item_id(item["item_id"])
        out.append(item)
    return out


def collect(judgement_globs: Sequence[str], feedback_globs: Sequence[str],
            judges: Optional[Sequence[str]] = None) -> List[Dict]:
    out: List[Dict] = []
    for pattern in judgement_globs:
        for path in sorted(glob.glob(pattern)):
            out.extend(from_judgements(path, judges=judges))
    for pattern in feedback_globs:
        for path in sorted(glob.glob(pattern)):
            out.extend(from_feedback(path))
    seen, unique = set(), []
    for item in out:
        if item["item_id"] not in seen:
            seen.add(item["item_id"])
            unique.append(item)
    return unique


def write_items(items: List[Dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def read_items(path: Path) -> List[Dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --------------------------------------------------------------------------
# classification


BATCH = 25


def _codebook_text() -> str:
    lines = []
    for kind in ("strength", "weakness", "gap", "other"):
        lines.append(f"[{kind}]")
        for c in CODEBOOK:
            if c.kind == kind:
                lines.append(f"  {c.id}: {c.label}. {c.definition}")
    return "\n".join(lines)


def classify_messages(batch: Sequence[Dict]) -> List[Dict]:
    system = (
        "You file reviewer comments about Japanese B2B truck sales scripts into a "
        "fixed codebook. Comments may be in English or Japanese. Assign each "
        "comment exactly ONE category id: the one that best describes what the "
        "comment is ABOUT. The comment's own kind (strength, weakness, gap) is a "
        "strong hint, but a comment filed as a gap that is really about honorifics "
        "is W_KEIGO. Use G_NODATA only when the comment is about missing knowledge "
        "of the customer itself. Use OTHER sparingly.\n\n"
        f"Codebook:\n{_codebook_text()}\n\n"
        "Return one JSON object: {\"labels\": [{\"i\": <index>, \"category\": <id>}, ...]} "
        "with one entry per comment, in order. Nothing else.")
    listing = "\n".join(
        f"{i}. [{item['kind']}] {item['text']}"
        + (f"  (quoted: {item['span'][:80]})" if item.get("span") else "")
        for i, item in enumerate(batch))
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Comments:\n{listing}"}]


def classify_schema(n: int) -> Dict:
    return {"type": "object", "additionalProperties": False, "required": ["labels"],
            "properties": {"labels": {
                "type": "array", "minItems": n, "maxItems": n,
                "items": {"type": "object", "additionalProperties": False,
                          "required": ["i", "category"],
                          "properties": {"i": {"type": "integer"},
                                         "category": {"type": "string", "enum": IDS}}}}}}


class LabelStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with sqlite3.connect(self.path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS labels (text_key TEXT PRIMARY KEY, "
                         "category TEXT, model TEXT, version TEXT, ts TEXT)")

    def all(self) -> Dict[str, str]:
        with sqlite3.connect(self.path) as conn:
            return dict(conn.execute("SELECT text_key, category FROM labels"))

    def put_many(self, rows: Iterable[Tuple[str, str]], model: str) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.executemany("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?)",
                             [(k, c, model, CODEBOOK_VERSION, now) for k, c in rows])


def text_key(item: Dict) -> str:
    """Identical comments are labelled once. Kind is part of the key: the same words filed
    as a strength and as a weakness are different statements.
    """
    return _item_id(item["kind"], item["text"].strip(), (item.get("span") or "").strip())


def classify(client: VLLMJudge, items: List[Dict], store: LabelStore, *,
             parallelism: int = 32, on_line: Callable[[str], None] = print) -> int:
    done = store.all()
    pending: Dict[str, Dict] = {}
    for item in items:
        key = text_key(item)
        if key not in done:
            pending.setdefault(key, item)
    keys = list(pending)
    batches = [keys[i:i + BATCH] for i in range(0, len(keys), BATCH)]
    on_line(f"classify: {len(items)} items, {len(keys)} unlabelled unique, "
            f"{len(batches)} call(s)")
    failed = 0

    def work(batch_keys):
        batch = [pending[k] for k in batch_keys]
        answer = client.ask(classify_messages(batch), classify_schema(len(batch)),
                            max_tokens=4000, temperature=0.0)
        labels = answer.json().get("labels", [])
        by_index = {int(l["i"]): l["category"] for l in labels
                    if isinstance(l, dict) and l.get("category") in IDS}
        return [(k, by_index[i]) for i, k in enumerate(batch_keys) if i in by_index]

    with ThreadPoolExecutor(max_workers=max(1, parallelism)) as pool:
        futures = [pool.submit(work, b) for b in batches]
        for n, future in enumerate(as_completed(futures), 1):
            try:
                rows = future.result()
            except JudgeUnavailable as exc:
                on_line(f"ABORT: {exc}")
                return 2
            except JudgeError as exc:
                failed += 1
                on_line(f"  batch failed: {str(exc)[:120]}")
                continue
            store.put_many(rows, client.model)
            if n % 20 == 0 or n == len(batches):
                on_line(f"  {n}/{len(batches)} batches, {failed} failed")
    return 1 if failed else 0


def attach(items: List[Dict], labels: Dict[str, str]) -> List[Dict]:
    return [dict(item, category=labels.get(text_key(item), "")) for item in items]


# --------------------------------------------------------------------------
# hand coding and agreement


def hand_sample(items: List[Dict], n: int = 150, seed: int = 20260924) -> List[Dict]:
    """Stratified by (judge, kind), so every judge and every list is coded."""
    rng = random.Random(seed)
    strata: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for item in items:
        strata[(item["judge"], item["kind"])].append(item)
    per = max(1, n // max(1, len(strata)))
    chosen = []
    for key in sorted(strata):
        pool = strata[key]
        chosen.extend(rng.sample(pool, min(per, len(pool))))
    rng.shuffle(chosen)
    return chosen[:n]


def write_hand_sheet(sample: List[Dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "kind", "comment", "quoted_span", "category"])
        for item in sample:
            writer.writerow([item["item_id"], item["kind"], item["text"],
                             item.get("span", ""), ""])
    with path.with_name("codebook.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "kind", "label", "definition"])
        for c in CODEBOOK:
            writer.writerow([c.id, c.kind, c.label, c.definition])
    return path


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> Optional[float]:
    if not a or len(a) != len(b):
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def agreement(hand_csv: Path, items: List[Dict]) -> Dict:
    """Kappa between the hand codes and the classifier, on the coded items."""
    by_id = {i["item_id"]: i for i in items}
    human, machine = [], []
    with Path(hand_csv).open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            code = (row.get("category") or "").strip()
            item = by_id.get(row["item_id"])
            if code and item and item.get("category"):
                human.append(code)
                machine.append(item["category"])
    kind = lambda cid: BY_ID[cid].kind if cid in BY_ID else "?"   # noqa: E731
    return {"n": len(human),
            "kappa": cohen_kappa(human, machine),
            "kappa_kind_level": cohen_kappa([kind(h) for h in human],
                                            [kind(m) for m in machine]),
            "raw_agreement": (sum(h == m for h, m in zip(human, machine)) / len(human)
                              if human else None)}


# --------------------------------------------------------------------------
# comparing judges


def _gammaincc(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x). Numerical Recipes 6.2."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        ap = a
        for _ in range(500):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-12:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - gln)
    b = x + 1 - a
    c = 1e300
    d = 1 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-12:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi_square(table: Dict[str, Counter]) -> Dict:
    """Rows = groups (e.g. judges), columns = categories. Test of independence, with
    adjusted standardised residuals to say WHICH cells drive it.
    """
    rows = sorted(table)
    cols = sorted({c for r in rows for c in table[r]})
    total = sum(sum(table[r].values()) for r in rows)
    if len(rows) < 2 or len(cols) < 2 or not total:
        return {"chi2": None, "df": 0, "p": None, "residuals": {}}
    row_sum = {r: sum(table[r].values()) for r in rows}
    col_sum = {c: sum(table[r][c] for r in rows) for c in cols}
    chi2, residuals = 0.0, {}
    for r in rows:
        for c in cols:
            expected = row_sum[r] * col_sum[c] / total
            if expected <= 0:
                continue
            observed = table[r][c]
            chi2 += (observed - expected) ** 2 / expected
            denom = math.sqrt(expected * (1 - row_sum[r] / total) * (1 - col_sum[c] / total))
            residuals[(r, c)] = (observed - expected) / denom if denom else 0.0
    df = (len(rows) - 1) * (len(cols) - 1)
    # Cramer's V: effect size, since with thousands of items p is always tiny.
    v = math.sqrt(chi2 / (total * (min(len(rows), len(cols)) - 1)))
    return {"chi2": chi2, "df": df, "p": _gammaincc(df / 2, chi2 / 2),
            "cramers_v": v, "n": total, "residuals": residuals}


def profile(items: List[Dict], by: Callable[[Dict], str],
            kinds: Sequence[str] = ("strength", "weakness", "gap")) -> Dict:
    """Category shares per group, per kind, plus the test between groups."""
    out = {}
    for kind in kinds:
        table: Dict[str, Counter] = defaultdict(Counter)
        for item in items:
            if item.get("category") and item["kind"] == kind:
                table[by(item)][item["category"]] += 1
        shares = {g: {c: n / sum(cnt.values()) for c, n in cnt.items()}
                  for g, cnt in table.items()}
        out[kind] = {"counts": {g: dict(c) for g, c in table.items()},
                     "shares": shares, "test": chi_square(table)}
    return out

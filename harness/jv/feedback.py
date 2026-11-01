"""Experiment 4b: what each judge says about a script, unscored."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .client import JudgeError, JudgeUnavailable, VLLMJudge
from .corpus import Script
from .rubric import NO_FACTS

FEEDBACK_VERSION = "fb-1"
KINDS = ("strengths", "weaknesses", "gaps")
MAX_POINTS = 8

_POINT = {"type": "object", "additionalProperties": False,
          "required": ["point", "span"],
          "properties": {"point": {"type": "string", "maxLength": 300},
                         # "" for a gap: there is nothing to quote for what is absent
                         "span": {"type": "string", "maxLength": 160}}}

JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": list(KINDS),
    "properties": {k: {"type": "array", "items": _POINT, "maxItems": MAX_POINTS}
                   for k in KINDS},
}

SYSTEM_EN = """You are an expert reviewer of Japanese B2B commercial-vehicle sales \
scripts, with native-level command of business honorifics (敬語) and long experience of \
how fleet operators actually buy trucks.

You are NOT scoring this script. You are giving the author specific feedback.

Rules:
- strengths: what works and should be kept. weaknesses: what is wrong or weak in the \
text as written. gaps: what is missing that the sales representative would need.
- One point is one short sentence about ONE thing. Do not merge several issues.
- For strengths and weaknesses, `span` is an exact quote from the script, at most 60 \
characters, that the point is about. For gaps, `span` is "".
- Give between 2 and {max_points} points per list, as many as are genuinely there. Do \
not pad a list, and do not repeat a point in different words.
- Be concrete. "Good personalisation" is not feedback; say what was personalised.
- You are NOT asked to fact-check product claims. Judge the language and whether the \
script does its job for this customer.
- Write every point in English.

Return ONE JSON object and nothing else: no prose, no markdown fence.

<schema>{schema}</schema>"""

SYSTEM_JA = """あなたは日本語のB2B商用車営業スクリプトをレビューする専門家です。ビジネス\
敬語をネイティブ並みに使いこなし、運送事業者が実際にどのようにトラックを購入するかを\
熟知しています。

このスクリプトを採点するのではありません。作成者に具体的なフィードバックを返します。

ルール:
- strengths は、うまくいっていて残すべき点。weaknesses は、書かれている本文の誤りや\
弱い点。gaps は、営業担当者が必要とするのに欠けている点です。
- 1つのポイントは、1つの事柄についての短い1文です。複数の問題をまとめないでください。
- strengths と weaknesses では、`span` にそのポイントが指すスクリプト中の箇所を60字\
以内でそのまま引用してください。gaps では `span` は "" にしてください。
- 各リストには2個から{max_points}個まで、実際にある分だけ挙げてください。水増しせず、\
同じ指摘を言い換えて繰り返さないでください。
- 具体的に書いてください。「パーソナライズが良い」ではフィードバックになりません。何が\
パーソナライズされていたかを書いてください。
- 製品に関する記述の事実確認は求められていません。言葉遣いと、このスクリプトがこの\
顧客に対して役割を果たしているかを評価してください。
- すべてのポイントを日本語で書いてください。

JSONオブジェクトを1つだけ返し、それ以外は何も書かないでください（説明文やマークダウン\
のコードフェンスは不可）。

<schema>{schema}</schema>"""

TEMPLATE_EN = """<customer_data>
The company this script was written about, and the vehicle it concerns. Background for \
your feedback, not a fact-checking reference.

{facts}
</customer_data>

<script id="{label}">
Each <section> carries an English `label` attribute. Those labels are metadata, not \
part of the script. Give feedback ONLY on the text inside each section.

{script}
</script>

Give your feedback as JSON only."""

TEMPLATE_JA = """<customer_data>
このスクリプトが対象としている企業と、その対象車両です。フィードバックの背景情報で\
あり、事実確認のための参照資料ではありません。

{facts}
</customer_data>

<script id="{label}">
各 <section> には英語の `label` 属性が付いています。これはメタデータであり本文では\
ありません。各 section の中のテキストだけについてフィードバックしてください。

{script}
</script>

フィードバックをJSONのみで返してください。"""

SYSTEM = {"en": SYSTEM_EN, "ja": SYSTEM_JA}
TEMPLATE = {"en": TEMPLATE_EN, "ja": TEMPLATE_JA}


def build_messages(lang: str, *, script: str, facts: str, label: str) -> List[Dict]:
    schema = json.dumps(JSON_SCHEMA, ensure_ascii=False, separators=(",", ":"))
    return [{"role": "system",
             "content": SYSTEM[lang].format(schema=schema, max_points=MAX_POINTS)},
            {"role": "user",
             "content": TEMPLATE[lang].format(facts=facts.strip() or NO_FACTS[lang],
                                              label=label or "unnamed", script=script)}]


def validate(payload) -> Tuple[Dict[str, List[Dict[str, str]]], List[str]]:
    """Keep well-formed points, record what was dropped. Never raises."""
    problems: List[str] = []
    clean: Dict[str, List[Dict[str, str]]] = {}
    if not isinstance(payload, dict):
        return {k: [] for k in KINDS}, [f"not an object: {type(payload).__name__}"]
    for kind in KINDS:
        items = payload.get(kind)
        if not isinstance(items, list):
            problems.append(f"{kind} missing or not a list")
            clean[kind] = []
            continue
        kept = []
        for item in items[:MAX_POINTS]:
            if isinstance(item, str):
                item = {"point": item, "span": ""}
            if not isinstance(item, dict) or not str(item.get("point", "")).strip():
                problems.append(f"{kind}: dropped a malformed point")
                continue
            kept.append({"point": str(item["point"]).strip(),
                         "span": str(item.get("span") or "").strip()})
        clean[kind] = kept
    return clean, problems


# --------------------------------------------------------------------------
# store


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    script_id TEXT, judge TEXT, lang TEXT, repeat INTEGER,
    arm TEXT, vin TEXT, family TEXT,
    payload TEXT, ok INTEGER, problems TEXT,
    model_id TEXT, version TEXT, temperature REAL,
    input_tokens INTEGER, output_tokens INTEGER, latency REAL, ts TEXT,
    PRIMARY KEY (script_id, judge, lang, repeat));
"""


class FeedbackStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with sqlite3.connect(self.path) as conn:
            conn.executescript(SCHEMA_SQL)

    def filled(self) -> set:
        with sqlite3.connect(self.path) as conn:
            return {tuple(r) for r in conn.execute(
                "SELECT script_id, judge, lang, repeat FROM feedback WHERE ok=1")}

    def put(self, row: Dict) -> None:
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO feedback VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (row["script_id"], row["judge"], row["lang"], row["repeat"],
                 row.get("arm", ""), row.get("vin", ""), row.get("family", ""),
                 json.dumps(row.get("payload", {}), ensure_ascii=False),
                 int(row["ok"]), json.dumps(row.get("problems", []), ensure_ascii=False),
                 row.get("model_id", ""), FEEDBACK_VERSION, row.get("temperature", 0.0),
                 row.get("input_tokens", 0), row.get("output_tokens", 0),
                 row.get("latency", 0.0), time.strftime("%Y-%m-%dT%H:%M:%S")))

    def rows(self, only_ok: bool = True) -> List[Dict]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            sql = "SELECT * FROM feedback" + (" WHERE ok=1" if only_ok else "")
            out = []
            for r in conn.execute(sql):
                d = dict(r)
                d["payload"] = json.loads(d["payload"] or "{}")
                d["problems"] = json.loads(d["problems"] or "[]")
                out.append(d)
            return out


# --------------------------------------------------------------------------
# run


def feedback_one(judge: VLLMJudge, script: Script, *, lang: str, repeat: int,
                 facts: str = "", max_tokens: int = 6000, temperature: float = 0.0,
                 seed: Optional[int] = None) -> Dict:
    row = {"script_id": script.script_id, "judge": judge.name, "lang": lang,
           "repeat": repeat, "arm": script.arm, "vin": script.vin,
           "family": script.family, "model_id": judge.model,
           "temperature": temperature, "ok": False, "problems": [], "payload": {}}
    messages = build_messages(lang, script=script.render(), facts=facts,
                              label=script.script_id)
    try:
        answer = judge.ask(messages, JSON_SCHEMA, max_tokens=max_tokens,
                           temperature=temperature,
                           seed=None if seed is None else seed + repeat)
        payload = answer.json()
    except JudgeUnavailable:
        raise
    except JudgeError as exc:
        row["problems"] = [f"call failed: {str(exc)[:300]}"]
        return row
    clean, problems = validate(payload)
    row.update(payload=clean, problems=problems,
               ok=any(clean[k] for k in KINDS),
               input_tokens=answer.input_tokens, output_tokens=answer.output_tokens,
               latency=answer.latency_seconds)
    return row


def run(judge: VLLMJudge, scripts: Sequence[Script], store: FeedbackStore, *,
        langs: Sequence[str] = ("en", "ja"), repeats: int = 1,
        facts_for: Optional[Callable[[Script], str]] = None,
        parallelism: int = 8, max_tokens: int = 6000, temperature: float = 0.0,
        seed: Optional[int] = None, on_line: Callable[[str], None] = print) -> int:
    """Fill every empty feedback cell for this judge. Resumable."""
    filled = store.filled()
    todo = [(s, lang, r) for s in scripts for lang in langs for r in range(repeats)
            if (s.script_id, judge.name, lang, r) not in filled]
    on_line(f"  {judge.name} feedback: {len(todo)} of "
            f"{len(scripts) * len(langs) * repeats} cell(s) to fill")
    written = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, parallelism)) as pool:
        futures = {pool.submit(feedback_one, judge, s, lang=lang, repeat=r,
                               facts=facts_for(s) if facts_for else "",
                               max_tokens=max_tokens, temperature=temperature,
                               seed=seed): (s, lang, r) for s, lang, r in todo}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
            except JudgeUnavailable as exc:
                on_line(f"  ABORT: {exc}")
                for other in futures:
                    other.cancel()
                break
            store.put(row)
            written += 1
            failed += 0 if row["ok"] else 1
            if n % 20 == 0 or n == len(todo):
                on_line(f"    {n}/{len(todo)} ({failed} failed)")
    on_line(f"  {judge.name} feedback: wrote {written}, {failed} failed")
    return written


def items(rows: Iterable[Dict]) -> List[Dict]:
    """One flat record per feedback point, for coding and analysis."""
    out = []
    for r in rows:
        for kind in KINDS:
            for i, p in enumerate(r["payload"].get(kind, [])):
                out.append({"item_id": f"fb|{r['judge']}|{r['lang']}|{r['script_id']}|"
                                       f"{r['repeat']}|{kind}|{i}",
                            "source": "feedback", "kind": kind,
                            "judge": r["judge"], "lang": r["lang"],
                            "script_id": r["script_id"], "arm": r["arm"],
                            "text": p["point"], "span": p["span"]})
    return out

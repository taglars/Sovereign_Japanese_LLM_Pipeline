"""The whole harness against a fake vLLM server, without a GPU."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jv.analyze import analyse                                     # noqa: E402
from jv.client import JudgeError, VLLMJudge                        # noqa: E402
from jv.corpus import Script                                       # noqa: E402
from jv.judge import run as run_judge, score_one                   # noqa: E402
from jv.rubric import JSON_SCHEMA, SCORED_KEYS, build_messages     # noqa: E402
from jv.store import Store                                         # noqa: E402

BEHAVIOUR = {"mode": "good", "seen": []}


def _judgement(offset: int = 0) -> dict:
    payload = {
        "keigo_errors": [{"span": "顧客は", "problem": "呼称が不適切です"}],
        "keigo_score": 6 + offset,
        "fluency_score": 4,
        "generic_spans": [{"span": "最新モデル", "why": "どの顧客にも使えます"}],
        "personalisation_score": 5,
        "missing_for_call": ["見積もり"],
        "readiness_score": 5,
        "objections": ["価格が高い"],
        "persuasion_score": 4 + offset,
        "has_cta": False,
        "non_japanese_sections": [],
        "overall_score": 4,
        "notes": "テスト",
    }
    return payload


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):                                  # quiet
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        BEHAVIOUR["seen"].append(body)
        mode = BEHAVIOUR["mode"]

        if mode == "reject_json_schema" and "response_format" in body \
                and body["response_format"].get("type") == "json_schema":
            self._json(400, {"error": {"message":
                                       "response_format json_schema is not supported"}})
            return

        finish = "stop"
        if mode == "fenced":
            content = "```json\n" + json.dumps(_judgement()) + "\n```"
        elif mode == "fence_inside":
            # A leaked ```html block quoted inside an evidence span.
            payload = _judgement()
            payload["generic_spans"][0]["span"] = "```html <p>foo</p> ```"
            content = json.dumps(payload, ensure_ascii=False)
        elif mode == "range":
            payload = _judgement()
            payload["keigo_score"] = "6-7"
            content = json.dumps(payload, ensure_ascii=False)
        elif mode == "truncated":
            content = json.dumps(_judgement())[:120]
            finish = "length"
        else:
            seed = body.get("seed") or 0
            content = json.dumps(_judgement(offset=seed % 2), ensure_ascii=False)

        self._json(200, {
            "choices": [{"message": {"role": "assistant", "content": content},
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": 9000, "completion_tokens": 700},
        })

    def _json(self, code: int, payload: dict):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class Fake:
    def __enter__(self):
        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def sample_scripts(n: int = 4):
    arms = ["batch", "qwen38_bf16", "qwen36_bf16", "qwen_only"]
    return [Script(script_id=f"S{i:02d}",
                   sections=[("intro", "現在ご使用中のトラックを更新することで、"
                                       "燃料費を削減できます。"),
                             ("technical_specs", "最新モデルは燃費効率が向上して"
                                                 "います。")],
                   arm=arms[i % len(arms)], vin=f"VIN{i:04d}", family="light")
            for i in range(1, n + 1)]


def _judge(url, mode="good", **kw):
    BEHAVIOUR["mode"] = mode
    BEHAVIOUR["seen"].clear()
    return VLLMJudge(url, model="fake/model", name="fake-judge", **kw)


def test_happy_path():
    with Fake() as url:
        judge = _judge(url)
        assert judge.health()
        record = score_one(judge, sample_scripts(1)[0], rubric_lang="en", repeat=0)
        assert record.ok, record.problems
        assert set(record.scores) == set(SCORED_KEYS)
        assert record.component_mean is not None
        assert record.context_note.startswith("NO CONTEXT")
        assert record.input_tokens == 9000


def test_json_schema_is_requested_first():
    with Fake() as url:
        judge = _judge(url)
        score_one(judge, sample_scripts(1)[0], rubric_lang="en", repeat=0)
        assert BEHAVIOUR["seen"][0]["response_format"]["type"] == "json_schema"


def test_degrades_when_schema_mode_rejected():
    with Fake() as url:
        judge = _judge(url, mode="reject_json_schema")
        record = score_one(judge, sample_scripts(1)[0], rubric_lang="en", repeat=0)
        assert record.ok, record.problems
        assert record.schema_mode == "guided_json"


def test_fenced_response_is_tolerated():
    with Fake() as url:
        record = score_one(_judge(url, mode="fenced"), sample_scripts(1)[0],
                           rubric_lang="en", repeat=0)
        assert record.ok, record.problems


def test_fence_inside_a_span_does_not_truncate():
    """A fence quoted inside an evidence span must not cut the JSON."""
    with Fake() as url:
        record = score_one(_judge(url, mode="fence_inside"), sample_scripts(1)[0],
                           rubric_lang="en", repeat=0)
        assert record.ok, record.problems
        assert "```html" in record.evidence["generic_spans"][0]["span"]


def test_a_range_is_recorded_as_a_problem():
    with Fake() as url:
        record = score_one(_judge(url, mode="range"), sample_scripts(1)[0],
                           rubric_lang="en", repeat=0)
        assert not record.ok
        assert "RANGE" in record.problems[0]


def test_truncation_is_named():
    with Fake() as url:
        record = score_one(_judge(url, mode="truncated"), sample_scripts(1)[0],
                           rubric_lang="en", repeat=0)
        assert not record.ok
        assert "max_tokens" in record.problems[0]


def test_japanese_rubric_reaches_the_wire():
    with Fake() as url:
        score_one(_judge(url), sample_scripts(1)[0], rubric_lang="ja", repeat=0)
        system = BEHAVIOUR["seen"][0]["messages"][0]["content"]
        user = BEHAVIOUR["seen"][0]["messages"][1]["content"]
        assert "あなたは日本語" in system
        assert "敬語の正しさと一貫性" in user
        # The schema and the field names stay English, or the two runs would not write the
        # same columns.
        assert "keigo_score" in user


def test_the_arm_never_reaches_the_prompt():
    """A judge told which model wrote a script is not judging blind."""
    scripts = sample_scripts(4)
    for lang in ("en", "ja"):
        for script in scripts:
            messages = build_messages(lang, script=script.render(), facts="",
                                      label=script.script_id)
            blob = json.dumps(messages, ensure_ascii=False)
            for arm in {s.arm for s in scripts}:
                assert arm not in blob, f"{arm} leaked into the {lang} prompt"
            assert script.vin not in blob


def test_full_run_is_resumable_and_analysable():
    scripts = sample_scripts(4)
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "j.db"
        with Fake() as url:
            store = Store(db)
            written = run_judge(_judge(url), scripts, store,
                                rubric_langs=("en", "ja"), repeats=2,
                                parallelism=2, seed=7, on_line=lambda _: None)
            assert written == 4 * 2 * 2

            # Resuming does nothing: every cell is filled.
            again = run_judge(_judge(url), scripts, store,
                              rubric_langs=("en", "ja"), repeats=2,
                              parallelism=2, seed=7, on_line=lambda _: None)
            assert again == 0

        result = analyse(db)
        assert result["records"] == 16
        assert ("fake-judge", "en") in result["cells"]
        assert ("fake-judge", "ja") in result["cells"]
        e2 = result["e2"]
        assert len(e2) == 1 and e2[0]["judge"] == "fake-judge"
        assert e2[0]["n"] == 4


def test_temperature_change_invalidates_resume():
    """A cell judged at another temperature is not a filled cell."""
    scripts = sample_scripts(2)
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        store = Store(db)
        with Fake() as url:
            run_judge(_judge(url), scripts, store, rubric_langs=("en",),
                      repeats=1, temperature=0.0, seed=1, on_line=lambda _: None)
            assert len(store.read()) == 2
            # Same cells, different temperature: must be re-judged, not skipped.
            written = run_judge(_judge(url), scripts, store, rubric_langs=("en",),
                                repeats=1, temperature=0.5, seed=1,
                                on_line=lambda _: None)
            assert written == 2, "temperature change must invalidate the cells"
            # Same temperature again: nothing to do.
            again = run_judge(_judge(url), scripts, store, rubric_langs=("en",),
                              repeats=1, temperature=0.5, seed=1,
                              on_line=lambda _: None)
            assert again == 0
        assert {r.temperature for r in store.read()} == {0.5}


def test_store_merge():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.db", Path(tmp) / "b.db"
        scripts = sample_scripts(2)
        with Fake() as url:
            run_judge(_judge(url), scripts, Store(a), rubric_langs=("en",),
                      repeats=1, on_line=lambda _: None)
            judge = _judge(url)
            judge.name = "other-judge"
            run_judge(judge, scripts, Store(b), rubric_langs=("en",),
                      repeats=1, on_line=lambda _: None)
        merged = Store(a)
        added = merged.merge(b)
        assert added == 2
        assert {r.judge for r in merged.read()} == {"fake-judge", "other-judge"}


def test_unreachable_server_raises_rather_than_looping():
    judge = VLLMJudge("http://127.0.0.1:9", model="m", name="dead", max_retries=1)
    assert not judge.health()
    try:
        score_one(judge, sample_scripts(1)[0], rubric_lang="en", repeat=0)
    except Exception as exc:
        assert "cannot reach" in str(exc)
    else:
        raise AssertionError("a dead server should raise JudgeUnavailable")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except Exception as exc:                                # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {exc}")
    print("end to end: " + ("all checks pass" if not failures
                            else f"{failures} FAILED"))
    raise SystemExit(1 if failures else 0)

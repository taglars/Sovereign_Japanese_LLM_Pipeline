"""The feedback pass, tested without a GPU."""

from __future__ import annotations

import csv
import json
import re
import sys
import tempfile
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jv import codebook as CB                                      # noqa: E402
from jv import feedback as FB                                      # noqa: E402
from jv.client import VLLMJudge                                    # noqa: E402
from jv.corpus import Script                                       # noqa: E402

SEEN = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        user = body["messages"][-1]["content"]
        if user.startswith("Comments:"):
            n = len(re.findall(r"(?m)^\d+\. \[", user))
            cats = ["W_KEIGO", "G_PRICE", "S_PERSONAL", "G_NODATA"]
            content = json.dumps({"labels": [{"i": i, "category": cats[i % 4]}
                                             for i in range(n)]})
        else:
            content = json.dumps({
                "strengths": [{"point": "Uses the customer's mileage.", "span": "1日101km"}],
                "weaknesses": [{"point": "Plain form mixed with keigo.", "span": "できる。"},
                               "a bare string point"],
                "gaps": [{"point": "No price or lease quote.", "span": ""}]},
                ensure_ascii=False)
        raw = json.dumps({"choices": [{"message": {"content": content},
                                       "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 50, "completion_tokens": 10}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class _Server(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


class Fake:
    def __enter__(self):
        self.server = _Server(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def scripts(n=3):
    return [Script(script_id=f"S{i:02d}", sections=[("intro", "本日はご提案に参りました。")],
                   arm="qwen38_bf16", vin=f"V{i}", family="light") for i in range(n)]


def test_feedback_pass_and_resume():
    SEEN.clear()
    with tempfile.TemporaryDirectory() as tmp, Fake() as url:
        store = FB.FeedbackStore(Path(tmp) / "fb.db")
        judge = VLLMJudge(url, "fake", name="fake-judge")
        assert FB.run(judge, scripts(), store, langs=("en", "ja"), on_line=lambda s: None) == 6
        rows = store.rows()
        assert len(rows) == 6 and all(r["ok"] for r in rows)
        # a bare string point is kept, as a point with no span
        assert rows[0]["payload"]["weaknesses"][1] == {"point": "a bare string point", "span": ""}
        # JA system prompt went out for the ja cells
        assert any("フィードバック" in b["messages"][0]["content"] for b in SEEN)
        # schema requested, arm never on the wire
        assert SEEN[0]["response_format"]["type"] == "json_schema"
        assert not any("qwen38" in json.dumps(b, ensure_ascii=False) for b in SEEN)
        before = len(SEEN)
        assert FB.run(judge, scripts(), store, langs=("en", "ja"), on_line=lambda s: None) == 0
        assert len(SEEN) == before
        flat = FB.items(rows)
        assert Counter(i["kind"] for i in flat) == {"strengths": 6, "weaknesses": 12, "gaps": 6}


def test_classify_dedup_and_constraint():
    SEEN.clear()
    items = [{"item_id": f"i{i}", "kind": "gap", "judge": "a" if i % 2 else "b",
              "lang": "en", "script_id": "S", "arm": "x", "source": "t",
              "text": f"comment {i % 30}", "span": ""} for i in range(60)]
    with tempfile.TemporaryDirectory() as tmp, Fake() as url:
        store = CB.LabelStore(Path(tmp) / "labels.db")
        client = VLLMJudge(url, "fake", name="classifier")
        assert CB.classify(client, items, store, parallelism=4, on_line=lambda s: None) == 0
        # 30 unique texts, 25 per call
        assert len(SEEN) == 2
        enum = SEEN[0]["response_format"]["json_schema"]["schema"]["properties"]["labels"][
            "items"]["properties"]["category"]["enum"]
        assert enum == CB.IDS
        labelled = CB.attach(items, store.all())
        assert all(i["category"] in CB.IDS for i in labelled)
        # resumable
        CB.classify(client, items, store, parallelism=4, on_line=lambda s: None)
        assert len(SEEN) == 2


def test_agreement_and_profile():
    items = [{"item_id": f"i{i}", "kind": "gap", "judge": "a" if i < 50 else "b",
              "lang": "en", "text": f"t{i}", "span": "",
              "category": "G_PRICE" if i < 40 or i >= 90 else "G_NODATA"} for i in range(100)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "hand.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            w = csv.writer(handle)
            w.writerow(["item_id", "kind", "comment", "quoted_span", "category"])
            for i in items[:20]:
                w.writerow([i["item_id"], i["kind"], i["text"], "", i["category"]])
            w.writerow(["i45", "gap", "t45", "", "G_PRICE"])      # one disagreement
        a = CB.agreement(path, items)
        assert a["n"] == 21 and a["raw_agreement"] == 20 / 21
    prof = CB.profile(items, lambda i: i["judge"], kinds=("gap",))["gap"]
    assert abs(prof["shares"]["a"]["G_PRICE"] - 0.8) < 1e-9
    assert prof["test"]["p"] < 1e-6 and prof["test"]["df"] == 1


def test_hand_sheet_is_blind_and_never_overwritten():
    items = [{"item_id": f"i{i}", "kind": k, "judge": j, "lang": "en", "text": f"t{i}",
              "span": ""} for i, (j, k) in enumerate(
                  [(j, k) for j in ("a", "b") for k in ("gap", "strength")] * 40)]
    sample = CB.hand_sample(items, n=40)
    assert len(sample) == 40
    assert Counter((s["judge"], s["kind"]) for s in sample) == {
        ("a", "gap"): 10, ("a", "strength"): 10, ("b", "gap"): 10, ("b", "strength"): 10}
    with tempfile.TemporaryDirectory() as tmp:
        path = CB.write_hand_sheet(sample, Path(tmp) / "hand.csv")
        header = path.read_text(encoding="utf-8-sig").splitlines()[0]
        assert header == "item_id,kind,comment,quoted_span,category"   # no judge column


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except Exception as exc:                                # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {exc!r}"[:400])
    print("feedback: " + ("all checks pass" if not failures else f"{failures} FAILED"))
    raise SystemExit(1 if failures else 0)

"""Talking to a vLLM server over its OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Response shapes, strongest guarantee first.
SCHEMA_MODES = ("json_schema", "guided_json", "json_object", "none")

#: Substrings in a 400 that mean "this deployment does not take that parameter".
_UNSUPPORTED = ("unsupported", "not supported", "unrecognized", "unexpected keyword",
                "extra fields not permitted", "invalid_request_error",
                "does not support", "unknown field")


class JudgeError(RuntimeError):
    """A call failed in a way the caller should record, not crash on."""


class JudgeUnavailable(JudgeError):
    """The server is not answering at all. Retrying the batch will not help."""


class JudgePermanent(JudgeError):
    """Retrying this exact request cannot change the outcome."""


@dataclass
class Answer:
    """One completion, with what it cost."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    schema_mode: str = ""

    def json(self) -> Dict[str, Any]:
        """Parsed, tolerating a fence around the whole response."""
        raw = self.text.strip()
        if raw.startswith("```"):
            fence = re.match(r"```(?:json)?\s*(.*?)\s*```$", raw, re.S)
            if fence:
                raw = fence.group(1).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise JudgeError(f"judge returned unparseable JSON: {exc}; "
                             f"first 200 chars: {raw[:200]!r}") from exc


class VLLMJudge:
    """A judge served by vLLM at `base_url`."""

    def __init__(self, base_url: str, model: str, *, name: str = "",
                 timeout: float = 600.0, max_retries: int = 4,
                 schema_mode: str = "json_schema",
                 extra_body: Optional[Dict[str, Any]] = None) -> None:
        self.base_url = base_url.rstrip("/")
        # Sampling fields the judge never sends (top_p, top_k, the chat template's
        # thinking switch).
        self.extra_body = dict(extra_body or {})
        self.model = model
        self.name = name or model
        self.timeout = timeout
        self.max_retries = max_retries
        if schema_mode not in SCHEMA_MODES:
            raise ValueError(f"schema_mode must be one of {SCHEMA_MODES}")
        self._schema_mode = schema_mode
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    # -- transport ---------------------------------------------------------

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise JudgeError(f"HTTP {exc.code} from {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            # Refused means nothing is listening: stop.
            reason = exc.reason
            if isinstance(reason, ConnectionError) and \
                    not isinstance(reason, ConnectionRefusedError):
                raise JudgeError(f"connection dropped: {reason}") from exc
            raise JudgeUnavailable(f"cannot reach {self.base_url}: {reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            # socket.timeout is not a TimeoutError before Python 3.10. Not retried: at
            # temperature 0 the call would run out again; the next chained job tries once more.
            raise JudgePermanent(f"timed out after {self.timeout:.0f}s") from exc
        except ConnectionError as exc:
            # A reset under load is transient; let ask() retry it.
            raise JudgeError(f"connection dropped: {exc}") from exc

    def health(self) -> bool:
        try:
            request = urllib.request.Request(f"{self.base_url}/health")
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status == 200
        except Exception:                                       # noqa: BLE001
            return False

    def wait_until_ready(self, *, timeout: float = 3600.0,
                         poll: float = 10.0, on_line=None) -> None:
        """Block until the server answers, or give up loudly."""
        deadline = time.monotonic() + timeout
        waited = 0.0
        while time.monotonic() < deadline:
            if self.health():
                if on_line and waited:
                    on_line(f"  {self.name} ready after {waited:.0f}s")
                return
            time.sleep(poll)
            waited += poll
            if on_line and waited % 60 == 0:
                on_line(f"  waiting for {self.name}, {waited:.0f}s")
        raise JudgeUnavailable(
            f"{self.name} did not become ready within {timeout:.0f}s. Check the "
            f"vLLM log in the job's stderr: an OOM or a missing weight shows there.")

    # -- one judgement -----------------------------------------------------

    def _body(self, messages: List[Dict], schema: Dict, max_tokens: int,
              temperature: float, seed: Optional[int]) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            body["seed"] = seed
        body.update(self.extra_body)
        mode = self._schema_mode
        if mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "judgement", "schema": schema,
                                "strict": True}}
        elif mode == "guided_json":
            body["guided_json"] = schema
        elif mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        return body

    def ask(self, messages: List[Dict], schema: Dict, *, max_tokens: int = 8000,
            temperature: float = 0.0, seed: Optional[int] = None) -> Answer:
        """One completion, retried on transient failure, degraded on a 400."""
        last: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._ask_once(messages, schema, max_tokens, temperature, seed)
            except (JudgeUnavailable, JudgePermanent):
                raise
            except JudgeError as exc:
                message = str(exc).lower()
                # "reduce the length" is a fact about the request, not a transient
                # failure. The same call will be rejected every time.
                if "maximum context length" in message or "reduce the length" in message:
                    raise JudgePermanent(str(exc)) from exc
                # A 400 naming the constraint parameter means this build wants a weaker
                # form.
                if "http 400" in message and any(m in message for m in _UNSUPPORTED):
                    if self._degrade():
                        logger.info("%s rejected %s, falling back", self.name, message[:80])
                        continue
                last = exc
                if attempt == self.max_retries - 1:
                    break
                wait = min(30.0, 2.0 * (2 ** attempt))
                logger.warning("judge call failed (%s), retrying in %.0fs", exc, wait)
                time.sleep(wait)
        raise JudgeError(f"judge failed after {self.max_retries} attempts: {last}")

    def _degrade(self) -> bool:
        """Step down to the next weaker output constraint. False when exhausted."""
        index = SCHEMA_MODES.index(self._schema_mode)
        if index >= len(SCHEMA_MODES) - 1:
            return False
        self._schema_mode = SCHEMA_MODES[index + 1]
        return True

    def _ask_once(self, messages: List[Dict], schema: Dict, max_tokens: int,
                  temperature: float, seed: Optional[int]) -> Answer:
        started = time.monotonic()
        payload = self._post("/v1/chat/completions",
                             self._body(messages, schema, max_tokens, temperature, seed))
        elapsed = time.monotonic() - started

        choices = payload.get("choices") or []
        if not choices:
            raise JudgeError(f"no choices in response: {str(payload)[:200]}")
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        finish = choices[0].get("finish_reason")
        if finish == "length":
            # Truncated output is not a judgement, it is half of one, and the JSON will
            # not close.
            raise JudgePermanent(
                f"response hit max_tokens ({max_tokens}) and was cut off. "
                f"Raise run.max_tokens or lower rubric MAX_EVIDENCE.")
        if not text.strip():
            raise JudgeError(f"empty content, finish_reason={finish!r}")

        usage = payload.get("usage") or {}
        self.calls += 1
        self.input_tokens += int(usage.get("prompt_tokens") or 0)
        self.output_tokens += int(usage.get("completion_tokens") or 0)
        return Answer(text=text, model=self.model,
                      input_tokens=int(usage.get("prompt_tokens") or 0),
                      output_tokens=int(usage.get("completion_tokens") or 0),
                      latency_seconds=elapsed, schema_mode=self._schema_mode)

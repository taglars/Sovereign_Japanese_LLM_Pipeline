"""The 12 scripts, recovered from the human rating form."""

from __future__ import annotations

import csv
import html
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

#: Where the array literal the form ships its scripts in begins.
_DATA_START = re.compile(r"const\s+DATA\s*=\s*(?=\[)")
_SECTION = re.compile(
    r"<section>\s*<h4>(?P<label>.*?)</h4>\s*"
    r'<div class="jp">(?P<body>.*?)</div>\s*</section>',
    re.S)

#: Tags whose content is not prose and must not be run together with it.
_BLOCK = {"p", "div", "br", "li", "ul", "ol", "tr", "table", "h1", "h2", "h3",
          "h4", "h5", "h6", "section", "blockquote"}


class _TextExtractor(HTMLParser):
    """Tags out, sentence boundaries kept."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCK:
            self._parts.append("\n")

    def text(self) -> str:
        joined = "".join(self._parts)
        joined = re.sub(r"[ \t　]+", " ", joined)
        joined = re.sub(r"\n\s*\n+", "\n", joined)
        return joined.strip()


def strip_markup(text: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(text)
        parser.close()
    except Exception:                                          # noqa: BLE001
        return text
    return parser.text()


@dataclass
class Script:
    """One blind-sampled script, with the arm it came from if the key is loaded."""

    script_id: str
    sections: List[Tuple[str, str]] = field(default_factory=list)
    arm: str = ""
    vin: str = ""
    family: str = ""

    @property
    def chars(self) -> int:
        return sum(len(body) for _, body in self.sections)

    def render(self, max_chars: int = 24000) -> str:
        """The script as the judge reads it."""
        parts: List[str] = []
        used = dropped = 0
        for label, body in self.sections:
            text = body.strip()
            if not text:
                continue
            block = f'<section label="{label}">\n{text}\n</section>'
            if used + len(block) > max_chars:
                dropped += 1
                continue
            parts.append(block)
            used += len(block)
        if dropped:
            parts.append(f"[{dropped} further section(s) omitted for length, do not "
                         f"treat the ending as abrupt]")
        return "\n\n".join(parts)

    def to_dict(self) -> Dict:
        return {"script_id": self.script_id, "arm": self.arm, "vin": self.vin,
                "family": self.family, "sections": self.sections,
                "chars": self.chars}


def read_rating_form(path: str | Path) -> List[Script]:
    """Pull the scripts out of the form, tags stripped, seams kept."""
    raw = Path(path).read_text(encoding="utf-8")
    match = _DATA_START.search(raw)
    if not match:
        raise ValueError(f"no `const DATA = [...]` array found in {path}. "
                         f"Is this the rating form the packet shipped?")
    try:
        entries, _ = json.JSONDecoder().raw_decode(raw, match.end())
    except json.JSONDecodeError as exc:
        raise ValueError(f"the DATA array in {path} is not valid JSON: {exc}") from exc
    if not isinstance(entries, list):
        raise ValueError(f"DATA in {path} is a {type(entries).__name__}, not a list")

    scripts: List[Script] = []
    for entry in entries:
        script_id = str(entry.get("id") or "").strip()
        if not script_id:
            continue
        sections: List[Tuple[str, str]] = []
        for chunk in _SECTION.finditer(entry.get("html") or ""):
            label = strip_markup(chunk.group("label")).strip()
            # Double-escaped: the form escapes the script's own HTML so the rater sees the
            # tags.
            body = strip_markup(html.unescape(chunk.group("body")))
            if body.strip():
                sections.append((label, body))
        if sections:
            scripts.append(Script(script_id=script_id, sections=sections))
    if not scripts:
        raise ValueError(f"the DATA array in {path} parsed but yielded no sections")
    return sorted(scripts, key=lambda s: s.script_id)


def attach_blind_key(scripts: Sequence[Script], path: str | Path) -> List[Script]:
    """Join arm, vin and family onto the scripts."""
    key: Dict[str, Dict[str, str]] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            script_id = (row.get("script_id") or "").strip()
            if script_id:
                key[script_id] = row
    missing = [s.script_id for s in scripts if s.script_id not in key]
    if missing:
        raise ValueError(f"blind key has no row for {', '.join(missing)}")

    # The packet writes this file twice under different column names: the CSV calls them
    # side/rec_type, the workbook sheet calls them arm/family.
    def pick(row: Dict[str, str], *names: str) -> str:
        for name in names:
            value = (row.get(name) or "").strip()
            if value:
                return value
        raise ValueError(
            f"blind key {path} has none of {names} filled in for "
            f"{row.get('script_id')!r}. Columns present: {fieldnames}")

    for script in scripts:
        row = key[script.script_id]
        script.arm = pick(row, "arm", "side")
        script.vin = pick(row, "vin")
        script.family = pick(row, "family", "rec_type")
    return list(scripts)


def build(rating_form: str | Path, blind_key: str | Path,
          out: str | Path) -> List[Script]:
    """Read the form, join the key, write the corpus the cluster job reads."""
    scripts = read_rating_form(rating_form)
    scripts = attach_blind_key(scripts, blind_key)
    payload = [s.to_dict() for s in scripts]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    return scripts


def load(path: str | Path) -> List[Script]:
    """Read back what `build` wrote."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Script(script_id=row["script_id"],
                   sections=[tuple(s) for s in row["sections"]],
                   arm=row.get("arm", ""), vin=row.get("vin", ""),
                   family=row.get("family", ""))
            for row in payload]

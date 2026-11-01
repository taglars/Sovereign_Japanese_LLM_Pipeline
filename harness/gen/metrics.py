"""Deterministic measures of each generated script. No model, so no judge bias."""

from __future__ import annotations

import csv
import re
import statistics as st
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional

from . import runner as R
from . import templates as T
from pipeline.common.prompts.registry import Fidelity, get as get_prompts  # noqa: E402

VOID = {"br", "hr", "img", "meta", "link", "input"}

#: The company's brand, product and feature names, which the prompts require in Latin
#: script.
BRAND_NAMES = r"BRAND_NAMES_WITHHELD"

#: Latin the prompts ask for or the data supplies. Not leakage.
ALLOWED_LATIN = re.compile(
    BRAND_NAMES + r"|"
    r"LDWS|AEBS|ESP|ABA|IHC|TCO|CO2|NOx|SOx|EV|AC|DC|"
    r"KPI|PC|HV|RV|LED|AMT|MT|Bluetooth|"
    r"Active\s+\w+(\s+Assist)?(\s+\d(\.\d)?)?|"
    r"[A-Z0-9]{2,}-[A-Z0-9]{3,}|"                       # model codes, 2RG-FBA60
    # (?![A-Za-z]) rather than \b: Python counts kana as word characters, so \b never
    # fires between "kWh" and "です".
    r"\d+(\.\d+)?\s?(kWh|kW|km/L|km|kg|Nm|dB|L|t|yen)(?![A-Za-z])|"
    r"(?<![A-Za-z])[SML](?![A-Za-z])")
LATIN = re.compile(r"[A-Za-z]")
LETTER = re.compile(r"[A-Za-z぀-ヿ一-鿿]")


class _Checker(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: List[str] = []
        self.errors = 0
        self.li = 0
        self.text: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)
        if tag == "li":
            self.li += 1

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        elif tag in self.stack:
            while self.stack and self.stack[-1] != tag:
                self.stack.pop()
                self.errors += 1
            self.stack.pop()
        else:
            self.errors += 1

    def handle_data(self, data):
        self.text.append(data)


def section_measures(markup: str) -> Dict[str, float]:
    checker = _Checker()
    try:
        checker.feed(markup)
        checker.close()
    except Exception:                                          # noqa: BLE001
        checker.errors += 1
    visible = "".join(checker.text)
    labelled = len(re.findall(r"<li>\s*(?:<b>|<strong>)?\s*[a-j][.．]", markup))
    return {"html_ok": float(checker.errors == 0 and not checker.stack),
            "fence": float("```" in markup),
            "li": checker.li, "li_labelled": labelled,
            "visible": visible}


#: The a./b./c. labels the output rules require are not English prose.
BULLET_LABEL = re.compile(r"(?<![A-Za-z])[a-j][.．)]")


def latin_share(text: str) -> float:
    stripped = ALLOWED_LATIN.sub("", BULLET_LABEL.sub("", text))
    letters = len(LETTER.findall(stripped))
    return len(LATIN.findall(stripped)) / letters if letters else 0.0


def _keigo_scorer():
    try:
        import sys
        sys.path.insert(0, str(T.VENDOR))
        from eval.honorific_scoring.keigo import HonorificScorer
        return HonorificScorer()
    except Exception:                                          # noqa: BLE001
        return None


def script_rows(store: R.GenStore, customers) -> List[Dict]:
    prompts = get_prompts(Fidelity.BATCH_PROD)
    scorer = _keigo_scorer()
    rows = []
    for c in customers:
        for cond in R.CONDITIONS:
            sections = _final_markup(store, prompts, c, cond)
            if sections is None:
                continue
            measures = [section_measures(m) for _, m in sections]
            lists = [m for m in measures if m["li"]]
            visible = "\n".join(m["visible"] for m in measures)
            row = {"customer": c.id, "family": c.family, "condition": cond,
                   "sections": len(sections), "chars": len(visible),
                   "html_ok_share": st.mean(m["html_ok"] for m in measures),
                   "fence_leaks": int(sum(m["fence"] for m in measures)),
                   "bullet_share": (st.mean(min(1.0, m["li_labelled"] / m["li"])
                                            for m in lists) if lists else ""),
                   "latin_share": round(latin_share(visible), 4)}
            if scorer is not None:
                score = scorer.score("\n".join(m for _, m in sections), is_html=True)
                d = score.as_dict()
                row.update({f"keigo_{k}": d.get(k) for k in (
                    "register_consistency", "defects_per_100_sentences",
                    "keigo_density", "referent_honorifics_per_100_sentences")})
            rows.append(row)
    return rows


def _final_markup(store, prompts, customer, condition):
    """Like runner.assemble, but keeps the markup the metrics look at."""
    out = []
    for spec in T.vehicle_graph(prompts, customer.row):
        if spec.task in ("review", "discount"):
            continue
        stage = R._stage(spec)
        review_stage = R._stage(T.PromptSpec("review", f"{spec.task}:{spec.section}",
                                             spec.family))
        reviewed = (prompts.review(spec.family, T.TaskType(spec.task), spec.section)
                    if spec.task != "intro" else None)
        text = store.done(customer.id, condition,
                          review_stage if reviewed is not None else stage)
        if text is None:
            return None
        out.append((stage, text))
    return out


def write(store: R.GenStore, customers, path: Path) -> Path:
    rows = script_rows(store, customers)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (
        ["customer", "family", "condition"].index(k)
        if k in ("customer", "family", "condition") else 9, k))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def summary(path: Path) -> str:
    """Paired means, EN vs JA, over customers present in both."""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by = {}
    for r in rows:
        by.setdefault(r["customer"], {})[r["condition"]] = r
    pairs = [v for v in by.values() if "en" in v and "ja" in v]
    keys = [k for k in rows[0] if k not in ("customer", "family", "condition")] if rows else []
    lines = [f"{len(pairs)} paired customer(s)",
             f"  {'measure':42} {'EN':>9} {'JA':>9} {'JA-EN':>9}"]
    for k in keys:
        try:
            en = [float(p["en"][k]) for p in pairs if p["en"][k] != "" and p["ja"][k] != ""]
            ja = [float(p["ja"][k]) for p in pairs if p["en"][k] != "" and p["ja"][k] != ""]
        except ValueError:
            continue
        if not en:
            continue
        diff = st.mean(j - e for e, j in zip(en, ja))
        lines.append(f"  {k:42} {st.mean(en):9.3f} {st.mean(ja):9.3f} {diff:+9.3f}")
    return "\n".join(lines)

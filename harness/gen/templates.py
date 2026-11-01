"""Production prompts as templates, and the Japanese arm built from them.

Each prompt is rendered with a sentinel in place of every value. The text between
sentinels is instruction text, translated line by line through a table; the data is
put back unchanged, so it never passes through translation.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

VENDOR = Path(__file__).resolve().parent / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from pipeline.common.prompts.registry import Fidelity, PromptSet, get as get_prompts  # noqa: E402
from pipeline.common.tasks import RecommendationType, TaskType, sections_for  # noqa: E402
from pipeline.common.vehicle import has_value  # noqa: E402

#: The production `lang` directive, verbatim from planner._lang_directive, including the
#: missing space after the first full stop.
LANG_EN = ("Your output must be in coherent, standard, natural, concise, "
           "technical Japanese."
           "[brand directive withheld from this copy, see README]")

#: Every value a prompt function can receive.
KWARG_KEYS = (
    "industry_pref_context", "current", "proposed", "customer", "tco",
    "prefecture_subsidy_details", "municipality_subsidy_details", "prefecture",
    "industry", "current_code", "proposed_code", "options", "history",
    "leasing", "leasing_package", "options_model_text", "options_industry_text",
    "cold_climate", "number_of_subsidies_in_this_prefecture",
)
POSITIONAL = ("lang", "draft", "discount_point", "sales_arg")

_OPEN, _CLOSE = "⟦", "⟧"          # ⟦ ⟧, never present in prompt text
_SENTINEL = re.compile(re.escape(_OPEN) + r"([a-z_]+)" + re.escape(_CLOSE))


def sentinel(key: str) -> str:
    return f"{_OPEN}{key}{_CLOSE}"


def sentinel_kwargs() -> Dict[str, str]:
    return {k: sentinel(k) for k in KWARG_KEYS}


@dataclass(frozen=True)
class PromptSpec:
    """One prompt in a vehicle's stage graph, independent of any customer."""

    task: str            # intro | tech | options | service | review | discount
    section: str         # the section label, "" for intro/discount
    family: RecommendationType

    @property
    def key(self) -> str:
        return f"{self.family.value}/{self.task}/{self.section}"


def stage_graph(prompts: PromptSet, family: RecommendationType) -> List[PromptSpec]:
    """Every prompt the planner emits for a vehicle of this family."""
    out = [PromptSpec("intro", "", family)]
    for task_type in (TaskType.TECH, TaskType.OPTIONS, TaskType.SERVICE):
        for section in sections_for(family, task_type):
            out.append(PromptSpec(task_type.value, section, family))
            if prompts.review(family, task_type, section) is not None:
                out.append(PromptSpec("review", f"{task_type.value}:{section}", family))
    out.append(PromptSpec("discount", "", family))
    return out


def applies(spec: PromptSpec, row: Mapping) -> bool:
    """planner._section_applies, verbatim in behaviour: which options sections a
    particular vehicle gets.
    """
    if spec.task != "options":
        return True
    if spec.section == "Popular for this region":
        return bool(row.get("cold_climate"))
    if not has_value(row.get("recommended_options_text")):
        return False
    if spec.section == "Popular for this industry":
        return has_value(row.get("recommended_options_text_industry"))
    if spec.section == "Popular for this model":
        return has_value(row.get("recommended_options_text_model"))
    return True


def _reviewed_section(spec: PromptSpec) -> str:
    return spec.section.split(":", 1)[1] if spec.task == "review" else spec.section


def vehicle_graph(prompts: PromptSet, row: Mapping) -> List[PromptSpec]:
    """The stage graph for one customer, as the company generated it."""
    family = RecommendationType(row["recommendation_type"])
    out = []
    for spec in stage_graph(prompts, family):
        if spec.task == "discount":
            continue
        base = spec if spec.task != "review" else PromptSpec(
            spec.section.split(":", 1)[0], _reviewed_section(spec), family)
        if applies(base, row):
            out.append(spec)
    for i, category in enumerate(row.get("discount_extra_info") or []):
        for j, _ in enumerate(category["discount_info"]):
            out.append(PromptSpec("discount", f"{i}.{j}", family))
    return out


def discount_inputs(row: Mapping, spec: PromptSpec,
                    service: Mapping[str, str]) -> Tuple[str, str]:
    """(points, sales argument) for one discount item, as the company passed them: the
    item's argumentation string, and the generated service section.
    """
    i, j = (int(x) for x in spec.section.split("."))
    info = row["discount_extra_info"][i]["discount_info"][j]
    return info["discount_argumentation"], str(dict(service))


def _call(prompts: PromptSet, spec: PromptSpec, kwargs: Mapping[str, str],
          lang: str, draft: str, discount_point: str, sales_arg: str,
          output_language: str) -> str:
    """Invoke the production prompt function exactly as the renderer does."""
    family = spec.family
    if spec.task == "intro":
        return prompts.intro(family)(lang, **kwargs)
    if spec.task in ("tech", "options", "service"):
        fn = prompts.construct(family, TaskType(spec.task), spec.section)
        return fn(lang, **kwargs)
    if spec.task == "review":
        task_value, section = spec.section.split(":", 1)
        fn = prompts.review(family, TaskType(task_value), section)
        return fn(draft, lang, **kwargs)
    if spec.task == "discount":
        return prompts.discount()(discount_point, sales_arg, output_language)
    raise ValueError(f"unknown task {spec.task!r}")


def render(prompts: PromptSet, spec: PromptSpec, kwargs: Mapping[str, str], *,
           draft: str = "", discount_point: str = "", sales_arg: str = "",
           lang: str = LANG_EN, output_language: str = "Japanese") -> str:
    """The production prompt with real data. This IS the EN condition."""
    return _call(prompts, spec, kwargs, lang, draft, discount_point, sales_arg,
                 output_language)


def template(prompts: PromptSet, spec: PromptSpec) -> str:
    """The prompt with a sentinel wherever data would go."""
    return _call(prompts, spec, sentinel_kwargs(), sentinel("lang"),
                 sentinel("draft"), sentinel("discount_point"),
                 sentinel("sales_arg"), "Japanese")


def segments(text: str) -> List[str]:
    """Instruction text between sentinels. Whitespace-only runs are dropped."""
    return [s for s in _SENTINEL.split(text)[0::2] if s.strip()]


def fill(text: str, values: Mapping[str, str]) -> str:
    """Put real data back where the sentinels are, as str() of each value."""
    def swap(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"no value for sentinel {key!r}")
        return str(values[key])
    return _SENTINEL.sub(swap, text)


# --------------------------------------------------------------------------
# the translation table


TABLE_PATH = Path(__file__).resolve().parent / "translations" / "batch_prod_ja.json"

#: A line is instruction prose, and needs translating, when Latin script dominates it.
_LATIN = re.compile(r"[A-Za-z]")
_JAPANESE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


def _units(part: str) -> List[Tuple[str, str, str]]:
    """Split instruction text into (leading ws, content, trailing ws) per line."""
    out = []
    for line in part.split("\n"):
        stripped = line.strip()
        lead = line[:len(line) - len(line.lstrip())]
        trail = line[len(line.rstrip()):] if stripped else ""
        out.append((lead, stripped, trail))
    return out


def needs_translation(unit: str) -> bool:
    latin = len(_LATIN.findall(unit))
    return latin > 0 and latin >= len(_JAPANESE.findall(unit))


def collect_units(prompts: Optional[PromptSet] = None) -> List[str]:
    """Every unique line of instruction text that needs translating."""
    prompts = prompts or get_prompts(Fidelity.BATCH_PROD)
    seen: Dict[str, None] = {}
    for family in RecommendationType:
        for spec in stage_graph(prompts, family):
            for part in _SENTINEL.split(template(prompts, spec))[0::2]:
                for _, content, _ in _units(part):
                    if content and needs_translation(content):
                        seen.setdefault(content, None)
    # The lang directive is data to the functions but instruction to the model.
    seen.setdefault(LANG_EN, None)
    # So are the lease texts: instruction blocks stored in the consumption table.
    for text in lease_units():
        for _, content, _ in _units(text):
            if content and needs_translation(content):
                seen.setdefault(content, None)
    return list(seen)


def load_table(path: Path = TABLE_PATH) -> Dict[str, str]:
    """{english line: japanese line}. Untranslated entries are an error."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    table = {row["en"]: row["ja"] for row in rows}
    untranslated = [en[:60] for en, ja in table.items() if not ja.strip()]
    if untranslated:
        raise ValueError(f"{len(untranslated)} line(s) have no translation yet, "
                         f"e.g. {untranslated[0]!r}")
    lost = [en[:60] for en, ja in table.items() if (TERM in en) != (TERM in ja)]
    if lost:
        raise ValueError(f"translation drops or adds the {TERM} placeholder: "
                         f"{lost[0]!r}")
    return table


def translate_template(text: str, table: Mapping[str, str]) -> str:
    """Translate every instruction line; sentinels and whitespace untouched."""
    parts = _SENTINEL.split(text)
    out: List[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:                          # a sentinel key
            out.append(sentinel(part))
            continue
        lines = []
        for lead, content, trail in _units(part):
            if content and needs_translation(content):
                if content not in table:
                    raise KeyError(f"line not in the translation table: "
                                   f"{content[:80]!r}")
                content = table[content]
            lines.append(lead + content + trail)
        out.append("\n".join(lines))
    return "".join(out)


def render_ja(prompts: PromptSet, spec: PromptSpec, kwargs: Mapping[str, str],
              table: Mapping[str, str], *, draft: str = "",
              discount_point: str = "", sales_arg: str = "") -> str:
    """The JA condition: same function, same data, instructions translated."""
    values = dict(kwargs)
    values.update(lang=table[LANG_EN], draft=draft,
                  discount_point=discount_point, sales_arg=sales_arg)
    return fill(translate_template(template(prompts, spec), table), values)


# --------------------------------------------------------------------------
# instruction text stored as data: the lease blocks


#: Where the lease term goes.
TERM = "{term}"

_LEASE_PATH = Path(__file__).resolve().parent / "vendor_company" / "lease_templates.json"


def _lease_templates() -> Dict[str, str]:
    return json.loads(_LEASE_PATH.read_text(encoding="utf-8"))["templates"]


def lease_texts(lease: Mapping) -> Tuple[str, str]:
    """(leasing, leasing_package) with the term still a placeholder."""
    t = _lease_templates()
    stem = lease["template"]                 # intro_lease | intro_lease_2
    return t[f"{stem}_a"] + TERM + t[f"{stem}_b"], t[lease["package"]]


def lease_units() -> List[str]:
    t = _lease_templates()
    return [t["intro_lease_a"] + TERM + t["intro_lease_b"],
            t["intro_lease_2_a"] + TERM + t["intro_lease_2_b"],
            t["green_lease"], t["power_lease"], t["maintenance_lease"]]


def translate_text(text: str, table: Mapping[str, str]) -> str:
    """Per-line translation of a block with no sentinels in it."""
    return translate_template(text, table)


def ja_kwargs(kwargs: Mapping, lease: Mapping, table: Mapping[str, str]) -> Dict:
    """The JA condition's data: identical, except the two instruction blocks."""
    leasing, package = lease_texts(lease)
    out = dict(kwargs)
    out["leasing"] = translate_text(leasing, table).replace(TERM, str(lease["term"]))
    out["leasing_package"] = translate_text(package, table)
    return out

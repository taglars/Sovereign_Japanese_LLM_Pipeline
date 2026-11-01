"""The rubric, in two languages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

SCALE_MIN, SCALE_MAX = 1, 10

#: Stamped on every record. Scores from different rubrics must never be averaged.
RUBRIC_VERSION = "2026-08-27-utility"

#: Evidence lists are capped, as at the company.
MAX_EVIDENCE = 10

LANGS = ("en", "ja")


@dataclass(frozen=True)
class Field:
    """One thing the judge returns, and the scale written out where it has one."""

    key: str
    name: str
    question: str
    anchors: Tuple[Tuple[int, str], ...] = ()
    evidence: str = ""

    def render(self, lang: str = "en") -> str:
        if lang == "ja":
            lines = [f"**{self.key}**（{self.name}）", f"  {self.question}"]
            if self.evidence:
                lines.append(f"  まず `{self.evidence}` に該当箇所をすべて原文のまま"
                             f"引用して列挙し、そのうえで採点してください。")
            for score, description in self.anchors:
                lines.append(f"  {score} = {description}")
            if self.anchors:
                lines.append(f"  -> `{self.key}` は {SCALE_MIN} から {SCALE_MAX} までの"
                             f"整数を1つだけ答えてください。範囲でも小数でも語でも"
                             f"ありません。")
            return "\n".join(lines)

        lines = [f"**{self.key}**, {self.name}", f"  {self.question}"]
        if self.evidence:
            lines.append(f"  List every item in `{self.evidence}` FIRST, with an exact "
                         f"quoted span, then score.")
        for score, description in self.anchors:
            lines.append(f"  {score} = {description}")
        if self.anchors:
            # The demand is repeated per field, not left to the schema.
            lines.append(f"  -> Answer `{self.key}` with ONE whole number from "
                         f"{SCALE_MIN} to {SCALE_MAX}. Not a range, not a decimal, "
                         f"not a word.")
        return "\n".join(lines)


SCORED_EN: Tuple[Field, ...] = (
    Field(
        key="keigo_score",
        name="Honorific correctness and consistency",
        question=("Is the honorific register correct for a written proposal to a "
                  "corporate customer, AND held consistently across the whole "
                  "document? Both too little and too much are errors, and so is "
                  "changing level partway through."),
        evidence="keigo_errors",
        anchors=(
            (1, "Plain form (常体) throughout, or honorifics applied to the seller's "
                "own actions. A customer would notice immediately."),
            (4, "Mostly 丁寧語 but the register slips between sections, a few 常体 "
                "sentences in an otherwise polite document."),
            (7, "Consistently polite throughout, but flat: no 尊敬語 or 謙譲語 where "
                "the context invites it."),
            (10, "Sustained appropriate register from first line to last: 丁寧語 "
                 "everywhere, 尊敬語 for the customer and 謙譲語 for the seller, used "
                 "correctly and without over-application."),
        ),
    ),
    Field(
        key="fluency_score",
        name="Natural professional Japanese",
        question=("Does it read as business Japanese written by a native speaker, with "
                  "correct commercial-vehicle terminology and consistent formatting?"),
        anchors=(
            (1, "Obviously machine-produced: broken particles, English word order, "
                "untranslated fragments, wrong terminology."),
            (4, "Grammatical but stiff, translationese, repeated sentence shapes, "
                "phrasing a Japanese salesperson would not choose."),
            (7, "Reads naturally, with occasional awkwardness or an inconsistent "
                "formatting convention."),
            (10, "Indistinguishable from a competent human writer of Japanese B2B "
                 "sales material."),
        ),
    ),
    Field(
        key="personalisation_score",
        name="Written for THIS customer",
        question=("How specific is this to the customer described in the data below, "
                  "as opposed to any company buying a truck?\n"
                  "  The test: remove the company name and the model code. Could this "
                  "be sent unchanged to a different customer in a different prefecture "
                  "in a different industry? The more of it that survives that removal "
                  "intact, the lower the score."),
        evidence="generic_spans",
        anchors=(
            (1, "Nothing specific survives the test. Generic product copy with a name "
                "pasted on."),
            (4, "Names the customer's details, industry, prefecture, fleet size, but "
                "does not reason from them. The industry is mentioned, not used."),
            (7, "Several arguments are genuinely shaped by this customer's situation, "
                "alongside generic passages."),
            (10, "The argument could not be sent to anyone else: mileage, fleet, "
                 "industry, region and contract history change what is emphasised and "
                 "why."),
        ),
    ),
    Field(
        key="readiness_score",
        name="Useful to a salesperson preparing for the call",
        question=("You are the sales representative who will make this call tomorrow. "
                  "How much of your preparation does this script do for you?\n"
                  "  Judge it as a working document, not as prose: does it give you "
                  "the specifics you would otherwise have to look up, in an order you "
                  "could actually speak from, covering what this customer will ask?"),
        evidence="missing_for_call",
        anchors=(
            (1, "You would have to prepare from scratch. Nothing here you could not "
                "have written yourself without the data."),
            (4, "A useful starting point, but you would need to look up the numbers "
                "yourself and reorder most of it."),
            (7, "You could walk into the call with this and be credible; a few gaps "
                "you would fill from the record."),
            (10, "You could take this to the meeting as it stands. The facts you need "
                 "are here, in the order you would raise them."),
        ),
    ),
    Field(
        key="persuasion_score",
        name="Would it move the customer",
        question=("Now read it as the CUSTOMER, the company described in the data "
                  "below, currently operating the vehicle named there.\n"
                  "  Having heard this from your sales representative, how much closer "
                  "are you to replacing your truck? Judge what would actually move you, "
                  "not whether the writing is enthusiastic."),
        evidence="objections",
        anchors=(
            (1, "It would not move you at all, or would make you trust the "
                "representative less."),
            (4, "You would listen politely. Nothing here addresses why you have not "
                "already replaced the vehicle."),
            (7, "It raises at least one point that genuinely applies to your "
                "operation and that you would want to discuss further."),
            (10, "You would ask for a quotation. The case is made in terms of your own "
                 "costs and your own operation."),
        ),
    ),
    Field(
        key="overall_score",
        name="Holistic judgement",
        question=("Would you send this to the customer? Your own judgement of the "
                  "document as a whole, not an average of the scores above."),
        anchors=(
            (1, "Not sendable. Something in it would embarrass the seller."),
            (5, "Sendable after real editing by a salesperson who knows the product."),
            (8, "Sendable after a light pass."),
            (10, "Sendable as written."),
        ),
    ),
)


SCORED_JA: Tuple[Field, ...] = (
    Field(
        key="keigo_score",
        name="敬語の正しさと一貫性",
        question=("法人顧客向けの提案書として敬語の使い方は適切であり、かつ文書全体を"
                  "通じて一貫していますか。過少も過剰も誤りであり、途中で文体が変わる"
                  "ことも誤りです。"),
        evidence="keigo_errors",
        anchors=(
            (1, "全体が常体、または自社の行為に尊敬語を用いている。顧客がすぐに"
                "気づく水準。"),
            (4, "おおむね丁寧語だが、節ごとに文体が揺れる。丁寧な文書の中に常体の文が"
                "いくつか混じる。"),
            (7, "全体を通して丁寧だが平板。文脈上ふさわしい箇所でも尊敬語・謙譲語が"
                "使われていない。"),
            (10, "冒頭から末尾まで適切な文体を保っている。全体に丁寧語を用い、顧客には"
                 "尊敬語、自社には謙譲語を、過剰にならず正しく使い分けている。"),
        ),
    ),
    Field(
        key="fluency_score",
        name="自然で専門的な日本語",
        question=("日本語を母語とする書き手による商用文として読めますか。商用車の用語は"
                  "正しく、書式は一貫していますか。"),
        anchors=(
            (1, "明らかに機械生成。助詞の誤り、英語の語順、未翻訳の断片、誤った用語が"
                "見られる。"),
            (4, "文法的には正しいが硬い。翻訳調で、同じ文型の繰り返しがあり、日本人の"
                "営業担当者なら選ばない言い回しが含まれる。"),
            (7, "自然に読める。ぎこちない箇所や書式の不統一が時折ある程度。"),
            (10, "日本語のB2B営業資料を書き慣れた人間の文章と区別がつかない。"),
        ),
    ),
    Field(
        key="personalisation_score",
        name="この顧客のために書かれているか",
        question=("下記データに記載された顧客に、どの程度固有の内容ですか。トラックを"
                  "購入するどの企業にも当てはまる内容とどこが違いますか。\n"
                  "  判定方法：企業名と型式コードを削除してください。別の都道府県の、"
                  "別の業種の、別の顧客にそのまま送れてしまいますか。削除後もそのまま"
                  "通用する部分が多いほど、点数は低くなります。"),
        evidence="generic_spans",
        anchors=(
            (1, "固有の内容が何も残らない。一般的な商品説明に名前を貼り付けただけ。"),
            (4, "顧客の情報、業種、都道府県、保有台数には言及するが、そこから論を"
                "組み立てていない。業種は「触れられている」だけで「使われていない」。"),
            (7, "この顧客の状況から実際に導かれた論点が複数あるが、一般的な記述も"
                "混在する。"),
            (10, "他社には送れない内容。走行距離、保有台数、業種、地域、契約履歴に"
                 "よって、何をどう強調するかが変わっている。"),
        ),
    ),
    Field(
        key="readiness_score",
        name="商談準備の道具として使えるか",
        question=("あなたは明日この商談を行う営業担当者です。この文書は準備作業を"
                  "どの程度代行してくれますか。\n"
                  "  文章としてではなく実務文書として判断してください。自分で調べ直す"
                  "必要がある具体的数値が揃っているか、実際に話せる順序になっているか、"
                  "この顧客が尋ねてくる点を押さえているか。"),
        evidence="missing_for_call",
        anchors=(
            (1, "一から準備し直す必要がある。データなしでも自分で書けた内容しかない。"),
            (4, "出発点としては有用だが、数値は自分で調べ直し、構成もほぼ組み替える"
                "必要がある。"),
            (7, "これを持って商談に臨んでも通用する。記録から補う小さな欠落がある程度。"),
            (10, "このまま商談に持参できる。必要な事実が、実際に切り出す順序で"
                 "揃っている。"),
        ),
    ),
    Field(
        key="persuasion_score",
        name="顧客の心を動かすか",
        question=("ここからは顧客の立場で読んでください。あなたは下記データに記載された"
                  "企業であり、そこに挙げられた車両を現在運用しています。\n"
                  "  営業担当者からこの説明を受けて、トラックの入れ替えにどれだけ"
                  "近づきましたか。文章が熱心かどうかではなく、実際に自分を動かすかで"
                  "判断してください。"),
        evidence="objections",
        anchors=(
            (1, "まったく動かされない。むしろ担当者への信頼が下がる。"),
            (4, "礼儀として聞きはする。なぜまだ入れ替えていないのかという点に、"
                "何も触れられていない。"),
            (7, "自社の運用に実際に当てはまる論点が少なくとも1つあり、もう少し"
                "話を聞きたいと思う。"),
            (10, "見積もりを依頼する。自社のコストと自社の運用の言葉で論が"
                 "組み立てられている。"),
        ),
    ),
    Field(
        key="overall_score",
        name="総合判断",
        question=("この文書を顧客に送りますか。上記の各点数の平均ではなく、文書全体に"
                  "対するあなた自身の判断を答えてください。"),
        anchors=(
            (1, "送れない。送れば売り手が恥をかく箇所がある。"),
            (5, "商品を理解している営業担当者が本格的に手を入れれば送れる。"),
            (8, "軽く手を入れれば送れる。"),
            (10, "このまま送れる。"),
        ),
    ),
)

BY_LANG: Dict[str, Tuple[Field, ...]] = {"en": SCORED_EN, "ja": SCORED_JA}

SCORED_KEYS: Tuple[str, ...] = tuple(f.key for f in SCORED_EN)
EVIDENCE_KEYS: Dict[str, str] = {f.evidence: f.key for f in SCORED_EN if f.evidence}
BOOLEAN_KEYS: Tuple[str, ...] = ("has_cta",)
LIST_KEYS: Tuple[str, ...] = tuple(EVIDENCE_KEYS) + ("non_japanese_sections",)

#: Field order is load-bearing.
JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["keigo_errors", "keigo_score", "fluency_score",
                 "generic_spans", "personalisation_score",
                 "missing_for_call", "readiness_score",
                 "objections", "persuasion_score",
                 "has_cta", "non_japanese_sections", "overall_score", "notes"],
    "properties": {
        "keigo_errors": {"type": "array", "maxItems": MAX_EVIDENCE, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["span", "problem"],
            "properties": {"span": {"type": "string"},
                           "problem": {"type": "string"}}}},
        "keigo_score": {"type": "integer", "minimum": SCALE_MIN, "maximum": SCALE_MAX},
        "fluency_score": {"type": "integer", "minimum": SCALE_MIN,
                          "maximum": SCALE_MAX},
        "generic_spans": {"type": "array", "maxItems": MAX_EVIDENCE, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["span", "why"],
            "properties": {"span": {"type": "string"},
                           "why": {"type": "string"}}}},
        "personalisation_score": {"type": "integer", "minimum": SCALE_MIN,
                                  "maximum": SCALE_MAX},
        "missing_for_call": {"type": "array", "maxItems": MAX_EVIDENCE,
                             "items": {"type": "string"}},
        "readiness_score": {"type": "integer", "minimum": SCALE_MIN,
                            "maximum": SCALE_MAX},
        "objections": {"type": "array", "maxItems": MAX_EVIDENCE,
                       "items": {"type": "string"}},
        "persuasion_score": {"type": "integer", "minimum": SCALE_MIN,
                             "maximum": SCALE_MAX},
        "has_cta": {"type": "boolean"},
        "non_japanese_sections": {"type": "array", "maxItems": MAX_EVIDENCE,
                                  "items": {"type": "string"}},
        "overall_score": {"type": "integer", "minimum": SCALE_MIN,
                          "maximum": SCALE_MAX},
        # Bounded, unlike every other field here.
        "notes": {"type": "string", "maxLength": 2000},
    },
}


SYSTEM_EN = """You are an expert evaluator of Japanese B2B commercial-vehicle sales \
scripts, with native-level command of business honorifics (敬語) and long experience of \
how fleet operators actually buy trucks.

Rules:
- List evidence BEFORE assigning the score it justifies, for every category that has an \
evidence list. Quote exact spans; never assert a problem without one.
- Longer is not better. Do not reward length, enthusiasm or the number of bullet points.
- Use the WHOLE 1-10 range. A script that is merely adequate is a 5, not a 2 and not \
an 8. Scores clustered at one end separate nothing and are of no use to anyone.
- You are NOT asked to fact-check. The generation prompts supply product and regulatory \
claims of their own, and figure-level checking happens elsewhere. Judge the language \
and whether the script does its job.
- Two of the questions ask you to take a role, the sales representative preparing for \
the call, and the customer receiving the pitch. Answer them from inside that role.
- has_cta is a fact you report, not a score, and it does not affect any score.

Output format, this is not negotiable, and a judgement that breaks it is discarded:
- Return ONE JSON object and nothing else. No prose before it, no commentary after it, \
no markdown fence, no explanation of your reasoning outside the `notes` field.
- Every *_score field is ONE WHOLE NUMBER between 1 and 10. Never a range ("6-7"), \
never a decimal ("7.5"), never a word ("high"), never null.
- Every field in the schema must be present. Use an empty array for an evidence list \
with nothing in it, never omit the key, and never write null.
- `has_cta` is true or false. `notes` is a single string.

<schema>{schema}</schema>"""


SYSTEM_JA = """あなたは日本語のB2B商用車営業スクリプトを評価する専門家です。ビジネス\
敬語を母語話者の水準で使いこなし、車両運行事業者が実際にどのようにトラックを購入\
するかについて長い経験があります。

規則:
- evidence（根拠）のある項目については、点数をつける前に必ず根拠を先に列挙して\
ください。該当箇所は原文のまま引用し、引用なしに問題を指摘してはいけません。
- 長ければ良いというものではありません。長さ、熱心さ、箇条書きの数を評価に\
加えないでください。
- 1から10の範囲全体を使ってください。可もなく不可もない文書は5であり、2でも8でも\
ありません。片側に固まった点数は何も区別せず、誰の役にも立ちません。
- 事実確認は求められていません。生成側のプロンプトが独自に製品情報や法規に関する\
記述を与えており、数値レベルの検証は別の仕組みで行われます。言語と、その文書が\
役目を果たしているかどうかを judge してください。
- 6つの設問のうち2つは、役割になりきって答えることを求めます。1つは商談を控えた\
営業担当者として、もう1つは提案を受ける顧客としてです。その立場の内側から\
答えてください。
- has_cta は報告する事実であって点数ではなく、どの点数にも影響しません。

出力形式。これは交渉の余地がなく、これに反する判定は破棄されます:
- JSONオブジェクトを1つだけ返してください。それ以外は何も書かないでください。前後の\
説明文、コメント、マークダウンのコードフェンスは不可です。推論の説明は `notes` \
フィールドの中だけに書いてください。
- すべての *_score フィールドは1から10までの整数1つです。範囲（「6-7」）、小数\
（「7.5」）、語（「高い」）、null はいずれも不可です。
- スキーマ上のすべてのフィールドを必ず含めてください。該当なしの evidence リストは\
空配列にし、キーを省略したり null を書いたりしないでください。
- `has_cta` は true か false です。`notes` は文字列1つです。

<schema>{schema}</schema>"""


TEMPLATE_EN = """{rubric}

<customer_data>
The company this script was written about, and the vehicle it concerns. Two of the
questions above ask you to judge from inside a role, read this as that person would
know it.

This is NOT a fact-checking reference and you are not asked to audit the script against
it.

{facts}
</customer_data>

<script id="{label}">
Each <section> carries an English `label` attribute. Those labels are the document's
own structural keys, they are metadata, not part of the script, and they are English in
the source system too. Judge ONLY the text inside each section. A section is "not in
Japanese" when its BODY is not Japanese; never because of its label.

{script}
</script>

Evaluate and return JSON only."""


TEMPLATE_JA = """{rubric}

<customer_data>
このスクリプトが対象としている企業と、その対象車両です。上記の設問のうち2つは役割の
内側からの判断を求めます。その立場の人物が知っている情報として読んでください。

これは事実確認のための参照資料ではなく、スクリプトをこれと突き合わせて監査することは
求められていません。

{facts}
</customer_data>

<script id="{label}">
各 <section> には英語の `label` 属性が付いています。これは文書自体の構造キーであり、
メタデータであって本文ではありません。元のシステムでも英語です。各 section の中の
テキストだけを評価してください。「日本語でない」と判定してよいのは本文が日本語でない
場合だけで、label が英語であることを理由にしてはいけません。

{script}
</script>

評価し、JSONのみを返してください。"""

SYSTEM_BY_LANG = {"en": SYSTEM_EN, "ja": SYSTEM_JA}
TEMPLATE_BY_LANG = {"en": TEMPLATE_EN, "ja": TEMPLATE_JA}

#: Shown where a script has no customer record.
NO_FACTS = {
    "en": "(no customer record is available for this script)",
    "ja": "（このスクリプトに対応する顧客情報は利用できません）",
}


def render_rubric(lang: str = "en") -> str:
    """The rubric as it appears in the prompt."""
    if lang not in BY_LANG:
        raise ValueError(f"unknown rubric language {lang!r}, expected one of {LANGS}")
    return "\n\n".join(field.render(lang) for field in BY_LANG[lang])


def render_schema() -> str:
    """Compact: it is sent on every call."""
    return json.dumps(JSON_SCHEMA, ensure_ascii=False, separators=(",", ":"))


def build_messages(lang: str, *, script: str, facts: str, label: str) -> List[Dict]:
    """The two messages exactly as they go over the wire."""
    system = SYSTEM_BY_LANG[lang].format(schema=render_schema())
    user = TEMPLATE_BY_LANG[lang].format(
        rubric=render_rubric(lang),
        facts=facts.strip() or NO_FACTS[lang],
        label=label or "unnamed",
        script=script)
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def validate(payload: Dict) -> Tuple[Dict[str, Any], List[str]]:
    """Type and range check. Records problems, never raises."""
    clean: Dict[str, Any] = {}
    problems: List[str] = []

    for key in SCORED_KEYS:
        if key not in payload:
            problems.append(f"missing: {key}")
            continue
        raw = payload[key]
        if isinstance(raw, bool):          # bool is an int subclass; not a score
            problems.append(f"{key}: boolean where a score was expected")
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            # Name the shape, because the fix differs.
            text = str(raw)
            if "-" in text[1:] or "–" in text:
                hint = ", a RANGE where one whole number was demanded"
            elif "." in text:
                hint = ", a DECIMAL where one whole number was demanded"
            else:
                hint = ", the judge did not return a number at all"
            problems.append(f"{key}: {raw!r} is not a number{hint}")
            continue
        if not SCALE_MIN <= value <= SCALE_MAX:
            problems.append(f"{key}: {value} is outside {SCALE_MIN}-{SCALE_MAX}")
            continue
        clean[key] = value

    for key in BOOLEAN_KEYS:
        if key not in payload:
            problems.append(f"missing: {key}")
        elif not isinstance(payload[key], bool):
            problems.append(f"{key}: {payload[key]!r} is not a boolean")
        else:
            clean[key] = payload[key]

    for key in LIST_KEYS:
        items = payload.get(key)
        if items is None:
            clean[key] = []
            continue
        if not isinstance(items, list):
            problems.append(f"{key}: {type(items).__name__} where a list was expected")
            clean[key] = []
            continue
        if len(items) > MAX_EVIDENCE:
            problems.append(f"{key}: {len(items)} items, capped at {MAX_EVIDENCE}")
        clean[key] = items[:MAX_EVIDENCE]

    clean["rubric_version"] = RUBRIC_VERSION
    return clean, problems


def component_mean(scores: Dict[str, int]) -> float | None:
    """Mean of the components, excluding overall_score."""
    keys = [k for k in SCORED_KEYS if k != "overall_score"]
    values = [scores[k] for k in keys if k in scores]
    if len(values) != len(keys):
        return None
    return round(sum(values) / len(values), 3)

"""The detailed ratings, drawn. Experiment 4a."""

from __future__ import annotations

import statistics as st
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .analyze import cells, load_reference
from .human import load_all
from .rubric import SCORED_KEYS
from .store import Record, Store

ROOT = Path(__file__).resolve().parents[1]

ACCENT = "#C0392B"
INK = "#1A1A1A"
MUTED = "#8C8C8C"
RULE = "#DDDDDD"

ARM_ORDER = ["batch", "qwen38_bf16", "qwen36_bf16", "qwen_only"]
ARM_LABEL = {"batch": "Azure batch", "qwen38_bf16": "Qwen3.8",
             "qwen36_bf16": "Qwen3.6", "qwen_only": "Qwen3-32B"}
ASPECT_LABEL = {"keigo_score": "Keigo", "fluency_score": "Fluency",
                "personalisation_score": "Personalisation",
                "readiness_score": "Readiness", "persuasion_score": "Persuasion",
                "overall_score": "Overall"}


def _instrument_label(judge: str, lang: str) -> str:
    if judge.startswith("human:"):
        return f"{judge.split(':', 1)[1]} (human)"
    name = {"gpt-5.6-sol": "GPT-5.6", "deepseek-v4-flash": "DeepSeek",
            "glm-4.7-flash": "GLM-4.7"}.get(judge, judge)
    return f"{name} / {lang.upper()}"


def _instrument_order(key: Tuple[str, str]) -> Tuple[int, int, str]:
    judge, lang = key
    rank = {"gpt-5.6-sol": 0, "deepseek-v4-flash": 1, "glm-4.7-flash": 2}
    if judge.startswith("human:"):
        return (9, 0, judge)
    return (rank.get(judge, 5), 0 if lang == "en" else 1, judge)


def collect(db: Path = ROOT / "out" / "judgements.db") -> Dict:
    records: List[Record] = list(Store(db).read(only_ok=True))
    records += load_reference(ROOT / "data" / "reference_scores.csv")
    arms = {r.script_id: r.arm for r in records if r.arm}
    humans = load_all(arms)
    by_key = cells(records + humans)
    keys = sorted(by_key, key=_instrument_order)

    # instrument -> aspect -> arm -> mean over that arm's scripts
    grid: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    for key in keys:
        cell = by_key[key]
        grid[key] = {}
        for aspect in SCORED_KEYS:
            per_arm: Dict[str, List[float]] = {}
            for sid, arm in cell.arm_of.items():
                value = cell.per_script.get(sid, {}).get(aspect)
                if arm and value is not None:
                    per_arm.setdefault(arm, []).append(value)
            grid[key][aspect] = {a: st.mean(v) for a, v in per_arm.items()}

    # raw per-judgement scores, for the distribution figure
    raw: Dict[Tuple[str, str], Dict[str, List[int]]] = {}
    for record in records + humans:
        bucket = raw.setdefault((record.judge, record.rubric_lang), {})
        for aspect, value in record.scores.items():
            bucket.setdefault(aspect, []).append(value)

    return {"keys": keys, "grid": grid, "raw": raw}


def _style(plt):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 10,
        "axes.titleweight": "bold", "svg.fonttype": "none",
    })


def aspects(data: Dict, out: Path) -> List[Path]:
    """Six aspects x four arms, one small heatmap per instrument."""
    import matplotlib.pyplot as plt
    _style(plt)
    keys = data["keys"]
    cols = 3
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.1 * rows))
    axes = [a for row in (axes if rows > 1 else [axes]) for a in row]

    for ax, key in zip(axes, keys):
        grid = data["grid"][key]
        values = [grid[a].get(arm) for a in SCORED_KEYS for arm in ARM_ORDER
                  if grid[a].get(arm) is not None]
        mu = st.mean(values)
        sd = st.pstdev(values) or 1.0
        matrix = [[((grid[a].get(arm, mu) - mu) / sd) for arm in ARM_ORDER]
                  for a in SCORED_KEYS]
        ax.imshow(matrix, cmap="Greys", vmin=-2.2, vmax=2.2, aspect="auto")
        for i, a in enumerate(SCORED_KEYS):
            for j, arm in enumerate(ARM_ORDER):
                v = grid[a].get(arm)
                if v is None:
                    continue
                z = matrix[i][j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if z > 0.9 else INK,
                        fontweight="bold" if arm == "batch" else "normal")
        ax.set_xticks(range(len(ARM_ORDER)))
        ax.set_xticklabels([ARM_LABEL[a] for a in ARM_ORDER], rotation=30,
                           ha="right")
        for tick, arm in zip(ax.get_xticklabels(), ARM_ORDER):
            if arm == "batch":
                tick.set_color(ACCENT)
                tick.set_fontweight("bold")
        ax.set_yticks(range(len(SCORED_KEYS)))
        ax.set_yticklabels([ASPECT_LABEL[a] for a in SCORED_KEYS])
        ax.set_title(_instrument_label(*key), loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    for ax in axes[len(keys):]:
        ax.axis("off")

    fig.suptitle("Mean score per aspect and arm. Shade is relative within each "
                 "panel; the number is the raw mean.", x=0.01, ha="left",
                 fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out / "aspects", plt)


def distributions(data: Dict, out: Path) -> List[Path]:
    """Per aspect, how each instrument uses the 1-10 scale."""
    import matplotlib.pyplot as plt
    _style(plt)
    keys = data["keys"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharey=True)
    flat = [a for row in axes for a in row]
    for index, (ax, aspect) in enumerate(zip(flat, SCORED_KEYS)):
        series = [data["raw"].get(k, {}).get(aspect, []) for k in keys]
        box = ax.boxplot(series, vert=False, widths=0.55, patch_artist=True,
                         medianprops={"color": INK, "linewidth": 1.4},
                         whiskerprops={"color": MUTED},
                         capprops={"color": MUTED},
                         flierprops={"marker": ".", "markersize": 3,
                                     "markerfacecolor": MUTED,
                                     "markeredgecolor": MUTED})
        for patch, key in zip(box["boxes"], keys):
            patch.set_facecolor("white" if key[0].startswith("human:") else "#EEEEEE")
            patch.set_edgecolor(ACCENT if key[0].startswith("human:") else MUTED)
        ax.set_yticks(range(1, len(keys) + 1))
        ax.set_yticklabels([_instrument_label(*k) for k in keys])
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
        ax.set_title(ASPECT_LABEL[aspect], loc="left")
        ax.grid(axis="x", color=RULE, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        # Shared y-axis hides the labels on inner columns but leaves the tick marks, which
        # read as stray dashes.
        ax.tick_params(axis="y", length=0)
        if index % 3:
            ax.tick_params(axis="y", labelleft=False)
        ax.invert_yaxis()
    fig.suptitle("How each instrument spreads its scores, per aspect. "
                 "Red outline: human rater.", x=0.01, ha="left", fontsize=10,
                 color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out / "distributions", plt)


def baseline_rank(data: Dict, out: Path) -> List[Path]:
    """Where the Azure baseline lands among the four arms, 1 = best."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _style(plt)
    keys = data["keys"]
    matrix = []
    for key in keys:
        row = []
        for aspect in SCORED_KEYS:
            means = data["grid"][key][aspect]
            order = sorted(means, key=lambda a: -means[a])
            row.append(order.index("batch") + 1 if "batch" in order else None)
        matrix.append(row)

    cmap = LinearSegmentedColormap.from_list("rank", [ACCENT, "#F4D6D2", "white"])
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(keys) + 1.4))
    shown = [[v if v is not None else 4 for v in row] for row in matrix]
    ax.imshow(shown, cmap=cmap, vmin=1, vmax=4, aspect="auto")
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            if v is not None:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=10,
                        fontweight="bold", color="white" if v == 1 else INK)
    ax.set_xticks(range(len(SCORED_KEYS)))
    ax.set_xticklabels([ASPECT_LABEL[a] for a in SCORED_KEYS])
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([_instrument_label(*k) for k in keys])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Rank of the Azure baseline among the four arms (1 = best)",
                 loc="left")
    fig.tight_layout()
    return _save(fig, out / "baseline_rank", plt)


def feedback(report: Dict, out: Path) -> List[Path]:
    """Exp 4b: category shares per judge and rubric language, one panel per kind."""
    import matplotlib.pyplot as plt
    from .codebook import BY_ID
    _style(plt)
    view = report.get("judge_lang", {})
    kinds = [k for k in ("strength", "weakness", "gap") if view.get(k, {}).get("shares")]
    if not kinds:
        return []
    groups = sorted({g for k in kinds for g in view[k]["shares"]},
                    key=lambda g: _instrument_order(tuple(g.split("|"))))
    shades = ["#1A1A1A", "#5A5A5A", "#9A9A9A", "#CFCFCF"][:len(groups)]
    heights = [len({c for g in groups for c in view[k]["shares"].get(g, {})}) for k in kinds]
    fig, axes = plt.subplots(len(kinds), 1, figsize=(8.5, 0.42 * sum(heights) + 1.6 * len(kinds)),
                             gridspec_kw={"height_ratios": heights})
    axes = axes if len(kinds) > 1 else [axes]
    width = 0.8 / len(groups)
    for ax, kind in zip(axes, kinds):
        shares = view[kind]["shares"]
        cats = sorted({c for g in groups for c in shares.get(g, {})},
                      key=lambda c: -sum(shares.get(g, {}).get(c, 0) for g in groups))
        for gi, (g, shade) in enumerate(zip(groups, shades)):
            ys = [i + (gi - (len(groups) - 1) / 2) * width for i in range(len(cats))]
            ax.barh(ys, [shares.get(g, {}).get(c, 0) for c in cats], height=width,
                    color=shade, label=_instrument_label(*g.split("|")))
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels([BY_ID[c].label if c in BY_ID else c for c in cats])
        for tick, c in zip(ax.get_yticklabels(), cats):
            if c == "G_NODATA":
                tick.set_color(ACCENT)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="x", color=RULE, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
        test = view[kind].get("test") or {}
        note = (f"   \u03c7\u00b2 p {test['p']:.1g}, Cram\u00e9r's V {test['cramers_v']:.2f}"
                if test.get("p") is not None else "")
        title = {"strength": "Strengths", "weakness": "Weaknesses", "gap": "Gaps"}[kind]
        ax.set_title(f"{title}{note}", loc="left")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Share of each judge's feedback points per category. Red label: "
                 "a confound of the setup, not a finding.", x=0.01, ha="left",
                 fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out / "feedback", plt)


def exp3_effects(result: Dict, out: Path) -> List[Path]:
    """Exp 3: one row per aspect, one marker per judge and rubric language."""
    import matplotlib.pyplot as plt
    _style(plt)
    rows = result.get("overall", [])
    if not rows:
        return []
    cells = sorted({(r["judge"], r["lang"]) for r in rows}, key=_instrument_order)
    shades = ["#1A1A1A", "#5A5A5A", "#8C8C8C", "#B5B5B5"]
    aspects = [a for a in SCORED_KEYS if any(r["aspect"] == a for r in rows)]
    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(aspects) + 1.4))
    step = 0.7 / max(1, len(cells))
    for ci, cell in enumerate(cells):
        for ai, aspect in enumerate(aspects):
            r = next((x for x in rows if (x["judge"], x["lang"]) == cell
                      and x["aspect"] == aspect), None)
            if r is None:
                continue
            y = ai + (ci - (len(cells) - 1) / 2) * step
            lo, hi = r["ci"]
            ax.plot([lo, hi], [y, y], color=shades[ci % 4], linewidth=1.4)
            sig = r["p_holm"] is not None and r["p_holm"] < 0.05
            ax.plot(r["diff"], y, "o", markersize=6, color=shades[ci % 4],
                    markerfacecolor=shades[ci % 4] if sig else "white",
                    label=_instrument_label(*cell) if ai == 0 else None)
    ax.axvline(0, color=ACCENT, linewidth=1)
    ax.set_yticks(range(len(aspects)))
    ax.set_yticklabels([ASPECT_LABEL[a] for a in aspects])
    ax.invert_yaxis()
    ax.grid(axis="x", color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Score difference, Japanese instructions minus English (points)")
    # Below the axis: inside, it sits on top of whichever interval ends lowest.
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=len(cells))
    n = max(r["n"] for r in rows)
    ax.set_title(f"Does instruction language change the score? {n} paired customers. "
                 "Filled: Holm p < 0.05.", loc="left")
    fig.tight_layout()
    return _save(fig, out / "exp3_effects", plt)


def relative_agreement(result: Dict, out: Path) -> List[Path]:
    """Left: pairwise preference agreement between every two raters. Right: each rater's
    ordering of the scripts, shaded by standardised score.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from .relative import _ranks, label as rlabel
    _style(plt)
    cmap = LinearSegmentedColormap.from_list("agree", ["#6B6B6B", "#FFFFFF", ACCENT])
    keys, scripts, humans = result["keys"], result["scripts"], result["humans"]
    names = [rlabel(k) for k in keys]
    n = len(keys)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.5, 0.62 * n + 2.6),
                                 gridspec_kw={"width_ratios": [n, len(scripts) + 1.5]})
    # -- agreement matrix ------------------------------------------------------
    grid = [[result["matrix"][(a, b)]["rate"] if a != b else None for b in keys]
            for a in keys]
    shown = [[0.5 if v is None else v for v in row] for row in grid]
    ax.imshow(shown, cmap=cmap, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            v = grid[i][j]
            if v is None:
                ax.text(j, i, "\u2013", ha="center", va="center", color=MUTED)
                continue
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v - 0.5) > 0.27 else INK)
    cut = len(humans) - 0.5
    for line in (ax.axhline, ax.axvline):
        line(cut, color=INK, linewidth=1.2)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    pairs = len(scripts) * (len(scripts) - 1) // 2
    ax.set_title(f"Do two raters prefer the same script?\nShare of {pairs} script "
                 "pairs ordered the same way (50% = chance)", loc="left")

    # -- each rater's ordering ---------------------------------------------------
    mean_z = {s: sum(result["z"][k][s] for k in keys) / n for s in scripts}
    order = sorted(scripts, key=lambda s: -mean_z[s])
    strip = [[max(-2.0, min(2.0, result["z"][k][s])) for s in order] for k in keys]
    bx.imshow([[(v + 2) / 4 for v in row] for row in strip], cmap=cmap, vmin=0, vmax=1,
              aspect="auto")
    for i, k in enumerate(keys):
        ranks, _ = _ranks([-result["scores"][k][s] for s in order])
        for j, r in enumerate(ranks):
            txt = f"{r:g}" if r == int(r) else f"{r:.1f}"
            bx.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if abs(strip[i][j]) > 1.1 else INK)
    bx.axhline(cut, color=INK, linewidth=1.2)
    short = {"batch": "Azure", "qwen38_bf16": "Q3.8", "qwen36_bf16": "Q3.6",
             "qwen_only": "Q3-32B"}
    bx.set_xticks(range(len(order)))
    bx.set_xticklabels([f"{short.get(result['arms'].get(s, ''), '?')}\n{s}" for s in order],
                       fontsize=8)
    for tick, s in zip(bx.get_xticklabels(), order):
        if result["arms"].get(s) == "batch":
            tick.set_color(ACCENT)
    bx.set_yticks(range(n))
    bx.set_yticklabels([f"{rlabel(k)}  \u00b7  ties {result['ties'][k]:.0%}" for k in keys],
                       fontsize=8.5)
    for spine in bx.spines.values():
        spine.set_visible(False)
    bx.tick_params(length=0)
    bx.set_title("Each rater's ordering of the scripts (1 = their best).\n"
                 "Shade: score on the rater's own scale (mean 0, equal spread)",
                 loc="left")

    g = result["groups"]
    foot = ("Standardised Krippendorff alpha: AI judges "
            f"{g['AI judges']['alpha_standardised']:+.2f}, all raters "
            f"{g['all raters']['alpha_standardised']:+.2f}. Scripts sorted by the "
            "average over all raters. Score: mean of the five component aspects; "
            "judges averaged over their repeats.")
    fig.text(0.01, 0.01, foot, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out / "relative", plt)


def exp3_checks(r: Dict, out: Path) -> List[Path]:
    """Three panels, one per robustness check that has a picture."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from .checks import CELLS, MORPH
    from .relative import label as rlabel
    _style(plt)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5.2),
                                     gridspec_kw={"width_ratios": [1.15, 1, 0.9]})

    pairs = r["agreement"]["pairs"]
    for i, p in enumerate(pairs):
        legacy = (p.get("legacy12") or {}).get("pairwise")
        now = p["exp3"]["pairwise"]
        if legacy is not None:
            a1.plot([legacy, now], [i, i], color=RULE, linewidth=1.5, zorder=1)
            a1.plot(legacy, i, "o", color=MUTED, markerfacecolor="white", markersize=7,
                    label="12 legacy scripts" if i == 0 else None, zorder=2)
        a1.plot(now, i, "o", color=INK, markersize=7,
                label=f"{r['n_scripts']} Exp 3 scripts" if i == 0 else None, zorder=3)
    a1.axvline(0.5, color=ACCENT, linewidth=1)
    a1.set_yticks(range(len(pairs)))
    a1.set_yticklabels([f"{rlabel(p['a'])} vs {rlabel(p['b'])}" for p in pairs], fontsize=8)
    a1.invert_yaxis()
    a1.set_xlim(0.3, 1.0)
    a1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a1.set_xlabel("Script pairs ordered the same way (50% = chance)")
    a1.set_title("Judges agree less on scripts\nfrom one model", loc="left")
    a1.legend(frameon=False, fontsize=8, loc="lower right")

    rows = [x for x in r["length"] if x["aspect"] == "component_mean"]
    for i, x in enumerate(rows):
        lo, hi = x["adjusted_ci"]
        a2.plot([lo, hi], [i, i], color=INK, linewidth=1.4)
        a2.plot(x["raw"], i, "o", color=MUTED, markerfacecolor="white", markersize=7,
                label="raw" if i == 0 else None)
        a2.plot(x["adjusted"], i, "o", color=INK, markersize=7,
                label="at equal length" if i == 0 else None)
    a2.axvline(0, color=ACCENT, linewidth=1)
    a2.set_yticks(range(len(rows)))
    a2.set_yticklabels([rlabel(x["cell"]) for x in rows], fontsize=8)
    a2.invert_yaxis()
    a2.set_xlabel("Score, JA minus EN instructions (points)")
    a2.set_title("Instruction-language effect,\nlength held equal", loc="left")
    a2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2)

    measures = [m for m in MORPH if m != "keigo_register_consistency"]
    cmap = LinearSegmentedColormap.from_list("rho", ["#6B6B6B", "#FFFFFF", ACCENT])
    grid = [[r["keigo"]["exp3"][k][m] or 0.0 for m in measures] for k in CELLS]
    a3.imshow(grid, cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            a3.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9, color=INK)
    a3.set_xticks(range(len(measures)))
    a3.set_xticklabels(["keigo errors", "honorifics to\ncustomer", "keigo density"],
                       fontsize=8)
    a3.set_yticks(range(len(CELLS)))
    a3.set_yticklabels([rlabel(k) for k in CELLS], fontsize=8)
    for spine in a3.spines.values():
        spine.set_visible(False)
    a3.tick_params(length=0)
    a3.set_title("Judge keigo score vs\nmorphological measure (rho)", loc="left")

    for ax in (a1, a2):
        ax.grid(axis="x", color=RULE, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.text(0.01, 0.01, "Score: mean of the five component aspects. Equal length: "
             "intercept of the per-customer score difference regressed on the length "
             "difference, 95% bootstrap interval over customers.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out / "exp3_checks", plt)


def _save(fig, stem: Path, plt) -> List[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("svg", "png"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def render_all(out: Path = ROOT / "out" / "figures") -> List[Path]:
    data = collect()
    return aspects(data, out) + distributions(data, out) + baseline_rank(data, out)

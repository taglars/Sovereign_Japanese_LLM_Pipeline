"""Thesis figures, drawn with seaborn from the result files."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from constants import PAYG_USD, SPOT_USD

THESIS = Path(__file__).resolve().parents[1]
JV = THESIS.parent / "harness"
DATA = THESIS / "data"
OUT = THESIS / "figures"
PREVIEW = OUT / "preview"

ACCENT = "#C0392B"
INK = "#1A1A1A"
MUTED = "#6E6E6E"
AXIS = "#9A9A9A"
RULE = "#E2E2E2"
GREYS = ["#B3B3B3", "#7A7A7A", "#3D3D3D"]  # light to dark, passes the ordinal checks

ARMS = ["batch", "qwen38_bf16", "qwen36_bf16", "qwen_only"]
ARM_LABEL = {"batch": "GPT-4o (batch)", "qwen38_bf16": "Qwen3.8-27B",
             "qwen36_bf16": "Qwen3.6-27B", "qwen_only": "Qwen3-32B AWQ"}
ARM_COLOR = {"batch": ACCENT, "qwen38_bf16": GREYS[2], "qwen36_bf16": GREYS[1],
             "qwen_only": GREYS[0]}
ARM_MARKER = {"batch": "D", "qwen38_bf16": "o", "qwen36_bf16": "s", "qwen_only": "^"}

ASPECTS = ["overall_score", "keigo_score", "fluency_score", "personalisation_score",
           "persuasion_score", "readiness_score"]
ASPECT_LABEL = {"overall_score": "Overall", "keigo_score": "Keigo",
                "fluency_score": "Fluency", "personalisation_score": "Personalisation",
                "persuasion_score": "Persuasion", "readiness_score": "Readiness"}
JUDGE_LABEL = {"gpt-5.6-sol": "GPT-5.6", "deepseek-v4-flash": "DeepSeek",
               "glm-4.7-flash": "GLM"}  # full model names go in the captions

WIDTH = 5.9  # inches, the text width of the DWS template

# Draw the human raters where a figure has them.
RATERS = True



def style() -> None:
    sns.set_theme(context="paper", style="ticks", rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
        "axes.titlelocation": "center", "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 0,
        "grid.color": RULE, "grid.linewidth": 0.6, "legend.frameon": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "lines.solid_capstyle": "round",
    })


def save(fig, name: str, pad: float = 0.02) -> None:
    OUT.mkdir(exist_ok=True)
    PREVIEW.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=pad)
    fig.savefig(PREVIEW / f"{name}.png", dpi=200, bbox_inches="tight",
                pad_inches=pad, facecolor="white")
    plt.close(fig)
    print(f"  {name}.pdf")


def xgrid(ax) -> None:
    """Vertical hairline grid, no left spine, no y ticks: a dot plot's frame."""
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=True)
    ax.tick_params(axis="y", length=0)


def arm_handles(arms=ARMS) -> list:
    return [Line2D([], [], ls="", marker=ARM_MARKER[a], color=ARM_COLOR[a], ms=5,
                   mec="white", mew=0.6, label=ARM_LABEL[a]) for a in arms]


def bootstrap(x, n: int = 10_000, seed: int = 0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(axis=1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def _jv_report() -> dict:
    return json.loads((JV / "out" / "report.json").read_text())


def rater_names() -> dict:
    """Display names "Human Rater 1" to "Human Rater 5", in a fixed order."""
    from results_macros import RATER_ORDER
    raters = _jv_report()["human"]["raters"]
    assert raters == RATER_ORDER, f"rater order changed: {raters}"
    return {r: f"Human Rater {i}" for i, r in enumerate(raters, 1)}


# ---- Experiment 1 -----------------------------------------------------------

def cost() -> None:
    """Cost of one script per system, next to the GPU time it was measured from."""
    comp = pd.read_csv(DATA / "exp1_compute.csv").set_index("arm")
    base = pd.read_csv(DATA / "baseline_cost.csv").set_index("approach")
    open_ = ["qwen38_bf16", "qwen36_bf16", "qwen_only"]
    rows = [("GPT-4o, synchronous\n(Approach 1)", "legacy"), ("GPT-4o, batch mode\n(Approach 2)", "batch")]
    rows += [(ARM_LABEL[a], a) for a in open_]
    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 2.35), sharey=True,
                                      gridspec_kw={"width_ratios": [1, 2.5]})
    for y, (_, key) in enumerate(rows):
        if key in open_:
            minutes = comp.loc[key, "node_hours_per_vehicle"] * 60
            left.barh(y, minutes, height=0.5, color=ARM_COLOR[key], zorder=2)
            left.text(minutes + 0.08, y, f"{minutes:.1f} min", va="center", fontsize=6.4, color=INK)
            lo = comp.loc[key, "node_hours_per_vehicle"] * SPOT_USD
            hi = comp.loc[key, "node_hours_per_vehicle"] * PAYG_USD
            right.plot([lo, hi], [y, y], color=GREYS[0], lw=3, solid_capstyle="round", zorder=1)
            right.plot(lo, y, "o", ms=5.5, color=ARM_COLOR[key], mec="white", mew=0.8, zorder=3)
            right.plot(hi, y, "o", ms=5.5, mfc="white", mec=GREYS[2], mew=1.1, zorder=3)
            right.text(lo - 0.011, y, f"${lo:.3f}", va="center", ha="right", fontsize=6.4, color=INK)
            right.text(hi + 0.013, y, f"${hi:.2f}", va="center", fontsize=6.4, color=INK)
        else:
            left.text(0.08, y, "API, no GPU", va="center", fontsize=6.2, color=MUTED, style="italic")
            usd = base.loc[key, "usd_per_vehicle"]
            right.plot([0, usd], [y, y], color=RULE, lw=1.6, zorder=1)
            right.plot(usd, y, "D", ms=6, color=ACCENT, mec="white", mew=0.8, zorder=3)
            right.text(usd + 0.013, y, f"${usd:.2f}" + (" or more" if key == "legacy" else ""),
                       va="center", fontsize=6.4, color=INK)
    for ax in (left, right):
        ax.axhline(1.5, color=AXIS, lw=0.6)
        xgrid(ax)
    left.set_yticks(range(len(rows)))
    left.set_yticklabels([r[0] for r in rows])
    left.set_ylim(len(rows) - 0.5, -0.5)
    left.set_xlim(0, 3.0)
    left.set_title("GPU time per script")
    left.set_xlabel("node-minutes, measured")
    right.set_xlim(-0.035, 0.42)
    right.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.2f}"))
    right.set_title("Cost of one script")
    right.set_xlabel("US dollars")
    fig.tight_layout(w_pad=1.0)
    key = [Line2D([], [], ls="", marker="o", color=GREYS[2], ms=5,
                  label=f"Azure spot, ${SPOT_USD:.2f} per node-hour"),
           Line2D([], [], ls="", marker="o", mfc="white", mec=GREYS[2], mew=1.1, ms=5,
                  label=f"Azure pay-as-you-go, ${PAYG_USD:.2f}"),
           Line2D([], [], ls="", marker="D", color=ACCENT, ms=5, label="GPT-4o, list prices")]
    fig.legend(handles=key, loc="upper center", bbox_to_anchor=(0.58, 0.0), ncol=3,
               fontsize=6.2, handletextpad=0.3, columnspacing=1.4)
    save(fig, "exp1_cost")


def judge_free() -> None:
    """Three judge-free measures per arm: pooled marker, one tick per family."""
    df = pd.read_csv(DATA / "exp1_judge_free.csv")
    panels = [("bad_markup", "Markup failures", "share of sections, lower is better", 100),
              ("register_consistency", "Register consistency", "higher is better", 1),
              ("plain_per_100", "Plain-form sentences", "per 100 sentences, lower is better", 1)]
    fams = [("light", "x", -0.25), ("electric", "+", 0.0), ("heavy", "*", 0.25)]  # shapes elsewhere mean systems
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.5), sharey=True)
    for ax, (col, title, unit, scale) in zip(axes, panels):
        for y, arm in enumerate(ARMS):
            rows = df[df.arm == arm]
            pooled = rows[rows.family == "all"][col].iloc[0] * scale
            ax.plot([pooled, pooled], [y - 0.38, y + 0.38], color=ARM_COLOR[arm], lw=1.2,
                    zorder=2)
            for fam, mk, dy in fams:
                v = rows[rows.family == fam][col].iloc[0] * scale
                ax.plot(v, y + dy, ls="", marker=mk, ms=4.5, mfc="white",
                        mec=ARM_COLOR[arm], mew=0.9, zorder=3)
        ax.set_title(title)
        ax.set_xlabel(unit)
        xgrid(ax)
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    axes[0].set_yticks(range(len(ARMS)))
    axes[0].set_yticklabels([ARM_LABEL[a] for a in ARMS])
    axes[0].set_ylim(len(ARMS) - 0.5, -0.5)
    fig.tight_layout(w_pad=1.2)
    key = [Line2D([], [], ls="", marker="|", color=GREYS[2], ms=9, mew=1.2,
                  label="all sections"),
           *[Line2D([], [], ls="", marker=mk, mfc="white", mec=GREYS[2], ms=4.5, mew=0.9,
                    label=lab) for (_, mk, _), lab in
             zip(fams, ["light-duty", "electric light-duty", "heavy-duty"])]]
    fig.legend(handles=key, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4,
               handletextpad=0.3, columnspacing=1.6, fontsize=6.3)
    save(fig, "exp1_judge_free")


def judged() -> None:
    """The GPT-5.6 judge's arm means per aspect, 40 vehicles, 2 repeats."""
    df = pd.read_csv(DATA / "exp1_judged.csv")
    order = ["mean", "overall", "keigo", "fluency", "personalisation", "persuasion",
             "readiness"]
    # "mean" is the notebook's five-component mean: the script score
    label = {"mean": "Script score", **{a: a.capitalize() for a in order[1:]}}
    df["row"] = df.aspect.map(order.index)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.5))
    for y, aspect in enumerate(order):
        vals = df[df.aspect == aspect].score
        ax.plot([vals.min(), vals.max()], [y, y], color=RULE, lw=2.5, zorder=1)
    sns.scatterplot(data=df, x="score", y="row", hue="arm", style="arm",
                    hue_order=ARMS, style_order=ARMS, palette=ARM_COLOR,
                    markers=ARM_MARKER, s=34, edgecolor="white", linewidth=0.8,
                    zorder=3, legend=False, ax=ax)
    ax.axhline(0.5, color=AXIS, lw=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label[a] for a in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel("Mean score, 1 to 10 (GPT-5.6 judge)")
    ax.set_ylabel("")
    xgrid(ax)
    ax.legend(handles=arm_handles(), loc="lower center", bbox_to_anchor=(0.45, 1.0),
              ncol=4, handletextpad=0.3, columnspacing=1.2)
    save(fig, "exp1_judged")


# ---- Experiment 2 -----------------------------------------------------------

def agreement() -> None:
    """Left: alpha within and across judges. Right: rank correlation per pair."""
    rep = _jv_report()
    self_rows = [("gpt-5.6-sol", "en"), ("deepseek-v4-flash", "en"),
                 ("deepseek-v4-flash", "ja"), ("glm-4.7-flash", "en"),
                 ("glm-4.7-flash", "ja")]
    left = [(f"{JUDGE_LABEL[j]}, {l.upper()} rubric", rep["self"][f"{j}|{l}"]["alpha"], "within")
            for j, l in self_rows]
    left += [("Three judges, EN rubric", rep["e1_en"]["alpha"], "across"),
             ("Two judges, JA rubric", rep["e1_ja"]["alpha"], "across")]

    short = {"gpt-5.6-sol": "GPT", "deepseek-v4-flash": "DeepSeek",
             "glm-4.7-flash": "GLM"}
    right = []
    for key in ("e1_en", "e1_ja"):
        lang = rep[key]["lang"].upper()
        for p in rep[key]["pairs"]:
            right.append((f"{short[p['left']]} and {short[p['right']]}, {lang}",
                          p["overall_spearman"], "judges"))
    # Judges only: the raters have their own figures (raters_matrix, human_range).

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 0.27 * max(len(left), len(right)) + 0.9))
    for ax, rows, xlabel, title in [
            (axes[0], left, "Krippendorff's alpha, script score", "Agreement"),
            (axes[1], right, "Spearman's rho, script score", "Rank correlation")]:
        groups = [r[2] for r in rows]
        for y, (name, value, group) in enumerate(rows):
            ax.plot(value, y, "o", ms=5.5, color=GREYS[2] if group in ("within", "judges") else GREYS[1],
                    mec="white", mew=0.8, zorder=3)
            ax.plot([0, value], [y, y], color=RULE, lw=1.6, zorder=1)
            # right of the marker, except for small negatives; far-left labels would run
            # into the row names, so those go inside the bar
            outside_left = -0.4 <= value < 0
            ax.text(value - 0.1 if outside_left else value + 0.1, y,
                    f"{value:.2f}".replace("-", "−"), va="center",
                    ha="right" if outside_left else "left", fontsize=6.4, color=INK, zorder=4)
        for y in range(1, len(rows)):
            if groups[y] != groups[y - 1]:
                ax.axhline(y - 0.5, color=AXIS, lw=0.6)
        ax.axvline(0, color=AXIS, lw=0.9, zorder=2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlim(-1, 1)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        xgrid(ax)
    fig.tight_layout(w_pad=2.0)
    save(fig, "exp2_agreement")


def arm_order() -> None:
    """Arm means per instrument, centred on that instrument's own mean."""
    rep = _jv_report()
    e2 = {r["judge"]: r for r in rep["e2"]}
    rows = [("GPT-5.6, EN rubric", rep["e1_en"]["arm_means"]["gpt-5.6-sol"]),
            ("DeepSeek, EN rubric", e2["deepseek-v4-flash"]["arm_means_en"]),
            ("DeepSeek, JA rubric", e2["deepseek-v4-flash"]["arm_means_ja"]),
            ("GLM, EN rubric", e2["glm-4.7-flash"]["arm_means_en"]),
            ("GLM, JA rubric", e2["glm-4.7-flash"]["arm_means_ja"])]
    names = rater_names() if RATERS else {}
    for rater, means in rep["human"]["arm_means"].items():
        if RATERS:
            rows.append((names[rater], means))

    fig, ax = plt.subplots(figsize=(WIDTH, 2.7))
    for y, (name, means) in enumerate(rows):
        level = np.mean([means[a] for a in ARMS])
        xs = [means[a] - level for a in ARMS]
        ax.plot([min(xs), max(xs)], [y, y], color=RULE, lw=2.5, zorder=1)
        for arm, x in zip(ARMS, xs):
            # Tied arms (equal means) are nudged apart vertically, else one hides the other.
            ties = [a for a, xx in zip(ARMS, xs) if abs(xx - x) < 0.02]
            dy = 0.36 * (ties.index(arm) - (len(ties) - 1) / 2)
            ax.scatter(x, y + dy, marker=ARM_MARKER[arm], s=34, color=ARM_COLOR[arm],
                       edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(1.03, y, f"{level:.1f}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=7, color=MUTED)
    ax.text(1.03, -0.62, "mean\nscore", transform=ax.get_yaxis_transform(), va="bottom",
            ha="left", fontsize=6.5, color=MUTED, linespacing=1.1)
    if names:  # judges above, raters below
        ax.axhline(len(rows) - len(names) - 0.5, color=AXIS, lw=0.6)
    ax.axvline(0, color=AXIS, lw=0.6, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("Arm mean minus the instrument's mean over the four arms (points)")
    xgrid(ax)
    ax.legend(handles=arm_handles(), loc="lower center", bbox_to_anchor=(0.45, 1.0),
              ncol=4, handletextpad=0.3, columnspacing=1.2)
    save(fig, "exp2_arm_order", pad=0.08)  # room for the right-hand labels


def scale_use() -> None:
    """How each instrument uses the 1 to 10 scale, all six aspects pooled."""
    sys.path.insert(0, str(JV))
    from jv.figures import collect, _instrument_order
    raw = collect()["raw"]
    keys = [k for k in sorted(raw, key=_instrument_order)
            if RATERS or not k[0].startswith("human:")]
    raters = [k for k in keys if k[0].startswith("human:")]
    fig, axes = plt.subplots(len(keys), 1, figsize=(WIDTH, 0.5 * len(keys) + 0.5),
                             sharex=True)
    for ax, key in zip(axes, keys):
        scores = [s for values in raw[key].values() for s in values]
        share = pd.Series(scores).value_counts(normalize=True).reindex(range(1, 11), fill_value=0)
        human = key in raters
        name = (rater_names()[key[0]] if human
                else f"{JUDGE_LABEL.get(key[0], key[0])}, {key[1].upper()}")
        sns.barplot(x=share.index, y=share.values, color=GREYS[2] if human else GREYS[0],
                    width=0.7, saturation=1, ax=ax, native_scale=True)
        ax.tick_params(axis="x", length=0)
        top = share.idxmax()
        ax.text(top, share.max(), f"{share.max():.0%}", ha="center", va="bottom", fontsize=6.5)
        ax.set_ylim(0, share.max() * 1.35)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=7.5)
        sns.despine(ax=ax, left=True)
    axes[-1].set_xticks(range(1, 11))
    axes[-1].set_xlabel("Score")
    fig.tight_layout(h_pad=0.2)
    save(fig, "exp4_scale_use")


# ---- Experiment 2: the human raters -----------------------------------------

def _instrument_label(key) -> str:
    if key[0].startswith("human:"):
        return rater_names()[key[0]]
    return f"{JUDGE_LABEL[key[0]]}, {key[1].upper()}"


def raters_matrix() -> None:
    """Pairwise preference agreement among raters and judges, lower triangle."""
    sys.path.insert(0, str(JV))
    from jv.relative import analyse
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle
    rel = analyse()
    keys = [k for k in rel["keys"] if k[0].startswith("human:")] + \
        [("gpt-5.6-sol", "en"), ("deepseek-v4-flash", "en"), ("deepseek-v4-flash", "ja"),
         ("glm-4.7-flash", "en"), ("glm-4.7-flash", "ja")]
    n = len(keys)
    grid = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i):
            grid[i, j] = rel["matrix"][(keys[i], keys[j])]["rate"] * 100
    cmap = LinearSegmentedColormap.from_list("agree", ["#FFFFFF", GREYS[2]])
    norm = Normalize(vmin=50, vmax=100, clip=True)
    fig, ax = plt.subplots(figsize=(WIDTH * 0.78, WIDTH * 0.66))
    image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect="equal")
    bar_ax = ax.inset_axes([0.66, 0.80, 0.30, 0.035])
    bar = fig.colorbar(image, cax=bar_ax, orientation="horizontal", ticks=[50, 75, 100])
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=6.5, length=2)
    bar.ax.set_title("% of script pairs ordered alike\n(50 = chance; hatched: below chance)",
                     fontsize=6.5,
                     fontweight="normal", loc="left", pad=3)
    for i in range(n):
        for j in range(i):
            v = grid[i, j]
            if v < 50:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                                       ec=GREYS[0], lw=0))
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 80 else INK,
                    bbox={"fc": "white", "ec": "none", "pad": 0.6} if v < 50 else None)
    labels = [_instrument_label(k) for k in keys]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    raters = sum(1 for k in keys if k[0].startswith("human:"))
    for pos in (raters - 0.5,):
        ax.axhline(pos, color=AXIS, lw=0.6)
        ax.axvline(pos, color=AXIS, lw=0.6)
    ax.set_xlim(-0.5, n - 1.5)
    ax.set_ylim(n - 0.5, 0.5)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    save(fig, "exp2_raters_matrix")


JUDGE_KEYS = [("gpt-5.6-sol", "en"), ("deepseek-v4-flash", "en"), ("deepseek-v4-flash", "ja"),
              ("glm-4.7-flash", "en"), ("glm-4.7-flash", "ja")]


def _range_values(group):
    """Each human rater in `group` against the others, each judge on the same footing.

    A rater's value is Spearman's rho with the consensus of the other raters in
    the group; a judge's value is its mean rho with those same leave-one-out
    consensuses, so both meet consensuses of the same size.
    """
    from results_macros import rater_data
    d = rater_data()
    S, R, sp, cons = d["scripts"], d["raters"], d["spearman"], d["consensus"]
    names = rater_names()
    vals = {}
    for n in group:
        rest = cons([m for m in group if m != n])
        vals[names[n]] = (sp([R[n].overall[s] for s in S], [rest[s] for s in S]), "rater")
    for key in JUDGE_KEYS:
        cell = d["judges"][key]
        rhos = [sp([cell.overall[s] for s in S], [cons([m for m in group if m != n])[s] for s in S])
                for n in group]
        vals[_instrument_label(key)] = (float(np.mean(rhos)), "judge")
    rv = [v for v, g in vals.values() if g == "rater"]
    return vals, min(rv), max(rv), float(np.median(rv))


def human_range() -> None:
    """All five human raters."""
    from results_macros import RATER_ORDER
    names = rater_names()
    labels = [names[n] for n in RATER_ORDER] + [_instrument_label(k) for k in JUDGE_KEYS]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 0.25 * len(labels) + 1.0), sharey=True)
    panels = [(RATER_ORDER, "All five human raters"),
              (RATER_ORDER[1:], "Without Human Rater 1")]
    for ax, (group, title) in zip(axes, panels):
        vals, lo, hi, med = _range_values(group)
        ax.axvspan(lo, hi, color=RULE, alpha=0.55, lw=0, zorder=0)
        ax.axvline(med, color=AXIS, lw=0.8, zorder=1)
        ax.axvline(0, color=AXIS, lw=0.6, zorder=1)
        for y, label in enumerate(labels):
            if label not in vals:
                ax.text(0.35, y, "left out", va="center", ha="center", fontsize=6.3,
                        color=MUTED, style="italic")
                continue
            v, kind = vals[label]
            ax.plot([0, v], [y, y], color=RULE, lw=1.6, zorder=2)
            if kind == "rater":
                ax.plot(v, y, "o", ms=5.5, color=INK, mec="white", mew=0.8, zorder=3)
            else:
                ax.plot(v, y, "o", ms=5, mfc="white", mec=GREYS[2], mew=1.1, zorder=3)
            ax.text(v + (0.04 if v >= 0 else -0.04), y, f"{v:.2f}".replace("-", "\u2212"),
                    va="center", ha="left" if v >= 0 else "right", fontsize=6.0, color=MUTED,
                    bbox={"fc": "white", "ec": "none", "pad": 0.3}, zorder=4)
        ax.axhline(len(RATER_ORDER) - 0.5, color=AXIS, lw=0.6)
        ax.set_xlim(-0.3, 1.05)
        ax.set_title(title)
        ax.set_xlabel("Spearman's rho with the human consensus")
        xgrid(ax)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].set_ylim(len(labels) - 0.5, -0.5)
    fig.tight_layout(w_pad=1.0)
    save(fig, "exp2_human_range")




def rater_scripts() -> None:
    """The 12 blind scripts by human consensus, with the raters' spread and two judges."""
    from results_macros import RATER_ORDER, rater_data
    sys.path.insert(0, str(JV))
    from jv.relative import standardise
    d = rater_data()
    S, z = d["scripts"], d["z"]
    cons = d["consensus"](RATER_ORDER)
    order = sorted(S, key=lambda s: cons[s])
    judges = [(("gpt-5.6-sol", "en"), "s", GREYS[1]), (("deepseek-v4-flash", "en"), "^", GREYS[2])]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    for x, s in enumerate(order):
        vals = [z[n][s] for n in RATER_ORDER]
        ax.plot([x, x], [min(vals), max(vals)], color=RULE, lw=4, solid_capstyle="round", zorder=1)
        ax.plot(x, cons[s], "o", ms=5.5, color=INK, mec="white", mew=0.8, zorder=3)
    for key, marker, color in judges:
        jz = standardise(d["judges"][key].overall, S)
        ax.plot(range(len(order)), [jz[s] for s in order], marker, ms=4.5, mfc="white",
                mec=color, mew=1.1, ls="", zorder=4)
    ax.axhline(0, color=AXIS, lw=0.6, zorder=0)
    ax.set_xticks(range(len(order)))
    short = {"batch": "GPT-4o", "qwen38_bf16": "Q3.8", "qwen36_bf16": "Q3.6",
             "qwen_only": "Q3-32B"}
    ax.set_xticklabels([f"{s}\n{short[d['arms'][s]]}" for s in order], fontsize=6.5)
    ax.set_ylabel("Standardised script score")
    sns.despine(ax=ax)
    handles = [Line2D([], [], color=RULE, lw=4, label="Range of the five human raters"),
               Line2D([], [], ls="", marker="o", color=INK, ms=5, label="Human consensus")]
    handles += [Line2D([], [], ls="", marker=m, mfc="white", mec=c, mew=1.1, ms=5,
                       label=_instrument_label(k)) for k, m, c in judges]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
              handletextpad=0.3, columnspacing=1.2)
    save(fig, "exp2_scripts", pad=0.08)  # room for the legend


# ---- Experiment 3 -----------------------------------------------------------

def exp3_judge_free() -> None:
    """Per customer, Japanese-instruction script minus English-instruction script."""
    m = pd.read_csv(JV / "out" / "exp3" / "metrics.csv")
    wide = m.pivot(index="customer", columns="condition")
    panels = [
        ("chars", "Length", "% change", lambda en, ja: (ja - en) / en * 100),
        ("html_ok_share", "Sections with valid markup", "percentage points",
         lambda en, ja: (ja - en) * 100),
        ("latin_share", "Latin-script share", "percentage points",
         lambda en, ja: (ja - en) * 100),
        ("keigo_referent_honorifics_per_100_sentences", "Customer-directed honorifics",
         "per 100 sentences", lambda en, ja: ja - en),
        ("keigo_defects_per_100_sentences", "Keigo errors", "per 100 sentences",
         lambda en, ja: ja - en),
        ("keigo_register_consistency", "Register consistency", "difference",
         lambda en, ja: ja - en),
    ]
    np.random.seed(20260924)  # the strip plot's jitter, so the figure reproduces
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.0))
    for ax, (col, title, unit, diff) in zip(axes.flat, panels):
        d = diff(wide[(col, "en")], wide[(col, "ja")]).dropna()
        sns.stripplot(x=d.values, ax=ax, color=GREYS[0], size=2.6, jitter=0.28,
                      zorder=1, orient="h")
        mean, lo, hi = bootstrap(d)
        ax.errorbar(mean, 0, xerr=[[mean - lo], [hi - mean]], fmt="o", color=INK, ms=5,
                    mec="white", mew=0.8, lw=1.6, capsize=0, zorder=4)
        ax.axvline(0, color=AXIS, lw=0.9, zorder=3)
        ax.set_title(title)
        ax.set_xlabel(f"JA minus EN, {unit}")
        ax.set_yticks([])
        ax.set_ylim(0.5, -0.5)
        sns.despine(ax=ax, left=True)
    fig.tight_layout(h_pad=1.4, w_pad=1.2)
    save(fig, "exp3_judge_free")


def exp3_judged() -> None:
    """Mean JA-minus-EN score difference with 95% interval, per judge and rubric."""
    from results_macros import exp3_script_score
    rows = pd.DataFrame(json.loads((JV / "out" / "exp3" / "report.json").read_text())["overall"])
    score = exp3_script_score()
    lang_style = {"en": (GREYS[2], "o", "EN rubric"), "ja": (GREYS[1], "s", "JA rubric")}
    labels = ["Script score"] + [ASPECT_LABEL[a] for a in ASPECTS]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.75), sharey=True, sharex=True)
    for ax, judge in zip(axes, ["deepseek-v4-flash", "glm-4.7-flash"]):
        for k, lang in enumerate(["en", "ja"]):
            color, marker, _ = lang_style[lang]
            s = score[(judge, lang)]
            cells = [(0, s["diff"], (s["lo"], s["hi"]), s["p_holm"])]
            for i, aspect in enumerate(ASPECTS, start=1):
                r = rows[(rows.judge == judge) & (rows.lang == lang) & (rows.aspect == aspect)]
                if not r.empty:
                    r = r.iloc[0]
                    cells.append((i, r["diff"], tuple(r["ci"]), r["p_holm"]))
            for i, diff, ci, p in cells:
                y = i + (k - 0.5) * 0.32
                ax.plot(ci, [y, y], color=color, lw=1.3, zorder=2)
                sig = p is not None and p < 0.05
                ax.plot(diff, y, marker, ms=5, color=color, mfc=color if sig else "white",
                        mew=1.1, zorder=3)
        ax.axvline(0, color=AXIS, lw=0.9, zorder=1)
        ax.axhline(0.5, color=AXIS, lw=0.6, zorder=1)
        ax.set_title(JUDGE_LABEL[judge])
        ax.set_xlabel("JA minus EN instructions (points)")
        xgrid(ax)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].set_ylim(len(labels) - 0.5, -0.5)
    handles = [Line2D([], [], ls="-", marker=m, color=c, ms=5, label=l)
               for c, m, l in lang_style.values()]
    handles += [Line2D([], [], ls="", marker="o", color=INK, ms=5, label="Holm p < 0.05"),
                Line2D([], [], ls="", marker="o", color=INK, mfc="white", ms=5,
                       label="not significant")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 1.0), ncol=4,
               handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout(w_pad=1.5)
    save(fig, "exp3_judged")


# ---- Experiment 4 -----------------------------------------------------------

def feedback() -> None:
    """Share of each judge's feedback points per category, one panel per kind."""
    sys.path.insert(0, str(JV))
    from jv.codebook import BY_ID
    rep = json.loads((JV / "out" / "codebook" / "report.json").read_text())
    if rep["n_labelled"] < rep["n_items"]:
        print(f"  feedback: PROVISIONAL, {rep['n_labelled']} of {rep['n_items']} points labelled")
    judges = [("deepseek-v4-flash", GREYS[2], "o"), ("glm-4.7-flash", GREYS[1], "s")]
    kinds = [("strength", "Strengths", "S_"), ("weakness", "Weaknesses", "W_"),
             ("gap", "Gaps", "G_")]
    tables = {}
    for kind, _, prefix in kinds:
        shares = rep["judge"][kind]["shares"]
        own = sorted(c for c in BY_ID if c.startswith(prefix))
        rows = {BY_ID[c].label: [shares.get(j, {}).get(c, 0) for j, _, _ in judges] for c in own}
        rows = dict(sorted(rows.items(), key=lambda kv: -sum(kv[1])))
        rows["Other"] = [1 - sum(v[i] for v in rows.values()) for i in range(len(judges))]
        tables[kind] = rows
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 0.19 * sum(map(len, tables.values())) + 1.3),
                             sharex=True,
                             gridspec_kw={"height_ratios": [len(tables[k]) for k, _, _ in kinds]})
    for ax, (kind, title, _) in zip(axes, kinds):
        rows = tables[kind]
        for y, vals in enumerate(rows.values()):
            ax.plot([min(vals), max(vals)], [y, y], color=RULE, lw=2.2, zorder=1)
            for (j, color, marker), v in zip(judges, vals):
                ax.plot(v, y, marker, ms=5, color=color, mec="white", mew=0.6, zorder=3)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(list(rows))
        ax.set_ylim(len(rows) - 0.5, -0.5)
        test = rep["judge"][kind].get("test") or {}
        if test.get("p") is not None:
            p = "p < 0.001" if test["p"] < 0.001 else f"p = {test['p']:.3f}"
            title += f"   (chi-square {p}, Cramér's V {test['cramers_v']:.2f})"
        ax.set_title(title)
        xgrid(ax)
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[-1].set_xlabel("Share of the judge's points of that kind")
    handles = [Line2D([], [], ls="", marker=m, color=c, ms=5, label=JUDGE_LABEL[j])
               for j, c, m in judges]
    fig.tight_layout(h_pad=0.8)
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.98, 1.0), ncol=2)
    save(fig, "exp4_feedback")


# ---- Diagrams ----------------------------------------------------------------

def case_pipeline() -> None:
    """Model calls behind one script in production."""
    from matplotlib.patches import FancyBboxPatch
    fam = pd.read_csv(DATA / "baseline_stage_calls.csv")
    per = fam.groupby("stage")["calls_per_vehicle"].agg(["min", "max"])
    tot = fam.groupby("family")["calls_per_vehicle"].sum()
    rng = lambda lo, hi: f"{lo:.0f}–{hi:.0f}"
    calls = {s: rng(per.loc[s, "min"], per.loc[s, "max"]) for s in per.index}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
    ax.set_xlim(-0.05, 10.3)
    ax.set_ylim(0.05, 3.85)
    ax.axis("off")

    def box(x, y, w, h, title, body="", lift=0.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="white", ec=AXIS, lw=0.8, zorder=2))
        ax.text(x + w / 2, y + h / 2 + lift + (0.15 if body else 0), title, ha="center",
                va="center", fontsize=6.9, weight="bold", color=INK, linespacing=1.2, zorder=3)
        if body:
            ax.text(x + w / 2, y + h / 2 + lift - 0.17, body, ha="center", va="center",
                    fontsize=6.0, color=MUTED, zorder=3)

    def badge(x, y, s):
        ax.text(x, y, s, ha="center", va="center", fontsize=5.2, weight="bold", color="white",
                zorder=6, bbox={"boxstyle": "round,pad=0.22,rounding_size=0.4", "fc": INK,
                                "ec": "none"})

    def arrow(x0, y0, x1, y1, rad=0.0, color=MUTED):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.8,
                                "shrinkA": 0, "shrinkB": 0, "mutation_scale": 8,
                                "connectionstyle": f"arc3,rad={rad}"})

    box(0.0, 2.15, 2.05, 0.85, "Vehicle record")
    box(0.0, 0.75, 2.05, 0.85, "Industry and\nprefecture context")

    # the section loop
    lx, ly, lw_, lh = 2.4, 0.45, 5.42, 3.3
    ax.add_patch(FancyBboxPatch((lx, ly), lw_, lh, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=GREYS[1], lw=0.9, zorder=0))
    ax.text(lx + 0.18, ly + lh - 0.14, "For each section", ha="left", va="top", fontsize=6.9,
            weight="bold", color=INK)
    bw, bh, by = 1.5, 1.0, 1.2
    xs = [2.58, 4.4, 6.22]
    for x, name, s in zip(xs, ["Draft", "Review", "English\ntranslation"], "ABC"):
        box(x, by, bw, bh, name, lift=0.12)
        badge(x + bw / 2, by, f"{calls[s]} model calls")
    arrow(xs[0] + bw, by + bh / 2, xs[1], by + bh / 2)
    arrow(xs[1] + bw, by + bh / 2, xs[2], by + bh / 2)
    arrow(xs[2] + bw / 2, by + bh + 0.12, xs[0] + bw / 2, by + bh + 0.12, rad=0.24, color=GREYS[2])
    ax.text((xs[0] + xs[2] + bw) / 2, by + bh + 0.7, "next section", ha="center", va="bottom",
            fontsize=6.0, color=GREYS[2])
    arrow(2.05, 2.55, xs[0], by + bh * 0.72)
    arrow(2.05, 1.15, xs[0], by + bh * 0.28)
    ax.text(lx + lw_ / 2, ly + 0.12, "Approach 1 only: one more model call repaired invalid HTML",
            ha="center", va="bottom", fontsize=5.8, color=MUTED, style="italic")

    # after the loop
    box(8.12, 2.1, 2.13, 1.05, "Discount argument", "per item, in Japanese\nand in English", lift=0.12)
    badge(8.12 + 2.13 / 2, 2.1, f"{calls['D']} model calls")
    arrow(lx + lw_, 2.62, 8.12, 2.62)
    box(8.12, 0.55, 2.13, 0.9, "Script in HTML", f"{rng(tot.min(), tot.max())} model calls")
    arrow(9.185, 1.93, 9.185, 1.4)
    ax.text(0.0, 0.55, "Model calls per script\n(API requests): range over\nthe three truck families", ha="left",
            va="top", fontsize=5.7, color=MUTED, linespacing=1.2)
    save(fig, "case_pipeline")


def pipeline_architecture() -> None:
    """Components of the Approach 3 pipeline around one spot node."""
    from matplotlib.patches import Circle, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
    ax.set_xlim(-0.05, 10.35)
    ax.set_ylim(-0.1, 6.55)
    ax.axis("off")

    def box(x, y, w, h, title, body="", z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="white", ec=AXIS, lw=0.8, zorder=z))
        ax.text(x + w / 2, y + h / 2 + (0.15 if body else 0), title, ha="center", va="center",
                fontsize=6.8, weight="bold", color=INK, linespacing=1.15, zorder=z + 1)
        if body:
            ax.text(x + w / 2, y + h / 2 - 0.2, body, ha="center", va="center", fontsize=6.0,
                    color=MUTED, zorder=z + 1)

    def arrow(x0, y0, x1, y1, label="", color=MUTED, ls="-", lx=None, ly=None, ha="center",
              va="bottom"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.8, "ls": ls,
                                "shrinkA": 0, "shrinkB": 0, "mutation_scale": 8})
        if label:
            ax.text((x0 + x1) / 2 if lx is None else lx, y0 + 0.06 if ly is None else ly, label,
                    ha=ha, va=va, fontsize=6.0, color=color, linespacing=1.15, zorder=5)

    def step(x, y, n):
        ax.add_patch(Circle((x, y), 0.17, fc=INK, ec="none", zorder=6))
        ax.text(x, y - 0.01, str(n), ha="center", va="center", fontsize=5.8, color="white",
                weight="bold", zorder=7)

    # control plane, left
    box(0.0, 4.7, 2.1, 0.9, "Planner", "one task per call")
    box(0.0, 2.55, 2.1, 1.3, "Task store", "Azure Table")
    arrow(1.05, 4.7, 1.05, 3.85)
    ax.text(0.0, 6.5, "Run from Databricks notebooks", ha="left", va="top", fontsize=6.0,
            color=MUTED)

    # the spot node
    nx, ny, nw, nh = 3.7, 0.95, 6.6, 5.2
    ax.add_patch(FancyBboxPatch((nx, ny), nw, nh, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=GREYS[2], lw=1.0, zorder=0))
    ax.text(nx + 0.18, ny + nh - 0.13, "Spot node with one A100", ha="left", va="top",
            fontsize=6.9, weight="bold", color=INK)
    ax.text(nx + 0.18, ny + nh - 0.45, "can be reclaimed at any time", ha="left", va="top",
            fontsize=6.0, color=MUTED)
    box(3.9, 4.3, 2.05, 0.8, "Eviction watcher")
    box(6.75, 4.3, 3.3, 0.8, "Container image", "worker, vLLM, routing")
    box(3.9, 2.55, 2.05, 1.3, "Worker")
    box(6.75, 2.55, 2.0, 1.3, "vLLM", "runs the model")
    box(9.05, 2.55, 1.0, 1.3, "GPU")
    box(6.75, 1.2, 2.0, 0.75, "Model mount", "read-only")

    arrow(2.1, 3.55, 3.9, 3.55, "lease a task", lx=2.9)
    arrow(3.9, 3.15, 2.1, 3.15, "done", lx=2.9)
    arrow(3.9, 2.75, 2.1, 2.75, "eviction:\nleases back", color=INK, ls="--", lx=2.9, ly=2.67,
          va="top")
    arrow(4.92, 4.3, 4.92, 3.85, color=INK, ls="--")
    arrow(5.95, 3.45, 6.75, 3.45, "prompt", lx=6.35)
    arrow(6.75, 2.95, 5.95, 2.95, "output", lx=6.35, ly=2.87, va="top")
    ax.annotate("", xy=(9.05, 3.2), xytext=(8.75, 3.2), zorder=4,
                arrowprops={"arrowstyle": "<|-|>", "color": MUTED, "lw": 0.8,
                            "shrinkA": 0, "shrinkB": 0, "mutation_scale": 7})
    arrow(7.75, 1.95, 7.75, 2.55)
    arrow(7.75, 0.7, 7.75, 1.2)
    arrow(4.92, 2.55, 4.92, 0.7, "result", lx=5.04, ly=1.6, ha="left", va="center")
    box(3.9, 0.0, 2.05, 0.7, "Blob: results", z=2)
    box(6.75, 0.0, 2.0, 0.7, "Blob: weights", z=2)

    step(7.5, 2.25, 1)
    step(2.9, 4.0, 2)
    step(4.62, 4.07, 3)
    ax.text(0.0, 2.15, "1  start: load the model\n2  work: lease, generate, save\n"
            "3  eviction: hand leases back", ha="left", va="top", fontsize=6.0, color=INK,
            linespacing=1.45)
    save(fig, "pipeline_architecture")


FIGURES = {f.__name__: f for f in [case_pipeline, pipeline_architecture, cost, judge_free,
                                     judged, agreement, arm_order, scale_use, raters_matrix,
                                     human_range, rater_scripts, exp3_judge_free, exp3_judged,
                                     feedback]}


def main(names) -> None:
    style()
    for name in names or FIGURES:
        FIGURES[name]()


if __name__ == "__main__":
    main(sys.argv[1:])

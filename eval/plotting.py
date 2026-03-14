"""
Visualization for RLM Evaluation Results

One concept per figure.  No crammed subplots.  No overlapping text.

Plots generated
---------------
1.  accuracy_per_task          - Grouped horizontal bars, sorted by task
2.  tokens_per_task            - Grouped bars (log scale) per task & runner
3.  latency_per_task           - Grouped bars per task & runner
4.  cost_per_task              - Grouped bars per task & runner (if cost available)
5.  accuracy_vs_cost           - Scatter: accuracy vs avg cost per query
6.  accuracy_vs_latency        - Scatter: accuracy vs avg latency per task
7.  token_savings              - Token savings factor vs Vanilla & Official per task
8.  summary_dashboard          - One-page headline numbers

Usage
-----
    # From a JSON results file
    uv run python -m eval.plotting path/to/eval.json [output_dir]

    # Auto-discovers the newest JSON in a directory tree
    uv run python -m eval.plotting logs/my_eval/ [output_dir]
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Tight layout not applied.*")
warnings.filterwarnings("ignore", message="This figure includes Axes.*tight_layout.*")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .metrics import EvalResult, aggregate_results, compute_statistics

# =============================================================================
# Style & identity
# =============================================================================

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 2,
        "lines.markersize": 9,
    }
)

# Runner display metadata — handles all known name variants
_RUNNER_META: dict[str, dict[str, Any]] = {
    "vanilla": {"label": "Vanilla LLM", "color": "#2980B9", "marker": "o", "zorder": 2},
    "official": {
        "label": "Official RLM",
        "color": "#E74C3C",
        "marker": "^",
        "zorder": 3,
    },
    "minrlm": {"label": "minRLM", "color": "#27AE60", "marker": "s", "zorder": 4},
    "ours": {"label": "minRLM", "color": "#27AE60", "marker": "s", "zorder": 4},
    "minrlm-reasoning": {
        "label": "minRLM",
        "color": "#27AE60",
        "marker": "D",
        "zorder": 5,
    },
}

_PREFERRED_ORDER = ["vanilla", "official", "minrlm", "ours", "minrlm-reasoning"]


def _meta(runner: str) -> dict[str, Any]:
    return _RUNNER_META.get(runner, {"label": runner, "color": "#95A5A6", "marker": "P", "zorder": 1})


def _label(runner: str) -> str:
    return _meta(runner)["label"]


def _color(runner: str) -> str:
    return _meta(runner)["color"]


def _sort_runners(runners: list[str]) -> list[str]:
    order = {r: i for i, r in enumerate(_PREFERRED_ORDER)}
    return sorted(runners, key=lambda r: order.get(r, 99))


def _task_label(name: str) -> str:
    return (
        name.upper()
        .replace("OFFICIAL_", "")
        .replace("_LONGBENCH_V2", " LONGBENCH V2")
        .replace("_V2", " V2")
        .replace("_2025", " 2025")
        .replace("_", " ")
        .strip()
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 1. Accuracy per task  (horizontal grouped bars)
# =============================================================================


def plot_accuracy_per_task(
    results: list[EvalResult],
    output_path: Path | None = None,
    title: str = "Accuracy by Task & Runner",
) -> plt.Figure:
    """Horizontal grouped bar chart — tasks on Y, one bar per runner."""
    agg = aggregate_results(results)
    tasks = sorted(agg.keys(), key=_task_label)
    runners = _sort_runners({r.runner_name for r in results})

    n_tasks = len(tasks)
    n_runners = len(runners)
    bar_h = 0.65 / n_runners

    fig_h = max(5, n_tasks * 0.7 + 2)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    y = np.arange(n_tasks)

    for i, runner in enumerate(runners):
        accs = [agg[t][runner].accuracy if runner in agg[t] else 0 for t in tasks]
        errs = [agg[t][runner].accuracy_std if runner in agg[t] else 0 for t in tasks]
        offsets = y + (i - (n_runners - 1) / 2) * bar_h

        bars = ax.barh(
            offsets,
            accs,
            bar_h * 0.85,
            label=_label(runner),
            color=_color(runner),
            xerr=errs if max(errs) > 0 else None,
            capsize=3,
            alpha=0.88,
            error_kw={"elinewidth": 1.2},
        )
        for bar, val in zip(bars, accs):
            if val > 2:
                ax.text(
                    min(val + 1.5, 103),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%",
                    va="center",
                    ha="left",
                    fontsize=9,
                )

    ax.set_xlabel("Accuracy (%)")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_yticks(y)
    ax.set_yticklabels([_task_label(t) for t in tasks])
    ax.set_xlim(0, 115)
    ax.axvline(x=100, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# =============================================================================
# 2. Tokens per task  (grouped bars, log scale)
# =============================================================================


def plot_tokens_per_task(
    results: list[EvalResult],
    output_path: Path | None = None,
    title: str = "Avg Tokens per Task",
) -> plt.Figure:
    """Grouped vertical bars, log-scale Y, one group per task."""
    agg = aggregate_results(results)
    tasks = sorted(agg.keys(), key=_task_label)
    runners = _sort_runners({r.runner_name for r in results})

    n_tasks = len(tasks)
    n_runners = len(runners)
    bar_w = 0.65 / n_runners

    fig_w = max(10, n_tasks * 1.1 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    x = np.arange(n_tasks)

    # Collect (runner_idx, x_pos, label) for zero-value annotations — placed after yscale is set
    na_annotations: list[tuple[float, str]] = []

    for i, runner in enumerate(runners):
        toks = [agg[t][runner].avg_total_tokens if runner in agg[t] else 0 for t in tasks]
        xs = x + (i - (n_runners - 1) / 2) * bar_w
        # Skip zero-value bars (log(0) = -inf breaks the axis); record them for N/A annotation
        valid_xs = [xi for xi, v in zip(xs, toks) if v > 0]
        valid_toks = [v for v in toks if v > 0]
        ax.bar(
            valid_xs,
            valid_toks,
            bar_w * 0.85,
            label=_label(runner),
            color=_color(runner),
            alpha=0.88,
        )
        for xi, val in zip(xs, toks):
            if val > 0:
                ax.text(
                    xi,
                    val * 1.08,
                    f"{int(val):,}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )
            else:
                na_annotations.append((xi, _color(runner)))

    ax.set_ylabel("Avg Tokens (log scale)")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([_task_label(t) for t in tasks], rotation=30, ha="right")
    ax.set_yscale("log")

    # Place N/A labels using axes-fraction transform so they sit just above the x-axis
    # regardless of the log-scale y limits (avoids canvas expansion)
    for xi, color in na_annotations:
        ax.annotate(
            "N/A",
            xy=(xi, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
            style="italic",
        )
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# =============================================================================
# 3. Latency per task  (grouped bars)
# =============================================================================


def plot_latency_per_task(
    results: list[EvalResult],
    output_path: Path | None = None,
    title: str = "Avg Latency per Task",
) -> plt.Figure:
    """Grouped vertical bars, one group per task, with std-dev error bars."""
    agg = aggregate_results(results)
    tasks = sorted(agg.keys(), key=_task_label)
    runners = _sort_runners({r.runner_name for r in results})

    n_tasks = len(tasks)
    n_runners = len(runners)
    bar_w = 0.65 / n_runners

    fig_w = max(10, n_tasks * 1.1 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    x = np.arange(n_tasks)

    for i, runner in enumerate(runners):
        times = [agg[t][runner].avg_time_seconds if runner in agg[t] else 0 for t in tasks]
        errs = [agg[t][runner].std_time_seconds if runner in agg[t] else 0 for t in tasks]
        xs = x + (i - (n_runners - 1) / 2) * bar_w
        bars = ax.bar(
            xs,
            times,
            bar_w * 0.85,
            label=_label(runner),
            color=_color(runner),
            alpha=0.88,
            yerr=errs if max(errs) > 0 else None,
            capsize=3,
            error_kw={"elinewidth": 1.2},
        )
        for bar, val in zip(bars, times):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(errs) * 0.1 + 0.5,
                    f"{val:.0f}s",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_ylabel("Avg Latency (seconds)")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([_task_label(t) for t in tasks], rotation=30, ha="right")
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# =============================================================================
# 4. Cost per task  (grouped bars)
# =============================================================================


def plot_cost_per_task(
    results: list[EvalResult],
    output_path: Path | None = None,
    title: str = "Avg Cost per Query by Task",
) -> plt.Figure | None:
    """Grouped bars showing avg cost per query (USD) per task."""
    if not any(r.cost_usd is not None for r in results):
        return None

    agg = aggregate_results(results)
    tasks = sorted(agg.keys(), key=_task_label)
    runners = _sort_runners({r.runner_name for r in results})

    n_tasks = len(tasks)
    n_runners = len(runners)
    bar_w = 0.65 / n_runners

    fig_w = max(10, n_tasks * 1.1 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    x = np.arange(n_tasks)

    for i, runner in enumerate(runners):
        costs_milli = []
        for t in tasks:
            m = agg[t].get(runner)
            costs_milli.append((m.avg_cost_usd or 0) * 1000 if m else 0)
        xs = x + (i - (n_runners - 1) / 2) * bar_w
        bars = ax.bar(
            xs,
            costs_milli,
            bar_w * 0.85,
            label=_label(runner),
            color=_color(runner),
            alpha=0.88,
        )
        for bar, val in zip(bars, costs_milli):
            if val > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"${val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_ylabel("Cost per Query  (milli-USD = $/1000 queries)")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([_task_label(t) for t in tasks], rotation=30, ha="right")
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# =============================================================================
# 5. Accuracy vs Cost  (scatter, efficiency frontier)
# =============================================================================


def plot_accuracy_vs_cost(
    results: list[EvalResult],
    output_path: Path | None = None,
    title: str = "Accuracy vs Cost — Efficiency Frontier",
) -> plt.Figure | None:
    """Scatter plot: each point = (task, runner).  Better = up-left corner."""
    if not any(r.cost_usd is not None for r in results):
        return None

    agg = aggregate_results(results)
    tasks = sorted(agg.keys())
    runners = _sort_runners({r.runner_name for r in results})

    # Per-runner label offsets to reduce overlap
    _label_offsets = [
        (6, 4),
        (6, -14),
        (-70, 4),
        (-70, -14),
        (6, 14),
    ]

    fig, ax = plt.subplots(figsize=(11, 7))

    for ri, runner in enumerate(runners):
        xs, ys, labels_pt = [], [], []
        for task in tasks:
            m = agg[task].get(runner)
            if m and m.avg_cost_usd is not None and m.avg_cost_usd > 0:
                xs.append(m.avg_cost_usd * 1000)
                ys.append(m.accuracy)
                labels_pt.append(_task_label(task))
        if not xs:
            continue
        meta = _meta(runner)
        ax.scatter(
            xs,
            ys,
            s=110,
            color=meta["color"],
            marker=meta["marker"],
            label=_label(runner),
            zorder=meta["zorder"],
            alpha=0.9,
            edgecolors="white",
            linewidths=0.7,
        )
        ox, oy = _label_offsets[ri % len(_label_offsets)]
        for x_pt, y_pt, lbl in zip(xs, ys, labels_pt):
            ax.annotate(
                lbl,
                (x_pt, y_pt),
                textcoords="offset points",
                xytext=(ox, oy),
                fontsize=8,
                color=meta["color"],
                alpha=0.85,
                arrowprops={
                    "arrowstyle": "-",
                    "color": meta["color"],
                    "alpha": 0.3,
                    "lw": 0.7,
                },
            )

    ax.set_xlabel("Avg Cost per Query (milli-USD)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_ylim(-5, 110)
    ax.annotate(
        "← cheaper + more accurate →\n(ideal: top-left)",
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        fontsize=9,
        color="gray",
        va="top",
    )
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# =============================================================================
# 6. Accuracy vs Latency  (scatter)
# =============================================================================


def plot_accuracy_vs_latency(
    results: list[EvalResult],
    output_path: Path | None = None,
    title: str = "Accuracy vs Latency",
) -> plt.Figure:
    """Scatter: accuracy (y) vs avg latency in seconds (x) per (task, runner)."""
    agg = aggregate_results(results)
    tasks = sorted(agg.keys())
    runners = _sort_runners({r.runner_name for r in results})

    _label_offsets = [
        (6, 4),
        (6, -14),
        (-70, 4),
        (-70, -14),
        (6, 14),
    ]

    fig, ax = plt.subplots(figsize=(11, 7))

    for ri, runner in enumerate(runners):
        xs, ys, labels_pt = [], [], []
        for task in tasks:
            m = agg[task].get(runner)
            if m and m.avg_time_seconds > 0:
                xs.append(m.avg_time_seconds)
                ys.append(m.accuracy)
                labels_pt.append(_task_label(task))
        if not xs:
            continue
        meta = _meta(runner)
        ax.scatter(
            xs,
            ys,
            s=110,
            color=meta["color"],
            marker=meta["marker"],
            label=_label(runner),
            zorder=meta["zorder"],
            alpha=0.9,
            edgecolors="white",
            linewidths=0.7,
        )
        ox, oy = _label_offsets[ri % len(_label_offsets)]
        for x_pt, y_pt, lbl in zip(xs, ys, labels_pt):
            ax.annotate(
                lbl,
                (x_pt, y_pt),
                textcoords="offset points",
                xytext=(ox, oy),
                fontsize=8,
                color=meta["color"],
                alpha=0.85,
                arrowprops={
                    "arrowstyle": "-",
                    "color": meta["color"],
                    "alpha": 0.3,
                    "lw": 0.7,
                },
            )

    ax.set_xlabel("Avg Latency (seconds)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_ylim(-5, 110)
    ax.annotate(
        "← faster + more accurate →\n(ideal: top-left)",
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        fontsize=9,
        color="gray",
        va="top",
    )
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# =============================================================================
# 7. Token savings vs baselines  (bar chart)
# =============================================================================


def plot_token_savings(
    results: list[EvalResult],
    output_path: Path | None = None,
    title: str = "Token Savings vs Baselines",
) -> plt.Figure | None:
    """
    For each task show how many × fewer tokens minRLM used vs Vanilla and Official.
    Bars > 1 mean we are cheaper.  Only shown when at least one baseline is present.
    """
    agg = aggregate_results(results)
    tasks = sorted(agg.keys(), key=_task_label)
    all_runners = {r.runner_name for r in results}

    rlm_runners = [r for r in _PREFERRED_ORDER if r in all_runners and r not in ("vanilla", "official")]
    baselines = [b for b in ("vanilla", "official") if b in all_runners]

    if not rlm_runners or not baselines:
        return None

    # For each (rlm_runner, baseline, task) compute ratio
    combos = [(rlm, base) for rlm in rlm_runners for base in baselines]
    n_combos = len(combos)
    if n_combos == 0:
        return None

    bar_w = 0.65 / n_combos
    x = np.arange(len(tasks))

    fig_w = max(10, len(tasks) * 1.2 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    palette = ["#2ECC71", "#1A8745", "#3498DB", "#1F618D"]
    for idx, (rlm, base) in enumerate(combos):
        ratios = []
        for t in tasks:
            rlm_m = agg[t].get(rlm)
            base_m = agg[t].get(base)
            if rlm_m and base_m and rlm_m.avg_total_tokens > 0:
                ratios.append(base_m.avg_total_tokens / rlm_m.avg_total_tokens)
            else:
                ratios.append(0)

        label = f"{_label(rlm)} vs {_label(base)}"
        color = palette[idx % len(palette)]
        xs = x + (idx - (n_combos - 1) / 2) * bar_w
        bars = ax.bar(xs, ratios, bar_w * 0.85, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, ratios):
            bx = bar.get_x() + bar.get_width() / 2
            if val >= 1.1:
                ax.text(
                    bx,
                    bar.get_height() + 0.05,
                    f"{val:.1f}×",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
            elif 0 < val < 1.0:
                # minRLM uses MORE tokens on this task (e.g. GDPVAL infrastructure overhead)
                ax.text(
                    bx,
                    bar.get_height() + 0.05,
                    f"⬆ {1 / val:.1f}×\noverhead",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#C0392B",
                    linespacing=1.3,
                )

    ax.axhline(
        1.0,
        color="gray",
        linestyle="--",
        alpha=0.5,
        linewidth=1.5,
        label="Break-even (1×)",
    )
    ax.set_ylabel("Token savings factor (higher = fewer tokens)")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([_task_label(t) for t in tasks], rotation=30, ha="right")
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# =============================================================================
# 8. Summary dashboard  (one-page headline numbers, clean layout)
# =============================================================================


def plot_summary_dashboard(
    results: list[EvalResult],
    output_path: Path | None = None,
    title: str = "RLM Evaluation Summary",
) -> plt.Figure:
    """
    3-col top row (accuracy | tokens | latency) + full-width stats row.
    """
    stats = compute_statistics(results)
    runners = _sort_runners(list(stats.get("by_runner", {}).keys()))
    by_r = stats.get("by_runner", {})

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)

    # GridSpec: 2 rows — top row 3 equal cols, bottom row full width
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35, height_ratios=[1.6, 1])

    ax_acc = fig.add_subplot(gs[0, 0])
    ax_tok = fig.add_subplot(gs[0, 1])
    ax_lat = fig.add_subplot(gs[0, 2])
    ax_txt = fig.add_subplot(gs[1, :])  # spans all 3 columns
    ax_txt.axis("off")

    def _bars(ax: plt.Axes, values: list[float], ylabel: str, fmt: str, title_: str) -> None:
        colors = [_color(r) for r in runners]
        bars = ax.bar(range(len(runners)), values, color=colors, alpha=0.88, width=0.55)
        vmax = max(values) if max(values) > 0 else 1
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + vmax * 0.02,
                fmt.format(val),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
        ax.set_xticks(range(len(runners)))
        ax.set_xticklabels([_label(r) for r in runners], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title_, fontweight="bold", pad=10)

    accs = [by_r.get(r, {}).get("overall_accuracy", 0) for r in runners]
    toks = [by_r.get(r, {}).get("avg_tokens_per_task", 0) for r in runners]
    lats = [by_r.get(r, {}).get("avg_time_per_task", 0) for r in runners]

    _bars(ax_acc, accs, "Accuracy (%)", "{:.1f}%", "Overall Accuracy")
    ax_acc.set_ylim(0, 115)
    _bars(ax_tok, toks, "Avg Tokens", "{:,.0f}", "Avg Tokens per Task")
    _bars(ax_lat, lats, "Avg Latency (s)", "{:.1f}s", "Avg Latency per Task")

    # ── Key stats text (full-width bottom row) ──────────────────────────────
    model = stats.get("model", "N/A")
    total = stats.get("total_evaluations", 0)
    lines = [f"Model: {model}   |   Total Evaluations: {total}", ""]

    our_runners = [r for r in runners if r not in ("vanilla", "official")]
    for our in our_runners:
        our_d = by_r.get(our, {})
        van_d = by_r.get("vanilla", {})
        off_d = by_r.get("official", {})
        our_tok = our_d.get("avg_tokens_per_task", 1) or 1
        our_acc = our_d.get("overall_accuracy", 0)

        bullets: list[str] = []
        if van_d:
            van_tok = van_d.get("avg_tokens_per_task", 0)
            van_acc = van_d.get("overall_accuracy", 0)
            if van_tok > our_tok:
                bullets.append(f"{van_tok / our_tok:.1f}× fewer tokens than Vanilla")
            if our_acc >= van_acc:
                bullets.append(f"matches or beats Vanilla accuracy ({our_acc:.1f}% vs {van_acc:.1f}%)")
        if off_d:
            off_tok = off_d.get("avg_tokens_per_task", 0)
            off_acc = off_d.get("overall_accuracy", 0)
            if off_tok > our_tok:
                bullets.append(f"{off_tok / our_tok:.1f}× fewer tokens than Official RLM")
            if our_acc >= off_acc:
                bullets.append(f"matches or beats Official RLM ({our_acc:.1f}% vs {off_acc:.1f}%)")
        our_cost = our_d.get("total_cost_usd")
        van_cost = van_d.get("total_cost_usd") if van_d else None
        if our_cost and van_cost and van_cost > 0:
            bullets.append(f"{van_cost / our_cost:.1f}× cheaper than Vanilla")

        for b in bullets:
            lines.append(f"✓  {_label(our)}  {b}")

    ax_txt.text(
        0.5,
        0.55,
        "\n".join(lines),
        transform=ax_txt.transAxes,
        fontsize=12,
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.9",
            "facecolor": "#EAF4FB",
            "alpha": 0.9,
            "edgecolor": "#2980B9",
        },
    )

    # Runner legend
    legend_patches = [mpatches.Patch(color=_color(r), label=_label(r)) for r in runners]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(runners),
        framealpha=0.9,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    if output_path:
        _save(fig, output_path)
    return fig


# =============================================================================
# Master function  — generates all plots
# =============================================================================


def plot_all(
    results: list[EvalResult],
    output_dir: Path,
    title_prefix: str = "RLM Evaluation",
) -> list[Path]:
    """
    Generate every plot and save to *output_dir*.
    Returns the list of paths that were actually written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def _run(name: str, fn, *args, **kwargs):
        path = output_dir / f"{name}.png"
        print(f"  Plotting {name}...", flush=True)
        fig = fn(*args, output_path=path, **kwargs)
        if fig is not None:
            saved.append(path)
            plt.close(fig)
        else:
            print("    (skipped — data not available)", flush=True)

    _run(
        "accuracy_per_task",
        plot_accuracy_per_task,
        results,
        title=f"{title_prefix} — Accuracy by Task",
    )
    _run(
        "tokens_per_task",
        plot_tokens_per_task,
        results,
        title=f"{title_prefix} — Avg Tokens per Task",
    )
    _run(
        "latency_per_task",
        plot_latency_per_task,
        results,
        title=f"{title_prefix} — Avg Latency per Task",
    )
    _run(
        "cost_per_task",
        plot_cost_per_task,
        results,
        title=f"{title_prefix} — Avg Cost per Query",
    )
    _run(
        "accuracy_vs_cost",
        plot_accuracy_vs_cost,
        results,
        title=f"{title_prefix} — Accuracy vs Cost",
    )
    _run(
        "accuracy_vs_latency",
        plot_accuracy_vs_latency,
        results,
        title=f"{title_prefix} — Accuracy vs Latency",
    )
    _run(
        "token_savings",
        plot_token_savings,
        results,
        title=f"{title_prefix} — Token Savings vs Baselines",
    )
    _run(
        "summary_dashboard",
        plot_summary_dashboard,
        results,
        title=f"{title_prefix} — Dashboard",
    )

    return saved


# =============================================================================
# Backward-compatible helpers (used by eval/run.py)
# =============================================================================


def plot_comprehensive_dashboard(
    results: list[EvalResult],
    output_dir: Path,
    title: str = "RLM Evaluation",
) -> list[Path]:
    """Alias kept for backward compatibility."""
    return plot_all(results, output_dir, title_prefix=title)


def load_and_visualize(json_path: Path, output_dir: Path) -> list[Path]:
    """Load results from *json_path* and generate all plots into *output_dir*."""
    with open(json_path) as f:
        data = json.load(f)
    results = [EvalResult.from_dict(d) for d in data]
    return plot_all(results, output_dir)


# =============================================================================
# CLI  —  python -m eval.plotting  <path>  [output_dir]
# =============================================================================


def _find_latest_json(root: Path) -> Path | None:
    """Recursively find the most-recently-modified .json file under *root*."""
    candidates = sorted(root.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate plots from an RLM evaluation JSON file or directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        help="Path to eval JSON file OR a directory containing one (auto-discovers the newest).",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output directory for plots. Defaults to <json_file_dir>/plots/.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.path)

    if input_path.is_dir():
        json_path = _find_latest_json(input_path)
        if json_path is None:
            print(f"Error: no .json files found under {input_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-selected: {json_path}")
    else:
        json_path = input_path

    if not json_path.exists():
        print(f"Error: {json_path} not found", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else json_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading  {json_path}")
    print(f"Output → {output_dir}/")
    paths = load_and_visualize(json_path, output_dir)

    print(f"\n✅  {len(paths)} plots saved:")
    for p in paths:
        print(f"   {p}")


if __name__ == "__main__":
    main()

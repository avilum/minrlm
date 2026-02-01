"""
Visualization Utilities for RLM Evaluation

Generates publication-quality plots:
- Accuracy comparison bar charts
- Token efficiency analysis
- Latency comparison
- Scaling analysis (context length vs accuracy)
- Comprehensive summary dashboard
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import EvalResult, aggregate_results, compute_statistics

# =============================================================================
# Style Configuration
# =============================================================================

STYLE = {
    "figure.figsize": (12, 8),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

# Color scheme - designed for clarity and accessibility
COLORS = {
    "vanilla": "#E74C3C",  # Red - baseline
    "ours": "#2ECC71",  # Green - our implementation
    "official": "#3498DB",  # Blue - official implementation
}

LABELS = {
    "vanilla": "Vanilla LLM",
    "ours": "minRLM",
    "official": "Official RLM",
}

MARKERS = {
    "vanilla": "o",
    "ours": "s",
    "official": "^",
}


def apply_style():
    """Apply consistent plot styling."""
    plt.rcParams.update(STYLE)


# =============================================================================
# Individual Plot Functions
# =============================================================================


def plot_accuracy_comparison(
    results: list[EvalResult], output_path: Path | None = None, title: str = "Accuracy Comparison"
) -> plt.Figure:
    """
    Bar chart comparing accuracy across methods and tasks.
    """
    apply_style()

    aggregated = aggregate_results(results)
    tasks = sorted(aggregated.keys())
    runners = sorted({r.runner_name for r in results})

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(tasks))
    width = 0.25

    for i, runner in enumerate(runners):
        accuracies = []
        errors = []
        for task in tasks:
            if runner in aggregated[task]:
                metrics = aggregated[task][runner]
                accuracies.append(metrics.accuracy)
                errors.append(metrics.accuracy_std)
            else:
                accuracies.append(0)
                errors.append(0)

        bars = ax.bar(
            x + i * width,
            accuracies,
            width,
            label=LABELS.get(runner, runner),
            color=COLORS.get(runner, f"C{i}"),
            yerr=errors if max(errors) > 0 else None,
            capsize=3,
            alpha=0.85,
        )

        # Add value labels
        for bar, val in zip(bars, accuracies):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"{val:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x + width * (len(runners) - 1) / 2)
    ax.set_xticklabels([t.upper().replace("_", " ") for t in tasks])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_token_efficiency(
    results: list[EvalResult], output_path: Path | None = None, title: str = "Token Usage Comparison"
) -> plt.Figure:
    """
    Stacked bar chart showing input vs output tokens.
    """
    apply_style()

    aggregated = aggregate_results(results)
    tasks = sorted(aggregated.keys())
    runners = sorted({r.runner_name for r in results})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Total tokens by task and method
    ax1 = axes[0]
    x = np.arange(len(tasks))
    width = 0.25

    for i, runner in enumerate(runners):
        tokens = []
        for task in tasks:
            if runner in aggregated[task]:
                tokens.append(aggregated[task][runner].avg_total_tokens)
            else:
                tokens.append(0)

        bars = ax1.bar(
            x + i * width,
            tokens,
            width,
            label=LABELS.get(runner, runner),
            color=COLORS.get(runner, f"C{i}"),
            alpha=0.85,
        )

        for bar, val in zip(bars, tokens):
            if val > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )

    ax1.set_ylabel("Total Tokens")
    ax1.set_title("Token Usage by Task", fontweight="bold")
    ax1.set_xticks(x + width * (len(runners) - 1) / 2)
    ax1.set_xticklabels([t.upper().replace("_", " ") for t in tasks])
    ax1.legend()
    ax1.set_yscale("log")

    # Right: Input vs Output breakdown
    ax2 = axes[1]

    # Calculate totals per runner
    runner_totals = {}
    for runner in runners:
        runner_results = [r for r in results if r.runner_name == runner]
        runner_totals[runner] = {
            "input": sum(r.input_tokens for r in runner_results),
            "output": sum(r.output_tokens for r in runner_results),
        }

    x = np.arange(len(runners))
    width = 0.6

    input_vals = [runner_totals[r]["input"] for r in runners]
    output_vals = [runner_totals[r]["output"] for r in runners]

    ax2.bar(x, input_vals, width, label="Input Tokens", color="#1976D2", alpha=0.8)
    ax2.bar(x, output_vals, width, bottom=input_vals, label="Output Tokens", color="#FFC107", alpha=0.8)

    for i, (inp, out) in enumerate(zip(input_vals, output_vals)):
        total = inp + out
        ax2.text(i, total, f"{total:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_ylabel("Total Tokens")
    ax2.set_title("Input vs Output Breakdown", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS.get(r, r) for r in runners])
    ax2.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_latency_comparison(
    results: list[EvalResult], output_path: Path | None = None, title: str = "Latency Comparison"
) -> plt.Figure:
    """
    Bar chart showing execution time across methods.
    """
    apply_style()

    aggregated = aggregate_results(results)
    tasks = sorted(aggregated.keys())
    runners = sorted({r.runner_name for r in results})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Time by task
    ax1 = axes[0]
    x = np.arange(len(tasks))
    width = 0.25

    for i, runner in enumerate(runners):
        times = []
        errors = []
        for task in tasks:
            if runner in aggregated[task]:
                metrics = aggregated[task][runner]
                times.append(metrics.avg_time_seconds)
                errors.append(metrics.std_time_seconds)
            else:
                times.append(0)
                errors.append(0)

        bars = ax1.bar(
            x + i * width,
            times,
            width,
            label=LABELS.get(runner, runner),
            color=COLORS.get(runner, f"C{i}"),
            yerr=errors if max(errors) > 0 else None,
            capsize=3,
            alpha=0.85,
        )

        for bar, val in zip(bars, times):
            if val > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}s",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Execution Time by Task", fontweight="bold")
    ax1.set_xticks(x + width * (len(runners) - 1) / 2)
    ax1.set_xticklabels([t.upper().replace("_", " ") for t in tasks])
    ax1.legend()

    # Right: Iterations (for RLM methods)
    ax2 = axes[1]

    for i, runner in enumerate(runners):
        iters = []
        for task in tasks:
            if runner in aggregated[task]:
                iters.append(aggregated[task][runner].avg_iterations)
            else:
                iters.append(0)

        bars = ax2.bar(
            x + i * width,
            iters,
            width,
            label=LABELS.get(runner, runner),
            color=COLORS.get(runner, f"C{i}"),
            alpha=0.85,
        )

    ax2.set_ylabel("Iterations")
    ax2.set_title("RLM Iterations by Task", fontweight="bold")
    ax2.set_xticks(x + width * (len(runners) - 1) / 2)
    ax2.set_xticklabels([t.upper().replace("_", " ") for t in tasks])
    ax2.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_scaling_analysis(
    results: list[EvalResult], output_path: Path | None = None, title: str = "Context Scaling Analysis"
) -> plt.Figure:
    """
    Line chart showing accuracy vs context size.
    Replicates Figure 1 from the RLM paper.
    """
    apply_style()

    # Group by runner and context size
    scaling_data: dict[str, dict[int, list[EvalResult]]] = {}

    for r in results:
        if r.runner_name not in scaling_data:
            scaling_data[r.runner_name] = {}
        size = r.context_size
        if size not in scaling_data[r.runner_name]:
            scaling_data[r.runner_name][size] = []
        scaling_data[r.runner_name][size].append(r)

    # Check if we have scaling data
    has_multiple_sizes = any(len(sizes) > 1 for sizes in scaling_data.values())

    if not has_multiple_sizes:
        # Fallback: just show accuracy comparison
        return plot_accuracy_comparison(results, output_path, title)

    fig, ax = plt.subplots(figsize=(10, 6))

    for runner, sizes in scaling_data.items():
        sorted_sizes = sorted(sizes.keys())
        x_vals = sorted_sizes
        y_vals = []

        for size in sorted_sizes:
            size_results = sizes[size]
            accuracy = sum(r.correct for r in size_results) / len(size_results) * 100
            y_vals.append(accuracy)

        ax.plot(
            x_vals,
            y_vals,
            marker=MARKERS.get(runner, "o"),
            color=COLORS.get(runner, "gray"),
            label=LABELS.get(runner, runner),
            linewidth=2,
            markersize=8,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Context Size (characters)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower left")
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)

    # Format x-axis with powers of 2
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"$2^{{{int(np.log2(x))}}}$" if x > 0 else "0"))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cost_comparison(
    results: list[EvalResult], output_path: Path | None = None, title: str = "Cost Analysis"
) -> plt.Figure | None:
    """
    Bar chart showing cost comparison across methods.
    Returns None if no cost data is available.
    """
    # Check if cost data is available
    costs_available = any(r.cost_usd is not None for r in results)
    if not costs_available:
        return None

    apply_style()

    runners = sorted({r.runner_name for r in results})

    # Calculate total cost per runner
    runner_costs = {}
    for runner in runners:
        runner_results = [r for r in results if r.runner_name == runner]
        costs = [r.cost_usd for r in runner_results if r.cost_usd is not None]
        if costs:
            runner_costs[runner] = {
                "total": sum(costs),
                "avg": sum(costs) / len(costs),
                "count": len(costs),
            }

    if not runner_costs:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Total cost
    ax1 = axes[0]
    runners_with_cost = list(runner_costs.keys())
    totals = [runner_costs[r]["total"] for r in runners_with_cost]
    colors = [COLORS.get(r, "gray") for r in runners_with_cost]

    bars = ax1.bar(runners_with_cost, totals, color=colors, alpha=0.85)
    for bar, val in zip(bars, totals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(), f"${val:.4f}", ha="center", va="bottom", fontsize=10
        )

    ax1.set_ylabel("Total Cost (USD)")
    ax1.set_title("Total Cost by Runner", fontweight="bold")
    ax1.set_xticks(range(len(runners_with_cost)))
    ax1.set_xticklabels([LABELS.get(r, r) for r in runners_with_cost])

    # Right: Cost per query
    ax2 = axes[1]
    avgs = [runner_costs[r]["avg"] * 1000 for r in runners_with_cost]  # Cost per 1000 queries

    bars = ax2.bar(runners_with_cost, avgs, color=colors, alpha=0.85)
    for bar, val in zip(bars, avgs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(), f"${val:.2f}", ha="center", va="bottom", fontsize=10
        )

    ax2.set_ylabel("Cost per 1000 Queries (USD)")
    ax2.set_title("Cost per 1000 Queries", fontweight="bold")
    ax2.set_xticks(range(len(runners_with_cost)))
    ax2.set_xticklabels([LABELS.get(r, r) for r in runners_with_cost])

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_comprehensive_dashboard(
    results: list[EvalResult], output_dir: Path, title: str = "RLM Evaluation Dashboard"
) -> list[Path]:
    """
    Generate a comprehensive set of plots.

    Returns list of saved plot paths.
    """
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    # 1. Accuracy comparison
    path = output_dir / "accuracy_comparison.png"
    plot_accuracy_comparison(results, path, "Accuracy Comparison by Method")
    saved_paths.append(path)

    # 2. Token efficiency
    path = output_dir / "token_efficiency.png"
    plot_token_efficiency(results, path, "Token Usage Analysis")
    saved_paths.append(path)

    # 3. Latency comparison
    path = output_dir / "latency_comparison.png"
    plot_latency_comparison(results, path, "Execution Time Analysis")
    saved_paths.append(path)

    # 4. Scaling analysis (if applicable)
    scaling_results = [r for r in results if "scaling" in r.task_name.lower()]
    if scaling_results:
        path = output_dir / "scaling_analysis.png"
        plot_scaling_analysis(scaling_results, path, "Context Scaling Analysis")
        saved_paths.append(path)

    # 5. Cost comparison (if cost data available)
    path = output_dir / "cost_comparison.png"
    fig = plot_cost_comparison(results, path, "Cost Analysis")
    if fig is not None:
        saved_paths.append(path)
        plt.close(fig)

    # 6. Summary dashboard
    path = output_dir / "summary_dashboard.png"
    _plot_summary_dashboard(results, path, title)
    saved_paths.append(path)

    return saved_paths


def _plot_summary_dashboard(results: list[EvalResult], output_path: Path, title: str):
    """Generate a summary dashboard with key metrics."""
    apply_style()

    stats = compute_statistics(results)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # Grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Accuracy by runner (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    runners = list(stats.get("by_runner", {}).keys())
    accuracies = [stats["by_runner"][r].get("overall_accuracy", 0) for r in runners]
    colors = [COLORS.get(r, "gray") for r in runners]

    bars = ax1.bar(runners, accuracies, color=colors, alpha=0.85)
    for bar, val in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.1f}%", ha="center", fontsize=10)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Overall Accuracy", fontweight="bold")
    ax1.set_ylim(0, 110)
    ax1.set_xticks(range(len(runners)))
    ax1.set_xticklabels([LABELS.get(r, r) for r in runners])

    # 2. Token usage (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    tokens = [stats["by_runner"][r].get("avg_tokens_per_task", 0) for r in runners]

    bars = ax2.bar(runners, tokens, color=colors, alpha=0.85)
    for bar, val in zip(bars, tokens):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,.0f}", ha="center", va="bottom", fontsize=9
        )
    ax2.set_ylabel("Avg Tokens per Task")
    ax2.set_title("Token Usage", fontweight="bold")
    ax2.set_xticks(range(len(runners)))
    ax2.set_xticklabels([LABELS.get(r, r) for r in runners])

    # 3. Time (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    times = [stats["by_runner"][r].get("avg_time_per_task", 0) for r in runners]

    bars = ax3.bar(runners, times, color=colors, alpha=0.85)
    for bar, val in zip(bars, times):
        ax3.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}s", ha="center", va="bottom", fontsize=9
        )
    ax3.set_ylabel("Avg Time (seconds)")
    ax3.set_title("Execution Time", fontweight="bold")
    ax3.set_xticks(range(len(runners)))
    ax3.set_xticklabels([LABELS.get(r, r) for r in runners])

    # 4-6. Task breakdown (middle row)
    tasks = list(stats.get("by_task", {}).keys())[:3]

    for i, task in enumerate(tasks):
        ax = fig.add_subplot(gs[1, i])
        task_data = stats["by_task"][task]
        task_runners = list(task_data.keys())
        task_accuracies = [task_data[r].get("accuracy", 0) for r in task_runners]
        task_colors = [COLORS.get(r, "gray") for r in task_runners]

        bars = ax.bar(range(len(task_runners)), task_accuracies, color=task_colors, alpha=0.85)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{task.upper().replace('_', ' ')}", fontweight="bold")
        ax.set_xticks(range(len(task_runners)))
        ax.set_xticklabels([LABELS.get(r, r) for r in task_runners], fontsize=8)
        ax.set_ylim(0, 110)

    # 7. Summary text (bottom row)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")

    # Build summary text
    summary_lines = [
        f"Model: {stats.get('model', 'N/A')}",
        f"Total Evaluations: {stats.get('total_evaluations', 0)}",
        "",
    ]

    # Token efficiency
    vanilla_tokens = stats.get("by_runner", {}).get("vanilla", {}).get("avg_tokens_per_task", 0)
    ours_tokens = stats.get("by_runner", {}).get("ours", {}).get("avg_tokens_per_task", 1)
    official_tokens = stats.get("by_runner", {}).get("official", {}).get("avg_tokens_per_task", 0)

    if vanilla_tokens > 0 and ours_tokens > 0:
        summary_lines.append(f"✓ minRLM uses {vanilla_tokens / ours_tokens:.1f}x fewer tokens than Vanilla LLM")
    if official_tokens > 0 and ours_tokens > 0:
        summary_lines.append(f"✓ minRLM uses {official_tokens / ours_tokens:.1f}x fewer tokens than Official RLM")

    # Cost efficiency
    vanilla_cost = stats.get("by_runner", {}).get("vanilla", {}).get("total_cost_usd")
    ours_cost = stats.get("by_runner", {}).get("ours", {}).get("total_cost_usd")

    if vanilla_cost is not None and ours_cost is not None and ours_cost > 0:
        cost_ratio = vanilla_cost / ours_cost
        summary_lines.append(
            f"✓ minRLM is {cost_ratio:.1f}x cheaper than Vanilla (${ours_cost:.4f} vs ${vanilla_cost:.4f})"
        )
    elif ours_cost is None:
        summary_lines.append("⚠ Cost data unavailable (model not in tokencost)")

    summary_text = "\n".join(summary_lines)
    ax7.text(
        0.5,
        0.5,
        summary_text,
        transform=ax7.transAxes,
        fontsize=12,
        ha="center",
        va="center",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Utility Functions
# =============================================================================


def load_and_visualize(json_path: Path, output_dir: Path) -> list[Path]:
    """Load results from JSON and generate all visualizations."""
    with open(json_path) as f:
        data = json.load(f)

    results = [EvalResult.from_dict(d) for d in data]
    return plot_comprehensive_dashboard(results, output_dir)

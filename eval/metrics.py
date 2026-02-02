"""
Metrics Collection and Analysis for RLM Evaluation

Provides:
- EvalResult: Structured result from a single evaluation run
- Aggregation functions for multiple runs
- Statistical analysis utilities
- Report generation
- Cost calculation via tokencost
"""

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# Cost calculation
try:
    from tokencost import calculate_completion_cost, calculate_prompt_cost

    TOKENCOST_AVAILABLE = True
except ImportError:
    TOKENCOST_AVAILABLE = False


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """
    Calculate cost for a completion using tokencost.

    Returns None if model is not supported or tokencost is unavailable.
    """
    if not TOKENCOST_AVAILABLE:
        return None

    try:
        # tokencost expects actual text, but we only have token counts
        # Use a workaround: create dummy strings of the right length
        # (tokencost will tokenize and may differ slightly, but close enough)
        input_cost = calculate_prompt_cost(prompt="x" * input_tokens, model=model)
        output_cost = calculate_completion_cost(completion="x" * output_tokens, model=model)
        return float(input_cost + output_cost)
    except Exception:
        # Model not in tokencost database
        return None


@dataclass
class EvalResult:
    """Result of evaluating one task with one method."""

    # Task info
    task_name: str
    task_instance_seed: int

    # Runner info
    runner_name: str
    model: str

    # Result
    correct: bool
    partial_score: float  # 0.0 - 1.0
    response: str
    expected: str

    # Metrics
    total_tokens: int
    input_tokens: int
    output_tokens: int
    time_seconds: float
    iterations: int

    # Metadata
    error: str | None = None
    context_size: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)

    # Cost (None if model not supported by tokencost)
    cost_usd: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalResult":
        return cls(**data)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple runs."""

    task_name: str
    runner_name: str
    model: str

    # Counts
    total_runs: int
    successful_runs: int

    # Accuracy
    accuracy: float  # percentage
    accuracy_std: float
    avg_partial_score: float

    # Tokens
    avg_total_tokens: float
    avg_input_tokens: float
    avg_output_tokens: float
    std_total_tokens: float

    # Time
    avg_time_seconds: float
    std_time_seconds: float

    # Iterations (for RLM methods)
    avg_iterations: float

    # Efficiency ratios (compared to baseline)
    token_efficiency: float | None = None  # baseline_tokens / our_tokens
    time_efficiency: float | None = None  # our_time / baseline_time

    # Cost (None if not calculable)
    avg_cost_usd: float | None = None
    total_cost_usd: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate_results(
    results: list[EvalResult], baseline_results: list[EvalResult] | None = None
) -> dict[str, dict[str, AggregatedMetrics]]:
    """
    Aggregate results by task and runner.

    Returns: {task_name: {runner_name: AggregatedMetrics}}
    """
    # Group by task and runner
    grouped: dict[str, dict[str, list[EvalResult]]] = {}
    for r in results:
        if r.task_name not in grouped:
            grouped[r.task_name] = {}
        if r.runner_name not in grouped[r.task_name]:
            grouped[r.task_name][r.runner_name] = []
        grouped[r.task_name][r.runner_name].append(r)

    # Compute baseline metrics if provided
    baseline_metrics: dict[str, AggregatedMetrics] = {}
    if baseline_results:
        baseline_grouped: dict[str, list[EvalResult]] = {}
        for r in baseline_results:
            if r.task_name not in baseline_grouped:
                baseline_grouped[r.task_name] = []
            baseline_grouped[r.task_name].append(r)

        for task, task_results in baseline_grouped.items():
            baseline_metrics[task] = _compute_metrics(task, "baseline", task_results[0].model, task_results)

    # Compute metrics for each group
    aggregated: dict[str, dict[str, AggregatedMetrics]] = {}
    for task, runners in grouped.items():
        aggregated[task] = {}

        # Get baseline for efficiency comparison
        baseline = baseline_metrics.get(task)

        for runner, runner_results in runners.items():
            metrics = _compute_metrics(task, runner, runner_results[0].model, runner_results)

            # Compute efficiency ratios vs baseline
            if baseline and runner != "vanilla":
                if metrics.avg_total_tokens > 0:
                    metrics.token_efficiency = baseline.avg_total_tokens / metrics.avg_total_tokens
                if baseline.avg_time_seconds > 0:
                    metrics.time_efficiency = metrics.avg_time_seconds / baseline.avg_time_seconds

            aggregated[task][runner] = metrics

    return aggregated


def _compute_metrics(task_name: str, runner_name: str, model: str, results: list[EvalResult]) -> AggregatedMetrics:
    """Compute aggregated metrics for a list of results."""
    if not results:
        return AggregatedMetrics(
            task_name=task_name,
            runner_name=runner_name,
            model=model,
            total_runs=0,
            successful_runs=0,
            accuracy=0,
            accuracy_std=0,
            avg_partial_score=0,
            avg_total_tokens=0,
            avg_input_tokens=0,
            avg_output_tokens=0,
            std_total_tokens=0,
            avg_time_seconds=0,
            std_time_seconds=0,
            avg_iterations=0,
        )

    n = len(results)
    correct_list = [1 if r.correct else 0 for r in results]

    # Calculate cost metrics
    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    avg_cost = statistics.mean(costs) if costs else None
    total_cost = sum(costs) if costs else None

    return AggregatedMetrics(
        task_name=task_name,
        runner_name=runner_name,
        model=model,
        total_runs=n,
        successful_runs=sum(1 for r in results if r.error is None),
        accuracy=sum(correct_list) / n * 100,
        accuracy_std=statistics.stdev(correct_list) * 100 if n > 1 else 0,
        avg_partial_score=statistics.mean(r.partial_score for r in results),
        avg_total_tokens=statistics.mean(r.total_tokens for r in results),
        avg_input_tokens=statistics.mean(r.input_tokens for r in results),
        avg_output_tokens=statistics.mean(r.output_tokens for r in results),
        std_total_tokens=statistics.stdev(r.total_tokens for r in results) if n > 1 else 0,
        avg_time_seconds=statistics.mean(r.time_seconds for r in results),
        std_time_seconds=statistics.stdev(r.time_seconds for r in results) if n > 1 else 0,
        avg_iterations=statistics.mean(r.iterations for r in results),
        avg_cost_usd=avg_cost,
        total_cost_usd=total_cost,
    )


def compute_statistics(results: list[EvalResult]) -> dict:
    """
    Compute comprehensive statistics from evaluation results.

    Returns a dict suitable for JSON serialization and reporting.
    """
    if not results:
        return {"error": "No results"}

    # Basic counts
    stats = {
        "total_evaluations": len(results),
        "unique_tasks": len({r.task_name for r in results}),
        "unique_runners": len({r.runner_name for r in results}),
        "model": results[0].model,
        "timestamp": datetime.now().isoformat(),
    }

    # Per-task, per-runner breakdown
    aggregated = aggregate_results(results, baseline_results=[r for r in results if r.runner_name == "vanilla"])

    stats["by_task"] = {}
    for task, runners in aggregated.items():
        stats["by_task"][task] = {runner: metrics.to_dict() for runner, metrics in runners.items()}

    # Overall summary per runner
    stats["by_runner"] = {}
    for runner_name in {r.runner_name for r in results}:
        runner_results = [r for r in results if r.runner_name == runner_name]

        # Calculate costs
        costs = [r.cost_usd for r in runner_results if r.cost_usd is not None]
        total_cost = sum(costs) if costs else None
        avg_cost = statistics.mean(costs) if costs else None

        stats["by_runner"][runner_name] = {
            "total_runs": len(runner_results),
            "overall_accuracy": sum(r.correct for r in runner_results) / len(runner_results) * 100,
            "total_tokens_used": sum(r.total_tokens for r in runner_results),
            "avg_tokens_per_task": statistics.mean(r.total_tokens for r in runner_results),
            "avg_input_tokens": statistics.mean(r.input_tokens for r in runner_results),
            "avg_output_tokens": statistics.mean(r.output_tokens for r in runner_results),
            "avg_time_per_task": statistics.mean(r.time_seconds for r in runner_results),
            "avg_iterations": statistics.mean(r.iterations for r in runner_results),
            "total_cost_usd": total_cost,
            "avg_cost_usd": avg_cost,
        }

    # Efficiency comparisons
    vanilla_stats = stats["by_runner"].get("vanilla", {})
    if vanilla_stats and vanilla_stats.get("avg_tokens_per_task", 0) > 0:
        baseline_tokens = vanilla_stats["avg_tokens_per_task"]

        for runner_name, runner_stats in stats["by_runner"].items():
            if runner_name != "vanilla" and runner_stats.get("avg_tokens_per_task", 0) > 0:
                runner_stats["token_efficiency_vs_vanilla"] = baseline_tokens / runner_stats["avg_tokens_per_task"]

    return stats


def save_results(results: list[EvalResult], output_dir: Path, prefix: str = "eval") -> tuple[Path, Path]:
    """
    Save results to JSON and generate a summary markdown report.

    Returns: (json_path, summary_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw results
    json_path = output_dir / f"{prefix}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    # Generate summary report
    summary_path = output_dir / f"summary_{timestamp}.md"
    stats = compute_statistics(results)

    with open(summary_path, "w") as f:
        f.write(_generate_markdown_report(stats, results))

    return json_path, summary_path


def _generate_markdown_report(stats: dict, results: list[EvalResult]) -> str:
    """Generate a markdown summary report."""
    # Check if cost data is available
    has_cost = any(data.get("total_cost_usd") is not None for data in stats.get("by_runner", {}).values())

    lines = [
        "# RLM Evaluation Report",
        "",
        f"**Generated**: {stats.get('timestamp', 'N/A')}",
        f"**Model**: {stats.get('model', 'N/A')}",
        f"**Total Evaluations**: {stats.get('total_evaluations', 0)}",
        "",
        "## Summary by Runner",
        "",
    ]

    if has_cost:
        lines.append("| Runner | Accuracy | Avg Tokens | Avg Time | Total Cost | Token Efficiency |")
        lines.append("|--------|----------|------------|----------|------------|------------------|")
    else:
        lines.append("| Runner | Accuracy | Avg Tokens | Avg Time | Token Efficiency |")
        lines.append("|--------|----------|------------|----------|------------------|")

    for runner, data in stats.get("by_runner", {}).items():
        efficiency = data.get("token_efficiency_vs_vanilla", 1.0)
        efficiency_str = f"{efficiency:.2f}x" if efficiency != 1.0 else "-"
        cost = data.get("total_cost_usd")
        cost_str = f"${cost:.6f}" if cost is not None else "N/A"

        if has_cost:
            lines.append(
                f"| {runner} | {data.get('overall_accuracy', 0):.1f}% | "
                f"{data.get('avg_tokens_per_task', 0):.0f} | "
                f"{data.get('avg_time_per_task', 0):.1f}s | "
                f"{cost_str} | "
                f"{efficiency_str} |"
            )
        else:
            lines.append(
                f"| {runner} | {data.get('overall_accuracy', 0):.1f}% | "
                f"{data.get('avg_tokens_per_task', 0):.0f} | "
                f"{data.get('avg_time_per_task', 0):.1f}s | "
                f"{efficiency_str} |"
            )

    lines.extend(
        [
            "",
            "## Results by Task",
            "",
        ]
    )

    for task, runners in stats.get("by_task", {}).items():
        lines.extend(
            [
                f"### {task.upper()}",
                "",
                "| Runner | Accuracy | Tokens | Time | Iterations |",
                "|--------|----------|--------|------|------------|",
            ]
        )

        for runner, data in runners.items():
            lines.append(
                f"| {runner} | {data.get('accuracy', 0):.1f}% | "
                f"{data.get('avg_total_tokens', 0):.0f} | "
                f"{data.get('avg_time_seconds', 0):.1f}s | "
                f"{data.get('avg_iterations', 0):.1f} |"
            )

        lines.append("")

    # Context size analysis
    lines.extend(
        [
            "",
            "## Context Size Analysis",
            "",
        ]
    )
    
    # Group results by context size and runner
    size_analysis: dict[int, dict[str, list[EvalResult]]] = {}
    for r in results:
        if r.context_size not in size_analysis:
            size_analysis[r.context_size] = {}
        if r.runner_name not in size_analysis[r.context_size]:
            size_analysis[r.context_size][r.runner_name] = []
        size_analysis[r.context_size][r.runner_name].append(r)
    
    if len(size_analysis) > 1:
        lines.append("| Context Size | Vanilla Accuracy | RLM Accuracy | RLM Advantage |")
        lines.append("|--------------|------------------|--------------|---------------|")
        
        for size in sorted(size_analysis.keys()):
            vanilla_results = size_analysis[size].get("vanilla", [])
            rlm_results = size_analysis[size].get("ours", []) or size_analysis[size].get("official", [])
            
            vanilla_acc = sum(r.correct for r in vanilla_results) / len(vanilla_results) * 100 if vanilla_results else 0
            rlm_acc = sum(r.correct for r in rlm_results) / len(rlm_results) * 100 if rlm_results else 0
            
            if vanilla_results and rlm_results:
                advantage = rlm_acc - vanilla_acc
                advantage_str = f"+{advantage:.1f}%" if advantage > 0 else f"{advantage:.1f}%"
                size_str = f"{size // 1024}K" if size < 1024 * 1024 else f"{size // (1024 * 1024)}M"
                lines.append(f"| {size_str} | {vanilla_acc:.1f}% | {rlm_acc:.1f}% | {advantage_str} |")
    
    # Key findings
    lines.extend(
        [
            "",
            "## Key Findings",
            "",
        ]
    )

    vanilla = stats.get("by_runner", {}).get("vanilla", {})
    ours = stats.get("by_runner", {}).get("ours", {})
    official = stats.get("by_runner", {}).get("official", {})

    if vanilla and ours:
        token_ratio = vanilla.get("avg_tokens_per_task", 0) / max(ours.get("avg_tokens_per_task", 1), 1)
        lines.append(f"- **minRLM** uses **{token_ratio:.1f}x fewer tokens** than vanilla LLM")

    if official and ours:
        official_tokens = official.get("avg_tokens_per_task", 0)
        ours_tokens = ours.get("avg_tokens_per_task", 1)
        if official_tokens > ours_tokens:
            lines.append(f"- **minRLM** uses **{official_tokens / ours_tokens:.1f}x fewer tokens** than official RLM")

    if ours:
        lines.append(f"- **Average iterations**: {ours.get('avg_iterations', 0):.1f} per task")

    # Cost findings
    vanilla_cost = vanilla.get("total_cost_usd")
    ours_cost = ours.get("total_cost_usd")
    official_cost = official.get("total_cost_usd")

    if vanilla_cost is not None and ours_cost is not None and ours_cost > 0:
        cost_ratio = vanilla_cost / ours_cost
        lines.append(
            f"- **Cost savings**: minRLM is **{cost_ratio:.1f}x cheaper** than vanilla (${ours_cost:.6f} vs ${vanilla_cost:.6f})"
        )
    elif ours_cost is None:
        lines.append("- **Cost**: Unable to calculate (model not in tokencost database)")

    return "\n".join(lines)


def load_results(path: Path) -> list[EvalResult]:
    """Load results from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [EvalResult.from_dict(d) for d in data]

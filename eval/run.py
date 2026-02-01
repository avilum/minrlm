#!/usr/bin/env python3
"""
RLM Evaluation Suite - Main Entry Point

A reproducible benchmark comparing:
1. Vanilla LLM (direct API calls)
2. Our minimal RLM implementation
3. Official RLM implementation (optional)

Usage:
    # Quick start
    uv run python eval/run.py --model gpt-5-nano

    # Full evaluation with multiple runs
    uv run python eval/run.py --model gpt-5-nano --runs 3 --tasks all

    # Skip official RLM (if not installed)
    uv run python eval/run.py --model gpt-5-nano --skip-official

    # Custom context sizes for scaling test
    uv run python eval/run.py --model gpt-5-nano --tasks scaling --context-sizes 8192,16384,32768

    # Extended evaluation (8K to 256K contexts)
    uv run python eval/run.py --model gpt-5-nano --tasks scaling --extended

    # Long context stress test (128K-256K)
    uv run python eval/run.py --model gpt-5-nano --tasks long_context --context-sizes 131072,262144

    # Multi-needle at large scale
    uv run python eval/run.py --model gpt-5-nano --tasks multi_needle_long

    # Output to specific directory
    uv run python eval/run.py --model gpt-5-nano --output-dir my_results/
"""

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# Ensure our module is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


from eval.metrics import EvalResult, calculate_cost, compute_statistics, save_results
from eval.runners import RUNNER_REGISTRY, get_runner
from eval.tasks import TASK_REGISTRY, get_task
from eval.visualize import plot_comprehensive_dashboard

# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
log = logging.getLogger(__name__)


# =============================================================================
# CLI Parser
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="RLM Evaluation Suite - Compare LLM implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python eval/run.py --model gpt-5-nano
  uv run python eval/run.py --model gpt-5-nano --runs 3 --tasks all
  uv run python eval/run.py --model gpt-5-nano --skip-official
        """,
    )

    parser.add_argument(
        "--model", "-m", default="gpt-5-nano", help="Model to use for evaluation (e.g., gpt-5-nano, gpt-5-nano)"
    )

    parser.add_argument(
        "--tasks",
        "-t",
        default="sniah,multi_needle,pairs",
        help=f"Tasks to run (comma-separated). Options: {', '.join(TASK_REGISTRY.keys())}, all",
    )

    parser.add_argument(
        "--runners",
        "-r",
        default="vanilla,ours,official",
        help=f"Runners to compare (comma-separated). Options: {', '.join(RUNNER_REGISTRY.keys())}, all",
    )

    parser.add_argument(
        "--runs", "-n", type=int, default=1, help="Number of runs per task/runner combination (default: 1)"
    )

    parser.add_argument("--output-dir", "-o", default="eval/results", help="Output directory for results and plots")

    parser.add_argument(
        "--skip-official", action="store_true", help="Skip official RLM runner (useful if not installed)"
    )

    parser.add_argument(
        "--context-sizes",
        default="8192,16384,32768,65536,131072",
        help="Context sizes for scaling benchmark (comma-separated). Use --context-sizes large for extended test.",
    )

    parser.add_argument(
        "--extended",
        action="store_true",
        help="Run extended scaling tests (8K to 256K contexts)",
    )

    parser.add_argument("--context-size", type=int, default=50000, help="Default context size for non-scaling tasks")

    parser.add_argument("--no-plot", action="store_true", help="Skip generating visualization plots")

    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output verbosity")

    return parser.parse_args()


# =============================================================================
# Main Evaluation Logic
# =============================================================================


def run_evaluation(
    model: str,
    tasks: list[str],
    runners: list[str],
    runs: int = 1,
    output_dir: Path | None = None,
    context_size: int = 50000,
    context_sizes: list[int] | None = None,
    verbose: bool = True,
) -> list[EvalResult]:
    """
    Run the full evaluation suite.

    Args:
        model: Model name to use
        tasks: List of task names to run
        runners: List of runner names to compare
        runs: Number of runs per configuration
        output_dir: Directory for output files
        context_size: Default context size
        context_sizes: List of sizes for scaling test
        verbose: Print progress

    Returns:
        List of EvalResult objects
    """
    all_results: list[EvalResult] = []

    if verbose:
        log.info("=" * 70)
        log.info("RLM EVALUATION SUITE")
        log.info("=" * 70)
        log.info(f"Model: {model}")
        log.info(f"Tasks: {tasks}")
        log.info(f"Runners: {runners}")
        log.info(f"Runs per config: {runs}")
        log.info("=" * 70)

    # Initialize runners
    active_runners = {}
    for runner_name in runners:
        try:
            runner = get_runner(runner_name, model)
            if runner.warmup():
                active_runners[runner_name] = runner
                if verbose:
                    log.info(f"✓ Runner '{runner_name}' ready")
            else:
                if verbose:
                    log.warning(f"✗ Runner '{runner_name}' not available")
        except Exception as e:
            if verbose:
                log.warning(f"✗ Runner '{runner_name}' failed to initialize: {e}")

    if not active_runners:
        log.error("No runners available!")
        return []

    # Run evaluations
    for task_name in tqdm(tasks, desc="Tasks", disable=not verbose):
        # Handle scaling task specially
        if task_name == "scaling":
            sizes = context_sizes or [8192, 16384, 32768, 65536]
            for size in sizes:
                results = _run_task_evaluations(
                    task_name=f"scaling_{size}",
                    task_kwargs={"context_size": size},
                    runners=active_runners,
                    model=model,
                    runs=runs,
                    verbose=verbose,
                )
                all_results.extend(results)
        elif task_name == "long_context":
            # Test at multiple large context sizes with position variation
            long_sizes = context_sizes or [131072, 262144]
            positions = ["start", "middle", "end"]
            for size in long_sizes:
                for pos in positions:
                    results = _run_task_evaluations(
                        task_name=f"long_context_{size // 1024}k_{pos}",
                        task_kwargs={"context_size": size, "position": pos},
                        runners=active_runners,
                        model=model,
                        runs=runs,
                        verbose=verbose,
                    )
                    all_results.extend(results)
        elif task_name == "multi_needle_long":
            # Test multi-needle at large context sizes
            long_sizes = context_sizes or [131072, 262144]
            for size in long_sizes:
                results = _run_task_evaluations(
                    task_name=f"multi_needle_{size // 1024}k",
                    task_kwargs={"context_size": size, "num_needles": 10},
                    runners=active_runners,
                    model=model,
                    runs=runs,
                    verbose=verbose,
                )
                all_results.extend(results)
        elif task_name == "json_extraction":
            # JSON extraction at various sizes
            json_sizes = context_sizes or [50000, 100000, 200000]
            for size in json_sizes:
                results = _run_task_evaluations(
                    task_name=f"json_extraction_{size // 1000}k",
                    task_kwargs={"context_size": size},
                    runners=active_runners,
                    model=model,
                    runs=runs,
                    verbose=verbose,
                )
                all_results.extend(results)
        elif task_name == "json_aggregation":
            # JSON aggregation at various sizes
            json_sizes = context_sizes or [50000, 100000, 200000]
            for size in json_sizes:
                results = _run_task_evaluations(
                    task_name=f"json_aggregation_{size // 1000}k",
                    task_kwargs={"context_size": size},
                    runners=active_runners,
                    model=model,
                    runs=runs,
                    verbose=verbose,
                )
                all_results.extend(results)
        else:
            results = _run_task_evaluations(
                task_name=task_name,
                task_kwargs={"context_size": context_size},
                runners=active_runners,
                model=model,
                runs=runs,
                verbose=verbose,
            )
            all_results.extend(results)

    return all_results


def _get_base_task_name(task_name: str) -> str:
    """Extract the registered task name from a parameterized task name."""
    # Map parameterized task names to their base registered names
    if task_name.startswith("scaling_"):
        return "scaling"
    elif task_name.startswith("long_context_"):
        return "long_context"
    elif task_name.startswith("multi_needle_") and task_name not in ("multi_needle", "multi_needle_long"):
        return "multi_needle_long"
    elif task_name.startswith("json_extraction_"):
        return "json_extraction"
    elif task_name.startswith("json_aggregation_"):
        return "json_aggregation"
    else:
        return task_name


def _run_task_evaluations(
    task_name: str,
    task_kwargs: dict,
    runners: dict,
    model: str,
    runs: int,
    verbose: bool,
) -> list[EvalResult]:
    """Run evaluations for a single task across all runners."""
    results = []

    # Get base task name (handle parameterized names like scaling_8192, json_extraction_100k)
    base_task_name = _get_base_task_name(task_name)

    try:
        task = get_task(base_task_name)
    except ValueError:
        log.warning(f"Skipping unknown task: {base_task_name}")
        return []

    if verbose:
        log.info(f"\n{'=' * 60}")
        log.info(f"TASK: {task_name.upper()}")
        log.info(f"{'=' * 60}")

    run_pbar = tqdm(range(runs), desc=f"{task_name}", leave=False, disable=not verbose)
    for run_idx in run_pbar:
        seed = 42 + run_idx * 100

        # Generate task instance
        instance = task.generate(seed=seed, **task_kwargs)

        run_pbar.set_postfix({"context": f"{len(instance.context):,} chars", "run": f"{run_idx + 1}/{runs}"})

        for runner_name, runner in tqdm(runners.items(), desc="  runners", leave=False, disable=not verbose):
            # Execute
            run_result = runner.run(instance.task, instance.context)

            # Check correctness
            correct = task.check(run_result.response, instance.expected)
            partial_score = task.check_partial(run_result.response, instance.expected)

            # Calculate cost
            cost = calculate_cost(model, run_result.input_tokens, run_result.output_tokens)

            # Create result
            eval_result = EvalResult(
                task_name=task_name,
                task_instance_seed=seed,
                runner_name=runner_name,
                model=model,
                correct=correct,
                partial_score=partial_score,
                response=run_result.response,
                expected=instance.expected,
                total_tokens=run_result.total_tokens,
                input_tokens=run_result.input_tokens,
                output_tokens=run_result.output_tokens,
                time_seconds=run_result.time_seconds,
                iterations=run_result.iterations,
                error=run_result.error,
                context_size=len(instance.context),
                metadata=instance.metadata or {},
                cost_usd=cost,
            )
            results.append(eval_result)

            if verbose:
                status = "✓" if correct else "✗"
                tqdm.write(
                    f"    {runner_name}: {status} | "
                    f"{run_result.input_tokens:,}+{run_result.output_tokens:,} tokens | "
                    f"{run_result.time_seconds:.1f}s | "
                    f"{run_result.iterations} iters"
                )

    return results


def print_summary(results: list[EvalResult]):
    """Print a summary table of results."""
    stats = compute_statistics(results)

    # Check if cost data is available
    has_cost = any(data.get("total_cost_usd") is not None for data in stats.get("by_runner", {}).values())

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\nModel: {stats.get('model', 'N/A')}")
    print(f"Total Evaluations: {stats.get('total_evaluations', 0)}")

    print("\n" + "-" * 80)
    if has_cost:
        print(
            f"{'Runner':<15} {'Accuracy':>10} {'Avg Tokens':>12} {'Avg Time':>10} {'Total Cost':>12} {'Token Eff':>10}"
        )
    else:
        print(f"{'Runner':<15} {'Accuracy':>10} {'Avg Tokens':>12} {'Avg Time':>10} {'Token Eff':>12}")
    print("-" * 80)

    for runner, data in stats.get("by_runner", {}).items():
        eff = data.get("token_efficiency_vs_vanilla", 1.0)
        eff_str = f"{eff:.2f}x" if eff != 1.0 else "-"
        cost = data.get("total_cost_usd")
        cost_str = f"${cost:.4f}" if cost is not None else "N/A"

        if has_cost:
            print(
                f"{runner:<15} "
                f"{data.get('overall_accuracy', 0):>9.1f}% "
                f"{data.get('avg_tokens_per_task', 0):>12,.0f} "
                f"{data.get('avg_time_per_task', 0):>9.1f}s "
                f"{cost_str:>12} "
                f"{eff_str:>10}"
            )
        else:
            print(
                f"{runner:<15} "
                f"{data.get('overall_accuracy', 0):>9.1f}% "
                f"{data.get('avg_tokens_per_task', 0):>12,.0f} "
                f"{data.get('avg_time_per_task', 0):>9.1f}s "
                f"{eff_str:>12}"
            )

    print("-" * 80)

    if not has_cost:
        print("\n⚠️  Cost calculation unavailable (model not in tokencost database)")

    # Per-task breakdown
    print("\nBy Task:")
    for task, runners_data in stats.get("by_task", {}).items():
        print(f"\n  {task.upper()}:")
        for runner, data in runners_data.items():
            status = "✓" if data.get("accuracy", 0) >= 80 else "✗"
            print(
                f"    {runner:<12} {status} "
                f"{data.get('accuracy', 0):>5.1f}% | "
                f"{data.get('avg_total_tokens', 0):>8,.0f} tokens | "
                f"{data.get('avg_time_seconds', 0):>6.1f}s"
            )

    print("\n" + "=" * 80)


# =============================================================================
# Entry Point
# =============================================================================


def main():
    args = parse_args()

    # Parse tasks
    tasks = [t.strip() for t in args.tasks.split(",")]
    if "all" in tasks:
        tasks = list(TASK_REGISTRY.keys())

    # Parse runners
    runners = [r.strip() for r in args.runners.split(",")]
    if "all" in runners:
        runners = list(RUNNER_REGISTRY.keys())

    # Handle skip_official flag
    if args.skip_official and "official" in runners:
        runners.remove("official")

    # Parse context sizes
    if args.extended:
        # Extended test: 8K to 256K (paper-style evaluation)
        context_sizes = [8192, 16384, 32768, 65536, 131072, 262144]
    elif args.context_sizes == "large":
        # Large preset: include 128K and 256K
        context_sizes = [32768, 65536, 131072, 262144]
    elif args.context_sizes:
        context_sizes = [int(s.strip()) for s in args.context_sizes.split(",")]
    else:
        context_sizes = None

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet

    # Run evaluation
    results = run_evaluation(
        model=args.model,
        tasks=tasks,
        runners=runners,
        runs=args.runs,
        output_dir=output_dir,
        context_size=args.context_size,
        context_sizes=context_sizes,
        verbose=verbose,
    )

    if not results:
        log.error("No results collected!")
        return

    # Save results
    json_path, summary_path = save_results(results, output_dir)
    log.info(f"\nResults saved to: {json_path}")
    log.info(f"Summary saved to: {summary_path}")

    # Generate plots
    if not args.no_plot:
        plots_dir = output_dir / "plots"
        plot_paths = plot_comprehensive_dashboard(results, plots_dir)
        log.info(f"Plots saved to: {plots_dir}")
        for path in plot_paths:
            log.info(f"  - {path.name}")

    # Print summary
    print_summary(results)

    return str(json_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
RLM Evaluation Suite - Main Entry Point

Implements benchmarks from the RLM paper (Zhang et al., 2025):
https://arxiv.org/abs/2512.24601

A reproducible benchmark comparing:
1. Vanilla LLM (direct API calls)
2. Our minimal RLM implementation
3. Official RLM implementation (optional)

Paper Tasks:
- S-NIAH: Single needle-in-a-haystack (basic retrieval)
- OOLONG: Information aggregation (Bertsch et al., 2025)
- OOLONG-Pairs: Pairwise matching (hardest task)
- CodeQA: Code repository understanding (Bai et al., 2025)
- BrowseComp+: Deep research / multi-hop (Chen et al., 2025)

Usage:
    # Quick start
    uv run python eval/run.py --model gpt-5-nano

    # Full evaluation with multiple runs
    uv run python eval/run.py --model gpt-5-nano --runs 3 --tasks all

    # Paper benchmarks (all core tasks from the RLM paper)
    uv run python eval/run.py --model gpt-5-nano --tasks paper

    # Paper scaling test (8K to 1M, Figure 1)
    uv run python eval/run.py --model gpt-5-nano --tasks scaling --paper-scale

    # Skip official RLM (if not installed)
    uv run python eval/run.py --model gpt-5-nano --skip-official

    # Custom context sizes for scaling test
    uv run python eval/run.py --model gpt-5-nano --tasks scaling --context-sizes 8192,16384,32768

    # Extended evaluation (8K to 256K contexts)
    uv run python eval/run.py --model gpt-5-nano --tasks scaling --extended

    # Output to specific directory
    uv run python eval/run.py --model gpt-5-nano --output-dir my_results/
"""

import argparse
import atexit
import gc
import logging
import resource
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def get_memory_mb() -> float:
    """Get current memory usage in MB."""
    try:
        # rusage returns memory in KB on Linux, bytes on macOS
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # maxrss is in KB on Linux, bytes on macOS
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)  # bytes -> MB
        else:
            return usage.ru_maxrss / 1024  # KB -> MB
    except Exception:
        return 0.0


from tqdm import tqdm

# Ensure our module is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


from eval.metrics import EvalResult, calculate_cost, compute_statistics, save_results
from eval.runners import RUNNER_REGISTRY, RunResult, get_runner
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
        help=(
            f"Tasks to run (comma-separated). Options: {', '.join(TASK_REGISTRY.keys())}, "
            "all, paper (core paper tasks: sniah, oolong, pairs, codeqa, browsecomp)"
        ),
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

    parser.add_argument("--log-dir", default=None, help="Directory to save RLM execution logs (default: None)")

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

    parser.add_argument(
        "--paper-scale",
        action="store_true",
        help="Use paper's context sizes for scaling (8K to 1M, as in Figure 1)",
    )

    parser.add_argument("--context-size", type=int, default=50000, help="Default context size for non-scaling tasks")

    parser.add_argument("--no-plot", action="store_true", help="Skip generating visualization plots")

    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output verbosity")

    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=3,
        help="Max parallel runners per task (default: 3 = all runners in parallel)",
    )

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
    results_accumulator: list[EvalResult] | None = None,
    max_parallel: int = 3,
    log_dir: str | None = None,
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
        results_accumulator: Optional mutable list to accumulate results (for crash recovery)

    Returns:
        List of EvalResult objects
    """
    # Use provided accumulator or create new list
    all_results = results_accumulator if results_accumulator is not None else []

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
            # Pass log_dir only to "ours" runner
            kwargs = {"log_dir": log_dir} if runner_name == "ours" and log_dir else {}
            runner = get_runner(runner_name, model, **kwargs)
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
            # Default: Test up to 1M as per paper (Figure 1)
            sizes = context_sizes or [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
            for size in sizes:
                results = _run_task_evaluations(
                    task_name=f"scaling_{size}",
                    task_kwargs={"context_size": size},
                    runners=active_runners,
                    model=model,
                    runs=runs,
                    verbose=verbose,
                    max_parallel=max_parallel,
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
                    max_parallel=max_parallel,
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
                    max_parallel=max_parallel,
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
                    max_parallel=max_parallel,
                )
                all_results.extend(results)
        elif task_name == "oolong":
            # OOLONG: Information aggregation at 131K (paper size)
            oolong_sizes = context_sizes or [131072]
            for size in oolong_sizes:
                results = _run_task_evaluations(
                    task_name=f"oolong_{size // 1024}k" if len(oolong_sizes) > 1 else "oolong",
                    task_kwargs={"context_size": size},
                    runners=active_runners,
                    model=model,
                    runs=runs,
                    verbose=verbose,
                    max_parallel=max_parallel,
                )
                all_results.extend(results)
        elif task_name == "codeqa":
            # CodeQA: Code repository understanding at various sizes (paper: 23K-4.2M)
            # Default: Include 1M+ contexts as per paper
            codeqa_sizes = context_sizes or [100000, 500000, 1000000, 2000000]
            for size in codeqa_sizes:
                results = _run_task_evaluations(
                    task_name=f"codeqa_{size // 1000}k",
                    task_kwargs={"context_size": size},
                    runners=active_runners,
                    model=model,
                    runs=runs,
                    verbose=verbose,
                    max_parallel=max_parallel,
                )
                all_results.extend(results)
        elif task_name == "browsecomp":
            # BrowseComp+: Deep research at large contexts (paper: 6M-11M)
            # Default to paper's range: 6M-11M
            browsecomp_sizes = context_sizes or [6000000, 8000000, 10000000, 11000000]
            for size in browsecomp_sizes:
                results = _run_task_evaluations(
                    task_name=f"browsecomp_{size // 1000000}M",
                    task_kwargs={"context_size": size},
                    runners=active_runners,
                    model=model,
                    runs=runs,
                    verbose=verbose,
                    max_parallel=max_parallel,
                )
                all_results.extend(results)
        elif task_name == "pairs":
            # PAIRS: Increase default runs for investigation (50% accuracy issue)
            # Use more runs to get better statistics on failure modes
            pairs_runs = max(runs, 5)  # At least 5 runs for PAIRS
            results = _run_task_evaluations(
                task_name=task_name,
                task_kwargs={"context_size": context_size},
                runners=active_runners,
                model=model,
                runs=pairs_runs,
                verbose=verbose,
                max_parallel=max_parallel,
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
    elif task_name.startswith("oolong_"):
        return "oolong"
    elif task_name.startswith("codeqa_"):
        return "codeqa"
    elif task_name.startswith("browsecomp_"):
        return "browsecomp"
    else:
        return task_name


def _run_single_runner(
    runner_name: str,
    runner,
    task,
    instance,
    model: str,
) -> tuple[str, RunResult, bool, float]:
    """Run a single runner on a task instance. Returns (runner_name, result, correct, partial_score)."""
    try:
        run_result = runner.run(instance.task, instance.context)
        correct = task.check(run_result.response, instance.expected)
        partial_score = task.check_partial(run_result.response, instance.expected)
        return runner_name, run_result, correct, partial_score
    except Exception as e:
        run_result = RunResult(
            response="",
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            time_seconds=0.0,
            iterations=0,
            error=str(e),
        )
        return runner_name, run_result, False, 0.0


def _run_task_evaluations(
    task_name: str,
    task_kwargs: dict,
    runners: dict,
    model: str,
    runs: int,
    verbose: bool,
    max_parallel: int = 3,
) -> list[EvalResult]:
    """Run evaluations for a single task across all runners (in parallel)."""
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
        log.info(f"TASK: {task_name.upper()} (parallel: {min(len(runners), max_parallel)})")
        log.info(f"{'=' * 60}")

    run_pbar = tqdm(range(runs), desc=f"{task_name}", leave=False, disable=not verbose)
    for run_idx in run_pbar:
        seed = 42 + run_idx * 100

        # Generate task instance
        instance = task.generate(seed=seed, **task_kwargs)

        run_pbar.set_postfix({"context": f"{len(instance.context):,} chars", "run": f"{run_idx + 1}/{runs}"})

        # Run all runners in parallel (up to max_parallel)
        gc.collect()  # GC before parallel execution

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all runners
            futures = {
                executor.submit(_run_single_runner, runner_name, runner, task, instance, model): runner_name
                for runner_name, runner in runners.items()
            }

            # Collect results as they complete
            for future in as_completed(futures):
                runner_name = futures[future]
                try:
                    runner_name, run_result, correct, partial_score = future.result()
                except KeyboardInterrupt:
                    tqdm.write(f"\n⚠️  Interrupted by user during {runner_name} on {task_name}")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as e:
                    tqdm.write(f"\n❌ ERROR in {runner_name} on {task_name}: {type(e).__name__}: {e}")
                    run_result = RunResult(
                        response="",
                        total_tokens=0,
                        input_tokens=0,
                        output_tokens=0,
                        time_seconds=0.0,
                        iterations=0,
                        error=str(e),
                    )
                    correct = False
                    partial_score = 0.0

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
                    error_info = f" | ⚠️ {run_result.error}" if run_result.error else ""
                    tqdm.write(
                        f"    {runner_name}: {status} | "
                        f"{run_result.input_tokens:,}+{run_result.output_tokens:,} tokens | "
                        f"{run_result.time_seconds:.1f}s | "
                        f"{run_result.iterations} iters{error_info}"
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

    print("\n" + "-" * 90)
    if has_cost:
        print(
            f"{'Runner':<15} {'Accuracy':>10} {'In+Out Tokens':>18} {'Avg Time':>10} {'Total Cost':>12} {'Token Eff':>10}"
        )
    else:
        print(f"{'Runner':<15} {'Accuracy':>10} {'In+Out Tokens':>18} {'Avg Time':>10} {'Token Eff':>12}")
    print("-" * 90)

    for runner, data in stats.get("by_runner", {}).items():
        eff = data.get("token_efficiency_vs_vanilla", 1.0)
        eff_str = f"{eff:.2f}x" if eff != 1.0 else "-"
        cost = data.get("total_cost_usd")
        cost_str = f"${cost:.6f}" if cost is not None else "N/A"

        # Show input+output tokens separately
        in_tok = int(data.get("avg_input_tokens", 0))
        out_tok = int(data.get("avg_output_tokens", 0))
        tokens_str = f"{in_tok:,}+{out_tok:,}"

        if has_cost:
            print(
                f"{runner:<15} "
                f"{data.get('overall_accuracy', 0):>9.1f}% "
                f"{tokens_str:>18} "
                f"{data.get('avg_time_per_task', 0):>9.1f}s "
                f"{cost_str:>12} "
                f"{eff_str:>10}"
            )
        else:
            print(
                f"{runner:<15} "
                f"{data.get('overall_accuracy', 0):>9.1f}% "
                f"{tokens_str:>18} "
                f"{data.get('avg_time_per_task', 0):>9.1f}s "
                f"{eff_str:>12}"
            )

    print("-" * 90)

    if not has_cost:
        print("\n⚠️  Cost calculation unavailable (model not in tokencost database)")

    # Per-task breakdown
    print("\nBy Task:")
    for task, runners_data in stats.get("by_task", {}).items():
        print(f"\n  {task.upper()}:")
        for runner, data in runners_data.items():
            status = "✓" if data.get("accuracy", 0) >= 80 else "✗"
            in_tok = int(data.get("avg_input_tokens", 0))
            out_tok = int(data.get("avg_output_tokens", 0))
            print(
                f"    {runner:<12} {status} "
                f"{data.get('accuracy', 0):>5.1f}% | "
                f"{in_tok:,}+{out_tok:,} tokens | "
                f"{data.get('avg_time_seconds', 0):>6.1f}s"
            )

    # Context size analysis
    print("\n" + "=" * 80)
    print("CONTEXT SIZE ANALYSIS")
    print("=" * 80)

    # Group by context size
    size_groups: dict[int, dict[str, list[EvalResult]]] = {}
    for r in results:
        if r.context_size not in size_groups:
            size_groups[r.context_size] = {}
        if r.runner_name not in size_groups[r.context_size]:
            size_groups[r.context_size][r.runner_name] = []
        size_groups[r.context_size][r.runner_name].append(r)

    if len(size_groups) > 1:
        print(
            f"\n{'Context Size':<15} {'Vanilla Acc':<12} {'RLM Acc':<12} {'Advantage':<12} {'RLM Tokens':<15} {'Vanilla Tokens':<15}"
        )
        print("-" * 80)

        for size in sorted(size_groups.keys()):
            vanilla_results = size_groups[size].get("vanilla", [])
            rlm_results = size_groups[size].get("ours", []) or size_groups[size].get("official", [])

            if vanilla_results and rlm_results:
                vanilla_acc = sum(r.correct for r in vanilla_results) / len(vanilla_results) * 100
                rlm_acc = sum(r.correct for r in rlm_results) / len(rlm_results) * 100
                advantage = rlm_acc - vanilla_acc

                vanilla_tokens = int(sum(r.total_tokens for r in vanilla_results) / len(vanilla_results))
                rlm_tokens = int(sum(r.total_tokens for r in rlm_results) / len(rlm_results))

                size_str = f"{size // 1024}K" if size < 1024 * 1024 else f"{size // (1024 * 1024)}M"
                advantage_str = f"+{advantage:.1f}%" if advantage > 0 else f"{advantage:.1f}%"

                print(
                    f"{size_str:<15} {vanilla_acc:>10.1f}% {rlm_acc:>10.1f}% {advantage_str:>11} {rlm_tokens:>14,} {vanilla_tokens:>14,}"
                )

        # Find crossover point where RLM starts outperforming
        crossover_sizes = []
        for size in sorted(size_groups.keys()):
            vanilla_results = size_groups[size].get("vanilla", [])
            rlm_results = size_groups[size].get("ours", []) or size_groups[size].get("official", [])
            if vanilla_results and rlm_results:
                vanilla_acc = sum(r.correct for r in vanilla_results) / len(vanilla_results) * 100
                rlm_acc = sum(r.correct for r in rlm_results) / len(rlm_results) * 100
                if rlm_acc > vanilla_acc:
                    crossover_sizes.append((size, rlm_acc - vanilla_acc))

        if crossover_sizes:
            min_crossover = min(crossover_sizes, key=lambda x: x[0])
            size_str = (
                f"{min_crossover[0] // 1024}K"
                if min_crossover[0] < 1024 * 1024
                else f"{min_crossover[0] // (1024 * 1024)}M"
            )
            print(
                f"\n✓ RLM starts outperforming vanilla at {size_str} context size (+{min_crossover[1]:.1f}% advantage)"
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
    elif "paper" in tasks:
        # Core tasks from the RLM paper (Table 1)
        tasks = ["sniah", "oolong", "pairs", "codeqa", "browsecomp"]

    # Parse runners
    runners = [r.strip() for r in args.runners.split(",")]
    if "all" in runners:
        runners = list(RUNNER_REGISTRY.keys())

    # Handle skip_official flag
    if args.skip_official and "official" in runners:
        runners.remove("official")

    # Parse context sizes
    if args.paper_scale:
        # Paper Figure 1: 8K to 1M (2^13 to 2^20)
        # 8K, 16K, 33K, 66K, 131K, 262K, 524K, 1M
        context_sizes = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    elif args.extended:
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

    # Run evaluation with crash protection
    # Use mutable list so partial results survive crashes
    results: list[EvalResult] = []

    def save_partial_results(sig_name: str = "unknown"):
        """Emergency save of partial results."""
        if results:
            log.warning(f"\n⚠️  Saving {len(results)} partial results (signal: {sig_name})...")
            try:
                json_path, summary_path = save_results(results, output_dir)
                log.info(f"Partial results saved to: {json_path}")
            except Exception as e:
                log.error(f"Failed to save partial results: {e}")

    def signal_handler(signum, frame):
        """Handle termination signals."""
        sig_name = signal.Signals(signum).name
        log.error(f"\n❌ Received signal {sig_name} ({signum})")
        save_partial_results(sig_name)
        sys.exit(128 + signum)

    # Register signal handlers for common termination signals
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        try:
            signal.signal(sig, signal_handler)
        except (ValueError, OSError):
            pass  # Some signals may not be available on all platforms

    # Register atexit handler for clean shutdown
    atexit.register(lambda: save_partial_results("atexit") if results and not hasattr(main, "_saved") else None)

    try:
        run_evaluation(
            model=args.model,
            tasks=tasks,
            runners=runners,
            runs=args.runs,
            output_dir=output_dir,
            context_size=args.context_size,
            context_sizes=context_sizes,
            verbose=verbose,
            results_accumulator=results,  # Pass mutable list
            max_parallel=args.parallel,
            log_dir=args.log_dir,
        )
    except KeyboardInterrupt:
        log.warning("\n⚠️  Evaluation interrupted by user")
        if results:
            log.info(f"Saving {len(results)} partial results...")
    except Exception as e:
        log.error(f"\n❌ Evaluation crashed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        if results:
            log.info(f"Saving {len(results)} partial results...")

    if not results:
        log.error("No results collected!")
        return

    # Mark as saved to prevent atexit double-save
    main._saved = True

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
    # Ensure unbuffered output for crash diagnosis
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    main()

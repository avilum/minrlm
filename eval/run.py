#!/usr/bin/env python3
"""
RLM Evaluation Suite - Main Entry Point

Implements benchmarks from the RLM paper (Zhang et al., 2025):
https://arxiv.org/abs/2512.24601

A reproducible benchmark comparing:
1. Vanilla LLM (direct API calls)
2. Our minimal RLM implementation
3. Official RLM implementation (optional)

Official Datasets:
- official_sniah: Needle-in-a-haystack (basic retrieval)
- official_oolong: Information aggregation
- official_repoqa: Code repository search
- official_codeqa: Code repository understanding
- official_browsecomp: Multi-hop research
- official_gdpval: Professional work tasks
- official_aime_2025: Competition math

Usage:
    # Quick start
    uv run python eval/quickstart.py

    # Single task
    uv run python eval/run.py --model gpt-5-mini --tasks official_sniah --runs 3

    # Multiple official tasks
    uv run python eval/run.py --model gpt-5-mini --tasks official_sniah,official_oolong --runs 5

    # Comprehensive benchmark (all official datasets)
    ./run_comprehensive_official_benchmark.sh

    # Output to specific directory
    uv run python eval/run.py --model gpt-5-mini --output-dir my_results/
"""

import argparse
import atexit
import gc
import logging
import resource
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import shutil


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
  uv run python eval/run.py --model gpt-5-mini
  uv run python eval/run.py --model gpt-5-mini --runs 3 --tasks all
  uv run python eval/run.py --model gpt-5-mini --skip-official
        """,
    )

    parser.add_argument(
        "--model", "-m", default="gpt-5-mini", help="Model to use for evaluation (e.g., gpt-5-mini, gpt-4o, gpt-4-turbo)"
    )

    parser.add_argument(
        "--tasks",
        "-t",
        default="official_sniah,official_oolong",
        help=(
            f"Tasks to run (comma-separated). Options: {', '.join(TASK_REGISTRY.keys())}, "
            "official (all official datasets), "
            "or specific tasks like official_sniah, official_oolong, official_repoqa, etc."
        ),
    )

    parser.add_argument(
        "--runners",
        "-r",
        default="vanilla,minrlm,minrlm-reasoning,official",
        help=f"Runners to compare (comma-separated). Options: {', '.join(RUNNER_REGISTRY.keys())}, all",
    )

    parser.add_argument(
        "--runs", "-n", type=int, default=1, help="Number of runs per task/runner combination (default: 1)"
    )

    parser.add_argument("--output-dir", "-o", default="eval/results", help="Output directory for results and plots")

    parser.add_argument("--log-dir", default=None, help="Directory to save RLM execution logs (default: <output-dir>/logs)")

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

    parser.add_argument(
        "--official-data-dir",
        default="evals/data",
        help="Root directory for official datasets (default: evals/data)",
    )

    parser.add_argument(
        "--official-split",
        default=None,
        help="Override split name for official datasets (e.g., test, validation, train)",
    )

    parser.add_argument(
        "--official-max-samples",
        type=int,
        default=None,
        help="Limit number of samples loaded per official dataset",
    )
    parser.add_argument(
        "--official-oolong-max-context-chars",
        type=int,
        default=None,
        help="Max OOLONG context length in characters (filters official_oolong samples)",
    )
    parser.add_argument(
        "--official-oolong-max-context-tokens",
        type=int,
        default=None,
        help="Max OOLONG context length in tokens (uses dataset context_len if present)",
    )

    parser.add_argument(
        "--official-longbench-max-context-tokens",
        type=int,
        default=None,
        help="Max LongBench-v2 context length in tokens (filters samples by context size)",
    )

    parser.add_argument(
        "--browsecomp-max-docs",
        type=int,
        default=None,
        help="Limit number of documents per BrowseComp+ query",
    )

    parser.add_argument("--no-plot", action="store_true", help="Skip generating visualization plots")

    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output verbosity")

    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=3,
        help="Max parallel runners per task instance (default: 3)",
    )

    parser.add_argument(
        "--task-parallel",
        type=int,
        default=4,
        help="Max parallel task instances (samples) per task (default: 4)",
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
    task_parallel: int = 4,
    log_dir: str | None = None,
    official_data_dir: str = "evals/data",
    official_split: str | None = None,
    official_max_samples: int | None = None,
    official_oolong_max_context_chars: int | None = None,
    official_oolong_max_context_tokens: int | None = None,
    official_longbench_max_context_tokens: int | None = None,
    browsecomp_max_docs: int | None = None,
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
    # Set default log_dir to output_dir/logs if not specified
    if log_dir is None and output_dir is not None:
        log_dir = str(output_dir / "logs")

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
            # Pass log_dir to all RLM runners (minrlm, official)
            _log_dir_runners = ("minrlm", "minrlm-reasoning", "official")
            kwargs = {"log_dir": log_dir} if runner_name in _log_dir_runners and log_dir else {}
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
    max_context_chars = official_oolong_max_context_chars
    max_context_tokens = official_oolong_max_context_tokens

    if max_context_chars is None and max_context_tokens is None and "gpt-5-mini" in model:
        max_context_tokens = 272000
        max_context_chars = 10_000_000

    official_kwargs = {
        "data_dir": official_data_dir,
        "split": official_split,
        "max_samples": official_max_samples,
        "max_docs": browsecomp_max_docs,
        "max_context_chars": max_context_chars,
        "max_context_tokens": max_context_tokens,
    }

    for task_name in tqdm(tasks, desc="Tasks", disable=not verbose):
        if task_name.startswith("official_"):
            # Use task-specific kwargs for longbench
            task_kwargs = official_kwargs.copy()
            if task_name == "official_longbench_v2" and official_longbench_max_context_tokens is not None:
                task_kwargs["max_context_tokens"] = official_longbench_max_context_tokens

            results = _run_task_evaluations(
                task_name=task_name,
                task_kwargs=task_kwargs,
                runners=active_runners,
                model=model,
                runs=runs,
                verbose=verbose,
                max_parallel=max_parallel,
                task_parallel=task_parallel,
            )
            all_results.extend(results)
            continue
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
                    task_parallel=task_parallel,
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
                        max_parallel=max_parallel,
                        task_parallel=task_parallel,
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
                    task_parallel=task_parallel,
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
                    task_parallel=task_parallel,
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
                    task_parallel=task_parallel,
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
                    task_parallel=task_parallel,
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
                    task_parallel=task_parallel,
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
                    task_parallel=task_parallel,
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
                task_parallel=task_parallel,
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
                max_parallel=max_parallel,
                task_parallel=task_parallel,
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


def _run_one_instance(
    run_idx: int,
    task_name: str,
    task,
    generate_kwargs: dict,
    runners: dict,
    model: str,
    max_parallel: int,
    verbose: bool,
) -> list[EvalResult]:
    """Run one task instance (one seed) across all runners. Returns list of EvalResult."""
    seed = 42 + run_idx * 100
    instance = task.generate(seed=seed, **generate_kwargs)
    results = []

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(_run_single_runner, rname, runner, task, instance, model): rname
            for rname, runner in runners.items()
        }
        for future in as_completed(futures):
            rname = futures[future]
            try:
                _, run_result, correct, partial_score = future.result()
            except Exception as e:
                if verbose:
                    tqdm.write(f"\n❌ ERROR in {rname} on {task_name} run_idx={run_idx}: {type(e).__name__}: {e}")
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

            cost = calculate_cost(model, run_result.input_tokens, run_result.output_tokens)
            results.append(
                EvalResult(
                    task_name=task_name,
                    task_instance_seed=seed,
                    runner_name=rname,
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
                    generated_code=run_result.generated_code,
                    log_file_path=run_result.log_file_path,
                )
            )
            if verbose:
                status = "✓" if correct else "✗"
                err = f" | ⚠️ {run_result.error}" if run_result.error else ""

                # Show expected vs actual preview (first 60 chars)
                expected_preview = str(instance.expected)[:60]
                actual_preview = str(run_result.response)[:60] if run_result.response else "(empty)"

                # Format output with preview
                tqdm.write(
                    f"    {rname}: {status} | "
                    f"{run_result.input_tokens:,}+{run_result.output_tokens:,} tokens | "
                    f"{run_result.time_seconds:.1f}s | {run_result.iterations} iters{err}"
                )
                if not correct:
                    # Show mismatch details for failures
                    tqdm.write(f"      Expected: {expected_preview}...")
                    tqdm.write(f"      Got:      {actual_preview}...")
    return results


def _run_task_evaluations(
    task_name: str,
    task_kwargs: dict,
    runners: dict,
    model: str,
    runs: int,
    verbose: bool,
    max_parallel: int = 3,
    task_parallel: int = 4,
) -> list[EvalResult]:
    """Run evaluations for a single task: multiple instances in parallel, runners per instance in parallel."""
    results = []

    base_task_name = _get_base_task_name(task_name)
    try:
        if task_name.startswith("official_"):
            task = get_task(base_task_name, **task_kwargs)
            generate_kwargs = {}
        else:
            task = get_task(base_task_name)
            generate_kwargs = task_kwargs
    except ValueError:
        log.warning(f"Skipping unknown task: {base_task_name}")
        return []

    workers = min(task_parallel, runs)
    if verbose:
        log.info(f"\n{'=' * 60}")
        log.info(f"TASK: {task_name.upper()} (task_parallel: {workers}, runner_parallel: {min(len(runners), max_parallel)})")
        log.info(f"{'=' * 60}")

    run_pbar = tqdm(total=runs, desc=f"{task_name}", leave=False, disable=not verbose)
    gc.collect()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_one_instance,
                run_idx,
                task_name,
                task,
                generate_kwargs,
                runners,
                model,
                max_parallel,
                verbose,
            ): run_idx
            for run_idx in range(runs)
        }
        for future in as_completed(futures):
            run_idx = futures[future]
            try:
                results.extend(future.result())
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception as e:
                tqdm.write(f"\n❌ ERROR on {task_name} run_idx={run_idx}: {type(e).__name__}: {e}")
            run_pbar.update(1)

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
            f"{'Runner':<15} {'Accuracy':>10} {'In+Out Tokens':>18} {'Avg Time':>10} {'Total Cost':>12} {'Cost Eff':>10}"
        )
    else:
        print(f"{'Runner':<15} {'Accuracy':>10} {'In+Out Tokens':>18} {'Avg Time':>10} {'Cost Eff':>12}")
    print("-" * 90)

    for runner, data in stats.get("by_runner", {}).items():
        eff = data.get("cost_efficiency_vs_vanilla", 1.0)
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
    elif "official" in tasks:
        tasks = [
            "official_sniah",
            "official_oolong",
            "official_codeqa",
            "official_longbench_v2",
            "official_repoqa",
            "official_browsecomp",
        ]

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

    # Save command line for reproducibility
    command_file = output_dir / "command.txt"
    with open(command_file, "w") as f:
        f.write("# Command used to generate this evaluation\n")
        f.write("# Run from: " + str(Path.cwd()) + "\n")
        f.write("# Date: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        f.write(" ".join(sys.argv) + "\n")

    # Backup minrlm/ folder for reproducibility (prompts, core logic, etc.)
    try:
        minrlm_src = Path(__file__).parent.parent / "minrlm"
        minrlm_backup = output_dir / "minrlm_backup"

        if minrlm_src.exists():
            # Copy entire minrlm folder
            shutil.copytree(minrlm_src, minrlm_backup, dirs_exist_ok=True)
            # Try to show relative path, fallback to absolute if not in cwd
            try:
                display_path = minrlm_backup.relative_to(Path.cwd())
            except ValueError:
                display_path = minrlm_backup
            log.info(f"✓ Backed up minrlm/ to {display_path}")

            # Create a snapshot info file
            snapshot_file = minrlm_backup / "SNAPSHOT_INFO.txt"
            with open(snapshot_file, "w") as f:
                f.write(f"# minrlm/ snapshot taken at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Original path: {minrlm_src}\n")
                f.write(f"# Evaluation run: {output_dir.name}\n\n")
                f.write("This backup captures the exact prompts and code used for this evaluation.\n")
                f.write("Key files:\n")
                f.write("  - prompts_reasoning.py: Reasoning-enhanced prompts\n")
                f.write("  - prompts.py: Standard prompts\n")
                f.write("  - core.py: RLM implementation\n")
                f.write("  - core_reasoning.py: Reasoning RLM wrapper\n")
        else:
            log.warning(f"⚠️  minrlm/ folder not found at {minrlm_src}")
    except Exception as e:
        log.warning(f"⚠️  Failed to backup minrlm/: {e}")

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
            task_parallel=args.task_parallel,
            log_dir=args.log_dir,
            official_data_dir=args.official_data_dir,
            official_split=args.official_split,
            official_max_samples=args.official_max_samples,
            official_oolong_max_context_chars=args.official_oolong_max_context_chars,
            official_oolong_max_context_tokens=args.official_oolong_max_context_tokens,
            official_longbench_max_context_tokens=args.official_longbench_max_context_tokens,
            browsecomp_max_docs=args.browsecomp_max_docs,
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

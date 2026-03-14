#!/usr/bin/env python3
"""
RLM Evaluation - Quick Start Example

Run a minimal evaluation to verify everything works.

Usage:
    uv run python eval/quickstart.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL = os.environ.get("MINRLM_MODEL", "gpt-5-mini")


def log(msg: str, *, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"{prefix}{msg}", flush=True)


def main():
    log("=" * 60)
    log("RLM Quick Evaluation")
    log("=" * 60)
    log(f"Model: {MODEL}")

    # ── Docker check ────────────────────────────────────────────
    log("\nChecking Docker...", indent=0)
    from minrlm.docker_repl import check_docker_available

    docker_available = check_docker_available()
    log(f"Docker: {'✓ Enabled (secure sandboxed execution)' if docker_available else '✗ Disabled (local execution)'}")

    # ── Load task ────────────────────────────────────────────────
    log("\nLoading task: official_sniah (this may download the dataset)...", indent=0)
    t0 = time.time()
    from eval.runners import get_runner, list_runners
    from eval.tasks import get_task

    task = get_task("official_sniah", max_samples=1)
    log(f"  Dataset loaded in {time.time() - t0:.1f}s")

    log("Generating task instance...", indent=0)
    instance = task.generate(seed=42)
    log(f"  Task:     {instance.task[:80]!r}")
    log(f"  Context:  {len(instance.context):,} characters")
    log(f"  Expected: {instance.expected!r}")

    # ── Available runners ────────────────────────────────────────
    available = list_runners()
    log(f"\nRegistered runners: {', '.join(available)}")

    runners_to_test = ["vanilla", "minrlm-reasoning"]
    log(f"Testing: {', '.join(runners_to_test)}")
    log("\n" + "-" * 60)

    # ── Run each runner ──────────────────────────────────────────
    for runner_name in runners_to_test:
        log(f"\n[{runner_name}] Initializing...", indent=0)
        try:
            runner = get_runner(runner_name, MODEL)
        except Exception as e:
            log(f"[{runner_name}] INIT ERROR: {e}", indent=0)
            continue

        log(f"[{runner_name}] Calling model ({MODEL})...", indent=0)
        t_run = time.time()
        try:
            result = runner.run(instance.task, instance.context)
            elapsed = time.time() - t_run
        except Exception as e:
            log(f"[{runner_name}] RUN ERROR ({time.time() - t_run:.1f}s): {e}", indent=0)
            continue

        if result.error:
            log(f"[{runner_name}] ERROR ({elapsed:.1f}s): {result.error}", indent=0)
            continue

        correct = task.check(result.response, instance.expected)
        status = "✓ PASS" if correct else "✗ FAIL"
        log(f"[{runner_name}] {status} ({elapsed:.1f}s)", indent=0)
        log(f"  Response:   {result.response[:120]!r}", indent=0)
        log(
            f"  Tokens:     {result.total_tokens:,} (in: {result.input_tokens:,}, out: {result.output_tokens:,})",
            indent=0,
        )
        log(f"  Iterations: {result.iterations}", indent=0)

    # ── Done ─────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("Quick test complete!")
    log("\nTo run a full benchmark:")
    log("  uv run python eval/run.py \\")
    log(f"      --model {MODEL} \\")
    log("      --tasks all \\")
    log("      --runners minrlm-reasoning \\")
    log("      --runs 50 \\")
    log("      --parallel 5 \\")
    log("      --task-parallel 5 \\")
    log("      --official-max-samples 500 \\")
    log("      --output-dir logs/my_eval")
    log("=" * 60)


if __name__ == "__main__":
    main()

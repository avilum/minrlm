#!/usr/bin/env python3
"""
RLM Evaluation - Quick Start Example

Run a minimal evaluation to verify everything works.

Usage:
    uv run python eval/quickstart.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.runners import get_runner
from eval.tasks import get_task


def main():
    model = "gpt-4o-mini"  # Change to your model

    print("=" * 60)
    print("RLM Quick Evaluation")
    print("=" * 60)

    # Generate a small task
    task = get_task("sniah")
    instance = task.generate(seed=42, context_size=10000)  # Small context for speed

    print("\nTask: S-NIAH (needle in haystack)")
    print(f"Context: {len(instance.context):,} characters")
    print(f"Expected: {instance.expected}")

    # Test each runner
    runners = ["vanilla", "ours"]

    print("\n" + "-" * 60)

    for runner_name in runners:
        print(f"\nRunning: {runner_name}...")

        try:
            runner = get_runner(runner_name, model)
            result = runner.run(instance.task, instance.context)

            correct = task.check(result.response, instance.expected)
            status = "✓ PASS" if correct else "✗ FAIL"

            print(f"  {status}")
            print(f"  Tokens: {result.total_tokens:,} (in: {result.input_tokens:,}, out: {result.output_tokens:,})")
            print(f"  Time: {result.time_seconds:.1f}s")
            print(f"  Iterations: {result.iterations}")
            print(f"  Response: {result.response[:100]}...")

        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("Quick test complete!")
    print("\nFor full evaluation, run:")
    print("  uv run python eval/run.py --model gpt-4o-mini --tasks sniah,multi_needle")
    print("=" * 60)


if __name__ == "__main__":
    main()

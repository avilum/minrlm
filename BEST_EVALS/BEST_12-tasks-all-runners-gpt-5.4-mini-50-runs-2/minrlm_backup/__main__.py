"""CLI entry point for minrlm.

Usage:
    uvx minrlm "What is the capital of France?"
    uvx minrlm "How many ERROR lines?" ./server.log
    uvx minrlm "Find the function that handles auth" ./src/**/*.py
    cat huge_file.txt | uvx minrlm "Summarize this"
    echo "1,2,3,4,5" | uvx minrlm "Return the sum"
"""

from __future__ import annotations

import sys


def _step_printer(event: str, data: dict) -> None:
    """Print RLM steps to stderr as they happen."""
    err = sys.stderr
    if event == "thinking":
        print(f"\n{'=' * 60}", file=err)
        print(f"  Iteration {data['iteration']}", file=err)
        print(f"{'=' * 60}", file=err)
    elif event == "reasoning":
        print(f"\n  Reasoning: {data['reasoning']}", file=err)
    elif event == "executing":
        print(f"\n  Code:\n{'─' * 40}", file=err)
        for line in data["code"].splitlines():
            print(f"  │ {line}", file=err)
        print(f"{'─' * 40}", file=err)
    elif event == "executed":
        stdout = data.get("stdout", "")
        error = data.get("error")
        if error:
            print(f"  Error: {error}", file=err)
        elif stdout:
            preview = stdout[:500] + ("..." if len(stdout) > 500 else "")
            print(f"  Output: {preview}", file=err)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="minrlm",
        description="Run a task through minRLM (Recursive Language Model).",
        epilog='Examples:\n  minrlm "What is 2+2?"\n  minrlm "Count errors" server.log\n  cat data.csv | minrlm "Sum column 3"',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task", help="The task / question to answer")
    parser.add_argument(
        "context", nargs="?", default=None, help="File path to use as context (reads file contents). Use - for stdin."
    )
    parser.add_argument("-m", "--model", default="gpt-5-mini", help="Model name (default: gpt-5-mini)")
    parser.add_argument("--no-docker", action="store_true", help="Disable Docker sandbox")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show token usage and timing")
    parser.add_argument("-s", "--steps", action="store_true", help="Show generated code and execution steps")

    args = parser.parse_args()

    # Read context
    context = ""
    if args.context == "-":
        context = sys.stdin.read()
    elif args.context:
        try:
            with open(args.context) as f:
                context = f.read()
        except FileNotFoundError:
            print(f"Error: file not found: {args.context}", file=sys.stderr)
            sys.exit(1)
    elif not sys.stdin.isatty():
        # Piped input
        context = sys.stdin.read()

    from .core_reasoning import RLMReasoning

    client = RLMReasoning(
        model=args.model,
        use_docker=not args.no_docker,
        debug=args.verbose,
        on_step=_step_printer if args.steps else None,
    )

    result = client.completion(task=args.task, context=context)

    if args.steps:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print("  Answer:", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)

    print(result.response)

    if args.verbose:
        print(f"\n---\nTokens: {result.total_tokens:,} | Iterations: {result.iterations}", file=sys.stderr)
        if context:
            print(f"Context: {len(context):,} chars", file=sys.stderr)


if __name__ == "__main__":
    main()

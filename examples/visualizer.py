#!/usr/bin/env python3
"""
RLM Live Visualizer - Gradio App with Real Benchmarks

Compare RLM vs Vanilla LLM on challenging long-context tasks.

Run with:
    uv run --with 'gradio>=5.0' python examples/visualizer.py
"""

import json
import os
import random
import string
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    print("=" * 60)
    print("ERROR: Gradio is not installed!")
    print()
    print("Run the visualizer with:")
    print()
    print("    uv run --with 'gradio>=5.0' python examples/visualizer.py")
    print()
    print("=" * 60)
    sys.exit(1)

import pandas as pd
import plotly.express as px
from openai import OpenAI

# Add parent to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from minrlm import RLM  # Our implementation


def is_official_rlm_available() -> bool:
    """Official RLM is always available via uv --with."""
    return True


# =============================================================================
# Real Benchmark Tasks from the Paper
# =============================================================================


def generate_noise(length: int, seed: int = 42) -> str:
    """Generate plausible-looking filler text."""
    random.seed(seed)
    words = [
        "data",
        "analysis",
        "result",
        "process",
        "study",
        "algorithm",
        "function",
        "method",
        "system",
        "network",
        "model",
        "output",
        "input",
        "value",
        "parameter",
        "variable",
        "structure",
        "pattern",
        "research",
        "finding",
        "evidence",
        "theory",
        "hypothesis",
        "approach",
    ]
    text = []
    while len(" ".join(text)) < length:
        sent_len = random.randint(8, 15)
        sentence = " ".join(random.choices(words, k=sent_len))
        text.append(sentence.capitalize() + ". ")
    return "".join(text)[:length]


@dataclass
class BenchmarkTask:
    name: str
    task: str
    context: str
    expected: str
    check_fn: callable
    description: str


def create_sniah_task(context_size: int = 30000, seed: int = None) -> BenchmarkTask:
    """S-NIAH: Single Needle-in-a-Haystack - Find one secret in noise."""
    if seed is None:
        seed = random.randint(0, 10000)
    random.seed(seed)

    secret = f"SECRET-{''.join(random.choices(string.ascii_uppercase + string.digits, k=8))}"
    needle = f"\n\n[IMPORTANT: The secret code is: {secret}]\n\n"

    noise = generate_noise(context_size, seed + 1)
    position = random.randint(len(noise) // 4, 3 * len(noise) // 4)
    context = noise[:position] + needle + noise[position:]

    task = """Find the secret code hidden in this text.
Look for a line like: "[IMPORTANT: The secret code is: SECRET-XXXXXXXX]"
Return ONLY the full secret code (e.g., SECRET-ABC12345)."""

    def check(response: str) -> bool:
        code = secret.split("-", 1)[1]
        return code.upper() in response.upper()

    return BenchmarkTask(
        name="S-NIAH",
        task=task,
        context=context,
        expected=secret,
        check_fn=check,
        description=f"Find 1 secret code hidden in {len(context):,} chars of noise",
    )


def create_multi_needle_task(num_needles: int = 5, context_size: int = 30000, seed: int = None) -> BenchmarkTask:
    """Multi-Needle: Find multiple secrets scattered in text."""
    if seed is None:
        seed = random.randint(0, 10000)
    random.seed(seed)

    secrets = [
        f"KEY-{''.join(random.choices(string.ascii_uppercase + string.digits, k=6))}" for _ in range(num_needles)
    ]

    noise = generate_noise(context_size, seed + 100)
    context = noise

    positions = sorted(random.sample(range(len(noise) // 10, 9 * len(noise) // 10), num_needles))
    offset = 0
    for i, pos in enumerate(positions):
        needle = f"\n\n[SECRET #{i + 1}: {secrets[i]}]\n\n"
        context = context[: pos + offset] + needle + context[pos + offset :]
        offset += len(needle)

    task = f"""Find ALL {num_needles} secret codes hidden in this text.
Each appears as: "[SECRET #N: KEY-XXXXXX]"
Return ALL codes as a comma-separated list."""

    def check(response: str) -> bool:
        found = sum(1 for s in secrets if s.split("-", 1)[1].upper() in response.upper())
        return found >= len(secrets) * 0.8

    return BenchmarkTask(
        name="Multi-Needle",
        task=task,
        context=context,
        expected=", ".join(secrets),
        check_fn=check,
        description=f"Find {num_needles} secrets scattered in {len(context):,} chars",
    )


def create_pairs_task(num_pairs: int = 6, context_size: int = 30000, seed: int = None) -> BenchmarkTask:
    """OOLONG-Pairs: Match definitions to concepts - the hardest task."""
    if seed is None:
        seed = random.randint(0, 10000)
    random.seed(seed)

    greek = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA", "ETA", "THETA"]
    concepts = [
        "quantum entanglement",
        "photosynthesis",
        "continental drift",
        "neural plasticity",
        "entropy",
        "fibonacci sequence",
        "golden ratio",
        "prime numbers",
    ]

    random.shuffle(greek)
    random.shuffle(concepts)
    pairs = [(f"DEF-{greek[i]}", concepts[i]) for i in range(num_pairs)]

    noise = generate_noise(context_size, seed + 1)
    context = noise

    positions = sorted(random.sample(range(len(noise) // 10, len(noise) // 2), num_pairs))
    offset = 0
    for i, pos in enumerate(positions):
        definition = f"\n\n[DEFINITION {pairs[i][0]}]: The concept of {pairs[i][1]}.\n\n"
        context = context[: pos + offset] + definition + context[pos + offset :]
        offset += len(definition)

    task = f"""Match each definition code to its concept.
Find all {num_pairs} definitions like "[DEFINITION DEF-XXXX]: The concept of <concept>."
Return as: DEF-XXXX=concept, DEF-YYYY=concept, ..."""

    def check(response: str) -> bool:
        found = 0
        for code, concept in pairs:
            if code.upper() in response.upper() and concept.lower() in response.lower():
                found += 1
        return found >= len(pairs) * 0.8

    return BenchmarkTask(
        name="OOLONG-Pairs",
        task=task,
        context=context,
        expected=", ".join(f"{c}={p}" for c, p in pairs),
        check_fn=check,
        description=f"Match {num_pairs} definition-concept pairs in {len(context):,} chars",
    )


# Available benchmark generators
BENCHMARKS = {
    "S-NIAH (Easy)": lambda: create_sniah_task(context_size=20000),
    "S-NIAH (Medium)": lambda: create_sniah_task(context_size=40000),
    "S-NIAH (Hard)": lambda: create_sniah_task(context_size=60000),
    "Multi-Needle (5 secrets)": lambda: create_multi_needle_task(num_needles=5),
    "Multi-Needle (8 secrets)": lambda: create_multi_needle_task(num_needles=8),
    "OOLONG-Pairs (6 pairs)": lambda: create_pairs_task(num_pairs=6),
    "OOLONG-Pairs (8 pairs)": lambda: create_pairs_task(num_pairs=8),
}


# =============================================================================
# Model Utilities
# =============================================================================


def get_available_models(base_url: str = None) -> list[str]:
    """Fetch available models from API."""
    try:
        client = OpenAI(**({"base_url": base_url} if base_url else {}))
        models = client.models.list()
        chat_models = [
            m.id for m in models.data if any(x in m.id.lower() for x in ["gpt", "claude", "llama", "o1", "o3"])
        ]
        return sorted(chat_models, reverse=True) if chat_models else ["gpt-5-nano"]
    except:
        return ["gpt-5-nano", "gpt-5-nano", "gpt-5-nano-mini"]


# =============================================================================
# Runners
# =============================================================================


@dataclass
class RunResult:
    response: str
    correct: bool
    tokens: int
    input_tokens: int
    output_tokens: int
    time_seconds: float
    iterations: int = 1
    trace: str = ""


def run_vanilla_llm(task: str, context: str, model: str, expected: str, check_fn: callable) -> RunResult:
    """Run with direct LLM call."""
    client = OpenAI()

    prompt = f"""{task}

Here is the text to analyze:

{context}"""

    trace = "## Vanilla LLM\n\n"
    trace += f"Sending full context ({len(context):,} chars) in one request.\n\n"

    start = time.time()
    try:
        kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        if "gpt-5" not in model.lower():
            kwargs["temperature"] = 0.7

        response = client.chat.completions.create(**kwargs)
        elapsed = time.time() - start

        resp_text = response.choices[0].message.content or ""
        usage = response.usage
        correct = check_fn(resp_text)

        trace += f"**Response:** `{resp_text[:100]}{'...' if len(resp_text) > 100 else ''}`\n\n"
        trace += f"{'✅ Correct' if correct else '❌ Incorrect'}\n\n"

        return RunResult(
            response=resp_text,
            correct=correct,
            tokens=usage.total_tokens if usage else 0,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            time_seconds=elapsed,
            trace=trace,
        )
    except Exception as e:
        trace += f"**Error:** {e}\n"
        return RunResult(
            response="",
            correct=False,
            tokens=0,
            input_tokens=0,
            output_tokens=0,
            time_seconds=time.time() - start,
            trace=trace,
        )


def run_our_rlm_streaming(task: str, context: str, model: str, expected: str, check_fn: callable):
    """Run RLM with streaming updates. Yields (status, trace) tuples."""
    trace_parts = ["## minRLM\n\n", f"Processing {len(context):,} chars...\n\n"]

    def build_trace():
        return "".join(trace_parts)

    def on_step(event: str, data: dict):
        if event == "thinking":
            trace_parts.append(f"### Iteration {data['iteration']}\n\n")
        elif event == "llm_response":
            response_preview = data.get("response", "")[:300]
            has_code = data.get("has_code", False)
            trace_parts.append(f"**LLM Response** ({len(data.get('response', ''))} chars):\n")
            if has_code:
                trace_parts.append("✅ Contains code block\n\n")
            else:
                trace_parts.append("⚠️ No code block found\n\n")
                trace_parts.append(
                    f"```\n{response_preview}{'...' if len(data.get('response', '')) > 300 else ''}\n```\n\n"
                )
        elif event == "executing":
            code = data.get("code", "")
            trace_parts.append(f"**Executing Code:**\n```python\n{code}\n```\n\n")
        elif event == "executed":
            if data.get("error"):
                trace_parts.append(f"❌ **Error:**\n```\n{data['error']}\n```\n\n")
            else:
                stdout = data.get("stdout", "")
                output = data.get("output")
                if stdout:
                    trace_parts.append(f"**stdout:**\n```\n{stdout[:500]}{'...' if len(stdout) > 500 else ''}\n```\n\n")
                if output:
                    trace_parts.append(f"✅ **set_output():** `{output}`\n\n")
                elif not stdout:
                    trace_parts.append("*(no output)*\n\n")

    start = time.time()
    try:
        rlm = RLM(model=model, max_iterations=10, on_step=on_step)
        result = rlm.completion(task=task, context=context)
        elapsed = time.time() - start

        response = result.response
        if response in ["No output", "Max iterations reached."] and result.history:
            last = (
                result.history[-1].get("content", "")
                if isinstance(result.history[-1], dict)
                else str(result.history[-1])
            )
            for line in last.split("\n"):
                if any(x in line.upper() for x in ["SECRET", "NEEDLE", "DEF-", "KEY-"]):
                    response = line.strip()
                    break

        correct = check_fn(response)

        trace_parts.append(f"\n**Final:** `{response[:100]}`\n")
        trace_parts.append(f"**Tokens:** {result.total_tokens:,} | **Time:** {elapsed:.1f}s\n")
        trace_parts.append(f"{'✅ Correct' if correct else '❌ Incorrect'}\n\n")

        return RunResult(
            response=response,
            correct=correct,
            tokens=result.total_tokens,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            time_seconds=elapsed,
            iterations=result.iterations,
            trace=build_trace(),
        )
    except Exception as e:
        trace_parts.append(f"\n**Error:** {e}\n")
        return RunResult(
            response="",
            correct=False,
            tokens=0,
            input_tokens=0,
            output_tokens=0,
            time_seconds=time.time() - start,
            trace=build_trace(),
        )


def run_our_rlm(task: str, context: str, model: str, expected: str, check_fn: callable) -> RunResult:
    """Run with RLM (wrapper for compatibility)."""
    return run_our_rlm_streaming(task, context, model, expected, check_fn)


def run_official_rlm(task: str, context: str, model: str, expected: str, check_fn: callable) -> RunResult:
    """Run with official RLM from github.com/alexzhang13/rlm via uv --with."""
    trace = "## Official RLM\n\n"
    trace += (
        f"Processing {len(context):,} chars via [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm).\n\n"
    )

    start = time.time()

    # Write context and task to temp files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(context)
        context_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(task)
        task_file = f.name

    # Script to run official RLM
    script = f"""
import json
import time

try:
    from rlm import RLM

    with open("{context_file}") as f:
        context = f.read()

    with open("{task_file}") as f:
        task = f.read()

    start = time.time()

    rlm = RLM(
        backend="openai",
        backend_kwargs={{"model_name": "{model}"}},
        environment="local",
        max_iterations=10,
        verbose=False,
    )

    result = rlm.completion(prompt=context, root_prompt=task)
    elapsed = time.time() - start

    total_input = 0
    total_output = 0
    if result.usage_summary and result.usage_summary.model_usage_summaries:
        for usage in result.usage_summary.model_usage_summaries.values():
            total_input += usage.total_input_tokens
            total_output += usage.total_output_tokens

    response = result.response or ""
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1]

    # Try to get iteration count from result
    iterations = getattr(result, 'num_iterations', None) or getattr(result, 'iterations', None) or 1

    print("<<<RESULT>>>")
    print(json.dumps({{
        "response": response,
        "elapsed": elapsed,
        "total_tokens": total_input + total_output,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "iterations": iterations,
    }}))
except Exception as e:
    import traceback
    print("<<<ERROR>>>")
    print(str(e))
    traceback.print_exc()
"""

    try:
        # Use uv run --with to install official RLM from GitHub
        result = subprocess.run(
            [
                "uv",
                "run",
                "--with",
                "git+https://github.com/alexzhang13/rlm",
                "python",
                "-c",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")},
        )
        elapsed = time.time() - start

        output = result.stdout
        if "<<<RESULT>>>" in output:
            json_str = output.split("<<<RESULT>>>")[1].strip()
            data = json.loads(json_str)
            resp_text = data.get("response", "")
            total_tokens = data.get("total_tokens", 0)
            input_tokens = data.get("input_tokens", 0)
            output_tokens = data.get("output_tokens", 0)
            iterations = data.get("iterations", 1)
        elif "<<<ERROR>>>" in output:
            error_msg = output.split("<<<ERROR>>>")[1].strip()
            trace += f"**Error:** {error_msg[:300]}\n"
            return RunResult(
                response="",
                correct=False,
                tokens=0,
                input_tokens=0,
                output_tokens=0,
                time_seconds=elapsed,
                trace=trace,
            )
        else:
            if result.stderr:
                trace += f"**Error:** {result.stderr[:500]}\n"
            return RunResult(
                response="",
                correct=False,
                tokens=0,
                input_tokens=0,
                output_tokens=0,
                time_seconds=elapsed,
                trace=trace,
            )

        correct = check_fn(resp_text)

        trace += f"**Tokens:** {total_tokens:,}\n\n"
        trace += f"**Response:** `{resp_text[:100]}{'...' if len(resp_text) > 100 else ''}`\n\n"
        trace += f"{'✅ Correct' if correct else '❌ Incorrect'}\n\n"

        return RunResult(
            response=resp_text,
            correct=correct,
            tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            time_seconds=elapsed,
            iterations=iterations,
            trace=trace,
        )

    except subprocess.TimeoutExpired:
        trace += "**Error:** Timeout (5 min limit)\n"
        return RunResult(
            response="",
            correct=False,
            tokens=0,
            input_tokens=0,
            output_tokens=0,
            time_seconds=300,
            trace=trace,
        )
    except Exception as e:
        trace += f"**Error:** {e}\n"
        return RunResult(
            response="",
            correct=False,
            tokens=0,
            input_tokens=0,
            output_tokens=0,
            time_seconds=time.time() - start,
            trace=trace,
        )
    finally:
        for f in [context_file, task_file]:
            try:
                os.unlink(f)
            except:
                pass


# =============================================================================
# Main UI Functions
# =============================================================================


def generate_task(benchmark_name: str) -> tuple[str, str, str, str]:
    """Generate a new task instance."""
    if benchmark_name not in BENCHMARKS:
        return "Select a benchmark", "", "", ""

    task_obj = BENCHMARKS[benchmark_name]()
    return (
        task_obj.description,
        task_obj.task,
        task_obj.context,
        task_obj.expected,
    )


def run_comparison(
    task: str,
    context: str,
    expected: str,
    model: str,
    run_vanilla: bool,
    run_rlm_flag: bool,
    run_official_flag: bool,
    benchmark_name: str,
) -> tuple[str, str]:
    """Run selected methods and compare."""
    if not task.strip() or not context.strip():
        return "⚠️ Generate a task first!", ""

    # Create check function based on the ACTUAL expected value (not a new random one)
    # This ensures we check against the task instance that was generated
    def check_fn(response: str) -> bool:
        # For S-NIAH and Multi-Needle: check if expected codes are in response
        if "SECRET-" in expected or "KEY-" in expected or "NEEDLE-" in expected:
            codes = [c.strip() for c in expected.replace(",", " ").split() if "-" in c]
            if not codes:
                codes = [expected]
            found = sum(1 for c in codes if c.split("-", 1)[1].upper() in response.upper())
            return found >= len(codes) * 0.8
        # For PAIRS: check if code=concept pairs are present
        elif "DEF-" in expected:
            pairs = [p.strip() for p in expected.split(",")]
            found = 0
            for pair in pairs:
                if "=" in pair:
                    code, concept = pair.split("=", 1)
                    if code.upper() in response.upper() and concept.lower() in response.lower():
                        found += 1
            return found >= len(pairs) * 0.8
        else:
            return expected.upper() in response.upper()

    vanilla_result = None
    rlm_result = None
    official_result = None

    # Run selected methods
    if run_vanilla:
        vanilla_result = run_vanilla_llm(task, context, model, expected, check_fn)

    if run_rlm_flag:
        rlm_result = run_our_rlm(task, context, model, expected, check_fn)

    if run_official_flag:
        official_result = run_official_rlm(task, context, model, expected, check_fn)

    # Build clean output
    results = []
    if vanilla_result:
        results.append(("Vanilla", vanilla_result))
    if rlm_result:
        results.append(("minRLM", rlm_result))
    if official_result:
        results.append(("Official", official_result))

    if not results:
        return "Select at least one method.", ""

    # Simple table
    output = f"**{benchmark_name}** · {len(context):,} chars · {model}\n\n"
    output += "| Method | Result | Tokens | Time |\n|--------|--------|--------|------|\n"

    for name, r in results:
        status = "✅" if r.correct else "❌"
        output += f"| {name} | {status} | {r.tokens:,} | {r.time_seconds:.1f}s |\n"

    # Quick insight
    if len(results) >= 2:
        tokens = [(name, r.tokens) for name, r in results if r.tokens > 0]
        if tokens:
            best = min(tokens, key=lambda x: x[1])
            worst = max(tokens, key=lambda x: x[1])
            if best[1] < worst[1]:
                savings = (1 - best[1] / worst[1]) * 100
                output += f"\n**{best[0]}** used {savings:.0f}% fewer tokens than {worst[0]}."

    # Traces
    traces = ""
    if vanilla_result:
        traces += vanilla_result.trace + "\n---\n\n"
    if rlm_result:
        traces += rlm_result.trace + "\n---\n\n"
    if official_result:
        traces += official_result.trace

    return output, traces


def refresh_models(base_url: str):
    """Refresh model list."""
    base_url = base_url.strip() if base_url and base_url.strip() else None
    models = get_available_models(base_url)
    default = "gpt-5-nano" if "gpt-5-nano" in models else models[0]
    return gr.Dropdown(choices=models, value=default)


# =============================================================================
# Gradio App
# =============================================================================


def build_app():
    initial_models = get_available_models()

    with gr.Blocks(title="RLM Benchmark") as demo:
        gr.Markdown(
            "# RLM Benchmark\nCompare implementations on long-context tasks. [Paper](https://arxiv.org/abs/2512.24601)"
        )

        with gr.Row():
            benchmark_dropdown = gr.Dropdown(
                choices=list(BENCHMARKS.keys()),
                value="S-NIAH (Medium)",
                label="Task",
                scale=2,
            )
            model_dropdown = gr.Dropdown(
                choices=initial_models,
                value=("gpt-5-nano" if "gpt-5-nano" in initial_models else initial_models[0]),
                label="Model",
                scale=1,
            )
            run_btn = gr.Button("Run", variant="primary", scale=1)

        with gr.Row():
            vanilla_checkbox = gr.Checkbox(label="Vanilla", value=True)
            rlm_checkbox = gr.Checkbox(label="minRLM", value=True)
            official_checkbox = gr.Checkbox(label="Official RLM", value=True)

        # Large status display
        status_html = gr.HTML(
            """<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px; margin: 10px 0;">
                <div style="font-size: 1.5em; color: #888;">Select a task and click <b>Run</b></div>
            </div>"""
        )
        results_output = gr.Markdown("")

        # Charts for tokens and time (using Plotly for reliability)
        with gr.Row():
            tokens_plot = gr.Plot(label="Token Usage")
            time_plot = gr.Plot(label="Execution Time")

        with gr.Accordion("Execution Traces", open=False):
            traces_output = gr.Markdown("*Run a benchmark to see detailed execution traces for each method.*")

        with gr.Accordion("Task Details", open=False):
            task_text = gr.Textbox(label="Task", lines=2, interactive=False)
            expected_text = gr.Textbox(label="Expected", interactive=False)

        full_context = gr.State("")

        def on_task_change(benchmark_name):
            desc, task, context, expected = generate_task(benchmark_name)
            return task, expected, context

        benchmark_dropdown.change(
            fn=on_task_change,
            inputs=[benchmark_dropdown],
            outputs=[task_text, expected_text, full_context],
        )

        def run_with_progress(
            task,
            context,
            expected,
            model,
            run_vanilla,
            run_rlm,
            run_official,
            benchmark_name,
        ):
            def status_box(
                title: str, subtitle: str = "", icon: str = "⏳", color: str = "#ff922b", pulse: bool = True
            ):
                """Generate styled HTML status box."""
                pulse_style = "animation: pulse 1.5s ease-in-out infinite;" if pulse else ""
                return f"""<div style="text-align: center; padding: 24px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px; margin: 10px 0; border-left: 4px solid {color};">
                    <div style="font-size: 2.5em; margin-bottom: 8px; {pulse_style}">{icon}</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: {color}; margin-bottom: 4px;">{title}</div>
                    <div style="font-size: 1.1em; color: #888;">{subtitle}</div>
                    <style>@keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}</style>
                </div>"""

            if not task or not context:
                yield status_box("No Task Selected", "Generate a task first", "📋", "#666", False), "", None, None, ""
                return

            results_list = []
            traces = ""

            # Check function
            def check_fn(response):
                if "SECRET-" in expected or "KEY-" in expected or "NEEDLE-" in expected:
                    codes = [c.strip() for c in expected.replace(",", " ").split() if "-" in c]
                    if not codes:
                        codes = [expected]
                    found = sum(1 for c in codes if c.split("-", 1)[1].upper() in response.upper())
                    return found >= len(codes) * 0.8
                elif "DEF-" in expected:
                    pairs = [p.strip() for p in expected.split(",")]
                    found = 0
                    for pair in pairs:
                        if "=" in pair:
                            code, concept = pair.split("=", 1)
                            if code.upper() in response.upper() and concept.lower() in response.lower():
                                found += 1
                    return found >= len(pairs) * 0.8
                else:
                    return expected.upper() in response.upper()

            def build_charts():
                if not results_list:
                    return None, None
                data = []
                for name, r in results_list:
                    data.append(
                        {
                            "Method": name,
                            "Tokens": float(r.tokens),
                            "Time": float(r.time_seconds),
                        }
                    )
                df = pd.DataFrame(data)

                # Create Plotly figures
                colors = {"Vanilla": "#4dabf7", "minRLM": "#ff922b", "Official": "#51cf66"}
                color_map = {m: colors.get(m, "#888") for m in df["Method"]}

                tokens_fig = px.bar(
                    df, x="Method", y="Tokens", color="Method", title="Token Usage", color_discrete_map=color_map
                )
                tokens_fig.update_layout(
                    showlegend=False,
                    height=300,
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )

                time_fig = px.bar(
                    df, x="Method", y="Time", color="Method", title="Execution Time", color_discrete_map=color_map
                )
                time_fig.update_layout(
                    showlegend=False,
                    height=300,
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis_title="Seconds",
                )

                return tokens_fig, time_fig

            methods_to_run = []
            if run_vanilla:
                methods_to_run.append(("Vanilla", "🔵", "#4dabf7", run_vanilla_llm))
            if run_rlm:
                methods_to_run.append(("minRLM", "🟠", "#ff922b", run_our_rlm))
            if run_official:
                methods_to_run.append(("Official RLM", "🟢", "#51cf66", run_official_rlm))

            if not methods_to_run:
                yield (
                    status_box("No Methods Selected", "Check at least one method above", "⚠️", "#fcc419", False),
                    "",
                    *build_charts(),
                    traces,
                )
                return

            total_methods = len(methods_to_run)
            total_elapsed = 0.0
            for i, (name, icon, color, run_fn) in enumerate(methods_to_run):
                progress = f"Step {i + 1} of {total_methods}"
                elapsed_str = f" · ⏱️ {total_elapsed:.1f}s elapsed" if total_elapsed > 0 else ""
                yield (
                    status_box(
                        f"Running {name}...",
                        f"{progress} · Processing {len(context):,} characters{elapsed_str}",
                        icon,
                        color,
                        True,
                    ),
                    "",
                    *build_charts(),
                    traces,
                )

                step_start = time.time()
                r = run_fn(task, context, model, expected, check_fn)
                step_elapsed = time.time() - step_start
                total_elapsed += step_elapsed

                results_list.append((name.replace(" RLM", ""), r))
                traces += r.trace + "\n---\n\n"

                # Show completion of this step before moving to next
                if i < total_methods - 1:
                    completed_names = [n for n, _ in results_list]
                    yield (
                        status_box(
                            f"✓ {name} completed in {step_elapsed:.1f}s",
                            f"Total: ⏱️ {total_elapsed:.1f}s · Completed: {', '.join(completed_names)}",
                            "✓",
                            color,
                            False,
                        ),
                        "",
                        *build_charts(),
                        traces,
                    )

            # Build final output
            output = f"**{benchmark_name}** · {len(context):,} chars · {model}\n\n"
            output += (
                "| Method | Result | Tokens | Time | Iterations |\n|--------|--------|--------|------|------------|\n"
            )
            for name, r in results_list:
                status = "✅" if r.correct else "❌"
                output += f"| {name} | {status} | {r.tokens:,} | {r.time_seconds:.1f}s | {r.iterations} |\n"

            # Compute savings message
            savings_msg = ""
            if len(results_list) >= 2:
                tokens = [(name, r.tokens) for name, r in results_list if r.tokens > 0]
                if tokens:
                    best = min(tokens, key=lambda x: x[1])
                    worst = max(tokens, key=lambda x: x[1])
                    if best[1] < worst[1]:
                        savings = (1 - best[1] / worst[1]) * 100
                        savings_msg = f"**{best[0]}** used {savings:.0f}% fewer tokens than {worst[0]}."
                        output += f"\n{savings_msg}"

            # Show completion status
            all_correct = all(r.correct for _, r in results_list)
            time_summary = f"⏱️ Total time: {total_elapsed:.1f}s"
            if all_correct:
                subtitle = f"{time_summary}"
                if savings_msg:
                    subtitle += f" · {savings_msg}"
                final_status = status_box(
                    "✓ Complete",
                    subtitle,
                    "🎉",
                    "#51cf66",
                    False,
                )
            else:
                failed = [n for n, r in results_list if not r.correct]
                final_status = status_box(
                    "Complete with Errors",
                    f"{time_summary} · {', '.join(failed)} got incorrect results",
                    "⚠️",
                    "#fcc419",
                    False,
                )

            yield final_status, output, *build_charts(), traces

        run_btn.click(
            fn=run_with_progress,
            inputs=[
                task_text,
                full_context,
                expected_text,
                model_dropdown,
                vanilla_checkbox,
                rlm_checkbox,
                official_checkbox,
                benchmark_dropdown,
            ],
            outputs=[status_html, results_output, tokens_plot, time_plot, traces_output],
        )

        # Initial load
        demo.load(
            fn=on_task_change,
            inputs=[benchmark_dropdown],
            outputs=[task_text, expected_text, full_context],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()

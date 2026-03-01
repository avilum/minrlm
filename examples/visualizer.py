#!/usr/bin/env python3
"""
RLM Live Visualizer - Gradio App with Evaluation Tasks

Compare RLM vs Vanilla LLM on evaluation tasks or custom prompts.

Run with:
    uv sync --extra visualizer
    uv run python examples/visualizer.py
"""

# Fix matplotlib backend before any imports (Gradio 6.x bug on Python 3.14)
import matplotlib

matplotlib.use("Agg")

import json
import os
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
    print("    uv sync --extra visualizer")
    print("    uv run python examples/visualizer.py")
    print()
    print("=" * 60)
    sys.exit(1)

import pandas as pd
import plotly.express as px
from openai import OpenAI

# Add parent to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import calculate_cost

# Import evaluation tasks
from eval.tasks import TASK_REGISTRY, get_task
from minrlm import RLM  # Our implementation
from minrlm.core_reasoning import RLMReasoning

# =============================================================================
# Build Benchmark Options from Eval Tasks
# =============================================================================


def get_benchmark_options() -> dict[str, dict]:
    """Build benchmark dropdown options from the eval task registry."""
    options = {}

    sizes = {
        "Small (20K)": 20000,
        "Medium (50K)": 50000,
        "Large (100K)": 100000,
        "XL (200K)": 200000,
    }

    core_tasks = ["sniah", "multi_needle", "pairs", "qa_retrieval"]
    for task_name in core_tasks:
        if task_name in TASK_REGISTRY:
            task_cls = TASK_REGISTRY[task_name]
            for size_name, size_val in sizes.items():
                display = f"{task_name.upper().replace('_', ' ')} - {size_name}"
                options[display] = {
                    "task": task_name,
                    "context_size": size_val,
                    "description": f"{task_cls.description} ({size_val:,} chars)",
                }

    # Paper's scaling sizes: 8K to 1M (Figure 1), plus 10M for extreme testing
    scaling_sizes = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 10485760]
    for size in scaling_sizes:
        if size < 1024 * 1024:
            size_label = f"{size // 1024}K"
        else:
            size_label = f"{size // (1024 * 1024)}M"
        options[f"SCALING - {size_label}"] = {
            "task": "official_sniah",
            "context_size": size,
            "description": f"S-NIAH at {size:,} chars",
        }

    json_sizes = {"50K": 50000, "100K": 100000, "200K": 200000}
    for size_name, size_val in json_sizes.items():
        if "json_extraction" in TASK_REGISTRY:
            options[f"JSON EXTRACTION - {size_name}"] = {
                "task": "json_extraction",
                "context_size": size_val,
                "description": f"Extract data from JSON ({size_val:,} chars)",
            }
        if "json_aggregation" in TASK_REGISTRY:
            options[f"JSON AGGREGATION - {size_name}"] = {
                "task": "json_aggregation",
                "context_size": size_val,
                "description": f"Aggregate data from JSON ({size_val:,} chars)",
            }

    # Extended long context sizes including paper's large contexts
    long_sizes = {
        "128K": 131072,
        "256K": 262144,
        "512K": 524288,
        "1M": 1048576,
        "10M": 10485760,
    }
    for size_name, size_val in long_sizes.items():
        if "long_context" in TASK_REGISTRY:
            for pos in ["start", "middle", "end"]:
                options[f"LONG CONTEXT {size_name} ({pos})"] = {
                    "task": "long_context",
                    "context_size": size_val,
                    "position": pos,
                    "description": f"Needle at {pos} of {size_val:,} chars",
                }

    if "multi_needle_long" in TASK_REGISTRY:
        for size_name, size_val in long_sizes.items():
            options[f"MULTI-NEEDLE LONG - {size_name}"] = {
                "task": "multi_needle_long",
                "context_size": size_val,
                "description": f"Find 10 needles in {size_val:,} chars",
            }

    if "official_oolong" in TASK_REGISTRY:
        options["OOLONG (Aggregation)"] = {
            "task": "official_oolong",
            "context_size": 131072,
            "description": "Count label occurrences (131K chars)",
        }

    if "official_codeqa" in TASK_REGISTRY:
        # Paper's CodeQA sizes: 23K-4.2M, include 1M+
        codeqa_sizes = {
            "Small (100K)": 100000,
            "Medium (500K)": 500000,
            "Large (1M)": 1000000,
            "XL (2M)": 2000000,
            "XXL (10M)": 10000000,
        }
        for size_name, size_val in codeqa_sizes.items():
            options[f"CODEQA - {size_name}"] = {
                "task": "official_codeqa",
                "context_size": size_val,
                "description": f"Code repository understanding ({size_val:,} chars)",
            }

    if "official_browsecomp" in TASK_REGISTRY:
        # Paper's BrowseComp+ sizes: 6M-11M
        browsecomp_sizes = {
            "Small (200K)": 200000,
            "Medium (1M)": 1000000,
            "Large (6M)": 6000000,
            "XL (10M)": 10000000,
            "XXL (11M)": 11000000,
        }
        for size_name, size_val in browsecomp_sizes.items():
            options[f"BROWSECOMP - {size_name}"] = {
                "task": "official_browsecomp",
                "context_size": size_val,
                "description": f"Multi-hop research ({size_val:,} chars)",
            }

    if "official_gdpval" in TASK_REGISTRY:
        # Real professional work tasks across multiple occupations
        options["GDPVAL (Professional Tasks)"] = {
            "task": "official_gdpval",
            "description": "Real professional work tasks (Accounting, Tax, Finance, etc.)",
        }

    if "official_aime_2025" in TASK_REGISTRY:
        # AIME 2025 competition math problems
        options["AIME 2025 (Competition Math)"] = {
            "task": "official_aime_2025",
            "description": "AIME 2025 - 30 competition-level math problems",
        }

    return options


BENCHMARKS = get_benchmark_options()


# =============================================================================
# Model Utilities
# =============================================================================


def get_available_models(base_url: str = None) -> list[str]:
    """Fetch available models from API, returning only known working chat models."""
    # Known working chat models (conservative list)
    fallback_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    ]

    try:
        client = OpenAI(**({"base_url": base_url} if base_url else {}))
        models = client.models.list()
        all_model_ids = [m.id for m in models.data]

        # Conservative filter: Only include models with known-good patterns
        # gpt-4o*, gpt-4-turbo*, gpt-4-*, gpt-3.5-turbo*
        chat_models = []
        for model_id in all_model_ids:
            m_lower = model_id.lower()
            # Include gpt-4o variants
            if m_lower.startswith("gpt-4o"):
                chat_models.append(model_id)
            # Include gpt-4-turbo variants
            elif m_lower.startswith("gpt-4-turbo"):
                chat_models.append(model_id)
            # Include gpt-4 (but not gpt-4-base, gpt-4-vision, etc.)
            elif m_lower == "gpt-4" or (m_lower.startswith("gpt-4-") and "base" not in m_lower and "vision" not in m_lower):
                chat_models.append(model_id)
            # Include gpt-3.5-turbo variants
            elif m_lower.startswith("gpt-3.5-turbo"):
                chat_models.append(model_id)
            # Include gpt-5 if it exists
            elif m_lower.startswith("gpt-5"):
                chat_models.append(model_id)

        # Sort: latest versions first
        if chat_models:
            return sorted(chat_models, reverse=True)
        else:
            return fallback_models
    except:
        return fallback_models


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
    cost_usd: float | None = None


def run_vanilla_llm(task: str, context: str, model: str, check_fn: callable = None) -> RunResult:
    """Run with direct LLM call."""
    client = OpenAI()

    if context:
        prompt = f"{task}\n\nHere is the text to analyze:\n\n{context}"
    else:
        prompt = task

    trace = "## Vanilla LLM\n\n"
    trace += f"Sending {'full context (' + str(len(context)) + ' chars)' if context else 'prompt'} in one request.\n\n"

    start = time.time()
    try:
        kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        if "gpt-5" not in model.lower():
            kwargs["temperature"] = 0.7

        response = client.chat.completions.create(**kwargs)
        elapsed = time.time() - start

        resp_text = response.choices[0].message.content or ""
        usage = response.usage
        correct = check_fn(resp_text) if check_fn else True

        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        cost = calculate_cost(model, input_tokens, output_tokens)

        trace += f"**Response:** `{resp_text[:200]}{'...' if len(resp_text) > 200 else ''}`\n\n"
        if check_fn:
            trace += f"{'✅ Correct' if correct else '❌ Incorrect'}\n\n"

        return RunResult(
            response=resp_text,
            correct=correct,
            tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            time_seconds=elapsed,
            cost_usd=cost,
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


def run_our_rlm(task: str, context: str, model: str, check_fn: callable = None) -> RunResult:
    """Run with minRLM."""
    trace_parts = ["## minRLM\n\n"]
    if context:
        trace_parts.append(f"Processing {len(context):,} chars...\n\n")

    def on_step(event: str, data: dict):
        if event == "thinking":
            trace_parts.append(f"### Iteration {data['iteration']}\n\n")
        elif event == "llm_response":
            has_code = data.get("has_code", False)
            trace_parts.append(f"**LLM Response** ({len(data.get('response', ''))} chars):\n")
            if has_code:
                trace_parts.append("✅ Contains code block\n\n")
            else:
                response_preview = data.get("response", "")[:300]
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
                    trace_parts.append(
                        f"**stdout:**\n```\n{stdout[:2000]}{'...' if len(stdout) > 2000 else ''}\n```\n\n"
                    )
                if output:
                    trace_parts.append(f"✅ **FINAL():** `{output}`\n\n")
                elif not stdout:
                    trace_parts.append("*(no output)*\n\n")

    start = time.time()
    try:
        rlm = RLM(model=model, max_iterations=10, on_step=on_step)
        if context:
            result = rlm.completion(task=task, context=context)
        else:
            result = rlm.completion(task=task)
        elapsed = time.time() - start

        response = result.response
        correct = check_fn(response) if check_fn else True

        cost = calculate_cost(model, result.input_tokens, result.output_tokens)

        trace_parts.append(f"\n**Final:** `{response[:200]}{'...' if len(response) > 200 else ''}`\n")
        trace_parts.append(
            f"**Tokens:** {result.input_tokens:,} in + {result.output_tokens:,} out = {result.total_tokens:,} total"
        )
        if cost is not None:
            trace_parts.append(f" | **Cost:** ${cost:.6f}")
        trace_parts.append(f" | **Time:** {elapsed:.1f}s\n")
        if check_fn:
            trace_parts.append(f"{'✅ Correct' if correct else '❌ Incorrect'}\n\n")

        return RunResult(
            response=response,
            correct=correct,
            tokens=result.total_tokens,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            time_seconds=elapsed,
            iterations=result.iterations,
            cost_usd=cost,
            trace="".join(trace_parts),
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
            trace="".join(trace_parts),
        )


def run_reasoning_rlm(task: str, context: str, model: str, check_fn: callable = None) -> RunResult:
    """Run with RLMReasoning (Reasoning approach)."""
    trace_parts = ["## Recursive minRLM\n\n"]
    if context:
        trace_parts.append(f"Processing {len(context):,} chars with reasoning-first approach...\n\n")

    def on_step(event: str, data: dict):
        if event == "thinking":
            trace_parts.append(f"### Iteration {data['iteration']}\n\n")
        elif event == "reasoning":
            reasoning = data.get("reasoning", "")
            trace_parts.append(f"**Reasoning:**\n> {reasoning[:500]}{'...' if len(reasoning) > 500 else ''}\n\n")
        elif event == "llm_response":
            has_code = data.get("has_code", False)
            trace_parts.append(f"**LLM Response** ({len(data.get('response', ''))} chars):\n")
            if has_code:
                trace_parts.append("Contains code block\n\n")
            else:
                response_preview = data.get("response", "")[:300]
                trace_parts.append("No code block found\n\n")
                trace_parts.append(
                    f"```\n{response_preview}{'...' if len(data.get('response', '')) > 300 else ''}\n```\n\n"
                )
        elif event == "executing":
            code = data.get("code", "")
            trace_parts.append(f"**Executing Code:**\n```python\n{code}\n```\n\n")
        elif event == "executed":
            if data.get("error"):
                trace_parts.append(f"**Error:**\n```\n{data['error']}\n```\n\n")
            else:
                stdout = data.get("stdout", "")
                output = data.get("output")
                if stdout:
                    trace_parts.append(
                        f"**stdout:**\n```\n{stdout[:2000]}{'...' if len(stdout) > 2000 else ''}\n```\n\n"
                    )
                if output:
                    trace_parts.append(f"**FINAL():** `{output}`\n\n")
                elif not stdout:
                    trace_parts.append("*(no output)*\n\n")

    start = time.time()
    try:
        rlm = RLMReasoning(model=model, max_iterations=10, on_step=on_step)
        if context:
            result = rlm.completion(task=task, context=context)
        else:
            result = rlm.completion(task=task)
        elapsed = time.time() - start

        response = result.response
        correct = check_fn(response) if check_fn else True

        cost = calculate_cost(model, result.input_tokens, result.output_tokens)

        if result.reasoning:
            trace_parts.append(f"\n**Reasoning Summary:** {result.reasoning[:300]}{'...' if len(result.reasoning) > 300 else ''}\n")
        trace_parts.append(f"\n**Final:** `{response[:200]}{'...' if len(response) > 200 else ''}`\n")
        trace_parts.append(
            f"**Tokens:** {result.input_tokens:,} in + {result.output_tokens:,} out = {result.total_tokens:,} total"
        )
        if cost is not None:
            trace_parts.append(f" | **Cost:** ${cost:.6f}")
        trace_parts.append(f" | **Time:** {elapsed:.1f}s\n")
        if check_fn:
            trace_parts.append(f"{'Correct' if correct else 'Incorrect'}\n\n")

        return RunResult(
            response=response,
            correct=correct,
            tokens=result.total_tokens,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            time_seconds=elapsed,
            iterations=result.iterations,
            cost_usd=cost,
            trace="".join(trace_parts),
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
            trace="".join(trace_parts),
        )


def run_official_rlm(task: str, context: str, model: str, check_fn: callable = None) -> RunResult:
    """Run with official RLM from github.com/alexzhang13/rlm via uv --with."""
    trace = "## Official RLM\n\n"
    if context:
        trace += f"Processing {len(context):,} chars via [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm).\n\n"
    else:
        trace += "Running via [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm).\n\n"

    start = time.time()

    # Write context and task to temp files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(context or "")
        context_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(task)
        task_file = f.name

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

    if context.strip():
        result = rlm.completion(prompt=context, root_prompt=task)
    else:
        result = rlm.completion(prompt=task)
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
        result = subprocess.run(
            ["uv", "run", "--with", "git+https://github.com/alexzhang13/rlm", "python", "-c", script],
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

            correct = check_fn(resp_text) if check_fn else True
            cost = calculate_cost(model, input_tokens, output_tokens)

            trace += f"**Tokens:** {input_tokens:,} in + {output_tokens:,} out = {total_tokens:,} total"
            if cost is not None:
                trace += f" | **Cost:** ${cost:.6f}"
            trace += "\n\n"
            trace += f"**Response:** `{resp_text[:200]}{'...' if len(resp_text) > 200 else ''}`\n\n"
            if check_fn:
                trace += f"{'✅ Correct' if correct else '❌ Incorrect'}\n\n"

            return RunResult(
                response=resp_text,
                correct=correct,
                tokens=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                time_seconds=elapsed,
                iterations=iterations,
                cost_usd=cost,
                trace=trace,
            )
        elif "<<<ERROR>>>" in output:
            error_msg = output.split("<<<ERROR>>>")[1].strip()
            trace += f"**Error:** {error_msg[:300]}\n"
        else:
            if result.stderr:
                trace += f"**Error:** {result.stderr[:500]}\n"

        return RunResult(
            response="", correct=False, tokens=0, input_tokens=0, output_tokens=0, time_seconds=elapsed, trace=trace
        )

    except subprocess.TimeoutExpired:
        trace += "**Error:** Timeout (5 min limit)\n"
        return RunResult(
            response="", correct=False, tokens=0, input_tokens=0, output_tokens=0, time_seconds=300, trace=trace
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
# Task Generation
# =============================================================================


def generate_task_instance(benchmark_name: str) -> tuple[str, str, str, str, callable]:
    """Generate a task instance from the eval task registry."""
    if benchmark_name not in BENCHMARKS:
        return "Select a benchmark", "", "", "", lambda x: False

    config = BENCHMARKS[benchmark_name]
    task_name = config["task"]
    task_obj = get_task(task_name)

    gen_kwargs = {"seed": int(time.time()) % 10000}
    if "context_size" in config:
        gen_kwargs["context_size"] = config["context_size"]
    if "position" in config:
        gen_kwargs["position"] = config["position"]

    instance = task_obj.generate(**gen_kwargs)

    def check_fn(response: str) -> bool:
        return task_obj.check(response, instance.expected)

    description = config.get("description", f"{task_name} task")

    return description, instance.task, instance.context, instance.expected, check_fn


# =============================================================================
# Shared UI Helpers
# =============================================================================


def create_status_box(
    title: str, subtitle: str = "", icon: str = "⏳", color: str = "#818cf8", pulse: bool = True
) -> str:
    pulse_opacity = "animation: status-pulse 2s ease-in-out infinite;" if pulse else ""
    return f"""
    <div class="status-box" style="--status-color: {color};">
        <div class="status-box__accent"></div>
        <div class="status-box__icon" style="{pulse_opacity}">{icon}</div>
        <div class="status-box__title">{title}</div>
        <div class="status-box__subtitle">{subtitle}</div>
    </div>"""


def build_charts(results_list: list) -> tuple:
    if not results_list:
        return None, None

    data = [{"Method": name, "Tokens": float(r.tokens), "Time": float(r.time_seconds)} for name, r in results_list]
    df = pd.DataFrame(data)

    colors = {"Vanilla": "#60a5fa", "minRLM": "#f97316", "minRLM with Reasoning": "#c084fc", "Official": "#22c55e"}
    color_map = {m: colors.get(m, "#94a3b8") for m in df["Method"]}

    layout_common = {
        "showlegend": False,
        "height": 280,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(24,24,27,0.6)",
        "font": {"family": "DM Sans, system-ui, sans-serif", "color": "#a1a1aa", "size": 12},
        "title_font": {"size": 14, "color": "#fafafa"},
        "xaxis": {"gridcolor": "rgba(255,255,255,0.06)", "linecolor": "rgba(255,255,255,0.08)", "tickfont": {"color": "#71717a"}},
        "yaxis": {"gridcolor": "rgba(255,255,255,0.06)", "linecolor": "rgba(255,255,255,0.08)", "tickfont": {"color": "#71717a"}},
        "margin": {"t": 44, "b": 36, "l": 52, "r": 16},
        "bargap": 0.36,
    }

    tokens_fig = px.bar(df, x="Method", y="Tokens", color="Method", color_discrete_map=color_map)
    tokens_fig.update_layout(**layout_common, title="Token usage", yaxis_title="Tokens")
    tokens_fig.update_traces(marker_line_width=0, opacity=0.92)

    time_fig = px.bar(df, x="Method", y="Time", color="Method", color_discrete_map=color_map)
    time_fig.update_layout(**layout_common, title="Time (s)", yaxis_title="Seconds")
    time_fig.update_traces(marker_line_width=0, opacity=0.92)

    return tokens_fig, time_fig


# =============================================================================
# Gradio App
# =============================================================================


def build_app():
    initial_models = get_available_models()
    benchmark_names = list(BENCHMARKS.keys())

    custom_css = """
    :root {
        --font-sans: "DM Sans", "Plus Jakarta Sans", system-ui, sans-serif;
        --bg-page: #09090b;
        --bg-card: #18181b;
        --border: #27272a;
        --text: #fafafa;
        --text-muted: #a1a1aa;
        --accent: #3b82f6;
    }
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
    .gradio-container { max-width: 1200px !important; margin: 0 auto !important; padding: 0 24px !important; }
    .contain { max-width: 1200px; margin: 0 auto; }
    /* Status box */
    .status-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 28px 24px;
        margin: 16px 0;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        position: relative;
        overflow: hidden;
    }
    .status-box__accent {
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        background: var(--status-color);
    }
    .status-box__icon { font-size: 2rem; line-height: 1; }
    .status-box__title {
        font-family: var(--font-sans);
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--status-color);
        letter-spacing: -0.01em;
    }
    .status-box__subtitle {
        font-size: 0.875rem;
        color: var(--text-muted);
    }
    @keyframes status-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
    /* Header */
    .app-header {
        text-align: center;
        padding: 40px 0 32px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 28px;
    }
    .app-header h1 {
        margin: 0 0 8px 0;
        font-family: var(--font-sans);
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: var(--text);
    }
    .app-header p {
        margin: 0 0 20px 0;
        font-size: 0.9375rem;
        color: var(--text-muted);
    }
    .app-header .pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8125rem;
        font-weight: 500;
    }
    .app-header .links {
        display: flex;
        gap: 20px;
        justify-content: center;
        font-size: 0.875rem;
    }
    .app-header .links a {
        color: var(--text-muted);
        text-decoration: none;
        transition: color 0.15s;
    }
    .app-header .links a:hover { color: var(--accent); }
    /* Blocks */
    .gr-block { border-radius: 12px !important; }
    .gr-input, .gr-box { border-radius: 10px !important; border-color: var(--border) !important; }
    .gr-button { border-radius: 10px !important; font-weight: 500 !important; }
    .gradio-container table {
        border-collapse: collapse;
        width: 100%;
        font-size: 0.8125rem;
        margin: 12px 0;
    }
    .gradio-container th,
    .gradio-container td {
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid var(--border);
    }
    .gradio-container th { color: var(--text-muted); font-weight: 500; }
    .gradio-container tbody tr:hover td { background: rgba(255,255,255,0.03); }
    /* Footer */
    .app-footer {
        text-align: center;
        padding: 28px 0;
        margin-top: 40px;
        border-top: 1px solid var(--border);
        font-size: 0.8125rem;
        color: var(--text-muted);
    }
    .app-footer a { color: var(--text-muted); text-decoration: none; }
    .app-footer a:hover { color: var(--accent); }
    """

    with gr.Blocks(
        title="RLM Visualizer",
        css=custom_css,
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="zinc",
            neutral_hue="zinc",
            font=gr.themes.GoogleFont("DM Sans"),
        ).set(
            body_background_fill="#09090b",
            body_background_fill_dark="#09090b",
            block_background_fill="#18181b",
            block_background_fill_dark="#18181b",
            block_border_color="#27272a",
            block_border_color_dark="#27272a",
            block_label_background_fill="#27272a",
            block_label_background_fill_dark="#27272a",
            block_title_text_color="#fafafa",
            block_title_text_color_dark="#fafafa",
            body_text_color="#fafafa",
            body_text_color_dark="#fafafa",
            input_background_fill="#27272a",
            input_background_fill_dark="#27272a",
            button_primary_background_fill="#3b82f6",
            button_primary_background_fill_dark="#3b82f6",
            button_primary_background_fill_hover="#2563eb",
            button_primary_background_fill_hover_dark="#2563eb",
        ),
    ) as demo:
        gr.HTML("""
        <header class="app-header">
            <h1>RLM Visualizer</h1>
            <p>
                Compare <span class="pill" style="background: rgba(59,130,246,0.15); color: #60a5fa;">Vanilla</span>
                <span class="pill" style="background: rgba(249,115,22,0.15); color: #fb923c;">minRLM</span>
                <span class="pill" style="background: rgba(192,132,252,0.15); color: #c084fc;">minRLM with Reasoning</span>
                <span class="pill" style="background: rgba(34,197,94,0.15); color: #4ade80;">Official RLM</span>
            </p>
            <div class="links">
                <a href="https://arxiv.org/abs/2512.24601" target="_blank">Paper</a>
                <a href="https://github.com/avilum/minrlm" target="_blank">GitHub</a>
            </div>
        </header>
        """)

        with gr.Accordion("About", open=False):
            gr.Markdown("""
            **RLM** = Recursive Language Model. The model writes Python code to search/process data in a REPL; data never enters context—only metadata. Token usage stays flat with context size.

            **Benchmarks (gpt-5-nano):** ~4.2× fewer tokens, ~4× cheaper; large contexts (128K+) often beat vanilla; 100% on 6M–11M where vanilla fails. [README](https://github.com/avilum/minrlm)
            """)

        with gr.Row(equal_height=True):
            # Use gpt-4o-mini as default (it's reliable and cost-effective)
            # Fallback to first model in list if gpt-4o-mini not available
            default_model = "gpt-4o-mini" if "gpt-4o-mini" in initial_models else (
                initial_models[0] if initial_models else "gpt-4o-mini"
            )

            model_dropdown = gr.Dropdown(
                choices=initial_models,
                value=default_model,
                label="Model",
                scale=3,
                container=True,
            )
            with gr.Column(scale=2, min_width=280):
                gr.Markdown("Methods", elem_classes=["method-label"])
                with gr.Row():
                    vanilla_checkbox = gr.Checkbox(label="Vanilla", value=True, scale=1)
                    rlm_checkbox = gr.Checkbox(label="minRLM", value=True, scale=1)
                    reasoning_checkbox = gr.Checkbox(label="minRLM with Reasoning", value=True, scale=1)
                    official_checkbox = gr.Checkbox(label="Official", value=True, scale=1)

        gr.HTML("<div style='height: 20px'></div>")

        # =================================================================
        # TABS
        # =================================================================
        with gr.Tabs():
            # =============================================================
            # TAB 1: Evaluation Tasks
            # =============================================================
            with gr.TabItem("Benchmarks", id="eval"):
                check_fn_state = gr.State(value=None)

                # Task Selection Row
                with gr.Group():
                    with gr.Row(equal_height=True):
                        benchmark_dropdown = gr.Dropdown(
                            choices=benchmark_names,
                            value=benchmark_names[0] if benchmark_names else None,
                            label="Benchmark",
                            scale=4,
                        )
                        generate_btn = gr.Button("New instance", variant="secondary", scale=1, min_width=120)
                        run_eval_btn = gr.Button("Run comparison", variant="primary", scale=1, min_width=140)

                task_description = gr.Markdown("*Select a benchmark and click New instance*")

                # Status & Results
                eval_status_html = gr.HTML(
                    create_status_box("Ready", "Select a task and run", "○", "#71717a", False)
                )
                eval_results_output = gr.Markdown("")

                with gr.Row(equal_height=True):
                    with gr.Column():
                        eval_tokens_plot = gr.Plot(label="Token usage", show_label=True)
                    with gr.Column():
                        eval_time_plot = gr.Plot(label="Time", show_label=True)

                with gr.Accordion("Execution traces", open=False):
                    eval_traces_output = gr.Markdown("*Run a benchmark to see traces.*")

                with gr.Accordion("Task details", open=False):
                    with gr.Row():
                        with gr.Column(scale=2):
                            task_text = gr.Textbox(label="Task", lines=3, interactive=False)
                        with gr.Column(scale=1):
                            expected_text = gr.Textbox(label="Expected", interactive=False)
                    context_preview = gr.Textbox(label="Context preview", lines=5, interactive=False)

                full_context = gr.State("")

                def on_generate(benchmark_name):
                    desc, task, context, expected, check_fn = generate_task_instance(benchmark_name)
                    preview = context[:500] + "..." if len(context) > 500 else context
                    return (
                        f"**{benchmark_name}**\n\n{desc}\n\n*Context: {len(context):,} chars*",
                        task,
                        expected,
                        preview,
                        context,
                        check_fn,
                    )

                generate_btn.click(
                    fn=on_generate,
                    inputs=[benchmark_dropdown],
                    outputs=[task_description, task_text, expected_text, context_preview, full_context, check_fn_state],
                )
                benchmark_dropdown.change(
                    fn=on_generate,
                    inputs=[benchmark_dropdown],
                    outputs=[task_description, task_text, expected_text, context_preview, full_context, check_fn_state],
                )

                def run_eval_task(task, context, model, run_vanilla, run_rlm, run_reasoning, run_official, benchmark_name, check_fn):
                    if not task:
                        yield (
                            create_status_box("No Task", "Generate a task first", "📋", "#666", False),
                            "",
                            None,
                            None,
                            "",
                        )
                        return

                    results_list = []
                    traces = ""

                    methods = []
                    if run_vanilla:
                        methods.append(("Vanilla", "🔵", "#4dabf7", run_vanilla_llm))
                    if run_rlm:
                        methods.append(("minRLM", "🟠", "#ff922b", run_our_rlm))
                    if run_reasoning:
                        methods.append(("minRLM with Reasoning", "🟣", "#c084fc", run_reasoning_rlm))
                    if run_official:
                        methods.append(("Official", "🟢", "#51cf66", run_official_rlm))

                    if not methods:
                        yield (
                            create_status_box("No Methods", "Select at least one method", "⚠️", "#fcc419", False),
                            "",
                            None,
                            None,
                            "",
                        )
                        return

                    total_elapsed = 0.0
                    for i, (name, icon, color, run_fn) in enumerate(methods):
                        yield (
                            create_status_box(
                                f"Running {name}…",
                                f"Step {i + 1}/{len(methods)} · {len(context):,} chars",
                                "⋯",
                                color,
                                True,
                            ),
                            "",
                            *build_charts(results_list),
                            traces,
                        )

                        step_start = time.time()
                        r = run_fn(task, context, model, check_fn)
                        step_elapsed = time.time() - step_start
                        total_elapsed += step_elapsed

                        results_list.append((name, r))
                        traces += r.trace + "\n---\n\n"

                    # Final output
                    output = f"**{benchmark_name}** · {len(context):,} chars · {model}\n\n"
                    output += (
                        "| Method | Result | Input Tokens | Output Tokens | Total Tokens | Cost | Time | Iters |\n"
                    )
                    output += (
                        "|--------|--------|--------------|---------------|--------------|------|------|-------|\n"
                    )
                    for name, r in results_list:
                        status = "✅" if r.correct else "❌"
                        cost_str = f"${r.cost_usd:.6f}" if r.cost_usd is not None else "N/A"
                        output += f"| {name} | {status} | {r.input_tokens:,} | {r.output_tokens:,} | {r.tokens:,} | {cost_str} | {r.time_seconds:.1f}s | {r.iterations} |\n"

                    if len(results_list) >= 2:
                        tokens = [(n, r.tokens) for n, r in results_list if r.tokens > 0]
                        costs = [(n, r.cost_usd) for n, r in results_list if r.cost_usd is not None]

                        if tokens:
                            best, worst = min(tokens, key=lambda x: x[1]), max(tokens, key=lambda x: x[1])
                            if best[1] < worst[1]:
                                output += f"\n**{best[0]}** used {(1 - best[1] / worst[1]) * 100:.0f}% fewer tokens."

                        if costs:
                            best_cost, worst_cost = min(costs, key=lambda x: x[1]), max(costs, key=lambda x: x[1])
                            if best_cost[1] < worst_cost[1]:
                                savings = (1 - best_cost[1] / worst_cost[1]) * 100
                                output += f"\n**{best_cost[0]}** is {savings:.0f}% cheaper (${best_cost[1]:.6f} vs ${worst_cost[1]:.6f})."

                    all_correct = all(r.correct for _, r in results_list)
                    final_status = create_status_box(
                        "Done" if all_correct else "Done",
                        f"{total_elapsed:.1f}s total",
                        "✓" if all_correct else "○",
                        "#22c55e" if all_correct else "#71717a",
                        False,
                    )

                    yield final_status, output, *build_charts(results_list), traces

                run_eval_btn.click(
                    fn=run_eval_task,
                    inputs=[
                        task_text,
                        full_context,
                        model_dropdown,
                        vanilla_checkbox,
                        rlm_checkbox,
                        reasoning_checkbox,
                        official_checkbox,
                        benchmark_dropdown,
                        check_fn_state,
                    ],
                    outputs=[
                        eval_status_html,
                        eval_results_output,
                        eval_tokens_plot,
                        eval_time_plot,
                        eval_traces_output,
                    ],
                )

            # =============================================================
            # TAB 2: Custom Task
            # =============================================================
            with gr.TabItem("Custom", id="custom"):
                gr.Markdown("Run your own task with optional context.")

                custom_task_input = gr.Textbox(
                    label="Task",
                    placeholder="e.g. Calculate 2^1000 or find all Engineering employees",
                    lines=3,
                )
                custom_context_input = gr.Textbox(
                    label="Context (optional)",
                    placeholder="Paste document, JSON, or text. Leave empty if no context.",
                    lines=8,
                )
                run_custom_btn = gr.Button("Run comparison", variant="primary", size="lg")

                custom_status_html = gr.HTML(
                    create_status_box("Ready", "Enter a task and run", "○", "#71717a", False)
                )

                with gr.Group():
                    custom_results_output = gr.Markdown("")

                with gr.Row(equal_height=True):
                    with gr.Column():
                        custom_tokens_plot = gr.Plot(label="Token usage", show_label=True)
                    with gr.Column():
                        custom_time_plot = gr.Plot(label="Time", show_label=True)

                with gr.Accordion("Execution traces", open=False):
                    custom_traces_output = gr.Markdown("*Run a task to see traces.*")

                with gr.Accordion("Responses", open=True):
                    custom_responses = gr.Markdown("*Responses appear here after running.*")

                def run_custom_task(task, context, model, run_vanilla, run_rlm, run_reasoning, run_official):
                    if not task.strip():
                        yield (
                            create_status_box("No Task", "Enter a task prompt", "✏️", "#666", False),
                            "",
                            None,
                            None,
                            "",
                            "",
                        )
                        return

                    results_list = []
                    traces = ""

                    methods = []
                    if run_vanilla:
                        methods.append(("Vanilla", "🔵", "#4dabf7", run_vanilla_llm))
                    if run_rlm:
                        methods.append(("minRLM", "🟠", "#ff922b", run_our_rlm))
                    if run_reasoning:
                        methods.append(("minRLM with Reasoning", "🟣", "#c084fc", run_reasoning_rlm))
                    if run_official:
                        methods.append(("Official", "🟢", "#51cf66", run_official_rlm))

                    if not methods:
                        yield (
                            create_status_box("No Methods", "Select at least one method", "⚠️", "#fcc419", False),
                            "",
                            None,
                            None,
                            "",
                            "",
                        )
                        return

                    context_info = f"{len(context):,} chars" if context else "no context"
                    total_elapsed = 0.0

                    for i, (name, icon, color, run_fn) in enumerate(methods):
                        yield (
                            create_status_box(
                                f"Running {name}…", f"Step {i + 1}/{len(methods)} · {context_info}", "⋯", color, True
                            ),
                            "",
                            *build_charts(results_list),
                            traces,
                            "",
                        )

                        step_start = time.time()
                        r = run_fn(task, context, model, None)  # No check_fn for custom tasks
                        step_elapsed = time.time() - step_start
                        total_elapsed += step_elapsed

                        results_list.append((name, r))
                        traces += r.trace + "\n---\n\n"

                    # Final output table
                    output = f"**Custom Task** · {context_info} · {model}\n\n"
                    output += "| Method | Input Tokens | Output Tokens | Total Tokens | Cost | Time | Iters |\n"
                    output += "|--------|--------------|---------------|--------------|------|------|-------|\n"
                    for name, r in results_list:
                        cost_str = f"${r.cost_usd:.6f}" if r.cost_usd is not None else "N/A"
                        output += f"| {name} | {r.input_tokens:,} | {r.output_tokens:,} | {r.tokens:,} | {cost_str} | {r.time_seconds:.1f}s | {r.iterations} |\n"

                    if len(results_list) >= 2:
                        tokens = [(n, r.tokens) for n, r in results_list if r.tokens > 0]
                        costs = [(n, r.cost_usd) for n, r in results_list if r.cost_usd is not None]

                        if tokens:
                            best, worst = min(tokens, key=lambda x: x[1]), max(tokens, key=lambda x: x[1])
                            if best[1] < worst[1]:
                                output += f"\n**{best[0]}** used {(1 - best[1] / worst[1]) * 100:.0f}% fewer tokens."

                        if costs:
                            best_cost, worst_cost = min(costs, key=lambda x: x[1]), max(costs, key=lambda x: x[1])
                            if best_cost[1] < worst_cost[1]:
                                savings = (1 - best_cost[1] / worst_cost[1]) * 100
                                output += f"\n**{best_cost[0]}** is {savings:.0f}% cheaper (${best_cost[1]:.6f} vs ${worst_cost[1]:.6f})."

                    # Build responses display
                    responses_md = "### Responses\n\n"
                    for name, r in results_list:
                        responses_md += (
                            f"**{name}:**\n```\n{r.response[:2000]}{'...' if len(r.response) > 2000 else ''}\n```\n\n"
                        )

                    final_status = create_status_box("Done", f"{total_elapsed:.1f}s total", "✓", "#22c55e", False)

                    yield final_status, output, *build_charts(results_list), traces, responses_md

                run_custom_btn.click(
                    fn=run_custom_task,
                    inputs=[
                        custom_task_input,
                        custom_context_input,
                        model_dropdown,
                        vanilla_checkbox,
                        rlm_checkbox,
                        reasoning_checkbox,
                        official_checkbox,
                    ],
                    outputs=[
                        custom_status_html,
                        custom_results_output,
                        custom_tokens_plot,
                        custom_time_plot,
                        custom_traces_output,
                        custom_responses,
                    ],
                )

        # Initial load for eval tab
        demo.load(
            fn=on_generate,
            inputs=[benchmark_dropdown],
            outputs=[task_description, task_text, expected_text, context_preview, full_context, check_fn_state],
        )

        gr.HTML("""
        <footer class="app-footer">
            <a href="https://gradio.app" target="_blank">Gradio</a> ·
            <a href="https://arxiv.org/abs/2512.24601" target="_blank">Paper</a> ·
            <a href="https://github.com/avilum/minrlm" target="_blank">GitHub</a>
        </footer>
        """)

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("🔬 RLM Visualizer")
    print("=" * 60)
    print(f"📊 Available eval tasks: {len(BENCHMARKS)}")
    print(f"🏷️  Task types: {sorted({c['task'] for c in BENCHMARKS.values()})}")
    print("=" * 60)
    build_app().launch()

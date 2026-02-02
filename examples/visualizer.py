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

from minrlm import RLM  # Our implementation

# Import evaluation tasks
from eval.tasks import TASK_REGISTRY, get_task
from eval.metrics import calculate_cost


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
            "task": "scaling",
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

    if "oolong" in TASK_REGISTRY:
        options["OOLONG (Aggregation)"] = {
            "task": "oolong",
            "context_size": 131072,
            "description": "Count label occurrences (131K chars)",
        }

    if "codeqa" in TASK_REGISTRY:
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
                "task": "codeqa",
                "context_size": size_val,
                "description": f"Code repository understanding ({size_val:,} chars)",
            }

    if "browsecomp" in TASK_REGISTRY:
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
                "task": "browsecomp",
                "context_size": size_val,
                "description": f"Multi-hop research ({size_val:,} chars)",
            }

    return options


BENCHMARKS = get_benchmark_options()


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
        return ["gpt-5-nano", "gpt-5-mini", "gpt-5-nano-mini"]


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
                    trace_parts.append(f"**stdout:**\n```\n{stdout[:2000]}{'...' if len(stdout) > 2000 else ''}\n```\n\n")
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
        trace_parts.append(f"**Tokens:** {result.input_tokens:,} in + {result.output_tokens:,} out = {result.total_tokens:,} total")
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

        return RunResult(response="", correct=False, tokens=0, input_tokens=0, output_tokens=0, time_seconds=elapsed, trace=trace)

    except subprocess.TimeoutExpired:
        trace += "**Error:** Timeout (5 min limit)\n"
        return RunResult(response="", correct=False, tokens=0, input_tokens=0, output_tokens=0, time_seconds=300, trace=trace)
    except Exception as e:
        trace += f"**Error:** {e}\n"
        return RunResult(response="", correct=False, tokens=0, input_tokens=0, output_tokens=0, time_seconds=time.time() - start, trace=trace)
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


def create_status_box(title: str, subtitle: str = "", icon: str = "⏳", color: str = "#818cf8", pulse: bool = True) -> str:
    pulse_style = "animation: pulse 1.5s ease-in-out infinite;" if pulse else ""
    glow = f"box-shadow: 0 0 40px {color}33;" if pulse else ""
    return f"""
    <div style="
        text-align: center;
        padding: 32px 24px;
        background: linear-gradient(135deg, rgba(30,41,59,0.9) 0%, rgba(15,23,42,0.95) 100%);
        border-radius: 16px;
        margin: 16px 0;
        border: 1px solid rgba(148,163,184,0.1);
        border-left: 4px solid {color};
        {glow}
    ">
        <div style="font-size: 3em; margin-bottom: 12px; {pulse_style}">{icon}</div>
        <div style="font-size: 1.5em; font-weight: 700; color: {color}; margin-bottom: 6px; letter-spacing: -0.01em;">{title}</div>
        <div style="font-size: 1rem; color: #94a3b8;">{subtitle}</div>
        <style>@keyframes pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.7; transform: scale(1.05); }} }}</style>
    </div>"""


def build_charts(results_list: list) -> tuple:
    if not results_list:
        return None, None

    data = [{"Method": name, "Tokens": float(r.tokens), "Time": float(r.time_seconds)} for name, r in results_list]
    df = pd.DataFrame(data)

    colors = {"Vanilla": "#60a5fa", "minRLM": "#f97316", "Official": "#22c55e"}
    color_map = {m: colors.get(m, "#94a3b8") for m in df["Method"]}

    # Common layout settings
    layout_common = dict(
        showlegend=False,
        height=320,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.5)",
        font=dict(family="Inter, system-ui, sans-serif", color="#e2e8f0"),
        title_font=dict(size=16, color="#e2e8f0"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)", title=""),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        margin=dict(t=50, b=40, l=60, r=20),
    )

    tokens_fig = px.bar(df, x="Method", y="Tokens", color="Method", color_discrete_map=color_map)
    tokens_fig.update_layout(**layout_common, title="Token Usage", yaxis_title="Tokens")
    tokens_fig.update_traces(marker_line_width=0, opacity=0.9)

    time_fig = px.bar(df, x="Method", y="Time", color="Method", color_discrete_map=color_map)
    time_fig.update_layout(**layout_common, title="Execution Time", yaxis_title="Seconds")
    time_fig.update_traces(marker_line_width=0, opacity=0.9)

    return tokens_fig, time_fig


# =============================================================================
# Gradio App
# =============================================================================


def build_app():
    initial_models = get_available_models()
    benchmark_names = list(BENCHMARKS.keys())

    # Custom CSS for professional look
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: auto !important;
    }
    .header-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        color: #94a3b8 !important;
        font-size: 1.1rem !important;
    }
    .header-links a {
        color: #818cf8 !important;
        text-decoration: none !important;
        font-weight: 500;
        transition: color 0.2s;
    }
    .header-links a:hover {
        color: #a5b4fc !important;
    }
    .control-panel {
        background: linear-gradient(180deg, rgba(30,41,59,0.8) 0%, rgba(15,23,42,0.9) 100%);
        border: 1px solid rgba(148,163,184,0.1);
        border-radius: 16px;
        padding: 20px;
    }
    .result-card {
        background: rgba(30,41,59,0.6);
        border: 1px solid rgba(148,163,184,0.1);
        border-radius: 12px;
        padding: 16px;
    }
    """

    with gr.Blocks(title="RLM Visualizer", css=custom_css, theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="*neutral_950",
        body_background_fill_dark="*neutral_950",
        block_background_fill="*neutral_900",
        block_background_fill_dark="*neutral_900",
        block_border_color="*neutral_800",
        block_border_color_dark="*neutral_800",
        block_label_background_fill="*primary_600",
        block_label_background_fill_dark="*primary_600",
        block_title_text_color="*neutral_100",
        block_title_text_color_dark="*neutral_100",
        input_background_fill="*neutral_800",
        input_background_fill_dark="*neutral_800",
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_dark="*primary_600",
        button_primary_background_fill_hover="*primary_500",
        button_primary_background_fill_hover_dark="*primary_500",
    )) as demo:

        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 32px 0 24px 0; border-bottom: 1px solid rgba(148,163,184,0.1); margin-bottom: 24px;">
            <h1 class="header-title" style="margin: 0; font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🔬 RLM Visualizer
            </h1>
            <p class="header-subtitle" style="margin: 8px 0 16px 0; color: #94a3b8; font-size: 1.1rem;">
                Compare <strong style="color:#60a5fa">Vanilla LLM</strong>, <strong style="color:#f97316">minRLM</strong>, and <strong style="color:#22c55e">Official RLM</strong> side-by-side
            </p>
            <div class="header-links" style="display: flex; gap: 24px; justify-content: center; font-size: 0.95rem;">
                <a href="https://arxiv.org/abs/2512.24601" target="_blank" style="color: #818cf8; text-decoration: none;">📄 Paper</a>
                <a href="https://github.com/avilum/minrlm" target="_blank" style="color: #818cf8; text-decoration: none;">⭐ GitHub</a>
            </div>
        </div>
        """)

        # Model & Method Selection Panel
        with gr.Group():
            with gr.Row(equal_height=True):
                model_dropdown = gr.Dropdown(
                    choices=initial_models,
                    value=("gpt-5-nano" if "gpt-5-nano" in initial_models else initial_models[0]),
                    label="🤖 Model",
                    scale=3,
                    container=True,
                )
                with gr.Column(scale=2, min_width=300):
                    gr.Markdown("**Methods to Compare**", elem_classes=["method-label"])
                    with gr.Row():
                        vanilla_checkbox = gr.Checkbox(label="🔵 Vanilla", value=True, scale=1)
                        rlm_checkbox = gr.Checkbox(label="🟠 minRLM", value=True, scale=1)
                        official_checkbox = gr.Checkbox(label="🟢 Official", value=True, scale=1)

        gr.HTML("<div style='height: 16px'></div>")

        # =================================================================
        # TABS
        # =================================================================
        with gr.Tabs() as tabs:
            # =============================================================
            # TAB 1: Evaluation Tasks
            # =============================================================
            with gr.TabItem("📊 Evaluation Benchmarks", id="eval"):
                check_fn_state = gr.State(value=None)

                # Task Selection Row
                with gr.Group():
                    with gr.Row(equal_height=True):
                        benchmark_dropdown = gr.Dropdown(
                            choices=benchmark_names,
                            value=benchmark_names[0] if benchmark_names else None,
                            label="📋 Select Benchmark",
                            scale=4,
                        )
                        generate_btn = gr.Button("🎲 New Instance", variant="secondary", scale=1, min_width=140)
                        run_eval_btn = gr.Button("▶️ Run Comparison", variant="primary", scale=1, min_width=140)

                # Task Description Card
                with gr.Group():
                    task_description = gr.Markdown("*Select a task and click 'New Instance' to generate*")

                # Status & Results
                eval_status_html = gr.HTML(create_status_box("Ready", "Select a task and click Run", "📋", "#64748b", False))

                with gr.Group():
                    eval_results_output = gr.Markdown("")

                # Charts Row
                with gr.Row(equal_height=True):
                    with gr.Column():
                        eval_tokens_plot = gr.Plot(label="📊 Token Usage", show_label=True)
                    with gr.Column():
                        eval_time_plot = gr.Plot(label="⏱️ Execution Time", show_label=True)

                # Expandable Sections
                with gr.Accordion("📜 Execution Traces", open=False):
                    eval_traces_output = gr.Markdown("*Run a benchmark to see detailed execution traces.*")

                with gr.Accordion("📝 Task Details", open=False):
                    with gr.Row():
                        with gr.Column(scale=2):
                            task_text = gr.Textbox(label="Task Prompt", lines=3, interactive=False)
                        with gr.Column(scale=1):
                            expected_text = gr.Textbox(label="Expected Answer", interactive=False)
                    context_preview = gr.Textbox(label="Context Preview", lines=5, interactive=False)

                full_context = gr.State("")

                def on_generate(benchmark_name):
                    desc, task, context, expected, check_fn = generate_task_instance(benchmark_name)
                    preview = context[:500] + "..." if len(context) > 500 else context
                    return (
                        f"**{benchmark_name}**\n\n{desc}\n\n*Context: {len(context):,} chars*",
                        task, expected, preview, context, check_fn,
                    )

                generate_btn.click(fn=on_generate, inputs=[benchmark_dropdown], outputs=[task_description, task_text, expected_text, context_preview, full_context, check_fn_state])
                benchmark_dropdown.change(fn=on_generate, inputs=[benchmark_dropdown], outputs=[task_description, task_text, expected_text, context_preview, full_context, check_fn_state])

                def run_eval_task(task, context, model, run_vanilla, run_rlm, run_official, benchmark_name, check_fn):
                    if not task:
                        yield create_status_box("No Task", "Generate a task first", "📋", "#666", False), "", None, None, ""
                        return

                    results_list = []
                    traces = ""

                    methods = []
                    if run_vanilla:
                        methods.append(("Vanilla", "🔵", "#4dabf7", run_vanilla_llm))
                    if run_rlm:
                        methods.append(("minRLM", "🟠", "#ff922b", run_our_rlm))
                    if run_official:
                        methods.append(("Official", "🟢", "#51cf66", run_official_rlm))

                    if not methods:
                        yield create_status_box("No Methods", "Select at least one method", "⚠️", "#fcc419", False), "", None, None, ""
                        return

                    total_elapsed = 0.0
                    for i, (name, icon, color, run_fn) in enumerate(methods):
                        yield create_status_box(f"Running {name}...", f"Step {i+1}/{len(methods)} · {len(context):,} chars", icon, color, True), "", *build_charts(results_list), traces

                        step_start = time.time()
                        r = run_fn(task, context, model, check_fn)
                        step_elapsed = time.time() - step_start
                        total_elapsed += step_elapsed

                        results_list.append((name, r))
                        traces += r.trace + "\n---\n\n"

                    # Final output
                    output = f"**{benchmark_name}** · {len(context):,} chars · {model}\n\n"
                    output += "| Method | Result | Input Tokens | Output Tokens | Total Tokens | Cost | Time | Iters |\n"
                    output += "|--------|--------|--------------|---------------|--------------|------|------|-------|\n"
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
                                output += f"\n**{best[0]}** used {(1-best[1]/worst[1])*100:.0f}% fewer tokens."
                        
                        if costs:
                            best_cost, worst_cost = min(costs, key=lambda x: x[1]), max(costs, key=lambda x: x[1])
                            if best_cost[1] < worst_cost[1]:
                                savings = (1 - best_cost[1] / worst_cost[1]) * 100
                                output += f"\n**{best_cost[0]}** is {savings:.0f}% cheaper (${best_cost[1]:.6f} vs ${worst_cost[1]:.6f})."

                    all_correct = all(r.correct for _, r in results_list)
                    final_status = create_status_box("✓ Complete" if all_correct else "Complete", f"⏱️ {total_elapsed:.1f}s", "🎉" if all_correct else "⚠️", "#51cf66" if all_correct else "#fcc419", False)

                    yield final_status, output, *build_charts(results_list), traces

                run_eval_btn.click(
                    fn=run_eval_task,
                    inputs=[task_text, full_context, model_dropdown, vanilla_checkbox, rlm_checkbox, official_checkbox, benchmark_dropdown, check_fn_state],
                    outputs=[eval_status_html, eval_results_output, eval_tokens_plot, eval_time_plot, eval_traces_output],
                )

            # =============================================================
            # TAB 2: Custom Task
            # =============================================================
            with gr.TabItem("✏️ Custom Task", id="custom"):
                gr.HTML("""
                <div style="padding: 12px 0 16px 0; border-bottom: 1px solid rgba(148,163,184,0.1); margin-bottom: 16px;">
                    <h3 style="margin: 0; color: #e2e8f0; font-weight: 600;">🧪 Custom Experiment</h3>
                    <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">Test your own prompts with optional context data</p>
                </div>
                """)

                with gr.Group():
                    custom_task_input = gr.Textbox(
                        label="💬 Task / Prompt",
                        placeholder="e.g., Calculate 2^1000 or Find all employees in the Engineering department",
                        lines=3,
                    )
                    custom_context_input = gr.Textbox(
                        label="📄 Context (optional)",
                        placeholder="Paste your document, JSON, code, or any text here. Leave empty for tasks without context.",
                        lines=8,
                    )
                    run_custom_btn = gr.Button("▶️ Run Comparison", variant="primary", size="lg")

                custom_status_html = gr.HTML(create_status_box("Ready", "Enter a task and click Run", "✏️", "#64748b", False))

                with gr.Group():
                    custom_results_output = gr.Markdown("")

                with gr.Row(equal_height=True):
                    with gr.Column():
                        custom_tokens_plot = gr.Plot(label="📊 Token Usage", show_label=True)
                    with gr.Column():
                        custom_time_plot = gr.Plot(label="⏱️ Execution Time", show_label=True)

                with gr.Accordion("📜 Execution Traces", open=False):
                    custom_traces_output = gr.Markdown("*Run a task to see detailed execution traces.*")

                with gr.Accordion("💬 Responses", open=True):
                    custom_responses = gr.Markdown("*Responses will appear here after running.*")

                def run_custom_task(task, context, model, run_vanilla, run_rlm, run_official):
                    if not task.strip():
                        yield create_status_box("No Task", "Enter a task prompt", "✏️", "#666", False), "", None, None, "", ""
                        return

                    results_list = []
                    traces = ""

                    methods = []
                    if run_vanilla:
                        methods.append(("Vanilla", "🔵", "#4dabf7", run_vanilla_llm))
                    if run_rlm:
                        methods.append(("minRLM", "🟠", "#ff922b", run_our_rlm))
                    if run_official:
                        methods.append(("Official", "🟢", "#51cf66", run_official_rlm))

                    if not methods:
                        yield create_status_box("No Methods", "Select at least one method", "⚠️", "#fcc419", False), "", None, None, "", ""
                        return

                    context_info = f"{len(context):,} chars" if context else "no context"
                    total_elapsed = 0.0

                    for i, (name, icon, color, run_fn) in enumerate(methods):
                        yield create_status_box(f"Running {name}...", f"Step {i+1}/{len(methods)} · {context_info}", icon, color, True), "", *build_charts(results_list), traces, ""

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
                                output += f"\n**{best[0]}** used {(1-best[1]/worst[1])*100:.0f}% fewer tokens."
                        
                        if costs:
                            best_cost, worst_cost = min(costs, key=lambda x: x[1]), max(costs, key=lambda x: x[1])
                            if best_cost[1] < worst_cost[1]:
                                savings = (1 - best_cost[1] / worst_cost[1]) * 100
                                output += f"\n**{best_cost[0]}** is {savings:.0f}% cheaper (${best_cost[1]:.6f} vs ${worst_cost[1]:.6f})."

                    # Build responses display
                    responses_md = "### Responses\n\n"
                    for name, r in results_list:
                        responses_md += f"**{name}:**\n```\n{r.response[:2000]}{'...' if len(r.response) > 2000 else ''}\n```\n\n"

                    final_status = create_status_box("✓ Complete", f"⏱️ {total_elapsed:.1f}s", "🎉", "#51cf66", False)

                    yield final_status, output, *build_charts(results_list), traces, responses_md

                run_custom_btn.click(
                    fn=run_custom_task,
                    inputs=[custom_task_input, custom_context_input, model_dropdown, vanilla_checkbox, rlm_checkbox, official_checkbox],
                    outputs=[custom_status_html, custom_results_output, custom_tokens_plot, custom_time_plot, custom_traces_output, custom_responses],
                )

        # Initial load for eval tab
        demo.load(fn=on_generate, inputs=[benchmark_dropdown], outputs=[task_description, task_text, expected_text, context_preview, full_context, check_fn_state])

        # Footer
        gr.HTML("""
        <div style="
            text-align: center;
            padding: 24px 0;
            margin-top: 32px;
            border-top: 1px solid rgba(148,163,184,0.1);
            color: #64748b;
            font-size: 0.85rem;
        ">
            <p style="margin: 0;">
                Built with 💜 using <a href="https://gradio.app" target="_blank" style="color: #818cf8;">Gradio</a> · 
                <a href="https://arxiv.org/abs/2512.24601" target="_blank" style="color: #818cf8;">Paper</a> · 
                <a href="https://github.com/avilum/minrlm" target="_blank" style="color: #818cf8;">GitHub</a>
            </p>
        </div>
        """)

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("🔬 RLM Visualizer")
    print("=" * 60)
    print(f"📊 Available eval tasks: {len(BENCHMARKS)}")
    print(f"🏷️  Task types: {sorted(set(c['task'] for c in BENCHMARKS.values()))}")
    print("=" * 60)
    build_app().launch()

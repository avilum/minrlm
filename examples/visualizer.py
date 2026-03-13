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
import queue
import subprocess
import sys
import tempfile
import threading
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
from minrlm import RLM, RLMBase

# =============================================================================
# Build Benchmark Options from Eval Tasks
# =============================================================================


def _display_name(task_name: str) -> str:
    """Convert a task registry key like 'official_gpqa_diamond' into 'GPQA DIAMOND'."""
    return task_name.upper().replace("OFFICIAL_", "").replace("_", " ")


def get_benchmark_options() -> dict[str, dict]:
    """Build benchmark dropdown options from the eval task registry.

    Every registered task appears automatically.  Parameterized size/position
    variants are added on top for tasks that benefit from them.
    """
    options: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # 1) Auto-discover ALL registered tasks
    # ------------------------------------------------------------------
    for task_name, task_cls in sorted(TASK_REGISTRY.items()):
        display = _display_name(task_name)
        options[display] = {
            "task": task_name,
            "description": getattr(task_cls, "description", task_name),
        }

    # ------------------------------------------------------------------
    # 2) Parameterized variants (scaling sizes, context sizes, positions)
    #    These add extra dropdown entries for tasks that support them.
    # ------------------------------------------------------------------

    # SNIAH scaling: 8K → 10M
    if "official_sniah" in TASK_REGISTRY:
        for size in [
            8192,
            16384,
            32768,
            65536,
            131072,
            262144,
            524288,
            1048576,
            10485760,
        ]:
            label = (
                f"{size // 1024}K"
                if size < 1024 * 1024
                else f"{size // (1024 * 1024)}M"
            )
            options[f"SCALING - {label}"] = {
                "task": "official_sniah",
                "context_size": size,
                "description": f"S-NIAH at {size:,} chars",
            }

    # JSON tasks at various sizes
    for json_task in ("json_extraction", "json_aggregation"):
        if json_task in TASK_REGISTRY:
            for label, val in {"50K": 50000, "100K": 100000, "200K": 200000}.items():
                display = json_task.upper().replace("_", " ")
                options[f"{display} - {label}"] = {
                    "task": json_task,
                    "context_size": val,
                    "description": f"{display} ({val:,} chars)",
                }

    # Long context with needle positions
    if "long_context" in TASK_REGISTRY:
        for label, val in {
            "128K": 131072,
            "256K": 262144,
            "512K": 524288,
            "1M": 1048576,
            "10M": 10485760,
        }.items():
            for pos in ("start", "middle", "end"):
                options[f"LONG CONTEXT {label} ({pos})"] = {
                    "task": "long_context",
                    "context_size": val,
                    "position": pos,
                    "description": f"Needle at {pos} of {val:,} chars",
                }

    # CodeQA at various sizes
    if "official_codeqa" in TASK_REGISTRY:
        for label, val in {
            "100K": 100000,
            "500K": 500000,
            "1M": 1000000,
            "2M": 2000000,
            "10M": 10000000,
        }.items():
            options[f"CODEQA - {label}"] = {
                "task": "official_codeqa",
                "context_size": val,
                "description": f"Code repository understanding ({val:,} chars)",
            }

    # BrowseComp at various sizes
    if "official_browsecomp" in TASK_REGISTRY:
        for label, val in {
            "200K": 200000,
            "1M": 1000000,
            "6M": 6000000,
            "10M": 10000000,
            "11M": 11000000,
        }.items():
            options[f"BROWSECOMP - {label}"] = {
                "task": "official_browsecomp",
                "context_size": val,
                "description": f"Multi-hop research ({val:,} chars)",
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
        "gpt-5-mini",
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
            elif m_lower == "gpt-4" or (
                m_lower.startswith("gpt-4-")
                and "base" not in m_lower
                and "vision" not in m_lower
            ):
                chat_models.append(model_id)
            # Include gpt-3.5-turbo variants
            elif m_lower.startswith("gpt-3.5-turbo"):
                chat_models.append(model_id)
            # Include gpt-5 chat/instruct variants but not bare base models.
            # Models like "gpt-5.2" (decimal version, no chat suffix) only support
            # v1/completions, not v1/chat/completions.
            elif m_lower.startswith("gpt-5"):
                import re as _re

                _chat_suffixes = ("-chat", "-mini", "-nano", "-turbo", "-instruct")
                has_chat_suffix = any(s in m_lower for s in _chat_suffixes)
                is_explicit_base = m_lower.endswith("-base")
                # Decimal-versioned names without a chat suffix are base/completion models
                has_decimal_version = bool(_re.search(r"gpt-5\.\d", m_lower))
                if not is_explicit_base and (
                    has_chat_suffix or not has_decimal_version
                ):
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


def run_vanilla_llm(
    task: str, context: str, model: str, check_fn: callable = None
) -> RunResult:
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


def run_our_rlm(
    task: str, context: str, model: str, check_fn: callable = None
) -> RunResult:
    """Run with minRLM."""
    trace_parts = ["## minRLM\n\n"]
    if context:
        trace_parts.append(f"Processing {len(context):,} chars...\n\n")

    def on_step(event: str, data: dict):
        if event == "thinking":
            trace_parts.append(f"### Iteration {data['iteration']}\n\n")
        elif event == "llm_response":
            has_code = data.get("has_code", False)
            trace_parts.append(
                f"**LLM Response** ({len(data.get('response', ''))} chars):\n"
            )
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
                trace_parts.append(f"❌ **Error:**\n```\n{data['error']}\n```\n\n")
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
        rlm = RLMBase(model=model, max_iterations=10, on_step=on_step)
        if context:
            result = rlm.completion(task=task, context=context)
        else:
            result = rlm.completion(task=task)
        elapsed = time.time() - start

        response = result.response
        correct = check_fn(response) if check_fn else True

        cost = calculate_cost(model, result.input_tokens, result.output_tokens)

        trace_parts.append(
            f"\n**Final:** `{response[:200]}{'...' if len(response) > 200 else ''}`\n"
        )
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


def run_reasoning_rlm(
    task: str, context: str, model: str, check_fn: callable = None
) -> RunResult:
    """Run with RLMReasoning (Reasoning approach)."""
    trace_parts = ["## minRLM with Reasoning\n\n"]
    if context:
        trace_parts.append(
            f"Processing {len(context):,} chars with reasoning-first approach...\n\n"
        )

    def on_step(event: str, data: dict):
        if event == "thinking":
            trace_parts.append(f"### Iteration {data['iteration']}\n\n")
        elif event == "reasoning":
            reasoning = data.get("reasoning", "")
            trace_parts.append(
                f"**Reasoning:**\n> {reasoning[:500]}{'...' if len(reasoning) > 500 else ''}\n\n"
            )
        elif event == "llm_response":
            has_code = data.get("has_code", False)
            trace_parts.append(
                f"**LLM Response** ({len(data.get('response', ''))} chars):\n"
            )
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
        rlm = RLM(model=model, max_iterations=10, on_step=on_step)
        if context:
            result = rlm.completion(task=task, context=context)
        else:
            result = rlm.completion(task=task)
        elapsed = time.time() - start

        response = result.response
        correct = check_fn(response) if check_fn else True

        cost = calculate_cost(model, result.input_tokens, result.output_tokens)

        if result.reasoning:
            trace_parts.append(
                f"\n**Reasoning Summary:** {result.reasoning[:300]}{'...' if len(result.reasoning) > 300 else ''}\n"
            )
        trace_parts.append(
            f"\n**Final:** `{response[:200]}{'...' if len(response) > 200 else ''}`\n"
        )
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


def run_official_rlm(
    task: str, context: str, model: str, check_fn: callable = None
) -> RunResult:
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
    from rlm.logger import RLMLogger

    with open("{context_file}") as f:
        context = f.read()

    with open("{task_file}") as f:
        task = f.read()

    start = time.time()

    logger = RLMLogger()  # in-memory trajectory capture

    rlm = RLM(
        backend="openai",
        backend_kwargs={{"model_name": "{model}"}},
        environment="local",
        max_iterations=10,
        verbose=False,
        logger=logger,
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

    # Extract slim trajectory (skip prompt — it's the full context, too large)
    slim_iterations = []
    trajectory = logger.get_trajectory() or {{}}
    for it in trajectory.get("iterations", []):
        slim_blocks = []
        for block in it.get("code_blocks", []):
            res = block.get("result", {{}})
            slim_blocks.append({{
                "code": block.get("code", ""),
                "stdout": (res.get("stdout") or "")[:3000],
                "stderr": (res.get("stderr") or "")[:500],
                "final_answer": res.get("final_answer"),
            }})
        slim_iterations.append({{
            "iteration": it.get("iteration"),
            "response": (it.get("response") or "")[:300],
            "response_len": len(it.get("response") or ""),
            "code_blocks": slim_blocks,
            "final_answer": it.get("final_answer"),
            "iteration_time": it.get("iteration_time"),
        }})

    print("<<<RESULT>>>")
    print(json.dumps({{
        "response": response,
        "elapsed": elapsed,
        "total_tokens": total_input + total_output,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "iterations": iterations,
        "trajectory": slim_iterations,
    }}))
except Exception as e:
    import traceback
    print("<<<ERROR>>>")
    print(str(e))
    traceback.print_exc()
"""

    try:
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
            trajectory = data.get("trajectory", [])

            correct = check_fn(resp_text) if check_fn else True
            cost = calculate_cost(model, input_tokens, output_tokens)

            # Render per-iteration trajectory
            for it in trajectory:
                it_num = it.get("iteration") or trajectory.index(it) + 1
                it_time = it.get("iteration_time")
                time_str = f" ({it_time:.1f}s)" if it_time else ""
                trace += f"### Iteration {it_num}{time_str}\n\n"

                resp_len = it.get("response_len", 0)
                resp_preview = it.get("response", "")
                has_code = bool(it.get("code_blocks"))
                trace += f"**LLM Response** ({resp_len:,} chars):\n"
                if has_code:
                    trace += "Contains code block\n\n"
                else:
                    trace += "No code block found\n\n"
                    if resp_preview:
                        trace += f"```\n{resp_preview}{'...' if resp_len > 300 else ''}\n```\n\n"

                for block in it.get("code_blocks", []):
                    code = block.get("code", "")
                    stdout = block.get("stdout", "")
                    stderr = block.get("stderr", "")
                    final_answer = block.get("final_answer")
                    trace += f"**Executing Code:**\n```python\n{code}\n```\n\n"
                    if stderr:
                        trace += f"❌ **Error:**\n```\n{stderr}\n```\n\n"
                    if stdout:
                        trace += f"**stdout:**\n```\n{stdout}\n```\n\n"
                    if final_answer is not None:
                        trace += f"**FINAL():** `{final_answer}`\n\n"

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
            response="",
            correct=False,
            tokens=0,
            input_tokens=0,
            output_tokens=0,
            time_seconds=elapsed,
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
# Real-time streaming generators
# Each yields (partial_trace: str, result: RunResult | None).
# None result = still in progress; non-None result = run finished.
# =============================================================================


def _rlm_trace_event(event: str, data: dict, include_reasoning: bool = False) -> str:
    """Convert an on_step event into a markdown trace fragment."""
    if event == "thinking":
        return f"### Iteration {data['iteration']}\n\n"
    if event == "reasoning" and include_reasoning:
        r = data.get("reasoning", "")
        return f"**Reasoning:**\n> {r[:500]}{'...' if len(r) > 500 else ''}\n\n"
    if event == "llm_response":
        rlen = len(data.get("response", ""))
        frag = f"**LLM Response** ({rlen:,} chars):\n"
        if data.get("has_code"):
            frag += "Contains code block\n\n"
        else:
            preview = data.get("response", "")[:300]
            frag += "No code block found\n\n"
            if preview:
                frag += f"```\n{preview}{'...' if rlen > 300 else ''}\n```\n\n"
        return frag
    if event == "executing":
        return f"**Executing Code:**\n```python\n{data.get('code', '')}\n```\n\n"
    if event == "executed":
        if data.get("error"):
            return f"❌ **Error:**\n```\n{data['error']}\n```\n\n"
        frag = ""
        stdout = data.get("stdout", "")
        output = data.get("output")
        if stdout:
            frag += f"**stdout:**\n```\n{stdout[:2000]}{'...' if len(stdout) > 2000 else ''}\n```\n\n"
        if output is not None:
            frag += f"**FINAL():** `{output}`\n\n"
        elif not stdout:
            frag += "*(no output)*\n\n"
        return frag
    return ""


def _stream_rlm_runner(
    label, task, context, model, check_fn, rlm_factory, include_reasoning=False
):
    """
    Generic streaming generator for on_step-based RLM runners (minRLM, minRLM-reasoning).

    Runs rlm.completion() in a background thread. on_step events push the current
    partial trace to a queue; the generator yields each snapshot to Gradio immediately.
    """
    q = queue.Queue()
    result_holder = [None]
    exc_holder = [None]
    trace_parts = [f"## {label}\n\n"]
    if context:
        trace_parts.append(f"Processing {len(context):,} chars...\n\n")
    start = time.time()

    def on_step(event: str, data: dict) -> None:
        frag = _rlm_trace_event(event, data, include_reasoning)
        if frag:
            trace_parts.append(frag)
            q.put(("step", "".join(trace_parts)))

    def run() -> None:
        try:
            rlm = rlm_factory(on_step)
            res = (
                rlm.completion(task=task, context=context)
                if context
                else rlm.completion(task=task)
            )
            result_holder[0] = res
        except Exception as e:
            exc_holder[0] = e
            trace_parts.append(f"\n**Error:** {e}\n")
        finally:
            q.put(("done", None))

    threading.Thread(target=run, daemon=True).start()

    while True:
        kind, payload = q.get()
        if kind == "done":
            break
        yield payload, None  # intermediate: partial trace, no result yet

    elapsed = time.time() - start

    if exc_holder[0] or result_holder[0] is None:
        yield "".join(trace_parts), RunResult(
            response="",
            correct=False,
            tokens=0,
            input_tokens=0,
            output_tokens=0,
            time_seconds=elapsed,
            trace="".join(trace_parts),
        )
        return

    result = result_holder[0]
    response = result.response
    correct = check_fn(response) if check_fn else True
    cost = calculate_cost(model, result.input_tokens, result.output_tokens)

    reasoning = getattr(result, "reasoning", "")
    if reasoning:
        trace_parts.append(
            f"\n**Reasoning Summary:** {reasoning[:300]}{'...' if len(reasoning) > 300 else ''}\n"
        )
    trace_parts.append(
        f"\n**Final:** `{response[:200]}{'...' if len(response) > 200 else ''}`\n"
    )
    trace_parts.append(
        f"**Tokens:** {result.input_tokens:,} in + {result.output_tokens:,} out"
        f" = {result.total_tokens:,} total"
    )
    if cost is not None:
        trace_parts.append(f" | **Cost:** ${cost:.6f}")
    trace_parts.append(f" | **Time:** {elapsed:.1f}s\n")
    if check_fn:
        trace_parts.append(f"{'✅ Correct' if correct else '❌ Incorrect'}\n\n")

    yield "".join(trace_parts), RunResult(
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


def stream_vanilla_llm(task: str, context: str, model: str, check_fn=None):
    """Streaming wrapper for vanilla LLM (single call — yield a status, then the result)."""
    trace = "## Vanilla LLM\n\n"
    trace += f"Sending {'full context (' + str(len(context)) + ' chars)' if context else 'prompt'} in one request.\n\n"
    trace += "*Calling API…*\n\n"
    yield trace, None  # show "calling API" immediately

    result = run_vanilla_llm(task, context, model, check_fn)
    yield result.trace, result


def stream_our_rlm(task: str, context: str, model: str, check_fn=None):
    """Streaming generator for minRLM."""
    yield from _stream_rlm_runner(
        label="minRLM",
        task=task,
        context=context,
        model=model,
        check_fn=check_fn,
        rlm_factory=lambda on_step: RLMBase(
            model=model, max_iterations=10, on_step=on_step
        ),
        include_reasoning=False,
    )


def stream_reasoning_rlm(task: str, context: str, model: str, check_fn=None):
    """Streaming generator for minRLM with Reasoning."""
    yield from _stream_rlm_runner(
        label="minRLM with Reasoning",
        task=task,
        context=context,
        model=model,
        check_fn=check_fn,
        rlm_factory=lambda on_step: RLM(
            model=model, max_iterations=10, on_step=on_step
        ),
        include_reasoning=True,
    )


def stream_official_rlm(task: str, context: str, model: str, check_fn=None):
    """Streaming generator for Official RLM — reads <<<ITER>>> markers from subprocess stdout."""
    trace = "## Official RLM\n\n"
    trace += (
        f"Processing {len(context):,} chars via "
        f"[github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm).\n\n"
        if context
        else "Running via [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm).\n\n"
    )
    start = time.time()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(context or "")
        context_file = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(task)
        task_file = f.name

    script = f"""
import json, sys, time

try:
    from rlm import RLM
    from rlm.logger import RLMLogger

    with open("{context_file}") as f:
        context = f.read()
    with open("{task_file}") as f:
        task = f.read()

    model = "{model}"
    start = time.time()
    logger = RLMLogger()

    # Monkey-patch to stream iteration events to stdout
    _orig_log = logger.log
    def _streaming_log(it):
        _orig_log(it)
        blocks = []
        for b in it.code_blocks:
            r = b.result
            blocks.append({{
                "code": b.code[:3000],
                "stdout": (r.stdout or "")[:2000],
                "stderr": (r.stderr or "")[:500],
                "final_answer": r.final_answer,
            }})
        entry = {{
            "iter": logger.iteration_count,
            "response_preview": (it.response or "")[:300],
            "response_len": len(it.response or ""),
            "has_code": bool(it.code_blocks),
            "code_blocks": blocks,
            "final_answer": it.final_answer,
            "iteration_time": it.iteration_time,
        }}
        print("<<<ITER>>> " + json.dumps(entry), flush=True)
    logger.log = _streaming_log

    rlm = RLM(
        backend="openai",
        backend_kwargs={{"model_name": model}},
        environment="local",
        max_iterations=10,
        verbose=False,
        logger=logger,
    )

    result = rlm.completion(prompt=context, root_prompt=task) if context.strip() else rlm.completion(prompt=task)
    elapsed = time.time() - start

    total_input = total_output = 0
    if result.usage_summary and result.usage_summary.model_usage_summaries:
        for usage in result.usage_summary.model_usage_summaries.values():
            total_input += usage.total_input_tokens
            total_output += usage.total_output_tokens

    response = result.response or ""
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1]

    iterations = getattr(result, 'num_iterations', None) or getattr(result, 'iterations', None) or 1

    slim_iters = []
    traj = logger.get_trajectory() or {{}}
    for it in traj.get("iterations", []):
        slim_blocks = []
        for block in it.get("code_blocks", []):
            res = block.get("result", {{}})
            slim_blocks.append({{
                "code": block.get("code", ""),
                "stdout": (res.get("stdout") or "")[:3000],
                "stderr": (res.get("stderr") or "")[:500],
                "final_answer": res.get("final_answer"),
            }})
        slim_iters.append({{
            "iteration": it.get("iteration"),
            "response": (it.get("response") or "")[:300],
            "response_len": len(it.get("response") or ""),
            "code_blocks": slim_blocks,
            "final_answer": it.get("final_answer"),
            "iteration_time": it.get("iteration_time"),
        }})

    print("<<<RESULT>>>")
    print(json.dumps({{
        "response": response, "elapsed": elapsed,
        "total_tokens": total_input + total_output,
        "input_tokens": total_input, "output_tokens": total_output,
        "iterations": iterations, "trajectory": slim_iters,
    }}))
except Exception as e:
    import traceback
    print("<<<ERROR>>>")
    print(str(e))
    traceback.print_exc()
"""

    try:
        proc = subprocess.Popen(
            [
                "uv",
                "run",
                "--with",
                "git+https://github.com/alexzhang13/rlm",
                "python",
                "-c",
                script,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env={
                **os.environ,
                "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
                "PYTHONUNBUFFERED": "1",
            },
        )

        result_json = None
        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if line.startswith("<<<ITER>>>"):
                try:
                    it = json.loads(line[len("<<<ITER>>> ") :])
                    it_num = it.get("iter", "?")
                    it_time = it.get("iteration_time")
                    time_str = f" ({it_time:.1f}s)" if it_time else ""
                    trace += f"### Iteration {it_num}{time_str}\n\n"
                    trace += (
                        f"**LLM Response** ({it.get('response_len', 0):,} chars):\n"
                    )
                    if it.get("has_code"):
                        trace += "Contains code block\n\n"
                        for block in it.get("code_blocks", []):
                            trace += f"**Executing Code:**\n```python\n{block['code']}\n```\n\n"
                            if block.get("stderr"):
                                trace += (
                                    f"❌ **Error:**\n```\n{block['stderr']}\n```\n\n"
                                )
                            if block.get("stdout"):
                                trace += f"**stdout:**\n```\n{block['stdout']}\n```\n\n"
                            if block.get("final_answer") is not None:
                                trace += f"**FINAL():** `{block['final_answer']}`\n\n"
                    else:
                        trace += "No code block found\n\n"
                        preview = it.get("response_preview", "")
                        if preview:
                            trace += f"```\n{preview}...\n```\n\n"
                    yield trace, None
                except Exception:
                    pass
            elif "<<<RESULT>>>" in line:
                next_line = proc.stdout.readline()
                try:
                    result_json = json.loads(next_line.strip())
                except Exception:
                    pass
                break
            elif line == "<<<ERROR>>>":
                err_msg = proc.stdout.readline().strip()
                trace += f"**Error:** {err_msg}\n"
                yield trace, None

        proc.wait()
        elapsed = time.time() - start

        if result_json:
            resp_text = result_json.get("response", "")
            input_tokens = result_json.get("input_tokens", 0)
            output_tokens = result_json.get("output_tokens", 0)
            total_tokens = result_json.get("total_tokens", 0)
            iterations = result_json.get("iterations", 1)
            correct = check_fn(resp_text) if check_fn else True
            cost = calculate_cost(model, input_tokens, output_tokens)

            trace += f"**Tokens:** {input_tokens:,} in + {output_tokens:,} out = {total_tokens:,} total"
            if cost is not None:
                trace += f" | **Cost:** ${cost:.6f}"
            trace += "\n\n"
            trace += f"**Response:** `{resp_text[:200]}{'...' if len(resp_text) > 200 else ''}`\n\n"
            if check_fn:
                trace += f"{'✅ Correct' if correct else '❌ Incorrect'}\n\n"

            yield trace, RunResult(
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
        else:
            trace += "**Error:** No result received from subprocess.\n"
            yield trace, RunResult(
                response="",
                correct=False,
                tokens=0,
                input_tokens=0,
                output_tokens=0,
                time_seconds=elapsed,
                trace=trace,
            )

    except Exception as e:
        trace += f"**Error:** {e}\n"
        yield trace, RunResult(
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
            except Exception:
                pass


def _consume_stream_in_thread(stream_gen, method_name, update_queue):
    """Consume a streaming generator in a background thread, pushing updates to a shared queue."""
    try:
        final_result = None
        for partial_trace, result in stream_gen:
            if result is None:
                update_queue.put(("trace", method_name, partial_trace))
            else:
                final_result = result
        update_queue.put(("done", method_name, final_result))
    except Exception as e:
        update_queue.put(("error", method_name, str(e)))


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
    title: str,
    subtitle: str = "",
    icon: str = "⏳",
    color: str = "#818cf8",
    pulse: bool = True,
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

    data = [
        {"Method": name, "Tokens": float(r.tokens), "Time": float(r.time_seconds)}
        for name, r in results_list
    ]
    df = pd.DataFrame(data)

    colors = {
        "Vanilla": "#60a5fa",
        "minRLM": "#f97316",
        "minRLM with Reasoning": "#c084fc",
        "Official": "#22c55e",
    }
    color_map = {m: colors.get(m, "#94a3b8") for m in df["Method"]}

    layout_common = {
        "showlegend": False,
        "height": 280,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(24,24,27,0.6)",
        "font": {
            "family": "DM Sans, system-ui, sans-serif",
            "color": "#a1a1aa",
            "size": 12,
        },
        "title_font": {"size": 14, "color": "#fafafa"},
        "xaxis": {
            "gridcolor": "rgba(255,255,255,0.06)",
            "linecolor": "rgba(255,255,255,0.08)",
            "tickfont": {"color": "#71717a"},
        },
        "yaxis": {
            "gridcolor": "rgba(255,255,255,0.06)",
            "linecolor": "rgba(255,255,255,0.08)",
            "tickfont": {"color": "#71717a"},
        },
        "margin": {"t": 44, "b": 36, "l": 52, "r": 16},
        "bargap": 0.36,
    }

    tokens_fig = px.bar(
        df, x="Method", y="Tokens", color="Method", color_discrete_map=color_map
    )
    tokens_fig.update_layout(**layout_common, title="Token usage", yaxis_title="Tokens")
    tokens_fig.update_traces(marker_line_width=0, opacity=0.92)

    time_fig = px.bar(
        df, x="Method", y="Time", color="Method", color_discrete_map=color_map
    )
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
            # Use gpt-5-mini as default (it's reliable and cost-effective)
            # Fallback to first model in list if gpt-5-mini not available
            default_model = (
                "gpt-5-mini"
                if "gpt-5-mini" in initial_models
                else (initial_models[0] if initial_models else "gpt-5-mini")
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
                    reasoning_checkbox = gr.Checkbox(
                        label="minRLM with Reasoning", value=True, scale=1
                    )
                    official_checkbox = gr.Checkbox(
                        label="Official", value=True, scale=1
                    )

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
                        generate_btn = gr.Button(
                            "New instance", variant="secondary", scale=1, min_width=120
                        )
                        run_eval_btn = gr.Button(
                            "Run comparison", variant="primary", scale=1, min_width=140
                        )

                task_description = gr.Markdown(
                    "*Select a benchmark and click New instance*"
                )

                # Status & Results
                eval_status_html = gr.HTML(
                    create_status_box(
                        "Ready", "Select a task and run", "○", "#71717a", False
                    )
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
                            task_text = gr.Textbox(
                                label="Task", lines=3, interactive=False
                            )
                        with gr.Column(scale=1):
                            expected_text = gr.Textbox(
                                label="Expected", interactive=False
                            )
                    context_preview = gr.Textbox(
                        label="Context preview", lines=5, interactive=False
                    )

                full_context = gr.State("")

                def on_generate(benchmark_name):
                    desc, task, context, expected, check_fn = generate_task_instance(
                        benchmark_name
                    )
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
                    outputs=[
                        task_description,
                        task_text,
                        expected_text,
                        context_preview,
                        full_context,
                        check_fn_state,
                    ],
                )
                benchmark_dropdown.change(
                    fn=on_generate,
                    inputs=[benchmark_dropdown],
                    outputs=[
                        task_description,
                        task_text,
                        expected_text,
                        context_preview,
                        full_context,
                        check_fn_state,
                    ],
                )

                def run_eval_task(
                    task,
                    context,
                    model,
                    run_vanilla,
                    run_rlm,
                    run_reasoning,
                    run_official,
                    benchmark_name,
                    check_fn,
                ):
                    if not task:
                        yield (
                            create_status_box(
                                "No Task", "Generate a task first", "📋", "#666", False
                            ),
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
                        methods.append(("Vanilla", "🔵", "#4dabf7", stream_vanilla_llm))
                    if run_rlm:
                        methods.append(("minRLM", "🟠", "#ff922b", stream_our_rlm))
                    if run_reasoning:
                        methods.append(
                            (
                                "minRLM with Reasoning",
                                "🟣",
                                "#c084fc",
                                stream_reasoning_rlm,
                            )
                        )
                    if run_official:
                        methods.append(
                            ("Official", "🟢", "#51cf66", stream_official_rlm)
                        )

                    if not methods:
                        yield (
                            create_status_box(
                                "No Methods",
                                "Select at least one method",
                                "⚠️",
                                "#fcc419",
                                False,
                            ),
                            "",
                            None,
                            None,
                            "",
                        )
                        return

                    method_names_ordered = [name for name, _, _, _ in methods]
                    parallel_start = time.time()
                    update_q = queue.Queue()
                    latest_traces: dict[str, str] = {}
                    results_dict: dict[str, RunResult] = {}

                    for name, _icon, _color, stream_fn in methods:
                        gen = stream_fn(task, context, model, check_fn)
                        threading.Thread(
                            target=_consume_stream_in_thread,
                            args=(gen, name, update_q),
                            daemon=True,
                        ).start()

                    done_count = 0
                    while done_count < len(methods):
                        try:
                            kind, mname, payload = update_q.get(timeout=0.5)
                        except queue.Empty:
                            continue

                        if kind == "trace":
                            latest_traces[mname] = payload
                        elif kind in ("done", "error"):
                            done_count += 1
                            if kind == "done" and payload is not None:
                                results_dict[mname] = payload
                                latest_traces[mname] = payload.trace
                            else:
                                err_trace = latest_traces.get(mname, "")
                                if kind == "error":
                                    err_trace += f"\n**Error:** {payload}\n"
                                results_dict[mname] = RunResult(
                                    response="",
                                    correct=False,
                                    tokens=0,
                                    input_tokens=0,
                                    output_tokens=0,
                                    time_seconds=time.time() - parallel_start,
                                    trace=err_trace,
                                )
                                latest_traces[mname] = err_trace

                        running = [
                            n for n in method_names_ordered if n not in results_dict
                        ]
                        finished = [
                            n for n in method_names_ordered if n in results_dict
                        ]
                        running_label = (
                            f"Running {len(running)} method{'s' if len(running) != 1 else ''} in parallel…"
                            if running
                            else "Finishing…"
                        )
                        step_status = create_status_box(
                            running_label,
                            f"{len(finished)}/{len(methods)} done · {len(context):,} chars",
                            "⋯",
                            "#818cf8",
                            bool(running),
                        )
                        combined_traces = "\n---\n\n".join(
                            latest_traces[n]
                            for n in method_names_ordered
                            if n in latest_traces
                        )
                        partial_results = [
                            (n, results_dict[n])
                            for n in method_names_ordered
                            if n in results_dict
                        ]
                        yield (
                            step_status,
                            "",
                            *build_charts(partial_results),
                            combined_traces,
                        )

                    results_list = [(n, results_dict[n]) for n in method_names_ordered]
                    total_elapsed = time.time() - parallel_start
                    traces = "\n---\n\n".join(
                        r.trace for _, r in results_list if r.trace
                    )

                    # Final output
                    output = (
                        f"**{benchmark_name}** · {len(context):,} chars · {model}\n\n"
                    )
                    output += "| Method | Result | Input Tokens | Output Tokens | Total Tokens | Cost | Time | Iters |\n"
                    output += "|--------|--------|--------------|---------------|--------------|------|------|-------|\n"
                    for name, r in results_list:
                        status = "✅" if r.correct else "❌"
                        cost_str = (
                            f"${r.cost_usd:.6f}" if r.cost_usd is not None else "N/A"
                        )
                        output += f"| {name} | {status} | {r.input_tokens:,} | {r.output_tokens:,} | {r.tokens:,} | {cost_str} | {r.time_seconds:.1f}s | {r.iterations} |\n"

                    if len(results_list) >= 2:
                        tokens = [
                            (n, r.tokens) for n, r in results_list if r.tokens > 0
                        ]
                        costs = [
                            (n, r.cost_usd)
                            for n, r in results_list
                            if r.cost_usd is not None
                        ]

                        if tokens:
                            best, worst = min(tokens, key=lambda x: x[1]), max(
                                tokens, key=lambda x: x[1]
                            )
                            if best[1] < worst[1]:
                                output += f"\n**{best[0]}** used {(1 - best[1] / worst[1]) * 100:.0f}% fewer tokens."

                        if costs:
                            best_cost, worst_cost = min(costs, key=lambda x: x[1]), max(
                                costs, key=lambda x: x[1]
                            )
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
                run_custom_btn = gr.Button(
                    "Run comparison", variant="primary", size="lg"
                )

                custom_status_html = gr.HTML(
                    create_status_box(
                        "Ready", "Enter a task and run", "○", "#71717a", False
                    )
                )

                with gr.Group():
                    custom_results_output = gr.Markdown("")

                with gr.Row(equal_height=True):
                    with gr.Column():
                        custom_tokens_plot = gr.Plot(
                            label="Token usage", show_label=True
                        )
                    with gr.Column():
                        custom_time_plot = gr.Plot(label="Time", show_label=True)

                with gr.Accordion("Execution traces", open=False):
                    custom_traces_output = gr.Markdown("*Run a task to see traces.*")

                with gr.Accordion("Responses", open=False):
                    custom_responses = gr.Markdown(
                        "*Responses appear here after running.*"
                    )

                def run_custom_task(
                    task,
                    context,
                    model,
                    run_vanilla,
                    run_rlm,
                    run_reasoning,
                    run_official,
                ):
                    if not task.strip():
                        yield (
                            create_status_box(
                                "No Task", "Enter a task prompt", "✏️", "#666", False
                            ),
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
                        methods.append(("Vanilla", "🔵", "#4dabf7", stream_vanilla_llm))
                    if run_rlm:
                        methods.append(("minRLM", "🟠", "#ff922b", stream_our_rlm))
                    if run_reasoning:
                        methods.append(
                            (
                                "minRLM with Reasoning",
                                "🟣",
                                "#c084fc",
                                stream_reasoning_rlm,
                            )
                        )
                    if run_official:
                        methods.append(
                            ("Official", "🟢", "#51cf66", stream_official_rlm)
                        )

                    if not methods:
                        yield (
                            create_status_box(
                                "No Methods",
                                "Select at least one method",
                                "⚠️",
                                "#fcc419",
                                False,
                            ),
                            "",
                            None,
                            None,
                            "",
                            "",
                        )
                        return

                    context_info = (
                        f"{len(context):,} chars" if context else "no context"
                    )
                    method_names_ordered = [name for name, _, _, _ in methods]
                    parallel_start = time.time()
                    update_q = queue.Queue()
                    latest_traces: dict[str, str] = {}
                    results_dict: dict[str, RunResult] = {}

                    for name, _icon, _color, stream_fn in methods:
                        gen = stream_fn(task, context, model, None)
                        threading.Thread(
                            target=_consume_stream_in_thread,
                            args=(gen, name, update_q),
                            daemon=True,
                        ).start()

                    done_count = 0
                    while done_count < len(methods):
                        try:
                            kind, mname, payload = update_q.get(timeout=0.5)
                        except queue.Empty:
                            continue

                        if kind == "trace":
                            latest_traces[mname] = payload
                        elif kind in ("done", "error"):
                            done_count += 1
                            if kind == "done" and payload is not None:
                                results_dict[mname] = payload
                                latest_traces[mname] = payload.trace
                            else:
                                err_trace = latest_traces.get(mname, "")
                                if kind == "error":
                                    err_trace += f"\n**Error:** {payload}\n"
                                results_dict[mname] = RunResult(
                                    response="",
                                    correct=False,
                                    tokens=0,
                                    input_tokens=0,
                                    output_tokens=0,
                                    time_seconds=time.time() - parallel_start,
                                    trace=err_trace,
                                )
                                latest_traces[mname] = err_trace

                        running = [
                            n for n in method_names_ordered if n not in results_dict
                        ]
                        finished = [
                            n for n in method_names_ordered if n in results_dict
                        ]
                        running_label = (
                            f"Running {len(running)} method{'s' if len(running) != 1 else ''} in parallel…"
                            if running
                            else "Finishing…"
                        )
                        step_status = create_status_box(
                            running_label,
                            f"{len(finished)}/{len(methods)} done · {context_info}",
                            "⋯",
                            "#818cf8",
                            bool(running),
                        )
                        combined_traces = "\n---\n\n".join(
                            latest_traces[n]
                            for n in method_names_ordered
                            if n in latest_traces
                        )
                        partial_results = [
                            (n, results_dict[n])
                            for n in method_names_ordered
                            if n in results_dict
                        ]
                        yield (
                            step_status,
                            "",
                            *build_charts(partial_results),
                            combined_traces,
                            "",
                        )

                    results_list = [(n, results_dict[n]) for n in method_names_ordered]
                    total_elapsed = time.time() - parallel_start
                    traces = "\n---\n\n".join(
                        r.trace for _, r in results_list if r.trace
                    )

                    # Final output table
                    output = f"**Custom Task** · {context_info} · {model}\n\n"
                    output += "| Method | Input Tokens | Output Tokens | Total Tokens | Cost | Time | Iters |\n"
                    output += "|--------|--------------|---------------|--------------|------|------|-------|\n"
                    for name, r in results_list:
                        cost_str = (
                            f"${r.cost_usd:.6f}" if r.cost_usd is not None else "N/A"
                        )
                        output += f"| {name} | {r.input_tokens:,} | {r.output_tokens:,} | {r.tokens:,} | {cost_str} | {r.time_seconds:.1f}s | {r.iterations} |\n"

                    if len(results_list) >= 2:
                        tokens = [
                            (n, r.tokens) for n, r in results_list if r.tokens > 0
                        ]
                        costs = [
                            (n, r.cost_usd)
                            for n, r in results_list
                            if r.cost_usd is not None
                        ]

                        if tokens:
                            best, worst = min(tokens, key=lambda x: x[1]), max(
                                tokens, key=lambda x: x[1]
                            )
                            if best[1] < worst[1]:
                                output += f"\n**{best[0]}** used {(1 - best[1] / worst[1]) * 100:.0f}% fewer tokens."

                        if costs:
                            best_cost, worst_cost = min(costs, key=lambda x: x[1]), max(
                                costs, key=lambda x: x[1]
                            )
                            if best_cost[1] < worst_cost[1]:
                                savings = (1 - best_cost[1] / worst_cost[1]) * 100
                                output += f"\n**{best_cost[0]}** is {savings:.0f}% cheaper (${best_cost[1]:.6f} vs ${worst_cost[1]:.6f})."

                    # Build responses display
                    responses_md = "### Responses\n\n"
                    for name, r in results_list:
                        responses_md += f"**{name}:**\n```\n{r.response[:2000]}{'...' if len(r.response) > 2000 else ''}\n```\n\n"

                    final_status = create_status_box(
                        "Done", f"{total_elapsed:.1f}s total", "✓", "#22c55e", False
                    )

                    yield final_status, output, *build_charts(
                        results_list
                    ), traces, responses_md

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
            outputs=[
                task_description,
                task_text,
                expected_text,
                context_preview,
                full_context,
                check_fn_state,
            ],
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

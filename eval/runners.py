"""
Runner Implementations for RLM Evaluation

Each runner wraps a different method:
- VanillaRunner: Direct LLM API calls
- OursRunner: Our minimal RLM implementation
- OfficialRunner: Official RLM implementation from the paper

To add a new runner, subclass BaseRunner and use @register_runner decorator.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

# Runner registry for extensibility
RUNNER_REGISTRY: dict[str, type] = {}


def register_runner(name: str):
    """Decorator to register a runner class."""

    def decorator(cls):
        RUNNER_REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


@dataclass
class RunResult:
    """Result of running a single task with a method."""

    response: str
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    time_seconds: float = 0.0
    iterations: int = 1
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None


class BaseRunner(ABC):
    """Base class for evaluation runners."""

    name: str = "base"
    description: str = "Base runner"

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    def run(self, task: str, context: str) -> RunResult:
        """Run the method on a task."""
        pass

    def warmup(self) -> bool:
        """Optional warmup call. Returns True if ready."""
        return True

    def cleanup(self) -> None:
        """Optional cleanup after evaluation."""
        pass


# =============================================================================
# Runner Implementations
# =============================================================================


@register_runner("vanilla")
class VanillaRunner(BaseRunner):
    """
    Vanilla LLM Runner

    Direct API call with full context - the baseline approach.
    No recursive decomposition, just send everything to the model.
    """

    description = "Direct LLM API call (no RLM)"

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        from openai import OpenAI

        self.client = OpenAI()

    def run(self, task: str, context: str) -> RunResult:
        start = time.time()

        prompt = f"""{task}

Here is the text to analyze:

{context}"""

        try:
            kwargs = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
            # Some models don't support temperature
            if "gpt-5" not in self.model.lower():
                kwargs["temperature"] = 0.7

            response = self.client.chat.completions.create(**kwargs)
            elapsed = time.time() - start

            usage = response.usage
            return RunResult(
                response=response.choices[0].message.content or "",
                total_tokens=usage.total_tokens if usage else 0,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                time_seconds=elapsed,
                iterations=1,
            )
        except Exception as e:
            return RunResult(
                response="",
                error=str(e),
                time_seconds=time.time() - start,
            )


@register_runner("ours")
class OursRunner(BaseRunner):
    """
    Our Minimal RLM Runner

    Uses our minimal RLM implementation (~400 LOC).
    Key features: sub_llm(), FINAL(), Python REPL.
    """

    description = "Our minimal RLM implementation"

    def __init__(self, model: str, max_iterations: int = 10, log_dir: str | None = None, **kwargs):
        super().__init__(model, **kwargs)
        self.max_iterations = max_iterations
        self.log_dir = log_dir

        # Import our RLM
        # Ensure our module path takes precedence
        module_path = str(Path(__file__).parent.parent)
        if module_path not in sys.path:
            sys.path.insert(0, module_path)

        from minrlm.core import RLM

        self.RLM = RLM

    def run(self, task: str, context: str) -> RunResult:
        start = time.time()

        try:
            rlm = self.RLM(
                model=self.model,
                max_iterations=self.max_iterations,
                log_dir=self.log_dir,
            )
            result = rlm.completion(task=task, context=context)
            elapsed = time.time() - start

            return RunResult(
                response=result.response,
                total_tokens=result.total_tokens,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                time_seconds=elapsed,
                iterations=result.iterations,
            )
        except Exception as e:
            return RunResult(
                response="",
                error=str(e),
                time_seconds=time.time() - start,
            )


@register_runner("official")
class OfficialRunner(BaseRunner):
    """
    Official RLM Runner

    Uses the official RLM implementation via uv run --with.
    Installs from github.com/alexzhang13/rlm on demand.
    """

    description = "Official RLM implementation from the paper"

    def __init__(self, model: str, timeout: int = 300, **kwargs):
        super().__init__(model, **kwargs)
        self.timeout = timeout

    def warmup(self) -> bool:
        """Official RLM is always available via uv --with."""
        return True

    def run(self, task: str, context: str) -> RunResult:
        start = time.time()

        # Write context and task to temp files (too large for command line)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(context)
            context_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(task)
            task_file = f.name

        # Script to run official RLM
        script = f'''
import json
import time

from rlm import RLM

with open("{context_file}") as f:
    context = f.read()

with open("{task_file}") as f:
    task = f.read()

model = "{self.model}"

start = time.time()

rlm = RLM(
    backend="openai",
    backend_kwargs={{"model_name": model}},
    environment="local",
    max_iterations=10,
    verbose=False
)

# Official API: prompt=context, root_prompt=task
result = rlm.completion(prompt=context, root_prompt=task)
elapsed = time.time() - start

# Usage comes from result.usage_summary
usage = result.usage_summary
total_input = 0
total_output = 0
if usage and usage.model_usage_summaries:
    for m_usage in usage.model_usage_summaries.values():
        total_input += m_usage.total_input_tokens
        total_output += m_usage.total_output_tokens

# Try to get iteration count
iterations = getattr(result, 'num_iterations', None) or getattr(result, 'iterations', None) or 1

response = result.response or ""
if response.startswith('"') and response.endswith('"'):
    response = response[1:-1]

print("<<<RESULT>>>")
print(json.dumps({{
    "response": response,
    "total_tokens": total_input + total_output,
    "input_tokens": total_input,
    "output_tokens": total_output,
    "time_seconds": elapsed,
    "iterations": iterations
}}))
'''

        try:
            # Use uv run --with to install official RLM from GitHub
            proc = subprocess.run(
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
                timeout=self.timeout,
                env={**os.environ, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")},
            )

            elapsed = time.time() - start

            if "<<<RESULT>>>" in proc.stdout:
                json_str = proc.stdout.split("<<<RESULT>>>")[1].strip()
                data = json.loads(json_str)

                return RunResult(
                    response=data["response"],
                    total_tokens=data["total_tokens"],
                    input_tokens=data["input_tokens"],
                    output_tokens=data["output_tokens"],
                    time_seconds=data["time_seconds"],
                    iterations=data["iterations"],
                )
            else:
                error_msg = proc.stderr[:500] if proc.stderr else proc.stdout[:500]
                return RunResult(
                    response="",
                    error=f"No result marker: {error_msg}",
                    time_seconds=elapsed,
                )

        except subprocess.TimeoutExpired:
            return RunResult(
                response="",
                error=f"Timeout after {self.timeout}s",
                time_seconds=self.timeout,
            )
        except Exception as e:
            return RunResult(
                response="",
                error=str(e),
                time_seconds=time.time() - start,
            )
        finally:
            try:
                os.unlink(context_file)
                os.unlink(task_file)
            except Exception:
                pass


# =============================================================================
# Runner Factory
# =============================================================================


def get_runner(name: str, model: str, **kwargs) -> BaseRunner:
    """Get a runner instance by name."""
    if name not in RUNNER_REGISTRY:
        available = ", ".join(RUNNER_REGISTRY.keys())
        raise ValueError(f"Unknown runner: {name}. Available: {available}")
    return RUNNER_REGISTRY[name](model=model, **kwargs)


def list_runners() -> list[str]:
    """List all registered runners."""
    return list(RUNNER_REGISTRY.keys())


def get_available_runners(model: str) -> list[str]:
    """List runners that are ready to use."""
    available = []
    for name in RUNNER_REGISTRY:
        try:
            runner = get_runner(name, model)
            if runner.warmup():
                available.append(name)
        except Exception:
            pass
    return available

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
    # Debug info (for RLM runners)
    generated_code: str | None = None
    log_file_path: str | None = None

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
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }
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


@register_runner("minrlm")
class OursRunner(BaseRunner):
    """
    minRLM Runner

    Uses our minimal RLM implementation (~400 LOC).
    Key features: sub_llm(), FINAL(), Python REPL.
    """

    description = "minRLM implementation"

    def __init__(
        self, model: str, max_iterations: int = 10, log_dir: str | None = None, use_docker: bool | None = None, **kwargs
    ):
        super().__init__(model, **kwargs)
        self.max_iterations = max_iterations
        self.log_dir = log_dir

        # Import our RLM
        # Ensure our module path takes precedence
        module_path = str(Path(__file__).parent.parent)
        if module_path not in sys.path:
            sys.path.insert(0, module_path)

        from minrlm.core import RLM
        from minrlm.docker_repl import check_docker_available

        self.RLM = RLM

        # Auto-detect Docker availability if not explicitly set
        if use_docker is None:
            self.use_docker = check_docker_available()
        else:
            self.use_docker = use_docker

    def run(self, task: str, context: str) -> RunResult:
        start = time.time()

        try:
            rlm = self.RLM(
                model=self.model,
                max_iterations=self.max_iterations,
                log_dir=self.log_dir,
                async_batch=True,
                use_docker=self.use_docker,
            )
            result = rlm.completion(task=task, context=context)
            elapsed = time.time() - start

            # Extract generated code from history (last assistant message with code)
            generated_code = None
            for msg in reversed(result.history):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if "```python" in content:
                        generated_code = content
                        break

            return RunResult(
                response=result.response,
                total_tokens=result.total_tokens,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                time_seconds=elapsed,
                iterations=result.iterations,
                generated_code=generated_code,
                log_file_path=result.log_file_path,
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

    Uses the official RLM implementation installed once at warmup into a
    dedicated venv, then reused for every sample.  Installing on every call
    via `uv run --with` causes concurrent git-fetch failures under parallel
    evaluation.
    """

    description = "Official RLM implementation from the paper"
    _RLM_REPO = "git+https://github.com/alexzhang13/rlm"

    def __init__(self, model: str, timeout: int = 300, log_dir: str | None = None, **kwargs):
        super().__init__(model, **kwargs)
        self.timeout = timeout
        self.log_dir = log_dir
        self._venv_python: str | None = None  # set by warmup()

    def warmup(self) -> bool:
        """Install rlm once into a persistent venv so run() never fetches git."""
        import shutil

        venv_dir = Path(tempfile.gettempdir()) / "rlm_official_venv"

        # Reuse an existing venv if the rlm package is already installed there.
        python_bin = venv_dir / "bin" / "python"
        if python_bin.exists():
            check = subprocess.run(
                [str(python_bin), "-c", "import rlm"],
                capture_output=True,
            )
            if check.returncode == 0:
                self._venv_python = str(python_bin)
                return True
            # Broken venv — remove and recreate
            shutil.rmtree(venv_dir, ignore_errors=True)

        print(f"[official] Installing rlm into {venv_dir} …", flush=True)
        subprocess.run(["uv", "venv", str(venv_dir)], check=True, capture_output=True)
        result = subprocess.run(
            ["uv", "pip", "install", self._RLM_REPO, "--python", str(python_bin)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[official] Install failed:\n{result.stderr}", flush=True)
            return False

        self._venv_python = str(python_bin)
        return True

    def run(self, task: str, context: str) -> RunResult:
        start = time.time()

        if not self._venv_python:
            return RunResult(
                response="",
                error="official runner not warmed up — rlm venv missing",
                time_seconds=0.0,
            )

        # Write context and task to temp files (too large for command line)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(context)
            context_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(task)
            task_file = f.name

        log_dir_escaped = json.dumps(self.log_dir) if self.log_dir else "None"

        # Script to run official RLM
        script = f"""
import json
import time

from rlm import RLM
from rlm.logger import RLMLogger

with open("{context_file}") as f:
    context = f.read()

with open("{task_file}") as f:
    task = f.read()

model = "{self.model}"
log_dir = {log_dir_escaped}

start = time.time()

logger = RLMLogger(log_dir=log_dir) if log_dir else None

rlm = RLM(
    backend="openai",
    backend_kwargs={{"model_name": model}},
    environment="local",
    max_iterations=10,
    verbose=False,
    logger=logger,
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

# Extract generated code from iterations (first code block across all iterations)
generated_code = None
if logger:
    trajectory = logger.get_trajectory()
    if trajectory:
        for it in trajectory.get("iterations", []):
            code_blocks = it.get("code_blocks", [])
            if code_blocks:
                generated_code = code_blocks[0].get("code")
                break

log_file_path = logger.log_file_path if logger else None

print("<<<RESULT>>>")
print(json.dumps({{
    "response": response,
    "total_tokens": total_input + total_output,
    "input_tokens": total_input,
    "output_tokens": total_output,
    "time_seconds": elapsed,
    "iterations": iterations,
    "generated_code": generated_code,
    "log_file_path": log_file_path,
}}))
"""

        try:
            proc = subprocess.run(
                [self._venv_python, "-c", script],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={
                    **os.environ,
                    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
                },
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
                    generated_code=data.get("generated_code"),
                    log_file_path=data.get("log_file_path"),
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


# Import reasoning runner to register it
try:
    from .runners_reasoning import OursReasoningRunner  # noqa: F401
except ImportError:
    pass  # OK if reasoning dependencies not available

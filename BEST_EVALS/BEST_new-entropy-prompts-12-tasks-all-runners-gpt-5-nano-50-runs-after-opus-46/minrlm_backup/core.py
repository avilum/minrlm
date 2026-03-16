"""
minrlm - Minimal Recursive Language Model
Based on https://arxiv.org/abs/2512.24601
Implemented by Avi Lumelsky
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAI

from .prompts import (
    SYSTEM_PROMPT_MINIMAL,
    SYSTEM_PROMPT_REASONING_MINIMAL,
    format_continue_prompt,
    format_system_prompt,
    format_user_prompt,
)

# Suppress HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Context chars to show before/after each search() match
SEARCH_CONTEXT_CHARS = 500

# Default timeout per completion (prevents infinite loops)
DEFAULT_MAX_TIME_SECONDS = 120


@dataclass
class RLMResult:
    """Result of an RLM completion."""

    response: str
    iterations: int
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    history: list[dict[str, str]] = field(default_factory=list)
    log_file_path: str | None = None


class _StopExecution(BaseException):
    """Raised by FINAL()/FINAL_var() to cleanly halt REPL execution.

    Inherits from BaseException (not Exception) so it is NOT caught by
    ``except Exception`` blocks inside user code, but IS caught by the
    explicit handler in PythonREPL.execute().
    """


class ProtectedNamespace(dict):
    """Dict that prevents reassignment of protected keys (data AND built-in tools)."""

    PROTECTED_DATA = {"input_0", "input_1", "input_2", "task_0"}
    PROTECTED_TOOLS = {
        "FINAL",
        "FINAL_var",
        "search",
        "peek",
        "sub_llm",
        "sub_llm_batch",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._allow_reassign = False

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self.PROTECTED_DATA and key in self and not self._allow_reassign:
            raise NameError(f"Cannot reassign '{key}' - it already contains your data. Use it directly.")
        if key in self.PROTECTED_TOOLS and key in self and not self._allow_reassign:
            return  # silently ignore — don't let LLM shadow built-in tools
        super().__setitem__(key, value)

    def allow_reassign(self, allow: bool) -> None:
        self._allow_reassign = allow


class PythonREPL:
    """Persistent Python REPL with sub_llm() and FINAL() support."""

    HIDDEN_KEYS = {
        "__builtins__",
        "__name__",
        "sub_llm",
        "sub_llm_batch",
        "FINAL",
        "FINAL_var",
        "peek",
        "search",
    }  # Don't include input_0/task_0 - they should be cleared on reset()

    def __init__(
        self,
        sub_llm_callback: Callable | None = None,
        sub_llm_batch_callback: Callable | None = None,
    ):
        self._output: str | None = None
        self._data_accessed = False  # Track if search() or input_0 was used
        self._namespace = ProtectedNamespace(
            {
                "__builtins__": __builtins__,
                "__name__": "__main__",  # So `if __name__ == "__main__":` works
                "sub_llm": self._make_sub_llm(sub_llm_callback),
                "sub_llm_batch": self._make_sub_llm_batch(sub_llm_batch_callback),
                "FINAL": self._set_output,
                "FINAL_var": self._set_output_var,
                "peek": self._peek,
                "search": self._search,
            }
        )

    def _make_sub_llm(self, callback: Callable[[str, str], str] | None) -> Callable[[str, str], str]:
        def sub_llm(task: str, context: str = "") -> str:
            """Call a sub-LLM with a task and optional context data."""
            if callback is None:
                raise RuntimeError("sub_llm() not available")
            return str(callback(task, context))

        return sub_llm

    def _make_sub_llm_batch(
        self, callback: Callable[[list[tuple[str, str]]], list[str]] | None
    ) -> Callable[[list[tuple[str, str]]], list[str]]:
        def sub_llm_batch(tasks: list[tuple[str, str]]) -> list[str]:
            """Run multiple sub-LLM calls in parallel. tasks = [(task, context), ...]"""
            if callback is None:
                raise RuntimeError("sub_llm_batch() not available")
            return list(callback(tasks))

        return sub_llm_batch

    def _set_output(self, value: str) -> None:
        # Reject None or empty values
        if value is None:
            raise ValueError("FINAL() called with None - provide a non-empty string value")

        # Enforce data grounding: if input_0 exists and is non-empty, must access data first
        input_0_val = self._namespace.get("input_0")
        if input_0_val and not self._data_accessed:
            # Allow patch outputs without forcing search()
            if self._looks_like_patch(str(value)):
                self._data_accessed = True
            else:
                raise ValueError(
                    "You must call search(input_0, 'keyword') first to find the data. Don't guess - search!"
                )

        self._output = str(value).strip()

        # Clean common artifacts from search() tuple returns
        # e.g., "['New York']" -> "New York", "[]" -> ""
        if self._output.startswith("[") and self._output.endswith("]"):
            inner = self._output[1:-1].strip()
            # Handle ['value'] or ["value"]
            if (inner.startswith("'") and inner.endswith("'")) or (inner.startswith('"') and inner.endswith('"')):
                inner = inner[1:-1]
            self._output = inner

        # Disallow placeholder outputs
        if self._output.strip().lower() in {"unknown", "n/a", "none"}:
            raise ValueError("FINAL() cannot be a placeholder (unknown/n/a/none). Extract the answer from input_0.")

        # Reject empty string after cleaning
        if self._output == "":
            raise ValueError("FINAL() called with empty string - provide a non-empty answer")

        # Halt execution cleanly - FINAL() is a terminal call
        raise _StopExecution()

    @staticmethod
    def _looks_like_patch(text: str) -> bool:
        """Detect patch-like outputs to relax grounding requirements."""
        return "diff --git " in text or "*** Begin Patch" in text

    def _search(self, text: str, pattern: str, context: int = SEARCH_CONTEXT_CHARS) -> list[tuple[str, str, str]]:
        """Search for literal pattern in text (case-insensitive).

        For regex, use: import re; re.findall(pattern, text)

        Args:
            text: The text to search in
            pattern: Literal string to find (case-insensitive)
            context: Characters to show around each match (default: SEARCH_CONTEXT_CHARS)

        Returns:
            List of tuples: (match, before_context, after_context)
        """
        self._data_accessed = True  # Mark data as accessed
        matches = []
        text_lower = text.lower()
        pattern_lower = pattern.lower()
        start = 0

        while True:
            pos = text_lower.find(pattern_lower, start)
            if pos == -1:
                break
            # Find the end of the "token" (continue until whitespace/punctuation)
            end = pos + len(pattern)
            while end < len(text) and text[end] not in " \t\n\r,;:!?()[]{}\"'<>":
                end += 1
            actual_match = text[pos:end]

            # Get context before and after
            ctx_before = text[max(0, pos - context) : pos]
            ctx_after = text[end : min(len(text), end + context)]
            matches.append((actual_match, ctx_before, ctx_after))
            start = pos + 1

        return matches

    def _set_output_var(self, var_name: str) -> None:
        """Set output from a variable in the namespace (FINAL_VAR from paper)."""
        # Enforce data grounding: if input_0 exists and is non-empty, must access data first
        input_0_val = self._namespace.get("input_0")
        if input_0_val and not self._data_accessed:
            raise ValueError("You must call search(input_0, 'keyword') first to find the data. Don't guess - search!")

        if var_name not in self._namespace:
            raise NameError(f"Variable '{var_name}' not found in REPL")

        value = self._namespace[var_name]
        if value is None:
            raise ValueError(f"Variable '{var_name}' is None - provide a non-empty value")

        self._output = str(value).strip()
        if self._output.strip().lower() in {"unknown", "n/a", "none"}:
            raise ValueError(
                f"Variable '{var_name}' resolves to a placeholder (unknown/n/a/none). Extract the answer from input_0."
            )
        if self._output == "":
            raise ValueError(f"Variable '{var_name}' contains empty string - provide a non-empty value")

        # Halt execution cleanly
        raise _StopExecution()

    def _peek(self, data: Any, max_len: int = 500, max_items: int = 5, depth: int = 0) -> str:
        """Efficient preview of data - truncates large strings/lists, recurses into structures."""
        if isinstance(data, str):
            if len(data) <= max_len:
                preview = repr(data)
            else:
                preview = f"str[{len(data):,}]"
            return preview

        elif isinstance(data, list | tuple):
            bracket = "[]" if isinstance(data, list) else "()"
            if len(data) == 0:
                return bracket
            for item in data[:max_items]:
                self._peek(item, max_len, max_items, depth + 1)
            return f"{bracket} ({len(data)} items)"

        elif isinstance(data, dict):
            if len(data) == 0:
                return "{}"
            for _k, v in list(data.items())[:max_items]:
                self._peek(v, max_len, max_items, depth + 2)
            return f"{{}} ({len(data)} keys)"

        else:
            preview = repr(data)
            if len(preview) > max_len:
                preview = preview[:max_len] + "..."
            return preview

    def save_output(self) -> str | None:
        """Save current output state."""
        return self._output

    def restore_output(self, saved: str | None) -> None:
        """Restore output state."""
        self._output = saved

    def execute(self, code: str) -> dict[str, Any]:
        """Execute code, return {stdout, output, error, state}."""
        self._output = None
        stdout_capture = StringIO()
        old_stdout, sys.stdout = sys.stdout, stdout_capture

        # Check if input_0 is accessed directly (e.g., json.loads(input_0), re.findall(..., input_0), etc.)
        # This allows structured data parsing and pattern matching without requiring search()
        input_0_val = self._namespace.get("input_0")
        if input_0_val and not self._data_accessed:
            import re

            # Look for common patterns that indicate input_0 is being used
            has_input_0 = "input_0" in code
            has_regex = bool(re.search(r"re\.(findall|search|finditer|match|fullmatch)", code))
            patterns = [
                r"json\.loads\s*\(\s*input_0",  # json.loads(input_0)
                r"input_0\s*\[",  # input_0[...]
                r"input_0\s*\.",  # input_0.method()
                r"=\s*input_0",  # var = input_0
                r"sub_llm\s*\([^)]*input_0",  # sub_llm(task, input_0)
                r"sub_llm_batch\s*\([^)]*input_0",  # sub_llm_batch([..., input_0, ...])
                r"[f]['\"][^'\"]*\{input_0",  # f"...{input_0}..."
            ]
            # If input_0 is used AND (regex function is called OR direct access pattern matches)
            if has_input_0 and (has_regex or any(re.search(p, code) for p in patterns)):
                self._data_accessed = True

        result: dict[str, Any] = {"stdout": "", "output": None, "error": None}
        try:
            exec(code, self._namespace)
        except _StopExecution:
            pass  # FINAL() was called - normal termination, output already set
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
        finally:
            sys.stdout = old_stdout
            result["stdout"] = stdout_capture.getvalue()
            # ALWAYS capture output - even if error occurred AFTER FINAL() was called
            result["output"] = self._output

        # Include current state
        result["state"] = self.get_state()
        return result

    def set_variable(self, name: str, value: Any, allow_override: bool = False) -> None:
        if allow_override and isinstance(self._namespace, ProtectedNamespace):
            prev = self._namespace._allow_reassign
            self._namespace.allow_reassign(True)
            try:
                self._namespace[name] = value
            finally:
                self._namespace.allow_reassign(prev)
            return
        self._namespace[name] = value

    def reset(self) -> None:
        """Reset REPL state for a new completion while preserving callbacks."""
        self._output = None
        self._data_accessed = False  # Reset data access tracking
        # Clear user variables, keep built-ins and special functions
        keys_to_keep = set(self.HIDDEN_KEYS) | {
            "sub_llm",
            "sub_llm_batch",
            "FINAL",
            "FINAL_var",
            "__builtins__",
            "__name__",
        }
        filtered_dict = {k: v for k, v in self._namespace.items() if k in keys_to_keep}
        self._namespace = ProtectedNamespace(filtered_dict)

    def get_state(self) -> dict[str, str]:
        """Get current namespace state as {name: type_and_preview}."""
        state = {}
        for name, value in self._namespace.items():
            if name in self.HIDDEN_KEYS or name.startswith("_"):
                continue
            state[name] = self._describe_value(value)
        return state

    def _describe_value(self, value: Any, max_len: int = 80) -> str:
        """Describe a value with type and preview."""
        t = type(value).__name__

        if isinstance(value, str):
            preview = value[:max_len] + "..." if len(value) > max_len else value
            return f"str({len(value)} chars) = {repr(preview)}"
        elif isinstance(value, list | tuple):
            return f"{t}({len(value)} items)"
        elif isinstance(value, dict):
            return f"dict({len(value)} keys: {list(value.keys())[:5]})"
        elif isinstance(value, int | float | bool):
            return f"{t} = {value}"
        elif callable(value):
            return f"function {getattr(value, '__name__', '?')}"
        else:
            try:
                s = str(value)
                if len(s) > max_len:
                    s = s[:max_len] + "..."
                return f"{t} = {s}"
            except Exception:
                return t


class RLM:
    """
    Recursive Language Model - Main Interface.

    Provide EITHER an initialized OpenAI client OR connection parameters (api_key/base_url).

    Usage:
        # Option 1: Let RLM create the client
        rlm = RLM(model="gpt-5-mini")

        # Option 2: Inject your own client
        from openai import OpenAI
        client = OpenAI(api_key="sk-...", base_url="https://my-proxy.com/v1")
        rlm = RLM(model="gpt-5-mini", client=client)

        # With Docker sandboxing (requires Docker):
        rlm = RLM(model="gpt-5-mini", use_docker=True)
    """

    def __init__(
        self,
        model: str | None = None,  # Falls back to MINRLM_MODEL or "gpt-5-mini"
        client: (OpenAI | None) = None,  # Pre-initialized sync client (skips api_key/base_url)
        async_client: AsyncOpenAI | None = None,  # Pre-initialized async client
        api_key: str | None = None,  # Falls back to MINRLM_API_KEY or OPENAI_API_KEY
        base_url: str | None = None,  # Falls back to MINRLM_BASE_URL or OPENAI_BASE_URL
        max_iterations: int = 6,  # Reduced from 20 - force early commitment
        max_time_seconds: int = DEFAULT_MAX_TIME_SECONDS,  # Timeout per completion
        max_output_tokens: (
            int | None
        ) = 3000,  # Allow complete code generation for complex tasks (was 1500, caused truncation)
        temperature: float = 0.0,  # Use 0 for deterministic code generation
        reasoning_effort: str = "low",  # For reasoning models: "low", "medium", "high"
        log_dir: str | None = None,
        async_batch: bool = True,  # Enable parallel sub_llm_batch calls
        on_step: (Callable[[str, dict], None] | None) = None,  # Callback for streaming steps
        max_sub_llm_calls: int = 100,  # Guardrail for recursive sub_llm usage
        # Docker options
        use_docker: bool = False,  # Run code in Docker container
        docker_image: str = "python:3.14-slim",
        docker_memory: str = "256m",
        docker_timeout: int = 60,
        debug: bool = False,
    ):
        model = model or os.environ.get("MINRLM_MODEL", "gpt-5-mini")
        self.model = model
        self.max_iterations = max_iterations
        self.max_time_seconds = max_time_seconds
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.log_dir = Path(log_dir) if log_dir else None
        self.async_batch = async_batch
        self.on_step = on_step
        self.use_docker = use_docker
        self.max_sub_llm_calls = max_sub_llm_calls
        self._sub_llm_calls = 0
        # Accumulators for tokens spent in sub_llm() / sub_llm_batch() calls.
        # Reset at the start of each top-level completion(); summed into the result.
        self._sub_llm_total_tokens: int = 0
        self._sub_llm_input_tokens: int = 0
        self._sub_llm_output_tokens: int = 0
        self.debug: bool = debug

        if client is not None:
            self.client = client
            self.async_client = async_client
        else:
            api_key = api_key or os.environ.get("MINRLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
            base_url = base_url or os.environ.get("MINRLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url) if async_batch else None
        self._depth = 0
        self._log_entries: list[dict[str, Any]] = []

        # Initialize REPL - use Docker or local based on config
        if use_docker:
            from .docker_repl import DockerREPL

            docker_repl = DockerREPL(
                image=docker_image,
                memory_limit=docker_memory,
                timeout=docker_timeout,
                network_disabled=True,
                sub_llm_callback=self._handle_sub_llm,
                sub_llm_batch_callback=self._handle_sub_llm_batch,
            )
            # Verify Docker is available, fall back to local REPL with warning if not
            if docker_repl.is_available():
                self._repl: PythonREPL | DockerREPL = docker_repl
            else:
                import warnings

                warnings.warn(
                    "⚠️  Docker is not available. Falling back to local PythonREPL (unsandboxed). "
                    "Install Docker for secure sandboxed execution: https://docs.docker.com/get-docker/",
                    UserWarning,
                    stacklevel=2,
                )
                self.use_docker = False  # Update flag to reflect actual state
                self._repl = PythonREPL(
                    sub_llm_callback=self._handle_sub_llm,
                    sub_llm_batch_callback=self._handle_sub_llm_batch,
                )
        else:
            self._repl = PythonREPL(
                sub_llm_callback=self._handle_sub_llm,
                sub_llm_batch_callback=self._handle_sub_llm_batch,
            )

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _extract_code(self, text: str) -> str | None:
        """Extract code blocks from LLM response. Tries multiple formats."""
        # Try ```python first
        matches = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if matches:
            code = "\n\n".join(matches)
            code = self._sanitize_patch_code(code)
            if self._is_valid_python(code):
                return code
            patch_wrapped = self._wrap_patch_block(code)
            if patch_wrapped is not None:
                return patch_wrapped
        # Try ```py
        matches = re.findall(r"```py\s*\n(.*?)```", text, re.DOTALL)
        if matches:
            code = "\n\n".join(matches)
            code = self._sanitize_patch_code(code)
            if self._is_valid_python(code):
                return code
            patch_wrapped = self._wrap_patch_block(code)
            if patch_wrapped is not None:
                return patch_wrapped
        # Try generic ``` blocks
        matches = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
        if matches:
            code = "\n\n".join(matches)
            code = self._sanitize_patch_code(code)
            if self._is_valid_python(code):
                return code
            patch_wrapped = self._wrap_patch_block(code)
            if patch_wrapped is not None:
                return patch_wrapped
        # Fallback: if response looks like pure code (starts with import/assignment)
        text = text.strip()
        if text.startswith("import ") or text.startswith("from ") or re.match(r"^[a-z_][a-z0-9_]*\s*=", text):
            sanitized = self._sanitize_patch_code(text)
            if self._is_valid_python(sanitized):
                return sanitized
            patch_wrapped = self._wrap_patch_block(sanitized)
            if patch_wrapped is not None:
                return patch_wrapped
        # Final fallback: extract patch-like blocks directly
        patch_wrapped = self._wrap_patch_block(text)
        if patch_wrapped is not None:
            return patch_wrapped
        return None

    def _is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python syntax."""
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError:
            return False

    def _sanitize_patch_code(self, code: str) -> str:
        """Make patch-holding code blocks valid when they contain triple quotes."""
        for delim in ('"""', "'''"):
            marker = f"patch = {delim}"
            if marker not in code:
                continue
            start = code.find(marker)
            if start == -1:
                continue
            start += len(marker)
            end = code.rfind(delim)
            if end <= start:
                continue
            inner = code[start:end]
            escaped = inner.replace(delim, f"\\{delim}")
            return code[:start] + escaped + code[end:]
        return code

    def _wrap_patch_block(self, text: str) -> str | None:
        """Wrap patch-like text into a FINAL(patch) Python snippet."""
        start = None
        end = None
        if "*** Begin Patch" in text:
            start = text.find("*** Begin Patch")
            end_marker = "*** End Patch"
            end = text.find(end_marker, start)
            if end != -1:
                end += len(end_marker)
        elif "diff --git" in text:
            start = text.find("diff --git")
            end = None

        if start is None:
            return None

        patch = text[start:end].strip() if end else text[start:].strip()
        if not patch:
            return None

        safe_patch = patch.replace("'''", "\\'\\'\\'")
        return f"patch = '''{safe_patch}'''\nFINAL(patch)"

    def _call_llm(self, messages: list[dict[str, str]]) -> tuple[str, int, int, int]:
        """Call LLM with retry logic for rate limits. Returns (response_text, total_tokens, input_tokens, output_tokens)."""
        import time

        # Build kwargs dynamically - only include params when they have valid values
        # (some providers reject null values for optional params)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        # TODO: Maintain this list as OpenAI releases new reasoning models
        # Currently: o1, o1-mini, o1-preview, o3, o3-mini, gpt-5, gpt-5-nano, gpt-5-mini
        is_reasoning_model = self._is_reasoning_model(self.model)

        # Set temperature (reasoning models don't support temperature != 1)
        if self.temperature is not None and not is_reasoning_model:
            kwargs["temperature"] = self.temperature

        # Limit output tokens to reduce latency
        # Use appropriate parameter based on model type
        if self.max_output_tokens:
            if is_reasoning_model:
                # Reasoning models (o1, o3, gpt-5) use max_completion_tokens
                kwargs["max_completion_tokens"] = self.max_output_tokens
            else:
                # Standard models use max_tokens
                kwargs["max_tokens"] = self.max_output_tokens

        # Use reasoning_effort for reasoning models to control token cost
        if is_reasoning_model and self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        # Retry logic with exponential backoff for rate limits
        max_retries = 5
        base_delay = 1.0  # Start with 1 second
        # Escalation order for models that reject lower reasoning_effort values
        _effort_levels = ["low", "medium", "high"]

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                usage = resp.usage
                return (
                    resp.choices[0].message.content or "",
                    usage.total_tokens if usage else 0,
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                )
            except Exception as e:
                error_str = str(e)

                # reasoning_effort value not supported by this model — escalate and retry immediately
                if (
                    "reasoning_effort" in error_str
                    and "unsupported_value" in error_str
                    and "reasoning_effort" in kwargs
                ):
                    current = kwargs["reasoning_effort"]
                    current_idx = _effort_levels.index(current) if current in _effort_levels else -1
                    if current_idx < len(_effort_levels) - 1:
                        kwargs["reasoning_effort"] = _effort_levels[current_idx + 1]
                        self._log(
                            "reasoning_effort_escalated",
                            {
                                "from": current,
                                "to": kwargs["reasoning_effort"],
                            },
                        )
                        continue
                    raise

                # Model does not support the chat completions endpoint
                if "404" in error_str and "not a chat model" in error_str:
                    raise ValueError(
                        f"Model '{self.model}' does not support the chat completions endpoint. "
                        "Use a chat-capable variant (e.g. add '-chat' suffix)."
                    ) from e

                # Check if it's a rate limit error (429)
                is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()

                if is_rate_limit and attempt < max_retries - 1:
                    # Extract wait time from error message if available
                    wait_time = base_delay * (2**attempt)  # Exponential backoff

                    # Try to parse wait time from error message (e.g., "try again in 133ms")
                    import re

                    match = re.search(r"try again in (\d+)ms", error_str)
                    if match:
                        suggested_ms = int(match.group(1))
                        wait_time = max(wait_time, suggested_ms / 1000.0)

                    match = re.search(r"try again in ([\d.]+)s", error_str)
                    if match:
                        suggested_s = float(match.group(1))
                        wait_time = max(wait_time, suggested_s)

                    self._log(
                        "rate_limit_retry",
                        {
                            "attempt": attempt + 1,
                            "wait_seconds": wait_time,
                            "error": error_str[:200],
                        },
                    )

                    time.sleep(wait_time)
                    continue

                # Not a rate limit error, or max retries exceeded
                raise

        raise RuntimeError(f"LLM call failed after {max_retries} retries")

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Check if model is a reasoning model that uses hidden chain-of-thought.

        TODO: Keep this list updated as OpenAI releases new reasoning models.
        These models have internal reasoning tokens that are billed but not visible.
        """
        model_lower = model.lower()
        reasoning_patterns = [
            "o1",
            "o3",  # o1, o1-mini, o1-preview, o3, o3-mini
            "gpt-5",  # gpt-5, gpt-5-nano, gpt-5-mini
        ]
        return any(pattern in model_lower for pattern in reasoning_patterns)

    @staticmethod
    def _extract_allowed_labels(task: str) -> list[str] | None:
        """Extract a label set from a task string, if present."""
        if not task:
            return None

        candidates: list[str] = []

        # Prefer quoted labels: 'label' or "label"
        for match in re.findall(r"'([^']+)'|\"([^\"]+)\"", task):
            token = match[0] or match[1]
            token = token.strip()
            if token and token not in candidates:
                candidates.append(token)

        # Pattern like: "labels: a, b, c" or "one of: a, b"
        if not candidates:
            m = re.search(r"(?:labels?|one of)\s*[:\-]\s*([^\n.]+)", task, re.IGNORECASE)
            if m:
                chunk = m.group(1)
                parts = re.split(r",|/|\bor\b", chunk)
                for p in parts:
                    t = p.strip().strip(".")
                    if t and t not in candidates:
                        candidates.append(t)

        # Multiple-choice format: detect actual lettered options present
        if not candidates:
            found = [c for c in "ABCDEFGHIJ" if f"{c})" in task]
            if len(found) >= 2:
                candidates = found

        # Common label pairs
        task_lower = task.lower()
        if not candidates:
            if "true" in task_lower and "false" in task_lower:
                candidates = ["True", "False"]
            elif "correct" in task_lower and "incorrect" in task_lower:
                candidates = ["correct", "incorrect"]
            elif "yes" in task_lower and "no" in task_lower:
                candidates = ["yes", "no"]

        # Sanitize: keep reasonably short tokens (allow spaces/hyphens)
        cleaned: list[str] = []
        for c in candidates:
            c = c.strip()
            # Python 3.14's re rejects "\\-" inside character classes; keep '-' at the end.
            if 0 < len(c) <= 60 and re.match(r"^[A-Za-z0-9_ -]+$", c):
                if c not in cleaned:
                    cleaned.append(c)

        if 1 <= len(cleaned) <= 12:
            return cleaned
        return None

    @staticmethod
    def _normalize_sub_llm_output(text: str, allowed: list[str] | None) -> str:
        """Normalize sub_llm output to a single label if possible."""
        if text is None:
            return ""

        raw = text.strip()
        if not raw:
            return ""

        # If code block contains FINAL("..."), extract it
        match = re.search(r"FINAL\(['\"](.+?)['\"]\)", raw)
        if match:
            raw = match.group(1).strip()

        # Strip code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\s*", "", raw)
            raw = re.sub(r"```\s*$", "", raw).strip()

        # Only keep first line
        raw = raw.splitlines()[0].strip()

        # Drop leading prefixes like "Label:" or "Answer:"
        raw = re.sub(r"^(?:label|answer)\s*:\s*", "", raw, flags=re.IGNORECASE).strip()

        if allowed:
            # Exact match (case-insensitive)
            for label in allowed:
                if raw.lower() == label.lower():
                    return label
            # Word match inside the response
            for label in sorted(allowed, key=len, reverse=True):
                if " " in label:
                    if label.lower() in raw.lower():
                        return label
                else:
                    if re.search(rf"\b{re.escape(label)}\b", raw, flags=re.IGNORECASE):
                        return label
        return raw

    @staticmethod
    def _is_suspect_label_task(task: str) -> bool:
        """Detect tasks that likely require label classification even without 'label' keyword."""
        if not task:
            return False
        t = task.lower()
        # Multiple-choice tasks: do not force label-classification heuristics.
        if "choices:" in t and re.search(r"\b[a-d]\)", t):
            return False
        # Do not treat secret-code extraction tasks as label classification.
        if re.search(r"\b(find|locate|return)\b.*\bsecret\s+code\b", t):
            return False
        keywords = [
            "label",
            "labels",
            "true",
            "false",
            "correct",
            "incorrect",
            "classify",
            "one of",
            "option",
            "options",
            "choice",
            "choices",
            "more common",
            "less common",
            "same frequency",
            "most common",
            "least common",
        ]
        if any(k in t for k in keywords):
            return True
        # Only treat quoted tokens as labels if there is a label intent keyword nearby
        if re.search(r"(label|classify|one of|option|choice)", t) and re.search(r"'[^']+'|\"[^\"]+\"", task):
            return True
        return False

    def _build_sub_llm_messages(self, task: str, context: str, allowed: list[str] | None) -> list[dict[str, str]]:
        """Build a minimal, tool-free prompt for sub_llm."""
        user = task.strip() if task else ""
        if allowed:
            user += f"\n\nReturn exactly one of: {', '.join(allowed)}. No other text."
        if context:
            user += f"\n\nContext:\n{context}"
            sys_prompt = SYSTEM_PROMPT_MINIMAL
        else:
            sys_prompt = SYSTEM_PROMPT_REASONING_MINIMAL
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ]

    def _call_sub_llm_raw(self, task: str, context: str) -> str:
        """Call LLM for sub_llm with minimal prompt and label enforcement."""
        task = self._sanitize_sub_llm_task(task, context)
        allowed = self._extract_allowed_labels(task)
        messages = self._build_sub_llm_messages(task, context, allowed)
        response_text, tok_t, tok_i, tok_o = self._call_llm(messages)
        self._sub_llm_total_tokens += tok_t
        self._sub_llm_input_tokens += tok_i
        self._sub_llm_output_tokens += tok_o
        normalized = self._normalize_sub_llm_output(response_text, allowed)

        if allowed and normalized not in allowed:
            # Retry once with stricter instruction
            retry_task = task + f"\n\nReturn exactly one of: {', '.join(allowed)}. Do not add any other text."
            messages = self._build_sub_llm_messages(retry_task, context, allowed)
            response_text, tok_t, tok_i, tok_o = self._call_llm(messages)
            self._sub_llm_total_tokens += tok_t
            self._sub_llm_input_tokens += tok_i
            self._sub_llm_output_tokens += tok_o
            normalized = self._normalize_sub_llm_output(response_text, allowed)
            if normalized not in allowed:
                self._log(
                    "sub_llm_invalid_label",
                    {"task": task[:120], "output": response_text[:120]},
                )
        return normalized

    def _guard_label_counting(self, task: str, code: str, context: str | None = None) -> str | None:
        """Guard against label-word counting in implicit-label tasks."""
        if not task or not code:
            return None
        # Code-retrieval tasks should not be treated as label-classification tasks.
        if self._is_code_retrieval_task(task):
            return None
        if not self._is_suspect_label_task(task):
            return None
        has_sub_llm = "sub_llm" in code
        # Allow if code explicitly parses Label: fields
        if re.search(r"label\s*:", code, flags=re.IGNORECASE):
            return None
        # Reject using search() fragments as items for line-based datasets
        if "search(input_0" in code and ("before" in code or "after" in code) and "splitlines" not in code:
            return (
                "Line-based dataset detected. Build items via input_0.splitlines() and filter lines. "
                "Do not use search() fragments (before/after) as items."
            )
        # Reject returning a label/Answer: 0 without classification
        if (not has_sub_llm) and (
            re.search(r"FINAL\(\s*['\"]Label:", code) or re.search(r"FINAL\(\s*['\"]Answer:\s*0", code)
        ):
            return (
                "Do not return a label or Answer: 0 without classifying at least one item. "
                "Extract items from lines, classify with sub_llm/sub_llm_batch, then aggregate."
            )
        # If the task states there are N items/pairs, require explicit limiting after filtering.
        # If context states there are N items/pairs, require explicit limiting after filtering.
        m = None
        if context:
            m = re.search(
                r"following\s+lines\s+contain\s+(\d+)\s+(?:pairs|items|lines)",
                context,
                flags=re.IGNORECASE,
            )
        if not m:
            m = re.search(r"contain\s+(\d+)\s+(?:pairs|items|lines)", task, flags=re.IGNORECASE)
        if m and ("splitlines" in code) and ("<-->" in code or "||" in code or "Date:" in code):
            n = m.group(1)
            if n not in code and f"[:{n}]" not in code:
                return (
                    f"Task states there are {n} items. After filtering data lines, "
                    "limit to exactly that many items (e.g., items = items[:N]) and avoid header lines."
                )
        # If labels mentioned and no sub_llm, reject counting heuristics
        if (not has_sub_llm) and re.search(
            r"\b(true|false|correct|incorrect|positive|negative|formal|informal)\b",
            code,
            flags=re.IGNORECASE,
        ):
            return (
                "Implicit-label task detected. Do NOT count label words or rely on regex heuristics. "
                "Use sub_llm/sub_llm_batch to classify each item then aggregate."
            )
        if has_sub_llm:
            return None
        return (
            "Label task requires explicit label extraction or sub_llm classification. "
            "Use sub_llm/sub_llm_batch unless labels are explicitly present next to each item."
        )

    @staticmethod
    def _is_code_retrieval_task(task: str) -> bool:
        """Heuristic for tasks that require exact code extraction."""
        if not task:
            return False
        t = task.lower()
        keywords = [
            "codebase",
            "code snippet",
            "exact snippet",
            "exact function",
            "exact method",
            "exact class",
            "exact identifier",
            "return the exact",
        ]
        return any(k in t for k in keywords)

    def _guard_multiple_choice(self, task: str, code: str) -> str | None:
        """Guard against guessing single-letter MCQA answers without sub_llm reasoning."""
        if not task or not code:
            return None
        task_lower = task.lower()
        # Only apply to multiple-choice tasks
        if not ("choices:" in task_lower and re.search(r"\b[a-d]\)", task_lower)):
            return None
        # Allow if sub_llm was used
        if "sub_llm" in code:
            return None
        # Detect direct single-letter FINAL without sub_llm
        if re.search(r'FINAL\s*\(\s*["\'][A-D]["\']', code):
            return (
                "Multiple-choice task: do NOT call FINAL('A'/'B'/'C'/'D') without sub_llm reasoning. "
                "Find a relevant snippet with search(), then: FINAL(sub_llm(task_0, snippet))"
            )
        return None

    def _guard_pipe_delimited_search(self, task: str, code: str, context: str | None) -> str | None:
        """Guard against using search() for pipe-delimited record-per-line data."""
        if not context or not code:
            return None
        # Only relevant if data looks like pipe-delimited records
        if "Date:" not in context[:2000] or "||" not in context[:2000]:
            return None
        # Detect the TOOL search(input_0, "User:/Date:") being used on pipe-delimited data.
        # Allow if splitlines() is used for actual parsing (search might just be for validation)
        # Must match search(input_0, ...) not re.search(r"User:...", ...).
        has_search = re.search(r'(?<!\w)search\s*\(\s*input_0\s*,\s*["\'](?:Date:|User:)', code)
        has_splitlines = "splitlines" in code
        if has_search and not has_splitlines:
            return (
                'Pipe-delimited data: do NOT use search(input_0, "User: X") or search(input_0, "Date:") '
                "— this splits records mid-line. Use splitlines() instead:\n"
                "  data_lines = [l for l in input_0.splitlines() if l.startswith('Date:')]\n"
                "  for line in data_lines:\n"
                "      parts = [p.strip() for p in line.split('||')]\n"
                "      # parts[0]=date, parts[1]=user, parts[2]=instance"
            )
        return None

    def _guard_repoqa(self, task: str, code: str) -> str | None:
        """Guard for code retrieval tasks: require sub_llm before any search()."""
        if not task or not code:
            return None
        task_lower = task.lower()
        # Only apply to code-retrieval tasks (REPOQA)
        if "return the exact function" not in task_lower and "return the exact code" not in task_lower:
            return None
        # Allow if sub_llm was used for discovery (correct pattern)
        if "sub_llm" in code:
            return None
        # Allow peek ONLY as a standalone step (no search alongside it)
        if "peek" in code and not re.search(r"\bsearch\s*\(", code):
            return None
        # Detect search() call without sub_llm — model is skipping the discovery step
        if re.search(r"\bsearch\s*\(", code):
            return (
                "Code retrieval task: do NOT search with description text — the codebase "
                "uses exact code identifiers. ALWAYS use sub_llm to find the function name FIRST:\n"
                "  preview = input_0[:2000]  # actual code; do NOT use peek() — it returns size metadata\n"
                "  func_name = sub_llm('Reply with ONLY the exact function name — just the name, nothing else.',\n"
                "                      preview + '\\n\\nTask: ' + task_0).strip()\n"
                "  res = search(input_0, func_name + '(')\n"
                "  if not res: res = search(input_0, func_name)\n"
                "  if res:\n"
                "      match, before, after = res[0]; FINAL(before[-400:] + match + after)\n"
                "  else:\n"
                "      pos = input_0.find(func_name); FINAL(input_0[max(0,pos-400):pos+3000])"
            )
        return None

    def _guard_stdin_usage(self, task: str, code: str) -> str | None:
        """Guard against executing sys.stdin.read() which hangs the REPL.
        Code generation tasks must return source code as a string, not execute it."""
        if not code:
            return None
        # Check if code calls sys.stdin outside of a string literal
        # Simple heuristic: if sys.stdin appears AND it's not inside a triple-quoted string
        if "sys.stdin" not in code:
            return None
        # If it's inside a string assignment (code = '''...sys.stdin...'''), that's fine
        if re.search(
            r"""(?:code|solution|program|my_answer|src|source)\s*=\s*(?:'''|\"\"\"|\"|')""",
            code,
        ):
            return None
        # If FINAL() is called with a string variable, it's likely wrapping code correctly
        if re.search(r"FINAL\s*\(\s*(?:code|solution|program|my_answer|src|source)", code):
            return None
        return (
            "sys.stdin.read() will HANG — there is no stdin in this REPL.\n"
            "This is a CODE GENERATION task. Wrap the solution in a STRING and return it:\n"
            "  code = '''\n"
            "  import sys\n"
            "  def main():\n"
            "      data = sys.stdin.read().split()\n"
            "      ...\n"
            "      print(result)\n"
            '  if __name__ == "__main__":\n'
            "      main()\n"
            "  '''\n"
            "  FINAL(code.strip())"
        )

    def _guard_code_extraction(self, task: str, context: str, output: str) -> str | None:
        """Ensure code outputs are exact substrings of input_0 for code-retrieval tasks."""
        if not output or not context:
            return None
        if not self._is_code_retrieval_task(task):
            return None
        # Skip verbatim-match check for multiple-choice tasks (single-letter answers A/B/C/D)
        task_lower = task.lower()
        if "choices:" in task_lower and re.search(r"\b[a-d]\)", task_lower):
            return None
        # Require verbatim substring match
        if output not in context:
            return (
                "Code retrieval task: output must be an exact substring of input_0. "
                "Use search() + direct extraction of the identifier/snippet from input_0; do not rewrite."
            )
        # If output is only an identifier but a def/class exists, require the full snippet.
        if "\n" not in output and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", output):
            if f"def {output}" in context or f"class {output}" in context:
                return (
                    "Code retrieval task: output is only an identifier. "
                    "Return the full def/class block from input_0 instead of just the name."
                )
        return None

    def _sanitize_sub_llm_task(self, task: str, context: str) -> str:
        """Prevent passing raw data as the sub_llm task by rewriting with labels."""
        if not task:
            return ""
        t = task.strip()
        if not context:
            return t
        # If task is identical to context or looks like raw data, rewrite.
        if t == context.strip():
            allowed = self._extract_allowed_labels(task)
            if allowed:
                return f"Return exactly one of: {', '.join(allowed)}. No other text."
            return "Return a single concise label. No other text."
        # Heuristic: task contains long raw data markers.
        if len(t) > 200 and ("Date:" in t or "<-->" in t):
            allowed = self._extract_allowed_labels(task)
            if allowed:
                return f"Return exactly one of: {', '.join(allowed)}. No other text."
        return t

    def _handle_sub_llm(self, task: str, context: str = "") -> str:
        """Handle recursive sub_llm() calls, preserving parent's output state."""
        # Save parent's output state to prevent contamination
        saved_output = self._repl.save_output() if self._repl else None
        saved_input_0 = None
        saved_data_accessed = None
        if self._repl:
            saved_input_0 = self._repl._namespace.get("input_0")
            saved_data_accessed = getattr(self._repl, "_data_accessed", None)

        if self.max_sub_llm_calls is not None:
            if self._sub_llm_calls >= self.max_sub_llm_calls:
                self._log(
                    "sub_llm_limit",
                    {"limit": self.max_sub_llm_calls, "depth": self._depth},
                )
                return "SUB_LLM_CALL_LIMIT_REACHED"
            self._sub_llm_calls += 1

        self._depth += 1
        try:
            if self.on_step:
                self.on_step(
                    "sub_llm_call",
                    {
                        "task": task[:2000],
                        "context_preview": context[:1000] if context else "",
                        "context_len": len(context),
                    },
                )
            result = self._call_sub_llm_raw(task, context)
            if self.on_step:
                self.on_step(
                    "sub_llm_response",
                    {
                        "response": result[:2000],
                    },
                )
            return result
        finally:
            self._depth -= 1
            # Restore parent's output state
            if self._repl:
                self._repl.restore_output(saved_output)
                # Restore parent context + access flag to avoid subcall pollution
                if saved_input_0 is not None:
                    self._repl.set_variable("input_0", saved_input_0, allow_override=True)
                if saved_data_accessed is not None and hasattr(self._repl, "_data_accessed"):
                    self._repl._data_accessed = saved_data_accessed

    def _handle_sub_llm_batch(self, tasks: list[tuple[str, str]]) -> list[str]:
        """Handle parallel sub_llm calls. tasks = [(task, context), ...]"""
        if not tasks:
            return []

        if self.max_sub_llm_calls is not None:
            remaining = self.max_sub_llm_calls - self._sub_llm_calls
            if remaining <= 0:
                self._log(
                    "sub_llm_limit",
                    {"limit": self.max_sub_llm_calls, "depth": self._depth},
                )
                return ["SUB_LLM_CALL_LIMIT_REACHED"] * len(tasks)
            if len(tasks) > remaining:
                tasks_to_run = tasks[:remaining]
                pending = len(tasks) - remaining
            else:
                tasks_to_run = tasks
                pending = 0
            self._sub_llm_calls += len(tasks_to_run)
        else:
            tasks_to_run = tasks
            pending = 0

        if self.async_batch and self.async_client:
            results = asyncio.run(self._run_batch_async(tasks_to_run))
        else:
            # Fallback: sequential
            results = [self._call_sub_llm_raw(task, ctx) for task, ctx in tasks_to_run]

        if pending:
            results.extend(["SUB_LLM_CALL_LIMIT_REACHED"] * pending)
        return results

    async def _run_batch_async(self, tasks: list[tuple[str, str]]) -> list[str]:
        """Run multiple sub_llm completions in parallel using async (minimal prompt)."""
        if self.async_client is None:
            raise RuntimeError("Async client not initialized")

        async_client = self.async_client  # Local ref for closure

        async def run_one(task: str, context: str) -> str:
            task = self._sanitize_sub_llm_task(task, context)
            allowed = self._extract_allowed_labels(task)
            messages = self._build_sub_llm_messages(task, context, allowed)
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }
            if "gpt-5" not in self.model.lower():
                kwargs["temperature"] = 0.7

            resp = await async_client.chat.completions.create(**kwargs)
            if resp.usage:
                self._sub_llm_total_tokens += resp.usage.total_tokens
                self._sub_llm_input_tokens += resp.usage.prompt_tokens
                self._sub_llm_output_tokens += resp.usage.completion_tokens
            text = resp.choices[0].message.content or ""
            return self._normalize_sub_llm_output(text, allowed)

        results = await asyncio.gather(*[run_one(task, ctx) for task, ctx in tasks])
        self._log("batch_call", {"count": len(tasks), "tasks": [t[0][:50] for t in tasks]})
        return list(results)

    def _log(self, event_type: str, data: dict[str, Any]) -> None:
        """Append to log entries."""
        if self.debug:
            print(f"[minrlm]{event_type}: {data}")
        if self.log_dir:
            self._log_entries.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": event_type,
                    "depth": self._depth,
                    "data": data,
                }
            )

    def _save_log(self, task: str) -> Path | None:
        """Save log entries to JSONL file."""
        if not self.log_dir or not self._log_entries:
            return None
        slug = "".join(c if c.isalnum() else "_" for c in task[:30])
        path = self.log_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{slug}.jsonl"
        with open(path, "w") as f:
            for entry in self._log_entries:
                f.write(json.dumps(entry) + "\n")
        self._log_entries = []
        return path

    def completion(self, task: str, context: str = "") -> RLMResult:
        """
        Run RLM completion for the given task.

        Args:
            task: The task/question to solve
            context: Optional context/input data (available as input_0)

        Returns:
            RLMResult with response, iterations, tokens, history
        """
        is_top_level = self._depth == 0

        if is_top_level:
            # Reset REPL state for new top-level completion (keep callbacks)
            self._repl.reset()
            self._log_entries = []
            self._log("start", {"task": task, "model": self.model})
            self._sub_llm_calls = 0
            self._sub_llm_total_tokens = 0
            self._sub_llm_input_tokens = 0
            self._sub_llm_output_tokens = 0

        # Set input_0 and task_0 for this completion (both top-level and sub_llm calls)
        # task_0 lets the model pass the full original task (with all choices) to sub_llm
        peek_output = ""
        if is_top_level and task:
            self._repl.set_variable("task_0", task, allow_override=False)
        self._repl.set_variable("input_0", context or "", allow_override=self._depth > 0)
        if context:
            # Auto-peek: show data preview in first prompt (saves an API call)
            peek_result = self._repl.execute("peek(input_0)")
            peek_output = peek_result.get("stdout", "")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": format_system_prompt(context)},
            {"role": "user", "content": format_user_prompt(task, context, peek_output)},
        ]

        history: list[dict[str, str]] = []
        total_tokens, input_tokens, output_tokens = 0, 0, 0
        final_output: str | None = None
        start_time = time.time()

        for iteration in range(self.max_iterations):
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.max_time_seconds:
                self._log("timeout", {"elapsed": elapsed, "max": self.max_time_seconds})
                if self.on_step:
                    self.on_step("timeout", {"elapsed": elapsed, "iteration": iteration + 1})
                break

            if self.on_step:
                self.on_step("thinking", {"iteration": iteration + 1})

            response_text, tok_total, tok_in, tok_out = self._call_llm(messages)
            total_tokens += tok_total
            input_tokens += tok_in
            output_tokens += tok_out
            history.append({"role": "assistant", "content": response_text})
            self._log(
                "llm_call",
                {
                    "iteration": iteration + 1,
                    "tokens": tok_total,
                    "input_tokens": tok_in,
                    "output_tokens": tok_out,
                },
            )

            code = self._extract_code(response_text)

            # Always log the raw response for debugging
            self._log("llm_response", {"response": response_text[:2000]})

            if self.on_step:
                self.on_step(
                    "llm_response",
                    {
                        "iteration": iteration + 1,
                        "response": response_text,
                        "has_code": code is not None,
                    },
                )

            if not code:
                cleaned = response_text.strip()
                auto_wrapped = False
                if cleaned:
                    if "FINAL(" in cleaned:
                        code = cleaned
                        auto_wrapped = True
                    elif (
                        self._is_code_retrieval_task(task)
                        or re.search(r"^(?:from|import|def|class)\b", cleaned)
                        or len(cleaned) <= 80
                    ):
                        # Wrap plain-text answers or code snippets to avoid format drift.
                        code = f"FINAL({cleaned!r})"
                        auto_wrapped = True

                if auto_wrapped:
                    self._log("no_code_autowrap", {"response_preview": cleaned[:200]})
                else:
                    # No code found - prompt to use FINAL()
                    self._log("no_code", {"response_preview": response_text[:500]})
                    messages += [
                        {"role": "assistant", "content": response_text},
                        {
                            "role": "user",
                            "content": "Use ```python and call FINAL('your answer').",
                        },
                    ]
                    continue

            # At this point code is guaranteed non-None (the `continue` above handles None)
            assert code is not None

            # Guard: reject code that counts label words instead of using sub_llm
            guard_msg = self._guard_stdin_usage(task, code)
            if not guard_msg:
                guard_msg = self._guard_label_counting(task, code, context)
            if not guard_msg:
                guard_msg = self._guard_pipe_delimited_search(task, code, context)
            if not guard_msg:
                guard_msg = self._guard_multiple_choice(task, code)
            if not guard_msg:
                guard_msg = self._guard_repoqa(task, code)
            if guard_msg:
                self._log(
                    "guard_label_counting",
                    {"message": guard_msg, "code_preview": code[:200]},
                )
                messages += [
                    {"role": "assistant", "content": response_text},
                    {
                        "role": "user",
                        "content": f"⚠️ {guard_msg}\nRewrite the code to fix this issue.",
                    },
                ]
                continue

            if self.on_step:
                self.on_step("executing", {"iteration": iteration + 1, "code": code})  # Full code for debugging

            result = self._repl.execute(code)
            self._log(
                "code_exec",
                {
                    "code": code[:500],
                    "output": result.get("output"),
                    "error": result.get("error"),
                    "state": result.get("state"),
                },
            )

            if self.on_step:
                self.on_step(
                    "executed",
                    {
                        "iteration": iteration + 1,
                        "stdout": result.get("stdout", ""),  # Full stdout for debugging
                        "output": result.get("output"),
                        "error": result.get("error"),
                    },
                )

            # Break if FINAL() was called (unless a guard rejects it)
            if result.get("output") is not None:
                candidate = result["output"]
                # Guard: ensure code-retrieval outputs are verbatim from input_0
                extract_guard = self._guard_code_extraction(task, context, candidate)
                if extract_guard:
                    self._log(
                        "guard_code_extraction",
                        {"message": extract_guard, "output": candidate[:200]},
                    )
                    messages += [
                        {"role": "assistant", "content": response_text},
                        {
                            "role": "user",
                            "content": f"⚠️ {extract_guard}\nFix the code.",
                        },
                    ]
                    continue
                final_output = candidate
                break

            # Pass state to continue prompt
            messages += [
                {"role": "assistant", "content": response_text},
                {
                    "role": "user",
                    "content": format_continue_prompt(
                        result.get("stdout", ""),
                        result.get("error", ""),
                        result.get("state", {}),
                        iteration=iteration + 1,
                        max_iterations=self.max_iterations,
                    ),
                },
            ]
        else:
            final_output = "Max iterations reached."

        log_path = None
        if is_top_level:
            self._log(
                "end",
                {
                    "response": final_output,
                    "total_tokens": total_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )
            log_path = self._save_log(task)

        # Only add sub_llm tokens at the top level to avoid double-counting across recursive depths
        if is_top_level:
            total_tokens += self._sub_llm_total_tokens
            input_tokens += self._sub_llm_input_tokens
            output_tokens += self._sub_llm_output_tokens

        return RLMResult(
            response=final_output if final_output is not None else "No output",
            iterations=len([h for h in history if h["role"] == "assistant"]),
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            history=history,
            log_file_path=str(log_path) if log_path else None,
        )

"""
minrlm - Minimal Recursive Language Model
Based on https://arxiv.org/abs/2512.24601
Implemented by Avi Lumelsky
"""

import asyncio
import json
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
    format_continue_prompt,
    format_system_prompt,
    format_user_prompt,
)

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


class ProtectedNamespace(dict):
    """Dict that prevents reassignment of protected keys (like input_0)."""
    
    PROTECTED = {"input_0", "input_1", "input_2"}  # Context variables are protected
    
    def __setitem__(self, key: str, value: Any) -> None:
        if key in self.PROTECTED and key in self:
            raise NameError(f"Cannot reassign '{key}' - it already contains your data. Use it directly.")
        super().__setitem__(key, value)


class PythonREPL:
    """Persistent Python REPL with sub_llm() and FINAL() support."""

    HIDDEN_KEYS = {"__builtins__", "__name__", "sub_llm", "sub_llm_batch", "FINAL", "FINAL_var", "peek", "search"}

    def __init__(self, sub_llm_callback: Callable | None = None, sub_llm_batch_callback: Callable | None = None):
        self._output: str | None = None
        self._data_accessed = False  # Track if search() or input_0 was used
        self._namespace = ProtectedNamespace({
            "__builtins__": __builtins__,
            "__name__": "__main__",  # So `if __name__ == "__main__":` works
            "sub_llm": self._make_sub_llm(sub_llm_callback),
            "sub_llm_batch": self._make_sub_llm_batch(sub_llm_batch_callback),
            "FINAL": self._set_output,
            "FINAL_var": self._set_output_var,
            "peek": self._peek,
            "search": self._search,
        })

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
        
        # Enforce data grounding: if input_0 exists, must access data first
        if "input_0" in self._namespace and not self._data_accessed:
            raise ValueError("You must call search(input_0, 'keyword') first to find the data. Don't guess - search!")
        
        self._output = str(value).strip()
        
        # Clean common artifacts from search() tuple returns
        # e.g., "['New York']" -> "New York", "[]" -> ""
        if self._output.startswith("[") and self._output.endswith("]"):
            inner = self._output[1:-1].strip()
            # Handle ['value'] or ["value"]
            if (inner.startswith("'") and inner.endswith("'")) or \
               (inner.startswith('"') and inner.endswith('"')):
                inner = inner[1:-1]
            self._output = inner
            if inner:
                print(f"ℹ️ Cleaned output: '{self._output}'")
        
        # Reject empty string after cleaning
        if self._output == "":
            raise ValueError("FINAL() called with empty string - provide a non-empty answer")

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
            ctx_before = text[max(0, pos - context):pos]
            ctx_after = text[end:min(len(text), end + context)]
            matches.append((actual_match, ctx_before, ctx_after))

            print(f"\n[Match {len(matches)}]: {actual_match}")
            print(f"<before>{ctx_before}</before>")
            print(f"<match>{actual_match}</match>")
            print(f"<after>{ctx_after}</after>")
            start = pos + 1

        if not matches:
            print(f"⚠️ NO MATCHES for '{pattern}'")
            print(f"   -> Try shorter/partial pattern")
        else:
            print(f"\n✓ Found {len(matches)} match(es)")

        return matches

    def _set_output_var(self, var_name: str) -> None:
        """Set output from a variable in the namespace (FINAL_VAR from paper)."""
        # Enforce data grounding: if input_0 exists, must access data first
        if "input_0" in self._namespace and not self._data_accessed:
            raise ValueError("You must call search(input_0, 'keyword') first to find the data. Don't guess - search!")
        
        if var_name not in self._namespace:
            raise NameError(f"Variable '{var_name}' not found in REPL")
        
        value = self._namespace[var_name]
        if value is None:
            raise ValueError(f"Variable '{var_name}' is None - provide a non-empty value")
        
        self._output = str(value).strip()
        if self._output == "":
            raise ValueError(f"Variable '{var_name}' contains empty string - provide a non-empty value")

    def _peek(self, data: Any, max_len: int = 500, max_items: int = 5, depth: int = 0) -> str:
        """Efficient preview of data - truncates large strings/lists, recurses into structures."""
        indent = "  " * depth

        if isinstance(data, str):
            if len(data) <= max_len:
                preview = repr(data)
                print(f"{indent}{preview}")
            else:
                # Show start, middle, AND end to reveal patterns throughout
                chunk_size = max_len // 3
                start = data[:chunk_size]
                mid_pos = len(data) // 2
                middle = data[mid_pos : mid_pos + chunk_size]
                end = data[-chunk_size:]
                print(f"{indent}Start: {repr(start)}...")
                print(f"{indent}Middle ({mid_pos:,}): {repr(middle)}...")
                print(f"{indent}End ({len(data) - chunk_size:,}): {repr(end)}")
                print(f"{indent}({len(data):,} chars total)")
                preview = f"str[{len(data):,}]"
            return preview

        elif isinstance(data, list | tuple):
            bracket = "[]" if isinstance(data, list) else "()"
            if len(data) == 0:
                print(f"{indent}{bracket}")
                return bracket
            print(f"{indent}{bracket[0]}  # {len(data)} items")
            for item in data[:max_items]:
                self._peek(item, max_len, max_items, depth + 1)
            if len(data) > max_items:
                print(f"{indent}  ... and {len(data) - max_items} more")
            return f"{bracket} ({len(data)} items)"

        elif isinstance(data, dict):
            if len(data) == 0:
                print(f"{indent}{{}}")
                return "{}"
            print(f"{indent}{{  # {len(data)} keys")
            for k, v in list(data.items())[:max_items]:
                print(f"{indent}  {repr(k)}:")
                self._peek(v, max_len, max_items, depth + 2)
            if len(data) > max_items:
                print(f"{indent}  ... and {len(data) - max_items} more keys")
            return f"{{}} ({len(data)} keys)"

        else:
            preview = repr(data)
            if len(preview) > max_len:
                preview = preview[:max_len] + "..."
            print(f"{indent}{preview}")
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
        if "input_0" in self._namespace and not self._data_accessed:
            import re
            # Look for common patterns that indicate input_0 is being used
            has_input_0 = 'input_0' in code
            has_regex = bool(re.search(r're\.(findall|search|finditer|match|fullmatch)', code))
            patterns = [
                r'json\.loads\s*\(\s*input_0',  # json.loads(input_0)
                r'input_0\s*\[',  # input_0[...]
                r'input_0\s*\.',  # input_0.method()
                r'=\s*input_0',  # var = input_0
            ]
            # If input_0 is used AND (regex function is called OR direct access pattern matches)
            if has_input_0 and (has_regex or any(re.search(p, code) for p in patterns)):
                self._data_accessed = True

        result: dict[str, Any] = {"stdout": "", "output": None, "error": None}
        try:
            exec(code, self._namespace)
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

    def set_variable(self, name: str, value: Any) -> None:
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
        self._namespace = {k: v for k, v in self._namespace.items() if k in keys_to_keep}

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

    Usage:
        rlm = RLM(model="gpt-5-nano")
        result = rlm.completion("Solve this problem...")
        print(result.response)

        # With Docker sandboxing (requires Docker):
        rlm = RLM(model="gpt-5-nano", use_docker=True)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",  # Default to non-reasoning model for cost efficiency
        api_key: str | None = None,
        base_url: str | None = None,
        max_iterations: int = 6,  # Reduced from 20 - force early commitment
        max_time_seconds: int = DEFAULT_MAX_TIME_SECONDS,  # Timeout per completion
        max_output_tokens: int | None = 1500,  # Reduced from 2000 - less verbose code
        temperature: float = 0.0,  # Use 0 for deterministic code generation
        reasoning_effort: str = "low",  # For reasoning models: "low", "medium", "high"
        log_dir: str | None = None,
        async_batch: bool = True,  # Enable parallel sub_llm_batch calls
        on_step: Callable[[str, dict], None] | None = None,  # Callback for streaming steps
        # Docker options
        use_docker: bool = False,  # Run code in Docker container
        docker_image: str = "python:3.11-slim",
        docker_memory: str = "256m",
        docker_timeout: int = 60,
    ):
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
            if self._is_valid_python(code):
                return code
        # Try ```py
        matches = re.findall(r"```py\s*\n(.*?)```", text, re.DOTALL)
        if matches:
            code = "\n\n".join(matches)
            if self._is_valid_python(code):
                return code
        # Try generic ``` blocks
        matches = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
        if matches:
            code = "\n\n".join(matches)
            if self._is_valid_python(code):
                return code
        # Fallback: if response looks like pure code (starts with import/assignment)
        text = text.strip()
        if text.startswith("import ") or text.startswith("from ") or re.match(r"^[a-z_][a-z0-9_]*\s*=", text):
            if self._is_valid_python(text):
                return text
        return None

    def _is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python syntax."""
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError:
            return False

    def _call_llm(self, messages: list[dict[str, str]]) -> tuple[str, int, int, int]:
        """Call LLM, return (response_text, total_tokens, input_tokens, output_tokens)."""
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

        # Limit output tokens to reduce latency (skip for reasoning models)
        if self.max_output_tokens and not is_reasoning_model:
            kwargs["max_tokens"] = self.max_output_tokens

        # Use reasoning_effort for reasoning models to control token cost
        if is_reasoning_model and self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        resp = self.client.chat.completions.create(**kwargs)
        usage = resp.usage
        return (
            resp.choices[0].message.content or "",
            usage.total_tokens if usage else 0,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Check if model is a reasoning model that uses hidden chain-of-thought.
        
        TODO: Keep this list updated as OpenAI releases new reasoning models.
        These models have internal reasoning tokens that are billed but not visible.
        """
        model_lower = model.lower()
        reasoning_patterns = [
            "o1", "o3",  # o1, o1-mini, o1-preview, o3, o3-mini
            "gpt-5",     # gpt-5, gpt-5-nano, gpt-5-mini
        ]
        return any(pattern in model_lower for pattern in reasoning_patterns)

    def _handle_sub_llm(self, task: str, context: str = "") -> str:
        """Handle recursive sub_llm() calls, preserving parent's output state."""
        # Save parent's output state to prevent contamination
        saved_output = self._repl.save_output() if self._repl else None

        self._depth += 1
        try:
            result = self.completion(task, context).response
            return result
        finally:
            self._depth -= 1
            # Restore parent's output state
            if self._repl:
                self._repl.restore_output(saved_output)

    def _handle_sub_llm_batch(self, tasks: list[tuple[str, str]]) -> list[str]:
        """Handle parallel sub_llm calls. tasks = [(task, context), ...]"""
        if not tasks:
            return []

        if self.async_batch and self.async_client:
            return asyncio.run(self._run_batch_async(tasks))
        else:
            # Fallback: sequential
            return [self._handle_sub_llm(task, ctx) for task, ctx in tasks]

    async def _run_batch_async(self, tasks: list[tuple[str, str]]) -> list[str]:
        """Run multiple completions in parallel using async."""
        if self.async_client is None:
            raise RuntimeError("Async client not initialized")

        async_client = self.async_client  # Local ref for closure

        async def run_one(task: str, context: str) -> str:
            messages: list[dict[str, str]] = [
                {"role": "system", "content": format_system_prompt(context)},
                {"role": "user", "content": format_user_prompt(task, context)},
            ]
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }
            if "gpt-5" not in self.model.lower():
                kwargs["temperature"] = 0.7

            resp = await async_client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""

            # Simple extraction: if FINAL pattern found, extract value
            match = re.search(r"FINAL\(['\"](.+?)['\"]\)", text)
            if match:
                return match.group(1)
            # Or code block with FINAL
            code_match = re.search(r'```python.*?FINAL\([\'"](.+?)[\'"]\).*?```', text, re.DOTALL)
            if code_match:
                return code_match.group(1)
            # Fallback: return the raw response
            return text.strip()

        results = await asyncio.gather(*[run_one(task, ctx) for task, ctx in tasks])
        self._log("batch_call", {"count": len(tasks), "tasks": [t[0][:50] for t in tasks]})
        return list(results)

    def _log(self, event_type: str, data: dict[str, Any]) -> None:
        """Append to log entries."""
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

        # Set input_0 for this completion (both top-level and sub_llm calls)
        peek_output = ""
        if context:
            self._repl.set_variable("input_0", context)
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
                {"iteration": iteration + 1, "tokens": tok_total, "input_tokens": tok_in, "output_tokens": tok_out},
            )

            code = self._extract_code(response_text)

            # Always log the raw response for debugging
            self._log("llm_response", {"response": response_text[:2000]})

            if self.on_step:
                self.on_step(
                    "llm_response",
                    {"iteration": iteration + 1, "response": response_text, "has_code": code is not None},
                )

            if not code:
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

            if self.on_step:
                self.on_step("executing", {"iteration": iteration + 1, "code": code})  # Full code for debugging

            result = self._repl.execute(code)
            self._log("code_exec", {"code": code[:500], "output": result.get("output"), "error": result.get("error"), "state": result.get("state")})

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

            # Break if FINAL() was called
            if result.get("output") is not None:
                final_output = result["output"]
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
            self._save_log(task)

        return RLMResult(
            response=final_output if final_output is not None else "No output",
            iterations=len([h for h in history if h["role"] == "assistant"]),
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            history=history,
        )

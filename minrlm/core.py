"""
minrlm - Minimal Recursive Language Model
Based on https://arxiv.org/abs/2512.24601
Implemented by Avi Lumelsky
"""

import asyncio
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAI

from .prompts import format_continue_prompt, format_system_prompt, format_user_prompt


@dataclass
class RLMResult:
    """Result of an RLM completion."""

    response: str
    iterations: int
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    history: list[dict[str, str]] = field(default_factory=list)


class PythonREPL:
    """Persistent Python REPL with sub_llm() and set_output() support."""

    HIDDEN_KEYS = {"__builtins__", "sub_llm", "sub_llm_batch", "set_output", "set_output_var", "peek", "search"}

    def __init__(self, sub_llm_callback: Callable | None = None, sub_llm_batch_callback: Callable | None = None):
        self._output: str | None = None
        self._namespace = {
            "__builtins__": __builtins__,
            "sub_llm": self._make_sub_llm(sub_llm_callback),
            "sub_llm_batch": self._make_sub_llm_batch(sub_llm_batch_callback),
            "set_output": self._set_output,
            "set_output_var": self._set_output_var,
            "peek": self._peek,
            "search": self._search,
        }

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
        self._output = str(value)

    def _search(self, text: str, pattern: str, context: int = 100) -> list[str]:
        """Search for pattern in text and return all matches with surrounding context.

        Args:
            text: The text to search in
            pattern: The pattern to find (case-insensitive)
            context: Number of characters to show around each match

        Returns:
            List of matches with context, prints them too
        """
        matches = []
        text_lower = text.lower()
        pattern_lower = pattern.lower()
        start = 0

        while True:
            pos = text_lower.find(pattern_lower, start)
            if pos == -1:
                break
            # Extract match with context
            ctx_start = max(0, pos - context)
            ctx_end = min(len(text), pos + len(pattern) + context)
            match_text = text[ctx_start:ctx_end]
            matches.append(match_text)
            print(f"[Match at {pos}]: ...{match_text}...")
            start = pos + 1

        if not matches:
            print(f"No matches found for '{pattern}'")
        else:
            print(f"\nFound {len(matches)} match(es) for '{pattern}'")

        return matches

    def _set_output_var(self, var_name: str) -> None:
        """Set output from a variable in the namespace (FINAL_VAR from paper)."""
        if var_name not in self._namespace:
            raise NameError(f"Variable '{var_name}' not found in REPL")
        value = self._namespace[var_name]
        self._output = str(value)

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

        result: dict[str, Any] = {"stdout": "", "output": None, "error": None}
        try:
            exec(code, self._namespace)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
        finally:
            sys.stdout = old_stdout
            result["stdout"] = stdout_capture.getvalue()
            # ALWAYS capture output - even if error occurred AFTER set_output() was called
            result["output"] = self._output

        # Include current state
        result["state"] = self.get_state()
        return result

    def set_variable(self, name: str, value: Any) -> None:
        self._namespace[name] = value

    def reset(self) -> None:
        """Reset REPL state for a new completion while preserving callbacks."""
        self._output = None
        # Clear user variables, keep built-ins and special functions
        keys_to_keep = set(self.HIDDEN_KEYS) | {
            "sub_llm",
            "sub_llm_batch",
            "set_output",
            "set_output_var",
            "__builtins__",
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
    """

    def __init__(
        self,
        model: str = "gpt-5-nano",
        api_key: str | None = None,
        base_url: str | None = None,
        max_iterations: int = 20,
        max_output_tokens: int | None = 2000,  # Limit output for speed (None = no limit)
        temperature: float = 0.0,  # Use 0 for deterministic code generation
        log_dir: str | None = None,
        async_batch: bool = True,  # Enable parallel sub_llm_batch calls
        on_step: Callable[[str, dict], None] | None = None,  # Callback for streaming steps
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.log_dir = Path(log_dir) if log_dir else None
        self.async_batch = async_batch
        self.on_step = on_step

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url) if async_batch else None
        self._depth = 0
        self._log_entries: list[dict[str, Any]] = []

        # Initialize REPL once at startup (not per-query)
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

        # Set temperature (gpt-5 models don't support temperature=0)
        if self.temperature is not None and "gpt-5" not in self.model.lower():
            kwargs["temperature"] = self.temperature

        # Limit output tokens to reduce latency (skip for gpt-5 models which may have issues)
        if self.max_output_tokens and "gpt-5" not in self.model.lower():
            kwargs["max_tokens"] = self.max_output_tokens

        resp = self.client.chat.completions.create(**kwargs)
        usage = resp.usage
        return (
            resp.choices[0].message.content or "",
            usage.total_tokens if usage else 0,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

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

            # Simple extraction: if set_output pattern found, extract value
            # For batch calls, we expect simple direct answers
            match = re.search(r"set_output\(['\"](.+?)['\"]\)", text)
            if match:
                return match.group(1)
            # Or code block with set_output
            code_match = re.search(r'```python.*?set_output\([\'"](.+?)[\'"]\).*?```', text, re.DOTALL)
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

        for iteration in range(self.max_iterations):
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
                    {"iteration": iteration + 1, "response": response_text[:500], "has_code": code is not None},
                )

            if not code:
                # No code found - log for debugging and prompt to continue
                self._log("no_code", {"response_preview": response_text[:500]})
                messages += [
                    {"role": "assistant", "content": response_text},
                    {
                        "role": "user",
                        "content": "Write Python code in a ```python block. Call set_output() with your answer.",
                    },
                ]
                continue

            if self.on_step:
                self.on_step("executing", {"iteration": iteration + 1, "code": code[:300]})

            result = self._repl.execute(code)
            self._log("code_exec", {"code": code[:500], "output": result.get("output"), "state": result.get("state")})

            if self.on_step:
                self.on_step(
                    "executed",
                    {
                        "iteration": iteration + 1,
                        "stdout": result.get("stdout", "")[:200],
                        "output": result.get("output"),
                        "error": result.get("error"),
                    },
                )

            # Only break if we have actual output (not empty string)
            if result.get("output"):
                final_output = result["output"]
                break

            # Pass state to continue prompt
            messages += [
                {"role": "assistant", "content": response_text},
                {
                    "role": "user",
                    "content": format_continue_prompt(
                        result.get("stdout", ""), result.get("error", ""), result.get("state", {})
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
            response=final_output or "No output",
            iterations=len([h for h in history if h["role"] == "assistant"]),
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            history=history,
        )

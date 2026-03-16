"""
Reasoning-enhanced RLM - wrapper around existing RLM with modified prompts.
"""

import re
from dataclasses import dataclass

from .core import RLM, RLMResult
from .prompts_reasoning import (
    compute_context_preview,
    compute_entropy_profile,
    format_continue_prompt_reasoning,
    format_system_prompt_reasoning,
    format_user_prompt_reasoning,
)


@dataclass
class RLMReasoningResult(RLMResult):
    """Result with reasoning captured."""

    reasoning: str = ""


class RLMReasoning(RLM):
    """RLM with reasoning step before code generation."""

    def __init__(self, *args, system_prompt=None, **kwargs):
        # Extract system_prompt before passing to parent
        self._custom_system_prompt = system_prompt
        super().__init__(*args, **kwargs)
        self._reasoning = ""

    def completion(self, task: str, context: str = "") -> RLMReasoningResult:
        """
        Complete a task using reasoning-first approach.

        Override to use reasoning prompts and extract reasoning.
        """
        # Build initial messages with reasoning prompt
        # Use custom prompt if provided, otherwise use default
        if self._custom_system_prompt:
            # Format custom prompt with context metadata + entropy profile
            if context:
                meta = f"string with {len(context)} chars"
                if "\n" in context:
                    lines = context.count("\n") + 1
                    meta += f", ~{lines} lines"
                preview = compute_context_preview(context)
                if preview:
                    meta += f"\n\n<context_preview>\n{preview}\n</context_preview>"
                profile = compute_entropy_profile(context)
                if profile:
                    meta += f"\n\n<entropy_map>\n{profile}\n</entropy_map>"
                system_prompt = self._custom_system_prompt.replace(
                    "{context_meta}", meta
                )
            else:
                system_prompt = self._custom_system_prompt.replace(
                    "{context_meta}", "string"
                )
        else:
            system_prompt = format_system_prompt_reasoning(
                context=context, context_type="string"
            )
        user_prompt = format_user_prompt_reasoning(task=task)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Reset state
        self._depth = 0
        self._log_entries = []
        self._sub_llm_calls = 0
        self._sub_llm_total_tokens = 0
        self._sub_llm_input_tokens = 0
        self._sub_llm_output_tokens = 0
        self._reasoning = ""
        self._repl.reset()

        # Set context in REPL (allow override for reuse)
        self._repl.set_variable("task_0", task, allow_override=True)
        self._repl.set_variable("input_0", context or "", allow_override=True)

        # Log start with full task and context for debugging
        start_data = {
            "task": task,  # Full task, not truncated
            "model": self.model,
            "context_length": len(context) if context else 0,
        }
        # Optionally include first 2000 chars of context for debugging
        if context:
            start_data["context_preview"] = (
                context[:2000] if len(context) > 2000 else context
            )
        self._log("start", start_data)

        # Iteration loop
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        iteration = 0

        import time

        start_time = time.time()

        for iteration in range(1, self.max_iterations + 1):
            # Check timeout
            if time.time() - start_time > self.max_time_seconds:
                self._log("timeout", {"iterations": iteration})
                if self.on_step:
                    self.on_step(
                        "timeout",
                        {"elapsed": time.time() - start_time, "iteration": iteration},
                    )
                break

            if self.on_step:
                self.on_step("thinking", {"iteration": iteration})

            # Call LLM - log full messages for debugging
            llm_call_data = {
                "iteration": iteration,
                "messages": messages,  # Full messages array including all prompts
            }
            self._log("llm_call", llm_call_data)
            response_text, t_total, t_in, t_out = self._call_llm(messages)
            total_tokens += t_total
            input_tokens += t_in
            output_tokens += t_out

            # Extract reasoning on first iteration
            if iteration == 1:
                # Look for # REASONING: comment in the code
                # Extract from code block if present
                code_block_match = re.search(
                    r"```(?:python)?\s*\n(.*?)```", response_text, re.DOTALL
                )
                if code_block_match:
                    code_content = code_block_match.group(1)
                    # Look for # REASONING: at the start of the code
                    reasoning_match = re.match(
                        r"#\s*REASONING:\s*(.+?)(?:\n|$)", code_content, re.IGNORECASE
                    )
                    if reasoning_match:
                        self._reasoning = reasoning_match.group(1).strip()
                        self._log(
                            "reasoning_extracted", {"reasoning": self._reasoning[:500]}
                        )
                        if self.on_step:
                            self.on_step("reasoning", {"reasoning": self._reasoning})

            # Log response - full response for debugging (not truncated)
            code = self._extract_code(response_text)
            self._log(
                "llm_response",
                {
                    "response": response_text,  # Full response, not truncated
                    "response_length": len(response_text),
                },
            )
            if self.on_step:
                self.on_step(
                    "llm_response",
                    {
                        "iteration": iteration,
                        "response": response_text,
                        "has_code": code is not None,
                    },
                )

            # Add assistant response to messages
            messages.append({"role": "assistant", "content": response_text})

            if not code:
                # Try fallback: accept raw Python code without code fences
                # (Some models like gpt-5-mini generate code directly without ```python markers)
                cleaned = response_text.strip()
                # Strip any leftover code fence markers that _extract_code failed to parse
                if cleaned.startswith("```python"):
                    cleaned = cleaned[len("```python"):].strip()
                elif cleaned.startswith("```py"):
                    cleaned = cleaned[len("```py"):].strip()
                elif cleaned.startswith("```"):
                    cleaned = cleaned[3:].strip()
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()
                if cleaned and "FINAL(" in cleaned:
                    # Response contains FINAL() call - treat as raw Python code
                    code = cleaned
                    self._log(
                        "code_auto_wrapped",
                        {"note": "Accepted raw Python without code fences"},
                    )

                if not code:
                    # Still no code - return error
                    self._log("no_code", {"response": response_text})
                    error_msg = (
                        "ERROR: LLM response did not contain executable Python code"
                    )
                    if len(response_text) > 1900:  # Likely truncated
                        error_msg += f" (response may be truncated at {len(response_text)} chars)"
                    return RLMReasoningResult(
                        response=error_msg,
                        iterations=iteration,
                        total_tokens=total_tokens + self._sub_llm_total_tokens,
                        input_tokens=input_tokens + self._sub_llm_input_tokens,
                        output_tokens=output_tokens + self._sub_llm_output_tokens,
                        history=messages,
                        reasoning=self._reasoning,
                    )

            # Execute code - log full code for debugging (not truncated)
            self._log("code_exec", {"code": code, "code_length": len(code)})
            if self.on_step:
                self.on_step("executing", {"iteration": iteration, "code": code})
            result = self._repl.execute(code)

            if self.on_step:
                self.on_step(
                    "executed",
                    {
                        "iteration": iteration,
                        "stdout": result.get("stdout", ""),
                        "error": result.get("error"),
                        "output": result.get("output"),
                    },
                )

            # Check if we got final answer
            if result.get("output") is not None:
                response = result["output"]
                self._log(
                    "end",
                    {
                        "response": response,
                        "total_tokens": total_tokens,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                )

                # Save log if configured
                log_path = None
                if self.log_dir:
                    log_path = self._save_log(task)

                return RLMReasoningResult(
                    response=response,
                    iterations=iteration,
                    total_tokens=total_tokens + self._sub_llm_total_tokens,
                    input_tokens=input_tokens + self._sub_llm_input_tokens,
                    output_tokens=output_tokens + self._sub_llm_output_tokens,
                    history=messages,
                    reasoning=self._reasoning,
                    log_file_path=str(log_path) if log_path else None,
                )

            # Continue with state
            continue_prompt = format_continue_prompt_reasoning(
                output=result.get("stdout", ""),
                error=result.get("error", ""),
                state=result.get("state", {}),
                reasoning_summary=self._reasoning[:200] if self._reasoning else "",
                iteration=iteration,
                max_iterations=self.max_iterations,
            )

            messages.append({"role": "user", "content": continue_prompt})

        # Max iterations reached
        self._log(
            "max_iterations",
            {
                "iterations": self.max_iterations,
                "total_tokens": total_tokens,
            },
        )

        # Return last stdout or error message (never empty)
        last_stdout = ""
        if messages and messages[-1]["role"] == "user":
            # Extract stdout from last continue prompt
            content = messages[-1]["content"]
            match = re.search(r"stdout: (.*?)(?:\n\nVariables:|$)", content, re.DOTALL)
            if match:
                last_stdout = match.group(1).strip()

        # If still empty, provide error message
        if not last_stdout:
            last_stdout = f"ERROR: Max iterations ({self.max_iterations}) reached without calling FINAL()"

        log_path = None
        if self.log_dir:
            log_path = self._save_log(task)

        return RLMReasoningResult(
            response=last_stdout,
            iterations=self.max_iterations,
            total_tokens=total_tokens + self._sub_llm_total_tokens,
            input_tokens=input_tokens + self._sub_llm_input_tokens,
            output_tokens=output_tokens + self._sub_llm_output_tokens,
            history=messages,
            reasoning=self._reasoning,
            log_file_path=str(log_path) if log_path else None,
        )

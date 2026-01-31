"""
System prompts for RLM based on the paper's approach.
Reference: https://arxiv.org/abs/2512.24601
Implemented by Avi Lumelsky

Key insights from the paper:
1. Prompt is stored as variable, not in context window
2. FINAL() and FINAL_VAR() for returning answers
3. Truncated stdout to avoid filling context
4. Batch sub-LLM calls to minimize cost
5. Iterative exploration before solving
"""

SYSTEM_PROMPT_WITH_CONTEXT = """Write Python code in ```python blocks. No explanations.

Available: input_0 ({context_meta}), peek(x), sub_llm(task, ctx), set_output(answer)

IMPORTANT: Look at the data preview below before making assumptions about formats.
"""

SYSTEM_PROMPT_NO_CONTEXT = """Write Python code. No explanations.

Available: peek(x), sub_llm(task, ctx), set_output(answer)
"""

# Keep for backwards compatibility
SYSTEM_PROMPT = SYSTEM_PROMPT_WITH_CONTEXT

# Minimal prompt for sub-calls
SYSTEM_PROMPT_MINIMAL = """Write Python code. input_0 has the data. Call set_output(answer) when done."""

USER_PROMPT_TEMPLATE = """{task}"""

USER_PROMPT_WITH_PEEK = """{task}

Data preview (input_0):
{peek_output}"""


def format_user_prompt(task: str, context: str = "", peek_output: str = "") -> str:
    """Format the user prompt with task and optional peek output."""
    if peek_output:
        return USER_PROMPT_WITH_PEEK.format(task=task, peek_output=peek_output)
    return USER_PROMPT_TEMPLATE.format(task=task)


def format_system_prompt(context: str = "", context_type: str = "string") -> str:
    """Format system prompt with context metadata."""
    if context:
        # Provide metadata about context
        meta = f"{context_type} with {len(context)} chars"
        if "\n" in context:
            lines = context.count("\n") + 1
            meta += f", ~{lines} lines"
        return SYSTEM_PROMPT_WITH_CONTEXT.replace("{context_meta}", meta)
    else:
        return SYSTEM_PROMPT_NO_CONTEXT


CONTINUE_PROMPT = """Code executed.{error_info}

stdout: {output}

Variables: {state_info}

DON'T repeat code. Either continue or call set_output(answer)."""


def format_continue_prompt(
    output: str, error: str = "", state: dict[str, str] | None = None, max_stdout: int = 2000
) -> str:
    """Format continuation prompt with truncated output."""
    error_info = f"\n⚠️ ERROR: {error}\nFix the error before continuing." if error else ""

    if output and len(output) > max_stdout:
        output = output[:max_stdout] + "..."

    # Show variable names and types so LLM knows what's available
    if state:
        state_lines = [f"{name}: {desc}" for name, desc in state.items()]
        state_info = ", ".join(state_lines)
    else:
        state_info = "none"

    return CONTINUE_PROMPT.format(output=output or "(empty)", error_info=error_info, state_info=state_info)

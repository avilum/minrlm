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

from typing import Any

SYSTEM_PROMPT_WITH_CONTEXT = """Write Python code in ```python blocks. No explanations.

input_0 = {context_meta}

# Important
You MUST access input_0 to find the answer. You CANNOT answer without examining the data first.

RULES:
1. For structured data (JSON, etc.): Parse it directly (e.g., import json; data = json.loads(input_0))
2. For unstructured text: Use search(input_0, "keyword") to find relevant sections
3. Build the context you need - do multiple searches or parse different parts if needed
4. You MUST extract the answer from the data (parsed JSON, search results, or input_0)
5. You CANNOT call FINAL() with a hardcoded guess - the answer MUST come from the data
6. If you try to call FINAL() without accessing input_0 first, it will be rejected
`
Required workflow examples:

For structured data (JSON, CSV, etc.):
```python
# Step 1: Parse the structured data
import json
data = json.loads(input_0)

# Step 2: Filter/aggregate based on the question
# Example: Count items matching certain criteria
matching = [item for item in data["items"]
            if item["category"] == "target_category" and item["status"] == "target_status"]
answer = len(matching)  # or sum(item["value"] for item in matching)

# Step 3: Return the answer
FINAL(answer)
```

For unstructured text (code, documents, etc.):

Option A: Using search() for keyword-based lookup:
```python
# Step 1: Search for the relevant data
results = search(input_0, "keyword_from_question")

# Step 2: Parse the results - IMPORTANT: 'match' is often incomplete!
# The actual data is usually in 'after' (or sometimes 'before')
answer = None
import re
for match, before, after in results:
    # CRITICAL: 'match' is just the literal matched text (often incomplete)
    # The full pattern you need is usually in 'after' or 'before'
    # Option 1: Search in 'after' (most common)
    m = re.search(r'pattern', after)
    # Option 2: Combine all fields for full context
    full = before + match + after
    m = re.search(r'pattern', full)
    if m:
        answer = m.group(1)
        break

# Step 3: Return the parsed answer
FINAL(answer)
```

Option B: Using regex directly on input_0 (better for pattern matching):
```python
# For well-defined patterns (e.g., "[ID: CODE-123]" or "Name: value"), use regex directly
import re
# Find all matches at once
matches = re.findall(r'\\[ID: ([A-Z0-9-]+)\\]', input_0)
# Or find a specific pattern with context
m = re.search(r'EntityName.*?Location:\\s*(\\w+)', input_0)
if m:
    answer = m.group(1)
FINAL(answer)
```

# For extracting pairs (multiple capture groups):
```python
# Example: Extract key-value pairs from structured text
import re
# Pattern: "[TAG CODE-123]: Description text."
pairs = re.findall(r'\\[TAG ([A-Z0-9-]+)\\]: ([^.]+)\\.', input_0)
# pairs = [('CODE-123', 'description1'), ('CODE-456', 'description2'), ...]
answer = ", ".join([f"{key}={value}" for key, value in pairs])
FINAL(answer)
```

# For processing search() results with nested comprehensions:
```python
# When you need to extract multiple items from each search result
results = search(input_0, "keyword")
import re
# Extract all matches from each result's 'after' field
all_matches = [m.group(1) for match, before, after in results
               for m in re.finditer(r'pattern', after)]
# Or extract pairs from each result
pairs = [(m.group(1), m.group(2)) for match, before, after in results
         for m in re.finditer(r'pattern1(.*?)pattern2(.*?)', after)]
FINAL(", ".join(all_matches))
```

Choose Option B when:
- You know the exact pattern format (e.g., "[ID: CODE-123]" or structured tags)
- You need to find multiple occurrences
- The pattern is well-defined and consistent

Choose Option A when:
- You need to find a keyword first, then extract nearby data
- The pattern location is uncertain
- You need context around the match

IMPORTANT: Choose the right approach based on data structure:
- For structured data (JSON, CSV, etc.): Parse it directly (json.loads(), csv.reader(), etc.)
- For pattern matching (e.g., structured tags or formatted codes): Use regex directly on input_0 (re.findall(), re.search())
- For keyword-based lookup: Use search() to find relevant sections, then parse 'after' field
- Build the context you need - do multiple searches or parse different parts if needed
- Remember: search() 'match' field is often incomplete - check 'after' for full patterns!

Tools:
- search(input_0, "keyword") -> [(match, before, after), ...]
  Example output:
  results = search(input_0, "keyword")
  # results[0] = ("keyword", "500 chars before...", "keyword: value\n500 chars after...")
  # match = literal matched text (may be incomplete fragment)
  # before = 500 chars before match
  # after = 500 chars after match (often contains the full pattern you need)

  Usage: Check 'after' field for full patterns, or combine fields:
  for match, before, after in results:
      m = re.search(r'pattern', after)  # Check 'after' for full pattern
      # OR: full = before + match + after; m = re.search(r'pattern', full)

- peek(input_0) -> preview structure
- sub_llm("question", chunk) -> semantic analysis
- FINAL("answer") -> return answer (MUST be parsed from search() results or input_0, NOT a guess)

💡 TIP: For well-defined patterns, use regex directly on input_0 instead of search():
import re
matches = re.findall(r'\\[Tag: ([A-Z0-9-]+)\\]', input_0)  # Generic pattern example

Remember: search() FIRST (maybe multiple times to build context), parse SECOND, FINAL() LAST. No guessing.
"""
# TODO: retrun answer with FINAL only when sure. If there is doung - must be cvalidated.


SYSTEM_PROMPT_NO_CONTEXT = """Write Python code in ```python blocks. No explanations.

FINAL("answer") -> return answer
"""

# Keep for backwards compatibility
SYSTEM_PROMPT = SYSTEM_PROMPT_WITH_CONTEXT

# Minimal prompt for sub-calls
SYSTEM_PROMPT_MINIMAL = (
    """Write Python code. input_0 has the data. Call FINAL("answer") or FINAL_var("varname") when done."""
)

USER_PROMPT_TEMPLATE = """{task}"""

USER_PROMPT_WITH_PEEK = """{task}

Data preview (input_0):
{peek_output}"""


def format_user_prompt(task: str, context: str = "", peek_output: str = "") -> str:
    """Format the user prompt with task and optional peek output."""
    if peek_output:
        return USER_PROMPT_WITH_PEEK.format(task=task, peek_output=peek_output)
    return USER_PROMPT_TEMPLATE.format(task=task)


def format_system_prompt(context: str = "", context_type: str = "string", **kwargs: Any) -> str:
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

{iteration_info}Call FINAL("answer") or FINAL_var("varname") to finish."""


def format_continue_prompt(
    output: str,
    error: str = "",
    state: dict[str, str] | None = None,
    max_stdout: int = 2000,
    iteration: int = 1,
    max_iterations: int = 10,
) -> str:
    """Format continuation prompt with truncated output."""
    error_info = f"\n⚠️ ERROR: {error}\nFix the error and try again." if error else ""

    if output and len(output) > max_stdout:
        output = output[:max_stdout] + f"... (truncated, {len(output)} chars total)"

    # Show variable names and types so LLM knows what's available
    if state:
        state_lines = [f"{name}: {desc}" for name, desc in state.items()]
        state_info = ", ".join(state_lines)
    else:
        state_info = "none"

    # Add urgency based on iteration count
    remaining = max_iterations - iteration
    if remaining <= 2:
        iteration_info = f"⚠️ Final attempt ({remaining} left). "
    elif remaining <= 4:
        iteration_info = f"[{iteration}/{max_iterations}] "
    else:
        iteration_info = ""

    return CONTINUE_PROMPT.format(
        output=output or "(empty)",
        error_info=error_info,
        state_info=state_info,
        iteration_info=iteration_info,
    )

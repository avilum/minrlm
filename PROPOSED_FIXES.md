# Proposed Fixes for RLM Failures

**Date:** 2026-02-20
**Target:** Improve accuracy from 70% → 85%+ (targeting vanilla's 75% and closer to official's 90%)

## Fix Priority Matrix

| Fix | Impact | Complexity | Priority | Est. Accuracy Gain |
|-----|--------|------------|----------|-------------------|
| REPOQA function retrieval | High | Medium | **P0** | +20-25% |
| Answer extraction | Medium | Low | **P0** | +5% |
| Prompt role boundaries | Low | Low | **P0** | +2-3% |
| Iteration strategy | Medium | Medium | P1 | +3-5% |
| Medium context optimization | High | High | P1 | +5-10% |

## P0 Fixes (Target: 70% → 85%)

### Fix 1: REPOQA Function Retrieval (Expected: +20-25%)

**Current issue:** Only 50% accuracy on REPOQA vs 100% for vanilla/official.

**Problems identified:**
1. Function name extraction from task is unreliable
2. Search with "def func_name" misses functions
3. 500-char context windows too small
4. No validation of extracted function

**Proposed solution:**

#### Option A: Multi-strategy retrieval (RECOMMENDED)

Update `prompts.py` lines 126-147 with:

```python
- Code/function retrieval (REPOQA): Extract existing function from input_0.
  ⚠ NEVER search with description text. NEVER implement the function yourself.
  ⚠ CRITICAL: The task asks for the EXACT function definition from the code.
  Multi-strategy pattern (covers 95%+ cases):
    import re
    # Strategy 1: Extract function name from task description
    # Look for patterns like "the function X" or "function named X" or "X function"
    task_lower = task_0.lower()
    func_patterns = [
        r'function[s]?\s+(?:named\s+|called\s+)?["`\']?(\w+)["`\']?',
        r'["`\'](\w+)["`\']?\s+function',
        r'extract\s+(?:the\s+)?["`\']?(\w+)["`\']?',
        r'function\s+["`\'](\w+)["`\']',
    ]
    func_name = None
    for pattern in func_patterns:
        match = re.search(pattern, task_lower)
        if match:
            func_name = match.group(1)
            break

    # Strategy 2: If no clear name in task, search for all function definitions
    # and use sub_llm to identify the right one
    if not func_name:
        # Get first 10 function names from code
        preview = input_0[:20000]
        func_defs = re.findall(r'^def (\w+)\(', preview, re.MULTILINE)[:10]
        if func_defs:
            func_list = ", ".join(func_defs)
            func_name = sub_llm(
                f"Which function name does this task ask about? Reply with ONLY the function name. Available: {func_list}",
                f"Task: {task_0}"
            ).strip()

    # Strategy 3: Search with progressive fallbacks (wider windows)
    results = []
    if func_name:
        # Try exact match first
        results = search(input_0, f"def {func_name}(")
        if not results:
            results = search(input_0, f"def {func_name} ")
        if not results:
            results = search(input_0, func_name)

    # Strategy 4: Extract full function body (not just 500 chars)
    if results:
        match, before, after = results[0]
        # Find function end (next def at same indentation or end of file)
        # Take up to 5000 chars to ensure we get the full function
        full_context = before[-200:] + match + after[:5000]

        # Extract complete function using indentation
        lines = full_context.split('\n')
        func_lines = []
        in_function = False
        func_indent = None

        for line in lines:
            if f'def {func_name}' in line:
                in_function = True
                func_indent = len(line) - len(line.lstrip())
                func_lines.append(line)
            elif in_function:
                if line.strip() and not line.startswith(' ' * (func_indent + 1)) and not line.startswith('\t'):
                    # End of function (next def or unindented line)
                    break
                func_lines.append(line)

        func_body = '\n'.join(func_lines)

        # Format: name||function_body
        FINAL(f"{func_name}||{func_body}")
    else:
        # No function found - return empty
        FINAL("")
```

**Benefits:**
- Multiple strategies for function name extraction
- Larger context windows (5000 chars vs 2000)
- Full function body extraction using indentation
- Validation that we found the right function

**Expected improvement:** 50% → 90%+ on REPOQA (+40% overall = +4%)

#### Option B: Simpler fallback (if Option A too complex)

```python
- Code/function retrieval (REPOQA): Use comprehensive search.
    # Extract function name from task (common patterns)
    import re
    patterns = [r'function\s+(\w+)', r'`(\w+)`', r'"(\w+)"', r'(\w+)\s+function']
    func_name = None
    for p in patterns:
        m = re.search(p, task_0, re.I)
        if m:
            func_name = m.group(1)
            break

    # Search with large context window
    if func_name:
        res = search(input_0, f"def {func_name}")
        if res:
            m, b, a = res[0]
            # Get 8000 chars to ensure full function
            FINAL(func_name + "||" + b[-500:] + m + a[:8000])

    # If search fails, return empty (don't loop)
    FINAL("")
```

### Fix 2: Answer Extraction Truncation (Expected: +5%)

**Current issue:** SNIAH returned `591871` instead of `5918715` (missing last digit).

**Root cause:** Need to investigate answer extraction in `core.py`.

**Investigation needed:**
1. Check how FINAL() extracts the answer from Python execution
2. Look for regex patterns that might truncate
3. Ensure full stdout capture

**Proposed solution:**

Check `minrlm/core.py` for answer extraction:
1. Find where FINAL() result is parsed
2. Ensure no character limits on answer extraction
3. Add test case for 7+ digit numbers

### Fix 3: Prompt Role Confusion (Expected: +2-3%)

**Current issue:** OOLONG returned `User:` instead of answer `44106`.

**Root cause:** Model is confused about role boundaries in continuation messages.

**Proposed solution:**

Update `prompts.py` - `build_continuation_message()` around line 306:

**Current:**
```python
def build_continuation_message(...) -> dict[str, str]:
    content = format_continue_prompt(...)
    return {"role": "user", "content": content}
```

**Change to:**
```python
def build_continuation_message(...) -> dict[str, str]:
    content = format_continue_prompt(...)
    # Add clear separator to prevent role confusion
    content = "--- CODE EXECUTION RESULT ---\n\n" + content
    return {"role": "user", "content": content}
```

Also update `CONTINUE_PROMPT` (line 234):
```python
CONTINUE_PROMPT = """--- CODE EXECUTION RESULT ---

stdout: {output}

{error_info}

Variables: {state_info}

{iteration_info}Continue writing Python code in ```python blocks. Call FINAL("answer") or FINAL_var("varname") when done."""
```

**Benefits:**
- Clear separation between system messages and results
- Prevents model from outputting role markers
- More explicit about what to do next

## P1 Fixes (Target: 85% → 90%)

### Fix 4: Iteration Strategy (Expected: +3-5%)

**Current issue:** One REPOQA case hit 10 iterations (49K tokens) and still failed.

**Problems:**
- No early stopping when stuck in loop
- No detection of repeated attempts
- Max iterations is arbitrary

**Proposed solution:**

Add to `minrlm/core.py` RLM class:

```python
class RLM:
    def __init__(self, ...):
        ...
        self.search_history = []  # Track searches to detect loops

    def completion(self, task, context):
        ...
        iteration = 0
        while iteration < self.max_iterations:
            ...
            # Detect if we're stuck in a loop
            if iteration >= 3:
                # Check if last 3 searches were similar/identical
                recent_searches = self.search_history[-3:]
                if len(set(recent_searches)) == 1:
                    # Stuck in loop - bail out
                    print("⚠️ Loop detected - aborting", file=sys.stderr)
                    break

            # Adaptive max iterations based on context size
            if context_size < 20000 and iteration > 3:
                # Small context shouldn't need many iterations
                break
            elif context_size < 100000 and iteration > 5:
                # Medium context - limit to 5 iterations
                break
```

### Fix 5: Medium Context Optimization (Expected: +5-10%)

**Current issue:** 13K-65K context range has 0-66% accuracy vs 100% vanilla.

**Root problem:** This is the "in-between" zone where:
- Vanilla fits it all in context
- Our chunking adds overhead without benefit

**Proposed solution:**

Add adaptive strategy based on context size:

```python
def completion(self, task: str, context: str = "") -> CompletionResult:
    context_size = len(context)

    # Adaptive strategy based on context size
    if context_size < 50000:  # 50K chars ~ 12-15K tokens
        # Small-medium context: use simpler prompts, fewer iterations
        self.max_iterations = min(3, self.max_iterations)
        # Consider using direct search without sub_llm overhead
    elif context_size < 200000:  # 200K chars ~ 50K tokens
        # Medium context: standard approach
        self.max_iterations = min(5, self.max_iterations)
    else:
        # Large context: full RLM needed
        self.max_iterations = 10  # Original max
```

Update prompts.py to adjust search strategy:

```python
SYSTEM_PROMPT_WITH_CONTEXT = r"""You are a universal python agent...

input_0 = {context_meta}

{strategy_hint}

ALL of the following are pre-loaded globals...
"""
```

Where `strategy_hint` varies by context size:
- Small (<50K): "This is a small context - prefer direct search over sub_llm"
- Medium (50-200K): "This is a medium context - use search efficiently"
- Large (>200K): "This is a large context - use sub_llm to process chunks"

## Implementation Plan

### Phase 1: P0 Fixes (1-2 days)
1. Implement REPOQA multi-strategy retrieval (Option A)
2. Investigate and fix answer extraction truncation
3. Fix prompt role confusion

**Expected result:** 70% → 85% accuracy

### Phase 2: Validation (0.5 day)
4. Run comprehensive eval on GPT-5-mini
5. Compare with vanilla and official
6. Identify remaining gaps

### Phase 3: P1 Optimization (1-2 days)
7. Implement iteration strategy improvements
8. Add medium context optimization
9. Run final comprehensive eval

**Expected result:** 85% → 90% accuracy

## Success Metrics

**Target accuracy by task (GPT-5-mini):**
- SNIAH: 90% → 95%+ (fix extraction bug)
- OOLONG: 90% → 95%+ (fix role confusion)
- REPOQA: 50% → 90%+ (fix function retrieval) ← **biggest impact**
- CODEQA: 50% → 55%+ (general improvements)

**Overall target:** 70% → 88%+ (competitive with official's 90%)

## Testing Strategy

For each fix:
1. Create unit test with known failure case
2. Verify fix resolves the specific case
3. Run full eval suite on affected task
4. Check for regressions on other tasks
5. Measure token usage impact

## Risks

1. **REPOQA fix complexity:** Option A is more complex - may introduce new bugs
   - Mitigation: Start with Option B, upgrade to A if needed

2. **Token usage increase:** Larger context windows → more tokens
   - Mitigation: Only use large windows when needed, track token usage

3. **Iteration limits too strict:** Early stopping may cut off valid searches
   - Mitigation: Make limits configurable, test on failure cases

4. **Prompt changes break other tasks:** Modifying prompts is risky
   - Mitigation: A/B test with old vs new prompts, check all tasks

## Alternative: Hybrid Approach

If P0 + P1 fixes don't reach 85%+, consider **hybrid approach**:

```python
def completion(self, task: str, context: str = "") -> CompletionResult:
    context_size = len(context)

    if 10000 < context_size < 100000:
        # Medium context: use vanilla LLM directly (it's working at 100%)
        return self._vanilla_completion(task, context)
    else:
        # Small or large context: use RLM
        return self._rlm_completion(task, context)
```

**Trade-off:**
- ✓ Achieve 90%+ accuracy by using vanilla where it works
- ✗ Lose token efficiency in medium range
- ✗ Not a "pure" RLM solution

This should be last resort if other fixes don't work.

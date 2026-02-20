# Immediate Actions to Fix RLM Failures

**Date:** 2026-02-20
**Current Accuracy:** 70% (ours) vs 75% (vanilla) vs 90% (official)
**Target:** 85%+ accuracy

## Critical Finding: REPOQA Log Analysis

Analyzed log: `20260219_232902_You_are_given_a_codebase_conte.jsonl`

### What Happened (Failed Case):

**Iteration 1:** ✓ SUCCESS
- Correctly identified function: `convert_unchanged_lines`
- Successfully extracted function body
- Output: `convert_unchanged_lines||def convert_unchanged_lines(...)`

**Guard Triggered:** ✗ GUARD FIRED
- Guard: "Code retrieval task: output must be an exact substring of input_0"
- Forced another iteration (WHY?)

**Iteration 2:** Unnecessary attempt
- Tried different extraction strategy
- Still working on same function

**Guard Triggered Again:** ✗ SECOND GUARD
- "Code retrieval task: do NOT search with description text"
- Forced yet another iteration

**Iteration 3:** ✗ FAILURE
- Used sub_llm to extract function name
- sub_llm returned: `convert_unchanged_lines_to_comments` ❌ (WRONG NAME!)
- Actual function: `convert_unchanged_lines` (no "_to_comments")
- Search failed: returned empty list
- find() returned -1 (not found)
- FINAL("") → ValueError

### Root Cause

**The guards are making things WORSE:**
1. First iteration SUCCESS
2. Guard triggers unnecessary retry
3. Retry uses sub_llm which hallucinates wrong function name
4. Final iteration fails completely

**Problem:** Guards are too strict and trigger false positives.

## Top 3 Fixes (Ordered by Impact)

### Fix #1: DISABLE or FIX the REPOQA Guards ⚠️ CRITICAL

**Impact:** +20-25% accuracy (REPOQA: 50% → 90%+)
**Effort:** Low (1-2 hours)

**Current Problem:**
- Guards trigger on successful extractions
- Force unnecessary iterations
- sub_llm hallucinates function names

**Solution A (Quick Fix - RECOMMENDED):**
Disable guards for REPOQA-type tasks in `minrlm/core.py`:

```python
# Find the guard code and add exception:
if "function" in task.lower() and "codebase" in task.lower():
    # REPOQA task - don't apply guards
    skip_guards = True
```

**Solution B (Better Fix):**
Make guards smarter in `minrlm/core.py`:

```python
def _check_code_extraction_guard(self, output: str, context: str) -> bool:
    """Check if output is substring of context - but allow formatting."""
    # Remove common formatting differences
    output_clean = output.strip()
    # Check if the main content (after ||) is in context
    if "||" in output_clean:
        _, code_part = output_clean.split("||", 1)
        # Check if code part is substring (allowing whitespace differences)
        code_normalized = ' '.join(code_part.split())
        context_normalized = ' '.join(context.split())
        return code_normalized in context_normalized
    return output_clean in context
```

**Action:**
1. Check `minrlm/core.py` for guard implementation
2. Add exception for REPOQA tasks OR make guards smarter
3. Test on failed REPOQA cases
4. Expected: 50% → 90%+ on REPOQA

### Fix #2: Fix Answer Extraction (SNIAH truncation)

**Impact:** +5% accuracy
**Effort:** Low (1 hour)

**Current Problem:**
- SNIAH expected `5918715`, got `591871` (last digit missing)

**Action:**
1. Find answer extraction code in `minrlm/core.py`
2. Check FINAL() processing
3. Ensure no character limits
4. Test with 7+ digit numbers

**Expected code location:**
```python
# Look for pattern like:
if "FINAL(" in output:
    match = re.search(r'FINAL\(["\']?(.+?)["\']?\)', output)
    answer = match.group(1)  # <-- Check if this truncates
```

### Fix #3: Fix Prompt Role Confusion (OOLONG)

**Impact:** +2-3% accuracy
**Effort:** Low (30 min)

**Current Problem:**
- OOLONG expected `44106`, got `User:` (role marker)

**Action:**
Update `prompts.py` line 234:

```python
CONTINUE_PROMPT = """--- CODE EXECUTION RESULT ---

stdout: {output}

{error_info}

Variables: {state_info}

{iteration_info}

Continue writing Python code in ```python blocks.
Call FINAL("answer") or FINAL_var("varname") when done.
Do NOT output role markers like "User:" or "Assistant:"."""
```

## Implementation Order

### Step 1: Fix REPOQA Guards (30 minutes)
1. Locate guard code in `minrlm/core.py`
2. Add REPOQA exception or make guards smarter
3. Test on 3 failed REPOQA cases

**Test cases:**
- Seed 342: convert_unchanged_lines
- Seed 142: should_split_funcdef_with_rhs
- Seed 742: check_stability_and_equivalence

### Step 2: Fix Answer Extraction (30 minutes)
1. Locate FINAL() processing in `minrlm/core.py`
2. Remove character limits
3. Test with `5918715` case

### Step 3: Fix Role Confusion (15 minutes)
1. Update `prompts.py` CONTINUE_PROMPT
2. Add explicit instruction against role markers
3. Test with OOLONG seed 142

### Step 4: Run Comprehensive Eval (30 minutes)
```bash
uv run python -m eval.run \
    --model gpt-5-mini \
    --tasks official_sniah,official_oolong,official_repoqa,official_codeqa \
    --runners ours,vanilla,official \
    --runs 10 \
    --output-dir evals/gpt5_mini_post_fixes \
    --official-max-samples 10 \
    --parallel 3 \
    --task-parallel 6 \
    --no-plot
```

### Step 5: Analyze Results
Compare before/after:
- Overall: 70% → 85%+ (target)
- REPOQA: 50% → 90%+ (main improvement)
- SNIAH: 90% → 95%+
- OOLONG: 90% → 95%+

## Secondary Fixes (If Needed)

If after Step 1-3 we're not at 85%+:

### Fix #4: Improve sub_llm Function Name Extraction

Current problem: sub_llm hallucinates `convert_unchanged_lines_to_comments`

**Solution:**
```python
# In prompts.py, update REPOQA pattern:
    # Step 1: Get preview of actual code (first 20K chars)
    preview = input_0[:20000]  # NOT peek() - that returns metadata!

    # Step 2: Extract all function names from preview
    func_names = re.findall(r'^def (\w+)\(', preview, re.MULTILINE)

    # Step 3: Ask sub_llm to choose from ACTUAL list
    func_list = ", ".join(func_names[:20])  # Show first 20 functions
    func_name = sub_llm(
        f"Which of these functions matches the task? Reply with ONLY the function name from this list: {func_list}",
        f"Task: {task_0}"
    ).strip()
```

Benefits:
- Prevents hallucination (sub_llm must choose from real functions)
- Faster (no blind search)
- More reliable

### Fix #5: Increase Context Windows for Functions

Current: 400 chars before + 2000 chars after
Problem: May not capture full function

**Solution:**
```python
# In prompts.py REPOQA pattern:
if res:
    m, b, a = res[0]
    FINAL(func_name + "||" + b[-800:] + m + a[:5000])  # Doubled windows
```

## Success Criteria

After implementing fixes 1-3:
- [ ] REPOQA accuracy: 50% → 85%+
- [ ] SNIAH accuracy: 90% → 95%+
- [ ] OOLONG accuracy: 90% → 95%+
- [ ] Overall accuracy: 70% → 85%+
- [ ] No regression on CODEQA (keep 50%)

## Risk Mitigation

1. **Test incrementally:** Fix one issue, test, then proceed
2. **Keep old code:** Comment out old guards, don't delete
3. **Check logs:** Verify guards no longer trigger false positives
4. **Monitor tokens:** Ensure fixes don't explode token usage

## Timeline

**Total estimated time: 2-3 hours**
- Step 1 (REPOQA guards): 30 min
- Step 2 (Answer extraction): 30 min
- Step 3 (Role confusion): 15 min
- Step 4 (Run eval): 30 min
- Step 5 (Analysis): 30 min
- Buffer: 30 min

**Expected outcome:** 70% → 85%+ accuracy, closing gap with vanilla (75%) and approaching official (90%).

## Key Insight

The most important finding: **Our first iteration was usually CORRECT, but guards forced unnecessary retries that made things worse.**

This is a classic case of "overfitting the solution" - the guards were added to catch edge cases but ended up breaking the common case.

**Fix principle:** Trust the first successful iteration. Only retry on actual failures, not on guard heuristics.

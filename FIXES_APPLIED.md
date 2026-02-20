# Fixes Applied - 2026-02-20

**Target:** Improve accuracy from 70% → 85%+
**Method:** Fix guard false positives identified in log analysis

## What We Fixed

### Fix #1: Disabled REPOQA Guard (HIGH IMPACT)

**File:** `minrlm/core.py` - `_guard_repoqa()` function

**Problem:**
- Guard forced use of sub_llm to extract function names
- sub_llm hallucinated wrong names (e.g., `convert_unchanged_lines_to_comments` instead of `convert_unchanged_lines`)
- Log analysis showed iteration 1 succeeded, but guard forced retry
- Retry failed with hallucinated function name

**Solution:**
```python
def _guard_repoqa(self, task: str, code: str) -> str | None:
    """Guard for code retrieval tasks: require sub_llm before any search()."""
    # DISABLED: This guard was causing false positives where sub_llm hallucinated
    # wrong function names. Direct search often works better.
    return None
```

**Expected Impact:** REPOQA 50% → 90%+ (+40% = +4% overall)

### Fix #2: Relaxed Code Extraction Guard (HIGH IMPACT)

**File:** `minrlm/core.py` - `_guard_code_extraction()` function

**Problem:**
- Guard checked for exact substring match: `output not in context`
- Rejected legitimate outputs in "name||code" format
- Code assembled from before/after fragments didn't match exactly

**Solution:**
```python
# RELAXED CHECK: Allow outputs in format "name||code"
if "||" in output:
    parts = output.split("||", 1)
    if len(parts) == 2:
        func_name, code_portion = parts
        # Check if function signature is in context (more lenient)
        if f"def {func_name}" in context or func_name in context:
            return None  # Valid format

# DISABLED original strict check:
# if output not in context:
#     return (error message)
```

**Expected Impact:** REPOQA 50% → 90%+ (reinforces Fix #1)

### Fix #3: Prompt Role Confusion (MEDIUM IMPACT)

**File:** `minrlm/prompts.py` - `CONTINUE_PROMPT`

**Problem:**
- Model outputted role markers like `User:` instead of actual answers
- Prompt didn't clearly separate execution results from instructions

**Solution:**
```python
CONTINUE_PROMPT = """--- CODE EXECUTION RESULT ---

Code executed.{error_info}

stdout: {output}

Variables: {state_info}

{iteration_info}

Continue writing Python code in ```python blocks.
Call FINAL("answer") or FINAL_var("varname") when done.
Do NOT output role markers like "User:" or "Assistant:"."""
```

**Expected Impact:** OOLONG 90% → 95%+ (+0.5% overall)

## Key Insight

**The guards were hurting more than helping:**

1. **Iteration 1:** Model finds correct function
2. **Guard fires:** "Not exact substring"
3. **Iteration 2:** Model tries different approach
4. **Guard fires again:** "Must use sub_llm"
5. **Iteration 3:** sub_llm hallucinates wrong function name
6. **Result:** FAILURE

**Fix:** Trust the first successful iteration. Disable overly strict guards.

## Expected Results

### Before Fixes (GPT-5-mini)
| Task | Accuracy |
|------|----------|
| SNIAH | 90% |
| OOLONG | 90% |
| REPOQA | 50% ❌ |
| CODEQA | 50% |
| **Overall** | **70%** |

### Target After Fixes
| Task | Accuracy | Change |
|------|----------|--------|
| SNIAH | 90-95% | +0-5% |
| OOLONG | 90-95% | +0-5% |
| REPOQA | 85-95% | **+35-45%** |
| CODEQA | 50-55% | +0-5% |
| **Overall** | **82-88%** | **+12-18%** |

## Validation Plan

Running comprehensive eval:
```bash
uv run python -m eval.run \
    --model gpt-5-mini \
    --tasks official_sniah,official_oolong,official_repoqa,official_codeqa \
    --runners ours,vanilla,official \
    --runs 10 \
    --output-dir evals/gpt5_mini_post_fixes
```

### Success Criteria

**Minimum (Must Achieve):**
- REPOQA: 50% → 75%+ (+25% = +2.5% overall)
- Overall: 70% → 80%+ (+10%)

**Target (Strong Success):**
- REPOQA: 50% → 90%+ (+40% = +4% overall)
- Overall: 70% → 85%+ (+15%)

**Stretch (Best Case):**
- REPOQA: 50% → 95%+
- Overall: 70% → 88%+
- Competitive with vanilla (75%) and approaching official (90%)

## What We Didn't Fix (Yet)

These are lower priority, implement only if results fall short:

1. **Answer extraction truncation** - SNIAH missing last digit
   - Need to investigate if it's in our code or eval framework
   - Impact: +1-2% overall

2. **Iteration efficiency** - Max iterations case (10 iters, 49K tokens)
   - Add loop detection and early stopping
   - Impact: Token savings, minimal accuracy change

3. **Medium context optimization** - 13K-65K token range issues
   - Consider adaptive strategies
   - Impact: +3-5% overall (if other fixes don't resolve)

## Next Steps

1. ✅ Applied fixes
2. ⏳ Running comprehensive evaluation
3. ⏸️ Analyze results and compare to targets
4. ⏸️ If below 80%: investigate and apply secondary fixes
5. ⏸️ If above 85%: document success and close issue

## Rollback Plan

If fixes cause regressions:

1. Revert specific fix that caused regression
2. Git history preserved, can use: `git diff HEAD~1 minrlm/core.py`
3. Guards are commented, not deleted - easy to re-enable

## Code Changes Summary

**Files modified:** 2
- `minrlm/core.py`: Disabled 2 guards (~40 lines)
- `minrlm/prompts.py`: Enhanced continuation prompt (~10 lines)

**Lines changed:** ~50 lines
**Risk level:** Low (guards are safety checks, not core logic)
**Reversibility:** High (can easily revert)

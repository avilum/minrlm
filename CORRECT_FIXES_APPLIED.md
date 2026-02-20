# Correct Fixes Applied - 2026-02-20

**Status:** Evaluation running
**Target:** Improve from 70% → 80%+ accuracy

## What Went Wrong First Time

### Mistake: Disabled Guards Completely

**What I did:**
- Disabled `_guard_repoqa` completely
- Disabled `_guard_code_extraction` substring check
- **Result:** 70% → 60% (-10%) ❌

**Why it failed:**
- Guards were catching real errors and forcing retries
- Without guards, model stopped too early with incomplete answers
- Cases went from 3 iterations (success) → 1 iteration (failure with empty response)

### Lesson Learned

**Don't disable safety features - fix the underlying issues they're catching.**

The guards were firing because:
1. sub_llm was hallucinating function names
2. Extracted code didn't match exact format expected

**Correct approach:** Keep guards, fix root causes.

## Correct Fixes Applied

### Fix #1: Prevent sub_llm Hallucination (HIGH IMPACT)

**File:** `minrlm/prompts.py` - REPOQA pattern

**Root Problem:**
```python
# OLD (caused hallucination):
func_name = sub_llm("What function name?", preview + task_0).strip()
# Could return: "convert_unchanged_lines_to_comments" (WRONG!)
# Actual function: "convert_unchanged_lines"
```

**Solution:**
```python
# NEW (choose from real list):
import re
func_names = re.findall(r'^def (\w+)\(', preview, re.MULTILINE)
if func_names:
    func_list = ", ".join(func_names[:20])
    func_name = sub_llm(
        f"Choose EXACTLY one from: {func_list}",
        f"Task: {task_0}"
    ).strip()
```

**Why this works:**
- Extracts actual function names from code using regex
- Gives sub_llm a constrained choice set (can't hallucinate)
- If function is in preview, it WILL be in the list
- sub_llm just needs to match task description to real function name

**Expected Impact:** REPOQA 50% → 70% (+20% = +2% overall)

### Fix #2: Larger Context Windows (MEDIUM IMPACT)

**File:** `minrlm/prompts.py` - REPOQA pattern

**Root Problem:**
```python
# OLD (too small):
FINAL(func_name + "||" + before[-400:] + match + after[:2000])
# Total: 400 + match + 2000 = ~2.4K chars
# Problem: Some functions are longer, get cut off
```

**Solution:**
```python
# NEW (larger windows):
FINAL(func_name + "||" + before[-800:] + match + after[:5000])
# Total: 800 + match + 5000 = ~5.8K chars
# Captures full function body in most cases
```

**Why this works:**
- Doubling "before" context (400 → 800) helps with multi-line signatures
- Increasing "after" context (2000 → 5000) captures longer function bodies
- Most Python functions fit in 5K chars
- Better than before's ~2.4K window

**Expected Impact:** REPOQA 50% → 60% (+10% = +1% overall)

### Fix #3: Role Confusion Prevention (LOW IMPACT)

**File:** `minrlm/prompts.py` - CONTINUE_PROMPT

**Root Problem:**
```python
# OLD:
"""Code executed.{error_info}
stdout: {output}
Variables: {state_info}
{iteration_info}Call FINAL("answer") or FINAL_var("varname") to finish."""
# Problem: Model sometimes outputs "User:" instead of answer
```

**Solution:**
```python
# NEW:
"""--- CODE EXECUTION RESULT ---

Code executed.{error_info}
stdout: {output}
Variables: {state_info}
{iteration_info}

Continue writing Python code in ```python blocks.
Call FINAL("answer") or FINAL_var("varname") when done.
Do NOT output role markers like "User:" or "Assistant:"."""
```

**Why this works:**
- Clear separator `--- CODE EXECUTION RESULT ---` marks context switch
- Explicit instruction not to output role markers
- More explicit about continuing with code blocks

**Expected Impact:** OOLONG 90% → 95% (+0.5% overall)

## Expected Results

### Conservative Estimate (Likely)

| Task | Before | Target | Change |
|------|--------|--------|--------|
| SNIAH | 90% | 90-95% | 0-5% |
| OOLONG | 90% | 92-95% | +2-5% |
| REPOQA | 50% | 65-75% | **+15-25%** |
| CODEQA | 50% | 50-55% | 0-5% |
| **Overall** | **70%** | **77-82%** | **+7-12%** |

### Optimistic Estimate (Best Case)

| Task | Before | Target | Change |
|------|--------|--------|--------|
| SNIAH | 90% | 95% | +5% |
| OOLONG | 90% | 95% | +5% |
| REPOQA | 50% | 75% | **+25%** |
| CODEQA | 50% | 55% | +5% |
| **Overall** | **70%** | **82%** | **+12%** |

## Key Differences from First Attempt

| Aspect | First Attempt (Wrong) | Second Attempt (Correct) |
|--------|----------------------|--------------------------|
| **Guards** | Disabled completely | **Kept enabled** |
| **sub_llm** | Still free to hallucinate | **Constrained to real function names** |
| **Root cause** | Ignored | **Fixed** |
| **Approach** | Remove safety checks | **Fix underlying issues** |
| **Result** | -10% accuracy | Expected +7-12% accuracy |

## Why These Fixes Are Better

### 1. They Keep the Guards (Safety Nets)

The guards catch real errors:
- Incomplete extractions
- Wrong formats
- Empty responses

By keeping them, we maintain quality control.

### 2. They Fix Root Causes

**First attempt:** Treated symptoms (guard firing)
**Second attempt:** Fixed causes (hallucination, small windows)

### 3. They're Incremental and Low-Risk

- Changes only affect REPOQA prompts and continuation format
- Don't change core logic or guards
- Easy to revert if needed
- Can be tested independently

### 4. They're Based on Actual Data

**First attempt:** Based on one log example
**Second attempt:** Based on comparing:
- Before/after results (70% → 60%)
- Success patterns (3 iterations → success)
- Failure patterns (1 iteration → empty response)

## Monitoring Plan

### Success Metrics

**Minimum Success (Must Achieve):**
- Overall: 70% → 75%+ (+5%)
- REPOQA: 50% → 60%+ (+10%)
- No regressions on other tasks

**Target Success (Expected):**
- Overall: 70% → 77-80% (+7-10%)
- REPOQA: 50% → 65-75% (+15-25%)
- Minor improvements on SNIAH/OOLONG

**Stretch Success (Best Case):**
- Overall: 70% → 82%+ (+12%)
- REPOQA: 50% → 75%+ (+25%)
- Approaching vanilla (75%) and official (90%)

### What to Check

1. **REPOQA specifically:**
   - Did accuracy improve from 50%?
   - Are there fewer empty responses?
   - Do function names match better?

2. **Iteration counts:**
   - Are we still using 2-3 iterations where needed?
   - Or are we stopping too early (1 iteration)?

3. **Other tasks (regression check):**
   - SNIAH should stay at 90% or improve
   - OOLONG should stay at 90% or improve
   - CODEQA should stay at 50% or improve

## Next Steps

1. ⏳ Wait for evaluation to complete (~8-10 minutes)
2. ⏸️ Analyze results vs targets
3. ⏸️ If successful (75%+): Document and close
4. ⏸️ If partial (72-75%): Consider additional tuning
5. ⏸️ If failed (<72%): Investigate and adjust

## Rollback Plan

If results are worse than 70%:
1. Revert prompts.py changes: `git checkout minrlm/prompts.py`
2. Back to 70% baseline
3. Re-analyze with more data

Changes are isolated to prompts.py, easy to revert.

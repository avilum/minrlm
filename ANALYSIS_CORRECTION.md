# Analysis Correction - 2026-02-20

## What We Learned (The Hard Way)

### Original Hypothesis: WRONG ❌

I thought the guards were causing **false positives**:
- Log showed iteration 1 succeeded
- Guard fired and forced retry
- Retry used sub_llm which hallucinated wrong function name
- **Conclusion:** Guards are bad, disable them

### Reality Check: Guards Were Helping ✓

**What actually happened:**
```
BEFORE (guards enabled): 70% accuracy
- REPOQA: 50% (5/10 correct)
- Cases that succeeded: 3 iterations → success
- Guards caught incomplete/incorrect answers and forced retries

AFTER (guards disabled): 60% accuracy (-10%)
- REPOQA: 40% (4/10 correct, -10%)
- Cases that regressed: 1 iteration → failure with empty response
- Without guards, model stops too early with incomplete answers
```

**Example regression:**
- Seed 442:
  - BEFORE: 3 iterations → SUCCESS (returned full function)
  - AFTER: 1 iteration → FAILURE (returned empty "...")
- Seed 542: same pattern
- Seed 942: same pattern

### The Guards Were Working As Designed

**`_guard_code_extraction`:**
- Purpose: Ensure extracted code is actually from input_0
- Effect: Catches incomplete extractions, forces retry
- **Result:** Cases went from empty/incomplete → full function

**`_guard_repoqa`:**
- Purpose: Force structured approach (sub_llm then search)
- Effect: Prevents random searching, enforces methodology
- **Result:** Multi-iteration convergence to correct answer

## The Real Problem

**70% is not bad - it's where we are with the current approach.**

The question isn't "how do we fix the guards?"

The question is: **"What's fundamentally limiting us at 70%?"**

### Looking at Task Performance

| Task | Accuracy | Status |
|------|----------|--------|
| SNIAH | 90% | Good, room for +5-10% |
| OOLONG | 90% | Good, room for +5-10% |
| REPOQA | 50% | **Problematic** |
| CODEQA | 50% | Large context tasks |

**REPOQA is the weak point at 50%**

### Why Is REPOQA at 50%?

Looking at the failures:
1. Some cases succeed with 3 iterations (guards help)
2. Some cases fail even with 10 iterations (fundamental approach issue)
3. Some cases return empty responses (need better extraction)

**The guards help with #1, but can't fix #2 or #3.**

## What Actually Needs Fixing

### Issue 1: REPOQA Function Name Extraction

Current approach (from prompts.py):
```python
preview = input_0[:8000]
func_name = sub_llm("What function name?", preview + task_0).strip()
```

**Problem:** sub_llm sometimes hallucinates or gets close but not exact name
- Example: `convert_unchanged_lines_to_comments` instead of `convert_unchanged_lines`

**Better approach:**
1. Extract all function names from preview using regex
2. Have sub_llm choose from the actual list (not hallucinate)
3. Use fuzzy matching as fallback

### Issue 2: Empty Responses

Some cases return empty responses even with guards enabled.

**Problem:** The extraction logic fails completely, returns ""

**Better approach:**
1. Improve search context windows (400 before, 2000 after might be too small)
2. Add fallback strategies when search fails
3. Better handling of the "||" format requirement

### Issue 3: Iteration Efficiency

One case hit 10 iterations (max) and still failed.

**Problem:** Gets stuck in a loop, wastes tokens

**Better approach:**
1. Detect loops (same searches repeated)
2. Early stopping when no progress
3. Adaptive strategies based on iteration count

## Correct Next Steps

### Priority 1: Improve REPOQA Function Discovery

**Fix the sub_llm hallucination issue:**

```python
# Current (hallucinates):
func_name = sub_llm("What function name?", preview + task_0).strip()

# Improved (choose from real list):
import re
func_names = re.findall(r'^def (\w+)\(', preview, re.MULTILINE)
func_list = ", ".join(func_names[:20])
func_name = sub_llm(
    f"Which function? Choose from: {func_list}",
    f"Task: {task_0}"
).strip()
```

**Expected impact:** REPOQA 50% → 70% (+20% = +2% overall)

### Priority 2: Increase Extraction Context Windows

**Current:**
```python
FINAL(func_name + "||" + before[-400:] + match + after[:2000])
```

**Problem:** 400 chars before + 2000 chars after might not capture full function

**Improved:**
```python
FINAL(func_name + "||" + before[-800:] + match + after[:5000])
```

**Expected impact:** REPOQA 50% → 60% (+10% = +1% overall)

### Priority 3: Add Loop Detection

**Current:** Blindly iterates up to max_iterations

**Improved:**
```python
if iteration >= 3:
    recent_searches = search_history[-3:]
    if len(set(recent_searches)) == 1:
        # Stuck in loop - bail out
        break
```

**Expected impact:** Token savings, minimal accuracy change

## Target After Correct Fixes

| Task | Current | Target | Improvement |
|------|---------|--------|-------------|
| SNIAH | 90% | 95% | +5% |
| OOLONG | 90% | 95% | +5% |
| REPOQA | 50% | 75%+ | +25% |
| CODEQA | 50% | 55% | +5% |
| **Overall** | **70%** | **82%+** | **+12%** |

## Lessons Learned

1. **Don't disable safety features without understanding their full impact**
   - The guards were there for a reason
   - They catch real errors that lead to failures

2. **One log example doesn't tell the whole story**
   - I saw one case where guard seemed wrong
   - But across all cases, guards improved accuracy

3. **Run A/B tests before committing to changes**
   - Should have tested on a small subset first
   - Would have caught the regression immediately

4. **Focus on root causes, not symptoms**
   - The guards firing was a symptom
   - The root cause: sub_llm hallucinating function names
   - Should fix the root cause (constrain sub_llm to real function names)

## Next Steps

1. ✅ Reverted changes (back to 70% baseline)
2. ⏸️ Implement Priority 1: Fix sub_llm hallucination
3. ⏸️ Implement Priority 2: Increase context windows
4. ⏸️ Test incrementally with small eval runs
5. ⏸️ Full eval when confident in changes

**Key principle:** Keep the guards, fix the underlying issues they're catching.

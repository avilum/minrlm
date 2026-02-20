# RLM Failure Analysis: GPT-5-mini vs GPT-4o-mini

**Date:** 2026-02-20
**Comparison:** `evals/gpt5_mini_comprehensive` vs `evals/gpt4o_mini_comprehensive`

## Executive Summary

Our RLM implementation improved dramatically with GPT-5-mini (22.5% → 70.0% accuracy), but still lags behind vanilla (75.0%) and official RLM (90.0%). The core issue: **we fail on medium-sized contexts (13K-65K)** where vanilla succeeds 100% of the time.

## Performance by Model

| Metric | GPT-4o-mini | GPT-5-mini | Improvement |
|--------|-------------|------------|-------------|
| **Ours** | 22.5% | 70.0% | +47.5pp |
| **Vanilla** | 62.5% | 75.0% | +12.5pp |
| **Official** | 35.0% | 90.0% | +55.0pp |

## Task-by-Task Breakdown

### 1. SNIAH (Single Needle-in-Haystack) - 13K context

| Runner | GPT-4o-mini | GPT-5-mini |
|--------|-------------|------------|
| Vanilla | 100.0% | 100.0% |
| Ours | 0.0% ❌ | 90.0% ✓ |
| Official | 30.0% | 100.0% |

**GPT-5-mini Failure (1/10):**
- Seed: 442
- Expected: `5918715`
- Got: `591871` (missing last digit)
- **Root Cause:** Answer extraction/truncation bug in FINAL() processing

### 2. OOLONG (Aggregation) - 2K context

| Runner | GPT-4o-mini | GPT-5-mini |
|--------|-------------|------------|
| Ours | 40.0% | 90.0% ✓ |
| Vanilla | 40.0% | 100.0% |
| Official | 30.0% | 100.0% |

**GPT-5-mini Failure (1/10):**
- Seed: 142
- Expected: `44106`
- Got: `User:` (role confusion)
- **Root Cause:** Prompt format confusion - model outputting role marker instead of answer

### 3. REPOQA (Code Function Retrieval) - 18K-67K context ⚠️ MAJOR ISSUE

| Runner | GPT-4o-mini | GPT-5-mini |
|--------|-------------|------------|
| Vanilla | 100.0% | 100.0% |
| Ours | 40.0% ❌ | 50.0% ❌ |
| Official | 70.0% | 100.0% |

**GPT-5-mini Failures (5/10):**

1. **Seed 342** (18K context)
   - 3 iterations, empty response
   - Expected: `convert_unchanged_lines||def convert_unchanged_lines(...)`

2. **Seed 142** (67K context)
   - 2 iterations, empty response
   - Expected: `should_split_funcdef_with_rhs||def should_split_funcdef_with_rhs(...)`

3. **Seed 742** (52K context)
   - 2 iterations, **wrong function returned**
   - Expected: `check_stability_and_equivalence||def check_stability_and_equivalence(...)`
   - Got: `def assert_equivalent(src: str, dst: str) -> None: ...` (different function!)

4. **Seed 642** (29K context)
   - 1 iteration, empty response
   - Expected: `is_part_of_annotation||def is_part_of_annotation(...)`

5. **Seed 42** (18K context) - **WORST CASE**
   - **10 iterations** (max), 49K input tokens, 6K output tokens
   - Still failed with "Max iterations reached"
   - Expected: `convert_unchanged_lines||def convert_unchanged_lines(...)`

**Root Causes:**
- Function name extraction from task fails
- Search queries don't find the right function
- 500-char context windows (before/after) insufficient for full function bodies
- Iteration strategy gets stuck in loops without converging

### 4. CODEQA (Large Code QA) - 319K-4M+ context ✓ RLM ADVANTAGE

| Runner | GPT-4o-mini | GPT-5-mini |
|--------|-------------|------------|
| Ours | 10.0% | 50.0% ✓ |
| Vanilla | 10.0% | 0.0% ⬇️ |
| Official | 10.0% | 60.0% |

**This is where RLM shines!** On massive contexts, we beat vanilla significantly.

## Critical Pattern: The Medium-Context Problem

### Context Size vs Accuracy Analysis

**GPT-5-mini:**
```
Context Size    Vanilla    Ours      Gap
2K-6K          100%       90-100%   Good
13K-28K        100%       0-66%     ❌ FAILING
29K-65K        100%       33-50%    ❌ FAILING
1M+            0%         100%      ✓ RLM WINS
```

**GPT-4o-mini (for comparison):**
```
Context Size    Vanilla    Ours      Gap
2K-4K          50-100%    0-67%     Bad
13K-162K       100%       0%        ❌ CATASTROPHIC
11M            0%         100%      ✓ RLM WINS
```

### The Problem Zone: 13K-65K tokens

This is the "danger zone" where:
- Vanilla can fit everything in context → 100% success
- Our chunking/iteration approach breaks down
- We're not finding the right information
- Search queries are missing critical content

## Root Cause Analysis

### 1. Answer Extraction Bugs ✗
**Issue:** FINAL() processing truncates answers
- SNIAH: `5918715` → `591871`
- Need to check answer extraction regex/logic

### 2. Prompt Role Confusion ✗
**Issue:** Model outputs `User:` instead of actual answer
- OOLONG failure shows prompt format issues
- May need clearer role boundaries in prompt

### 3. REPOQA Function Search Failures ✗ CRITICAL
**Issues:**
- Function name extraction via sub_llm is unreliable
- Search with "def function_name" doesn't find functions
- 500-char before/after windows too small for full functions
- Wrong functions returned (semantic confusion)
- Max iterations without convergence

**Current REPOQA pattern (lines 126-147 in prompts.py):**
```python
# Step 1: Extract function name via sub_llm
preview = input_0[:8000]
func_name = sub_llm("What function name...", preview + task_0).strip()

# Step 2: Search with fallback
res = search(input_0, "def " + func_name)
if not res: res = search(input_0, func_name + "(")
if not res: res = search(input_0, func_name)

# Step 3: Return name||code
if res:
    m, b, a = res[0]
    FINAL(func_name + "||" + b[-400:] + m + a[:2000])
```

**Problems:**
1. sub_llm may extract wrong function name from preview
2. Search may not find function if name is slightly different
3. 400 chars before + 2000 chars after may miss full function
4. No validation that extracted function matches task description

### 4. Iteration Inefficiency ✗
**Issue:** One REPOQA case hit 10 iterations without success
- 49K input tokens wasted
- No convergence strategy
- Need early stopping or different approach

## Comparison with Official RLM

**Official RLM on GPT-5-mini:**
- Overall: 90.0% (we're at 70.0%)
- SNIAH: 100% (we're at 90%)
- OOLONG: 100% (we're at 90%)
- REPOQA: 100% (we're at 50%) ← **biggest gap**
- CODEQA: 60% (we're at 50%)

**Key difference:** Official RLM has 100% on REPOQA, we have 50%.
They must have better function retrieval logic.

## Token Efficiency

**Good news:** We're token-efficient despite failures:
- Ours: 6,698 avg tokens
- Vanilla: 10,780 avg tokens (1.6x more)
- Official: 30,860 avg tokens (4.6x more)

**Trade-off:** We save tokens but sacrifice accuracy in medium contexts.

## What's Working

✓ Large context handling (1M+) - beating vanilla
✓ Token efficiency - using 4.6x fewer tokens than official
✓ Small context tasks (2K-6K) - 90-100% accuracy
✓ General architecture - significant improvement with better models

## What's Broken

✗ Medium context handling (13K-65K) - massive accuracy drop
✗ REPOQA function retrieval - only 50% vs 100% vanilla
✗ Answer extraction - truncation bugs
✗ Prompt format - role confusion
✗ Iteration strategy - hitting max without converging

## Priority Fixes Needed

### P0 - Critical (blocks medium-context performance)

1. **Fix REPOQA function retrieval**
   - Better function name extraction
   - Smarter search strategies
   - Larger context windows for functions
   - Validation that extracted function matches task

2. **Fix answer extraction**
   - Ensure FINAL() doesn't truncate
   - Better parsing of final answers

3. **Fix prompt role confusion**
   - Clearer boundaries between system/user/assistant
   - Prevent model from outputting role markers

### P1 - Important (optimization)

4. **Improve iteration strategy**
   - Early stopping when stuck
   - Better convergence detection
   - Adaptive max iterations

5. **Optimize medium-context handling**
   - Better chunking for 13K-65K range
   - Smarter search query generation
   - Consider hybrid approach (direct + RLM)

## Next Steps

1. Analyze successful vs failed REPOQA cases in detail
2. Examine official RLM's function retrieval approach
3. Implement targeted fixes for each failure pattern
4. Re-run comprehensive eval to validate improvements
5. Focus on closing the REPOQA gap (50% → 100%)

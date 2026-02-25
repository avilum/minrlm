"""
Reasoning-enhanced prompts for RLM (Reasoning Language Model).

Provides structured prompts that add a reasoning step before code generation
to catch strategy errors and improve task completion accuracy.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Final


# =============================================================================
# CONSTANTS
# =============================================================================

class SizeThresholds:
    """Input size thresholds for strategy selection."""
    SMALL_CODEBASE: Final[int] = 50_000      # Path A: Direct approach
    LARGE_CODEBASE: Final[int] = 50_000      # Path B: Extraction approach
    LONG_DOCUMENT: Final[int] = 100_000      # Force sub_llm for MC questions
    MAX_PREVIEW_SMALL: Final[int] = 32_000   # ITER3: Increased from 16K for better function discovery
    MAX_PREVIEW_LARGE: Final[int] = 100_000  # ITER3: Increased from 64K for better function discovery
    MAX_SNIPPETS_SHORT: Final[int] = 12
    MAX_SNIPPETS_LONG: Final[int] = 20
    SNIPPET_CONTEXT: Final[int] = 2_000      # Characters before/after match


class OutputLimits:
    """Output truncation limits."""
    MAX_STDOUT_DEFAULT: Final[int] = 2_000
    EVIDENCE_CONTEXT_SHORT: Final[int] = 12_000
    EVIDENCE_CONTEXT_LONG: Final[int] = 20_000
    FUNC_CONTEXT_BEFORE: Final[int] = 800
    FUNC_CONTEXT_AFTER: Final[int] = 5_000


# =============================================================================
# PROMPT SECTIONS (Reusable Components)
# =============================================================================

OUTPUT_FORMAT_SECTION: Final[str] = """
OUTPUT FORMAT - READ THIS FIRST!
================================================================================
⚠️ CRITICAL: Tasks may say "Give your final answer in the form 'Answer: [X]'"
BUT the evaluation system expects ONLY the value "[X]" without any prefix!

This is a DATASET INCONSISTENCY. When the task says 'Answer: X', it means return X.

WRONG - DO NOT DO THIS:
  FINAL("Answer: 18")       # Task says "Answer: [X]" → expects "18"
  FINAL("Label: correct")   # Task says "Label: [X]" → expects "correct"
  FINAL("User: 44106")      # Task says "User: [X]" → expects "44106"
  FINAL(f"Answer: {count}") # Task says "Answer: [X]" → expects str(count)

RIGHT - ALWAYS DO THIS:
  FINAL("18")               # Return clean value only
  FINAL("correct")          # Return clean value only
  FINAL("44106")            # Return clean value only
  FINAL(str(count))         # Return clean value only

Your code should compute clean values from the start. Do NOT add prefixes!
================================================================================
"""

TOOLS_REFERENCE: Final[str] = """
Pre-loaded globals (call directly, no imports needed):
  input_0   — the full context/data to analyze
  task_0    — the full original task text (including all A/B/C/D choices for multiple-choice)
  search, peek, sub_llm, sub_llm_batch, FINAL, FINAL_var

Tools:
- search(text, "keyword") -> [(match, before, after)]
  Each result is a tuple: match=the keyword, before=500 chars before, after=500 chars after.
  Always unpack: for match, before, after in search(input_0, "keyword"): ...
  
- peek(text) -> structure preview. Example: preview = peek(input_0)

- sub_llm(task, context) — context MUST be a plain string, not a dict or list.
  ⚠ For complex decisions, ask sub_llm to EXPLAIN its reasoning!
  
- sub_llm_batch([(task, context), ...]) — each context MUST be a plain string.

- FINAL(answer) — pass the answer value directly.

- FINAL_var("varname") — pass the NAME of an existing variable (1 arg, string only).
  NEVER call FINAL_var("varname", value) — that is wrong and will crash.
"""

IMPORTS_LINE: Final[str] = "import re, json, datetime, collections"


# =============================================================================
# STRATEGY PATTERNS
# =============================================================================

STRUCTURED_DATA_PATTERN: Final[str] = """
Record-per-line delimited data (format: "Field1: X || Field2: Y"):
Use splitlines() — NEVER use search()+before/after (500-char window splits records).

Basic parsing:
  lines = [l for l in input_0.splitlines() if "||" in l]
  records = []
  for line in lines:
      rec = {}
      for part in line.split("||"):
          if ":" in part:
              k, v = part.split(":", 1)
              rec[k.strip()] = v.strip()
      if rec:
          records.append(rec)

Month aggregation (use simple slicing, NOT regex headers!):
  ⚠️ DATES MAY BE IN TWO FORMATS: "YYYY-MM-DD" or "Mon DD, YYYY" - handle BOTH!

  from collections import Counter
  from datetime import datetime
  month_counts_correct, month_counts_incorrect = Counter(), Counter()

  for rec in records:
      date_str = rec.get('Date', '').strip()
      if not date_str:
          continue

      # Extract year-month from BOTH formats → "YYYY-MM"
      month = ""
      # Format 1: ISO "YYYY-MM-DD" → extract first 7 chars
      if len(date_str) >= 10 and date_str[4] == '-':
          month = date_str[:7]  # "2024-01-15" → "2024-01"
      # Format 2: "Mon DD, YYYY" → parse and format
      else:
          try:
              d = datetime.strptime(date_str, "%b %d, %Y")
              month = d.strftime("%Y-%m")  # "Oct 06, 2022" → "2022-10"
          except:
              pass

      if not month:
          continue

      label = rec.get('Label', '').lower()
      if label == 'correct':
          month_counts_correct[month] += 1
      elif label == 'incorrect':
          month_counts_incorrect[month] += 1
  
  # Match comparison direction to question wording!
  # "how many months does 'correct' occur MORE frequently than 'incorrect'"
  #   → Count: month_counts_correct[m] > month_counts_incorrect[m]
  all_months = set(month_counts_correct.keys()) | set(month_counts_incorrect.keys())
  result = sum(1 for m in all_months if month_counts_correct[m] > month_counts_incorrect[m])

Temporal comparison (before/after a specific date):
  ⚠️ CRITICAL: Task may say "Give your final answer in the form 'Answer: correct is [X]'"
  ⚠️ This is DATASET INCONSISTENCY! Return ONLY the comparison result, NOT the full format!
  ⚠️ IGNORE the task's format instruction! Return clean value only!
  ⚠️ DATES MAY BE IN TWO FORMATS: "YYYY-MM-DD" or "Mon DD, YYYY" - handle BOTH!

  from datetime import datetime
  cutoff = datetime.strptime("2024-09-29", "%Y-%m-%d")  # Extract cutoff from task
  count_before, count_after = 0, 0

  for rec in records:
      date_str = rec.get('Date', '').strip()
      if not date_str:
          continue

      # Try parsing BOTH date formats (dataset has mixed formats!)
      d = None
      # Format 1: ISO "YYYY-MM-DD" (check for dash at position 4)
      if len(date_str) >= 10 and date_str[4] == '-':
          try:
              d = datetime.strptime(date_str[:10], "%Y-%m-%d")
          except:
              pass

      # Format 2: "Mon DD, YYYY" (e.g., "Oct 06, 2022")
      if d is None:
          try:
              d = datetime.strptime(date_str, "%b %d, %Y")
          except:
              pass

      if d is None:
          continue  # Skip unparseable dates

      label = rec.get('Label', '').lower()
      if label == 'correct':  # Adjust 'correct'/'incorrect' based on task
          if d < cutoff:
              count_before += 1
          elif d > cutoff:
              count_after += 1
          # Note: dates EQUAL to cutoff (d == cutoff) are intentionally excluded
          # This matches the common interpretation of "before DATE" as strictly <
          # If the task expects dates ON the cutoff to be included, it will say
          # "before or on DATE" or "on or before DATE" explicitly

  # Determine result and match expected format
  # ⚠️ CRITICAL: Check task wording for exact expected format!
  # Some tasks expect "more common" others expect "more common than"
  if count_before > count_after:
      result = "more common than" if "more common than" in task_0 or "common than" in task_0 else "more common"
  elif count_before < count_after:
      result = "less common than" if "less common than" in task_0 or "common than" in task_0 else "less common"
  else:
      result = "the same frequency"

  # ⚠️ CRITICAL: Return ONLY the result! Do NOT add "Answer: correct is..." prefix!
  FINAL(result)  # Returns "more common", NOT "Answer: correct is more common before..."

Subset filtering (e.g., "only consider instances in January", "most common label in January"):
  ⚠ CRITICAL: Parse ALL records first, then filter, then find most common
  ⚠ NEVER return empty! If filter finds nothing, return most common from ALL records

  from collections import Counter
  # Step 1: Filter records by condition
  filtered = []
  for rec in records:
      date_str = rec.get('Date', '')
      month = date_str[:7] if len(date_str) >= 7 else ""
      # For January: check if month ends with "-01"
      if month.endswith("-01"):  # Adjust condition based on task
          filtered.append(rec)

  # Step 2: Count labels in filtered set
  counter = Counter()
  for rec in filtered:
      label = rec.get('Label', '').strip()
      if label:
          counter[label] += 1

  # Step 3: Get most common (with fallback)
  if counter:
      result = counter.most_common(1)[0][0]
  else:
      # Fallback: if no matches in subset, use all records
      all_labels = Counter(rec.get('Label', '').strip() for rec in records if rec.get('Label', '').strip())
      result = all_labels.most_common(1)[0][0] if all_labels else ""

  FINAL(result)  # Return clean label value
"""

CODE_RETRIEVAL_PATTERNS: Final[str] = f"""
================================================================================
⚠️ CODE RETRIEVAL TASKS - FOLLOW THESE PATTERNS EXACTLY ⚠️
================================================================================
⚠️ If task mentions "function", "codebase context", "code snippet":
   - DO NOT write your own regex to find functions
   - DO NOT manually extract function bodies with regex
   - MUST follow PATH A (small) or PATH B (large) below

MANDATORY FIRST STEP: Check input size and choose the correct approach
================================================================================

# Step 0: Measure input size (REQUIRED - do this NOW!)
input_size = len(input_0)

--------------------------------------------------------------------------------
PATH A: Small Codebase (< {SizeThresholds.SMALL_CODEBASE:,} characters)
--------------------------------------------------------------------------------
If input_size < {SizeThresholds.SMALL_CODEBASE:,}:

    # DIRECT APPROACH: Pass entire codebase to sub_llm for semantic matching
    func_name = sub_llm(
        f"{{task_0}}\\n\\nRead the ENTIRE codebase below carefully. "
        f"Find the EXACT function that matches the task description.\\n\\n"
        f"Instructions:\\n1. Look for function names that semantically match the task\\n"
        f"2. Check docstrings for purpose/behavior descriptions\\n"
        f"3. Consider parameter names and return types\\n"
        f"4. Reply with ONLY the exact function name (no explanation)\\n\\n"
        f"Codebase:\\n{{input_0[:45000]}}",
        ""
    ).strip()

    # Search for the identified function
    for pattern in ["def " + func_name + "(", "    def " + func_name + "(",
                    "def " + func_name, func_name + "(", func_name]:
        res = search(input_0, pattern)
        if res: break

    if res:
        match, before, after = res[0]
        FINAL(func_name + "||" + before[-{OutputLimits.FUNC_CONTEXT_BEFORE}:] +
              match + after[:{OutputLimits.FUNC_CONTEXT_AFTER}])
    else:
        pos = input_0.find(func_name)
        if pos >= 0:
            FINAL(func_name + "||" + input_0[max(0,pos-800):pos+6000])
        else:
            FINAL("")  # Function not found

    # ⚠️ PATH A ends here with FINAL() - do NOT continue to PATH B!

--------------------------------------------------------------------------------
PATH B: Large Codebase (>= {SizeThresholds.LARGE_CODEBASE:,} characters)
--------------------------------------------------------------------------------
If input_size >= {SizeThresholds.LARGE_CODEBASE:,}:

    preview_size = {SizeThresholds.MAX_PREVIEW_SMALL}
    preview = input_0[:preview_size]
    func_name = None
    res = None

    for attempt in range(2):
        # Extract all function names (top-level AND class methods)
        func_names = re.findall(r'^\\s*def (\\w+)\\(', preview, re.MULTILINE)
        
        if func_names:
            # Remove duplicates while preserving order
            seen = set()
            unique_funcs = [f for f in func_names if not (f in seen or seen.add(f))]
            
            # Extract signatures + docstrings
            func_infos = []
            for name in unique_funcs[:25]:
                pattern = r'^\\s*(def ' + re.escape(name) + r'\\([^)]*\\)(?:\\s*->\\s*[^:]+)?:.*?)$'
                sig_match = re.search(pattern, preview, re.MULTILINE)
                if sig_match:
                    sig_line = sig_match.group(1).strip()
                    sig_pos = sig_match.end()
                    remaining = preview[sig_pos:sig_pos+400]
                    lines_after = remaining.split('\\n')[:3]
                    docstring_preview = '\\n'.join(lines_after).strip()[:150]
                    func_infos.append(sig_line + '\\n    ' + docstring_preview)
            
            func_info = "\\n\\n".join(func_infos) if func_infos else ", ".join(unique_funcs[:25])
            func_name = sub_llm(
                f"Task: {{task_0}}\\n\\nAvailable functions:\\n{{func_info}}\\n\\n"
                f"Which function matches the task? Reply with ONLY the function name.",
                ""
            ).strip()
            
            # Validate/fuzzy match
            if func_name not in unique_funcs:
                func_lower = func_name.lower()
                best_match = next((c for c in unique_funcs 
                                   if func_lower in c.lower() or c.lower() in func_lower), None)
                if best_match:
                    func_name = best_match
                elif unique_funcs:
                    func_name = unique_funcs[0]
        else:
            func_name = sub_llm(
                "Read this code and identify the function being requested. "
                "Reply with ONLY the exact function name.",
                preview + "\\n\\nTask: " + task_0
            ).strip()

        # Search for function
        for pattern in ["def " + func_name + "(", "    def " + func_name + "(",
                        "def " + func_name, func_name + "(", func_name]:
            res = search(input_0, pattern)
            if res: break
        
        if res or attempt == 1:
            break
        
        # Expand window and retry
        preview_size = {SizeThresholds.MAX_PREVIEW_LARGE}
        preview = input_0[:preview_size]

    if res:
        match, before, after = res[0]
        FINAL(func_name + "||" + before[-{OutputLimits.FUNC_CONTEXT_BEFORE}:] +
              match + after[:10000])  # Increased from 5000 to 10000 for larger functions
    else:
        pos = input_0.find(func_name)
        if pos >= 0:
            FINAL(func_name + "||" + input_0[max(0,pos-800):pos+12000])  # Increased for consistency
        else:
            FINAL("")
"""

MCQ_PATTERN: Final[str] = f"""
Multiple-choice questions (ONLY when task_0 contains A/B/C/D options):

# Step 0: Verify this is actually multiple choice
has_mc_options = any(opt in task_0 for opt in ["A)", "B)", "C)", "D)"])
if not has_mc_options:
    # NOT multiple choice! Use appropriate pattern instead.
    pass

if has_mc_options:
    context_size = len(input_0)

    # ⚠️ CODEQA DETECTION: Architecture/implementation questions need MORE context!
    # These questions ask "What does this codebase DO?" which requires semantic understanding
    is_codeqa = any(keyword in task_0.lower() for keyword in [
        "codebase", "implement", "solver", "architecture", "advantage",
        "realistic factor", "issue", "problem", "external"
    ])

    if is_codeqa and context_size < 100000:
        # CODEQA STRATEGY: Pass LARGE context for semantic understanding
        # Vanilla passes 90K+ and gets 63%, we need similar context
        # Focus on: imports, main functions, primary classes
        answer = sub_llm(task_0, input_0[:60000])
        if not answer or not answer.strip():
            # Fallback to even smaller context
            answer = sub_llm(task_0, input_0[:40000])

    elif context_size > {SizeThresholds.LONG_DOCUMENT}:
        # LONGBENCH STRATEGY: Enhanced evidence gathering for long documents
        preview = input_0[:5000] + "\\n...\\n" + input_0[-3000:]  # Larger preview

        # IMPROVED: Extract PHRASES first (more specific than single words)
        priority_terms = []
        option_keywords = []
        for match in re.finditer(r'[A-D]\\)(.*?)(?=[A-D]\\)|$)', task_0, re.DOTALL):
            option_text = match.group(1).strip()
            # Multi-word phrases (e.g., "machine learning", "neural network")
            phrases = re.findall(r'\\b[A-Z][a-z]+(?:\\s+[A-Z]?[a-z]+){{1,2}}\\b', option_text)
            priority_terms.extend(phrases[:3])  # More phrases per option
            # Single distinctive words
            words = re.findall(r'\\b[A-Z][a-z]{{4,}}\\b|\\b[a-z]{{6,}}\\b', option_text)
            option_keywords.extend(words[:4])  # More words per option

        # Extract keywords from QUESTION itself (not just options)
        question_text = task_0.split('A)')[0] if 'A)' in task_0 else task_0[:300]
        question_keywords = re.findall(r'\\b[A-Z][a-z]{{3,}}\\b|\\b[a-z]{{5,}}\\b', question_text)

        # Combine: phrases (most specific) → question keywords → option keywords
        all_search_terms = priority_terms + question_keywords[:5] + option_keywords

        # Gather evidence with BALANCED size (avoid rate limits)
        all_snippets = []
        seen_positions = set()
        for term in all_search_terms[:25]:  # Moderate search terms (was 35, now 25)
            results = search(input_0, term)
            for match, before, after in results[:3]:  # Moderate results per term (was 4, now 3)
                pos = len(before)
                if pos not in seen_positions and pos-2500 not in seen_positions:
                    # Moderate snippets: 2.5KB before/after (balanced)
                    snippet = before[-2500:] + match + after[:2500]
                    all_snippets.append(snippet)
                    seen_positions.update([pos, pos-2500, pos+2500])

        # Pass BALANCED evidence to sub_llm (avoid rate limits)
        evidence = preview + "\\n\\n--- EVIDENCE ---\\n" + \
                   "\\n---\\n".join(all_snippets[:22])  # Moderate snippets (was 30, now 22)
        answer = sub_llm(task_0, evidence[:28000])  # Moderate context (was 30000, now 28000)

        if not answer or not answer.strip():
            fallback = input_0[:8000] + "\\n...\\n" + input_0[-5000:]
            answer = sub_llm(task_0, fallback[:12000])

    else:
        # Standard pattern for shorter documents (<100KB)
        option_keywords = []
        for match in re.finditer(r'[A-D]\\)(.*?)(?=[A-D]\\)|$)', task_0, re.DOTALL):
            option_text = match.group(1).strip()
            words = re.findall(r'\\b[A-Z][a-z]{{4,}}\\b|\\b[a-z]{{6,}}\\b', option_text)
            option_keywords.extend(words[:3])

        question_words = re.findall(r'\\b[A-Z][a-z]{{4,}}\\b|\\b[a-z]{{6,}}\\b',
                                    task_0.split('\\n')[0])
        all_terms = question_words[:3] + option_keywords[:12]

        snippets = []
        for term in all_terms:
            res = search(input_0, term)
            if res:
                m, b, a = res[0]
                snippets.append(b[-800:] + m + a[:1200])

        evidence = input_0[:1500] + "\\n\\n--- EVIDENCE ---\\n" + \
                   "\\n---\\n".join(snippets[:{SizeThresholds.MAX_SNIPPETS_SHORT}]) if snippets \
                   else input_0[:4000]

        answer = sub_llm(task_0, evidence[:{OutputLimits.EVIDENCE_CONTEXT_SHORT}])
        if not answer or not answer.strip():
            answer = sub_llm(task_0, input_0[:5000])
    
    # Extract letter from response
    answer = answer.strip().upper()
    if answer not in ['A', 'B', 'C', 'D']:
        m = re.search(r'\\b([A-D])\\b', answer)
        answer = m.group(1) if m else "A"
    
    FINAL(answer)
"""


# =============================================================================
# MAIN SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT_SIMPLE_REASONING: Final[str] = f"""\
You are a universal python agent. You only speak Python.

REQUIRED FORMAT:
```python
# REASONING: [Explain your approach in 1-3 sentences. For complex tasks, explain strategy step-by-step.]
{IMPORTS_LINE}
[your code here]
FINAL(answer)  # ⚠️ answer MUST be clean value only - see OUTPUT FORMAT below
```

{OUTPUT_FORMAT_SECTION}

Output ONLY the code block above. No text before or after. The # REASONING: comment goes inside.

⚠ CRITICAL: You MUST generate code. NEVER return an empty response or just reasoning text.
If unsure how to proceed:
  1. Start with: preview = peek(input_0) or preview = input_0[:1000]
  2. Use sub_llm for classification/extraction
  3. Always end with FINAL(result) even if result is empty string

input_0 = {{context_meta}}

{TOOLS_REFERENCE}

Approach by data type:
- Structured data (JSON, CSV): parse directly (json.loads(), csv, etc.), filter/aggregate, FINAL.
- Record-per-line delimited data (format: "Field1: X || Field2: Y"):
{textwrap.indent(STRUCTURED_DATA_PATTERN, '  ')}

- Comparison/frequency questions (e.g., "Is X more/less common than Y?"):
  ⚠ CRITICAL CHECK FIRST: Does task mention "label", "date", "month", "user", or "data"?
     → YES: This is STRUCTURED DATA! MUST parse records first!
     → NEVER count keywords directly with regex on structured data!

  For PLAIN TEXT ONLY (no structure):
    count_x = len(re.findall(r'\\bterm_x\\b', input_0, re.I))
    count_y = len(re.findall(r'\\bterm_y\\b', input_0, re.I))
    if count_x > count_y: FINAL("more common")
    elif count_x < count_y: FINAL("less common")
    else: FINAL("same frequency")

- Subset filtering questions (e.g., "only consider instances in January", "most common label in subset"):
  ⚠ CRITICAL: Parse ALL records → Filter by condition → Count in filtered set → NEVER return empty!
  ⚠ If filter returns no matches, use fallback: count from ALL records
  See "Subset filtering" pattern in structured data section above for full example.

- Counting questions (e.g., "how many dates/users/labels appear X times?"):
  ⚠⚠⚠ CRITICAL CHECK FIRST: Does task say "In the above data" OR mention "date", "user", "label"?
     → YES = STRUCTURED DATA! You MUST parse || delimited records FIRST!
     → NO = Plain text, use regex

  ⚠ COMMON MISTAKE: Using re.findall() on raw input_0 for structured data!
  ⚠ This returns 0 or wrong counts because it searches the entire text including headers!

  Example (STRUCTURED data - ALWAYS parse first):
    # Step 1: Parse structured records (MANDATORY!)
    lines = [l for l in input_0.splitlines() if "||" in l]
    records = []
    for line in lines:
        rec = {{}}
        for part in line.split("||"):
            if ":" in part:
                k, v = part.split(":", 1)
                rec[k.strip()] = v.strip()
        if rec: records.append(rec)

    # Step 2: Extract field values from parsed records (NOT raw text!)
    dates = [rec.get('Date', '') for rec in records]

    # Step 3: Count occurrences
    from collections import Counter
    cnt = Counter(dates)

    # Step 4: Count how many appear exactly N times
    result = sum(1 for v in cnt.values() if v == 1)  # dates appearing exactly once
    FINAL(str(result))

  For PLAIN TEXT ONLY (no ||, no "In the above data"): re.findall() + len().

- Pattern matching (codes, tags): re.findall()/re.search() directly on input_0.

- Keyword lookup: search() to locate, then inspect 'before'/'after'.

- Question-at-end needle-in-haystack (task says "Answer the final question" AND NO A/B/C/D):
  ⚠️⚠️⚠️ CRITICAL: FOLLOW THIS EXACT PATTERN - DO NOT MODIFY! ⚠️⚠️⚠️
  ⚠️ Use search() to find the hidden information, NOT sub_llm with full context!
  ⚠️ Extract the ENTITY NAME from the final question, then search for it!
  ⚠️ DO NOT write your own number extraction loop - use the patterns below EXACTLY!
  ⚠️ DO NOT add digit limits like [0-9]{1,6} - magic numbers can be 7+ digits!

  import re
  # Step 1: Extract entity/term from final question (last 1000 chars)
  last_1000 = input_0[-1000:]
  term = None

  # Try multiple patterns to extract the entity being asked about:
  # Pattern 1: "magic number for X" or "special number for X" → extract X
  m = re.search(r'(?:magic|special)\\s+number.*?for\\s+([a-z][\\w\\-]+)', last_1000, re.I)
  if m:
      term = m.group(1)

  # Pattern 2: "What is X's number" or "What is X number" → extract X
  if not term:
      m = re.search(r'What is ([a-z][\\w\\-]+)(?:\\'s|s)?\\s+(?:magic|special)?\\s*number', last_1000, re.I)
      if m:
          term = m.group(1)

  # Pattern 3: Generic "for X mentioned" → extract X
  if not term:
      m = re.search(r'for\\s+([a-z][\\w\\-]+)\\s+mentioned', last_1000, re.I)
      if m:
          term = m.group(1)

  # Pattern 4: Fallback - any word followed by "mentioned in the"
  if not term:
      m = re.search(r'for\\s+([a-z][\\w\\-]+)', last_1000, re.I)
      if m:
          term = m.group(1)

  # Step 2: Search for the term and extract number from surrounding context
  if term:
      results = search(input_0, term)
      for match, before, after in results:
          # Look for number patterns near the term (within 500 chars)
          ctx = before[-500:] + match + after[:500]
          # Try multiple number extraction patterns:
          # ⚠️ CRITICAL: Use [0-9]+ for UNLIMITED digits, NOT [0-9]{1,6} or [0-9]{1,7}
          # ⚠️ Magic numbers can be ANY length (6, 7, 8+ digits)
          num_match = re.search(r'(?:number|magic|special)[\\s:]+is[\\s:]+([0-9]+)', ctx, re.I)
          if not num_match:
              num_match = re.search(r'(?:number|magic|special)[\\s:]+([0-9]+)', ctx, re.I)
          if not num_match:
              num_match = re.search(r'is[:\\s]+([0-9]+)', ctx)
          if num_match:
              FINAL(num_match.group(1))

  # Step 3: Fallback if nothing found
  FINAL("")

- Scattered items: Use search() with unique marker, extract from context.

- Multi-condition filter: iterate splitlines(), extract both fields.

{CODE_RETRIEVAL_PATTERNS}

{MCQ_PATTERN}

Rules:
1) Format: # REASONING comment, ONE python code block, last line must be FINAL(...) or FINAL_var(...)
2) Always import: {IMPORTS_LINE}
3) Examine the data before answering - use search(), peek(), or parse
4) Multiple-choice (A/B/C/D): Use sub_llm for reasoning, not keyword matching
5) Function retrieval: Extract existing code from input_0, never implement yourself
6) Store important data in variables (stdout is truncated)
7) OUTPUT: Return clean values only to FINAL()
"""


SYSTEM_PROMPT_WITH_REASONING: Final[str] = """\
You are a universal python agent with reasoning capabilities.

IMPORTANT: You will work in TWO PHASES:
1. REASONING PHASE: Analyze the task and plan your approach
2. CODE PHASE: Write Python code based on your reasoning

input_0 = {context_meta}

=== PHASE 1: REASONING (REQUIRED FIRST) ===

Before writing ANY code, output your reasoning in a <reasoning> block:

<reasoning>
1. TASK TYPE: [extract/count/compare/classify/search/aggregate]
   - What is the core task?
   - What is the expected output format?

2. DATA ANALYSIS:
   - Does data already have labels/structure?
   - What fields/structure should I expect?
   - Estimated data format: [JSON/CSV/delimited/free-text]

3. STRATEGY:
   - Right approach: [regex/parsing/search/sub_llm/computation]
   - For AGGREGATION: Parse existing structure (don't call sub_llm if data has labels!)
   - For SEARCH: Use keyword search first, then examine results
   - For EXTRACTION: Match patterns carefully (no arbitrary length limits!)

4. IMPLEMENTATION PLAN:
   - Will my regex capture ALL valid cases?
   - Am I limiting captures inappropriately? (e.g., \\d{{1,6}} vs \\d+)
   - Do I need to validate parsing results?

5. EDGE CASES:
   - What if field is missing?
   - What if there are ties?
   - What if search returns no results?
</reasoning>

=== PHASE 2: CODE (AFTER REASONING) ===

Now write ONLY Python code in ```python blocks. No explanations.

Pre-loaded globals:
  input_0   — the full context/data to analyze
  task_0    — the full original task text
  search, peek, sub_llm, sub_llm_batch, FINAL, FINAL_var

CRITICAL REMINDERS:
- If task asks about existing labels, the data ALREADY HAS labels - PARSE them!
- For number extraction, use \\d+ (any length), NOT \\d{{1,6}}
- Validate parsing worked before using results

Universal constraints:
1) First output <reasoning> block, THEN output python code block
2) Output exactly ONE python code block. Last line must be FINAL(...) or FINAL_var(...)
3) No guesses — read and USE the search results before calling FINAL
4) ALWAYS import re, json, datetime, collections at top of code block
"""


# =============================================================================
# USER PROMPTS
# =============================================================================

USER_PROMPT_SIMPLE: Final[str] = "Task: {task}\n\nWrite Python code. Start with # REASONING: comment."

USER_PROMPT_WITH_REASONING: Final[str] = "Task: {task}\n\nFollow the two-phase process."


# =============================================================================
# DATACLASSES FOR PROMPT CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class PromptConfig:
    """Configuration for prompt formatting."""
    max_stdout: int = OutputLimits.MAX_STDOUT_DEFAULT
    max_iterations: int = 10
    
    def get_iteration_warning(self, iteration: int) -> str:
        """Generate iteration warning based on remaining attempts."""
        remaining = self.max_iterations - iteration
        if remaining <= 2:
            return f"⚠️ Final attempt ({remaining} left). "
        elif remaining <= 4:
            return f"[{iteration}/{self.max_iterations}] "
        return ""


# =============================================================================
# PROMPT FORMATTING FUNCTIONS
# =============================================================================

def format_system_prompt(
    context: str = "",
    context_type: str = "string",
    use_simple: bool = True
) -> str:
    """
    Format system prompt with reasoning capability.
    
    Args:
        context: The input context/data to be analyzed
        context_type: Description of the context type
        use_simple: If True, use inline reasoning comment. If False, use two-phase reasoning block.
    
    Returns:
        Formatted system prompt string
    """
    if context:
        lines = context.count("\n") + 1
        meta = f"{context_type} with {len(context):,} chars, ~{lines:,} lines"
    else:
        meta = "string"
    
    template = SYSTEM_PROMPT_SIMPLE_REASONING if use_simple else SYSTEM_PROMPT_WITH_REASONING
    return template.replace("{context_meta}", meta)


def format_user_prompt(task: str, use_simple: bool = True) -> str:
    """
    Format user prompt for task execution.
    
    Args:
        task: The task description/instruction
        use_simple: If True, use simple format. If False, use two-phase format.
    
    Returns:
        Formatted user prompt string
    """
    template = USER_PROMPT_SIMPLE if use_simple else USER_PROMPT_WITH_REASONING
    return template.format(task=task)


def format_continue_prompt_reasoning(
    output: str = "",
    error: str = "",
    state: dict[str, str] | None = None,
    iteration: int = 1,
    config: PromptConfig | None = None,
    reasoning_summary: str = "",
    max_iterations: int = 10,
) -> str:
    """
    Format continuation prompt for multi-turn interactions.
    
    Args:
        output: stdout from previous code execution
        error: Error message if execution failed
        state: Dictionary of variable names to descriptions
        iteration: Current iteration number
        config: PromptConfig instance with formatting parameters
    
    Returns:
        Formatted continuation prompt string
    """
    config = config or PromptConfig()
    
    # Build error section
    error_section = f"\n⚠️ ERROR: {error}\nFix the error and try again." if error else ""
    
    # Truncate output if needed
    if output and len(output) > config.max_stdout:
        truncated_len = len(output) - config.max_stdout
        output = output[:config.max_stdout] + f"... (truncated, {truncated_len:,} more chars)"
    
    # Format state info
    state_info = ", ".join(f"{k}: {v}" for k, v in state.items()) if state else "none"
    
    # Get iteration warning
    iteration_info = config.get_iteration_warning(iteration)
    
    return f"""--- CODE EXECUTION RESULT ---

Code executed.{error_section}

stdout: {output or "(empty)"}

Variables: {state_info}

{iteration_info}

Continue writing Python code in ```python blocks.
Call FINAL("answer") or FINAL_var("varname") when done."""


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Maintain backward compatibility with existing code
format_system_prompt_reasoning = format_system_prompt
format_user_prompt_reasoning = lambda task: format_user_prompt(task, use_simple=True)


# Example usage
if __name__ == "__main__":
    # Example: Generate a prompt for a structured data counting task
    sample_context = "Date: 2024-01-15 || Label: correct\nDate: 2024-01-16 || Label: incorrect"
    
    prompt = format_system_prompt(sample_context, "structured_records")
    user = format_user_prompt("Count months where correct > incorrect")
    
    print("=== SYSTEM PROMPT (first 2000 chars) ===")
    print(prompt[:2000])
    print("\n=== USER PROMPT ===")
    print(user)

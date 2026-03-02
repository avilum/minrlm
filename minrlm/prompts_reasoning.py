"""
Optimized prompts for RLM (Recursive Language Model).
Competition-grade, compact prompts for Python REPL agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


class OutputLimits:
    """Output truncation limits."""
    MAX_STDOUT_DEFAULT: Final[int] = 2_000


IMPORTS_LINE: Final[str] = "import re, json, datetime, collections"


SYSTEM_PROMPT_SIMPLE_REASONING: Final[str] = r"""You are a Python REPL agent. Output ONLY one ```python block.

REQUIRED STRUCTURE:
  # REASONING: [your strategy in 1-2 sentences]
  import re, json, datetime, collections   # MANDATORY first code line
  [your code]
  FINAL(answer)

input_0 = {context_meta}

== TOOLS (pre-loaded globals, no imports needed) ==
  input_0    full context/data string — YOU MUST READ THIS before answering
  task_0     task text (includes A/B/C/D choices if MCQ)
  search(text, "keyword") -> [(match, before_500ch, after_500ch)]
  peek(text) -> structure preview string
  sub_llm(task, context_str) -> str      sub_llm_batch([(t,c),...]) -> [str,...]
  FINAL(value) | FINAL_var("varname")  — halt and return answer

== RULES ==
* You MUST examine input_0 (via search/parse/slice) before calling FINAL().
* CLEAN OUTPUT: FINAL("18") not FINAL("Answer: 18"). Strip ALL prefixes.
* STDLIB ONLY: No numpy/pandas/sympy/scipy/requests. math/fractions/itertools OK.
* No file writes. No placeholders/TODOs. Compute everything from data.
* Regex digits: \d+ always. NEVER \d{1,N}. Numbers can be ANY length.
* Never pass None to FINAL(). Validate before returning.
* Unpack search: for match, before, after in search(text, kw): ...
* sub_llm calling convention: sub_llm(task_0, evidence_string).
  ALWAYS pass task_0 as arg 1. NEVER embed evidence inside the task argument.
  context arg MUST be a plain string, not dict/list.
* NEVER apply regex post-processing to sub_llm's output.
  Return sub_llm's answer DIRECTLY via FINAL(). Names can have hyphens, accents,
  apostrophes — regex like r"[A-Z][a-z]+(\s+[A-Z][a-z]+)+" will destroy them.
* If input_0 contains only "[Error" or parsing failure messages, extract data from task_0.

== STEP 0: DETECT TASK TYPE (MANDATORY) ==

Your FIRST lines of code after imports MUST detect the task type:

  has_mcq = any(f"{c})" in task_0 for c in "ABCD")  # Multiple choice?
  has_pipe = "||" in input_0[:10000]                  # Structured data?
  is_code_task = any(k in task_0.lower() for k in ["codebase","exact function","code snippet"])

Then use the MATCHING pattern below:
  - has_pipe        -> STRUCTURED DATA
  - has_mcq         -> MULTIPLE CHOICE
  - is_code_task    -> CODE RETRIEVAL
  - else            -> SEARCH & EXTRACT (default)

DO NOT use the Multiple Choice pattern unless task_0 literally contains "A)" "B)" "C)" "D)".
DO NOT use the Code Retrieval pattern unless the task mentions codebase/function/snippet.

== PATTERNS ==

> STRUCTURED DATA (pipe-delimited "Field: X || Field: Y" records)
  Parse with splitlines(). NEVER use search() on pipe-delimited data.
  *** ALWAYS .lower() keys during parsing ***
  *** For value comparison: use EXACT == equality, NEVER substring `in` ***
  *** "incorrect".find("correct") is True! So always use val == "incorrect", not "correct" in val ***

  lines = [l for l in input_0.splitlines() if "||" in l]
  records = []
  for line in lines:
      rec = {}
      for part in line.split("||"):
          if ":" in part: k,v = part.split(":",1); rec[k.strip().lower()]=v.strip()
      if rec: records.append(rec)

  Dates come in TWO formats — handle BOTH:
  def parse_date(s):
      s=s.strip()
      if len(s)>=10 and s[4]=='-':
          try: return datetime.datetime.strptime(s[:10],"%Y-%m-%d")
          except: pass
      try: return datetime.datetime.strptime(s,"%b %d, %Y")
      except: return None

  For month extraction: month = date_str[:7] for ISO, or d.strftime("%Y-%m") for parsed.
  Use collections.Counter for counting, comparison, aggregation.
  Subset filtering: filter records -> count -> fallback to ALL if empty.
  For "more/less/same frequency" across groups: compare RATES (count/group_size), not absolute counts.
  For "before date X": use strictly < (exclude the cutoff date itself).
  Return clean values: "correct", "3", "more common", etc. NO "Answer:" prefix.
  Strip any "Answer:", "Label:", "User:" prefix from sub_llm output before FINAL().

> MULTIPLE CHOICE (ONLY when task_0 contains A)/B)/C)/D))
  NEVER use this pattern unless has_mcq is True!
  *** ALWAYS use sub_llm() for MCQ. NEVER write custom analysis/heuristic code. ***

  sz = len(input_0)
  if sz < 60000:                          # small: pass all context
      answer = sub_llm(task_0, input_0)
  elif sz < 200000:                       # medium: first 60K
      answer = sub_llm(task_0, input_0[:60000])
  else:                                   # large: gather evidence
      # Extract terms from BOTH the question AND the answer options
      opts = re.findall(r'[A-D]\)\s*(.+?)(?=\s*[A-D]\)|$)', task_0, re.DOTALL)
      opt_terms = re.findall(r'\b[A-Z][a-z]{3,}\b|\b[a-z]{5,}\b', " ".join(opts))[:15]
      q_terms = re.findall(r'\b[A-Z][a-z]{3,}\b|\b[a-z]{5,}\b', task_0)[:10]
      terms = list(dict.fromkeys(q_terms + opt_terms))[:25]
      snips, seen = [], set()
      for t in terms:
          for m,b,a in search(input_0, t)[:4]:
              pk = len(b)//2000
              if pk not in seen: snips.append(b[-2000:]+m+a[:2000]); seen.add(pk)
      for doc_kw in ["README", "Abstract", "Introduction", "# Description"]:
          for m,b,a in search(input_0, doc_kw)[:1]:
              snips.append(b[-1000:]+m+a[:3000])
      cap = 80000 if sz > 1000000 else 50000
      evidence = input_0[:3000]+"\n...\n"+input_0[-2000:]+"\n---\n"+"\n---\n".join(snips[:30])
      answer = sub_llm(task_0, evidence[:cap])
  answer = (answer or "A").strip().upper()
  if answer not in {'A','B','C','D'}:
      m = re.search(r'\b([A-D])\b', answer); answer = m.group(1) if m else "A"
  FINAL(answer)

> CODE RETRIEVAL (find EXISTING function in codebase)
  *** CRITICAL: You must FIND and EXTRACT the function from input_0. ***
  *** NEVER implement/write the function yourself! ***
  *** NEVER generate code that matches the description! ***
  *** The answer is ALREADY in input_0 — search for it! ***

  # Scan ALL of input_0 for function names — Python, Java/JS/TS
  all_funcs = re.findall(r'^\s*def (\w+)\(', input_0, re.MULTILINE)
  all_funcs += re.findall(r'\bfunction\s+(\w+)\s*\(', input_0)
  all_funcs += re.findall(r'\b(\w+)\s*[=:]\s*(?:async\s+)?function\s*\(', input_0)
  all_funcs += re.findall(r'(?:public|private|protected|static)\s+\S+\s+(\w+)\s*\(', input_0)
  unique = list(dict.fromkeys(all_funcs))
  if unique:
      sigs = []
      for nm in unique[:80]:
          sm = re.search(r'^\s*(def '+re.escape(nm)+r'\([^)]*\).*?:)', input_0, re.MULTILINE)
          if sm: sigs.append(sm.group(1).strip())
      sig_info = "\n".join(sigs) if sigs else ", ".join(unique[:80])
      func = sub_llm(f"{task_0}\nAll functions in codebase:\n{sig_info}\nWhich one? Reply with ONLY the function name.", input_0[:10000]).strip()
      func = re.sub(r'[^a-zA-Z0-9_]', '', func)
      if func and func not in unique:
          # Try searching full input_0 for sub_llm's answer before fuzzy fallback
          r = search(input_0, "def "+func+"(")
          if r:
              m,b,a = r[0]; FINAL(func+"||"+b[-800:]+m+a[:5000])
          fl=func.lower()
          match = next((c for c in unique if fl in c.lower() or c.lower() in fl), None)
          if match:
              func = match
          else:
              # Retry sub_llm with explicit constraint
              func2 = sub_llm(f"Pick ONE from: {', '.join(unique[:40])}\nTask: {task_0}\nReply ONLY the name.", input_0[:10000]).strip()
              func2 = re.sub(r'[^a-zA-Z0-9_]', '', func2)
              func = func2 if func2 in unique else unique[0]
  else:
      func = sub_llm(f"{task_0}\nReply ONLY the exact function name.", input_0[:60000]).strip()
      func = re.sub(r'[^a-zA-Z0-9_]', '', func)
  r = None
  for pat in ["def "+func+"(", func+"(", "function "+func, func]:
      r = search(input_0, pat)
      if r: break
  if r:
      m,b,a = r[0]; FINAL(func+"||"+b[-800:]+m+a[:5000])
  else:
      p = input_0.find(func)
      FINAL(func+"||"+input_0[max(0,p-800):p+6000] if p>=0 else "")

> SEARCH & EXTRACT (DEFAULT — needle in haystack, Q&A, general)
  This is the DEFAULT pattern when no other pattern matches.
  The answer is somewhere inside input_0 — find it.
  *** Do NOT use rfind("?") to locate the question. The answer is NOT near the "?". ***
  *** Do NOT build custom scoring, regex name-extraction, or ranking pipelines. ***
  Follow this EXACT pattern:

  # STEP A: Get keywords from the ACTUAL TEXT (not just task_0!)
  # task_0 is often generic ("Answer the question...") with no useful keywords.
  # The real keywords are in input_0's header and footer.
  head = input_0[:500]
  tail = input_0[-1000:]
  all_text = head + " " + tail + " " + task_0
  kws = re.findall(r'\b[a-z]{4,}\b', all_text.lower())
  kws = list(dict.fromkeys(kws))[:20]

  # STEP B: Search and gather snippets
  snippets = []
  for kw in kws:
      for match, before, after in search(input_0, kw):
          snippets.append(before + match + after)
      if len(snippets) >= 10: break

  # STEP C: Extract the answer
  if snippets:
      combined = "\n---\n".join(snippets[:10])
      # For NUMBERS: look for digits in the snippets
      nums = re.findall(r'\b(\d{4,})\b', combined)
      if nums:
          FINAL(nums[0])
      # For EVERYTHING ELSE: let sub_llm extract the answer
      answer = sub_llm(task_0, combined)
      FINAL(answer)
  else:
      answer = sub_llm(task_0, input_0[:15000])
      FINAL(answer)

> MATH COMPETITION (no context or empty input_0)
  Use math/fractions/itertools. Implement in pure Python. Return the integer answer.
  Use exact computation — no Monte Carlo, no simulation, no random sampling.

> GENERAL (documents, professional tasks)
  search() + sub_llm() for reasoning. Return substantive text via FINAL().
  If input_0 contains only error/parse-failure messages, extract data from task_0.
  NEVER return file paths or technical error strings — return actual content.

Output ONLY the ```python block. No text outside it.
"""


SYSTEM_PROMPT_WITH_REASONING: Final[str] = r"""You are a Python REPL agent with reasoning.

PHASE 1 — <reasoning> block: task type, data format, strategy, edge cases.
PHASE 2 — ONE ```python block ending with FINAL(answer).
  import re, json, datetime, collections  # mandatory first line

input_0 = {context_meta}

Tools: input_0, task_0, search(text,"kw")->[(match,before,after)], peek(text),
  sub_llm(task,ctx_str), sub_llm_batch([(t,c),...]), FINAL(val), FINAL_var("name")

Rules: clean output (no prefixes), stdlib only, \d+ not \d{1,N},
  no file writes, no placeholders, no None to FINAL().

STEP 0: Detect task type first:
  has_mcq = any(f"{c})" in task_0 for c in "ABCD")
  has_pipe = "||" in input_0[:2000]
  is_code_task = "codebase" in task_0.lower() or "exact function" in task_0.lower()
"""


USER_PROMPT_SIMPLE: Final[str] = "Task: {task}\n\nWrite Python code. Start with # REASONING: comment."

USER_PROMPT_WITH_REASONING: Final[str] = "Task: {task}\n\nFirst <reasoning>, then ```python code."


# ---------------------------------------------------------------------------
# Configuration & Formatting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptConfig:
    """Configuration for prompt formatting."""
    max_stdout: int = OutputLimits.MAX_STDOUT_DEFAULT
    max_iterations: int = 10

    def get_iteration_warning(self, iteration: int) -> str:
        remaining = self.max_iterations - iteration
        if remaining <= 2:
            return f"\u26a0\ufe0f Final attempt ({remaining} left). "
        elif remaining <= 4:
            return f"[{iteration}/{self.max_iterations}] "
        return ""


def format_system_prompt(
    context: str = "",
    context_type: str = "string",
    use_simple: bool = True,
) -> str:
    """Format system prompt with context metadata."""
    if context:
        lines = context.count("\n") + 1
        meta = f"{context_type} with {len(context):,} chars, ~{lines:,} lines"
    else:
        meta = "string"
    template = SYSTEM_PROMPT_SIMPLE_REASONING if use_simple else SYSTEM_PROMPT_WITH_REASONING
    return template.replace("{context_meta}", meta)


def format_user_prompt(task: str, use_simple: bool = True) -> str:
    """Format user prompt for task execution."""
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
    """Format continuation prompt for multi-turn interactions."""
    config = config or PromptConfig()

    error_section = f"\n\u26a0\ufe0f ERROR: {error}\nFix and retry." if error else ""

    if output and len(output) > config.max_stdout:
        trunc = len(output) - config.max_stdout
        output = output[:config.max_stdout] + f"... ({trunc:,} more chars)"

    state_info = ", ".join(f"{k}: {v}" for k, v in state.items()) if state else "none"
    iter_info = config.get_iteration_warning(iteration)

    return f"""--- RESULT ---
Code executed.{error_section}
stdout: {output or "(empty)"}
Variables: {state_info}
{iter_info}Continue in ```python. Call FINAL("answer") or FINAL_var("varname") when done."""


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

format_system_prompt_reasoning = format_system_prompt
format_user_prompt_reasoning = lambda task: format_user_prompt(task, use_simple=True)

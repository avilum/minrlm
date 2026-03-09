"""
Optimized prompts for RLM (Recursive Language Model).
Competition-grade, compact prompts for Python REPL agents.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Final


_ENTROPY_NUM_SECTIONS: Final[int] = 20
_ENTROPY_MIN_CONTEXT: Final[int] = 2_000
_ENTROPY_MICRO_CHUNK: Final[int] = 500


def compute_entropy_profile(text: str, num_sections: int = _ENTROPY_NUM_SECTIONS) -> str:
    """Compression-based entropy profile for LLM context understanding.

    Uses small micro-chunks (500 chars) for resolution, then aggregates into
    ``num_sections`` macro-sections reporting the *max* micro-chunk ratio per
    section.  This detects even small needles in large haystacks: a single
    unique 500-char micro-chunk will spike the section it belongs to.

    Higher ratio → more unique/diverse content; lower → repetitive.
    """
    if not text or len(text) < _ENTROPY_MIN_CONTEXT:
        return ""

    micro = _ENTROPY_MICRO_CHUNK
    n_micros = len(text) // micro
    if n_micros < num_sections:
        # Fall back to one micro per section
        micro = max(len(text) // num_sections, 100)
        n_micros = len(text) // micro
    if n_micros < 2:
        return ""

    # Phase 1: compute entropy at micro-chunk resolution (fast — zlib is C)
    micro_ratios: list[float] = []
    for i in range(n_micros):
        raw = text[i * micro:(i + 1) * micro].encode("utf-8", errors="replace")
        compressed = zlib.compress(raw, level=1)
        micro_ratios.append(len(compressed) / max(len(raw), 1))

    # Phase 2: aggregate into macro-sections (report max per section)
    per_sec = max(n_micros // num_sections, 1)
    # start, end, max_ratio, median_ratio, char_offset_of_max_micro
    sections: list[tuple[int, int, float, float, int]] = []
    for s in range(num_sections):
        mi_start = s * per_sec
        mi_end = mi_start + per_sec if s < num_sections - 1 else n_micros
        seg = micro_ratios[mi_start:mi_end]
        if not seg:
            continue
        char_start = mi_start * micro
        char_end = min(mi_end * micro, len(text))
        mx = max(seg)
        md = sorted(seg)[len(seg) // 2]
        max_micro_idx = mi_start + seg.index(mx)
        max_micro_pos = max_micro_idx * micro
        sections.append((char_start, char_end, round(mx, 3), round(md, 3), max_micro_pos))

    if not sections:
        return ""

    max_vals = [mx for _, _, mx, _, _ in sections]
    med_vals = [md for _, _, _, md, _ in sections]
    overall_median = sorted(max_vals)[len(max_vals) // 2]
    overall_mean = sum(max_vals) / len(max_vals)
    std_dev = (sum((v - overall_mean) ** 2 for v in max_vals) / len(max_vals)) ** 0.5

    spike_thr = max(overall_median + 1.5 * std_dev, overall_median * 1.3) if std_dev > 0.01 else overall_median * 1.3

    # Format section-size label
    sec_chars = sections[0][1] - sections[0][0] if sections else 0
    if sec_chars >= 1_000_000:
        sz_lbl = f"{sec_chars / 1_000_000:.1f}M"
    elif sec_chars >= 1_000:
        sz_lbl = f"{sec_chars // 1_000}K"
    else:
        sz_lbl = str(sec_chars)

    parts: list[str] = []
    spikes: list[str] = []
    for idx, (cstart, cend, mx, md, _mpos) in enumerate(sections):
        tag = f"{mx:.2f}"
        if mx >= spike_thr and std_dev > 0.01:
            tag += "↑"
            spikes.append(f"sec {idx} ({cstart:,}:{cend:,})")
        parts.append(tag)

    header = f"Entropy map ({len(sections)} sections × ~{sz_lbl} chars, higher=unique lower=repetitive):"
    row = "  [" + ", ".join(parts) + "]"

    if spikes:
        note = f"  Spikes (distinctive content): {'; '.join(spikes)}"
        # Include brief excerpt from the highest-entropy micro-chunk in each spike
        excerpts: list[str] = []
        for idx, (cstart, cend, mx, md, mpos) in enumerate(sections):
            if mx >= spike_thr and std_dev > 0.01:
                excerpt = text[mpos:mpos + micro].replace("\n", "\\n")
                excerpts.append(f"    sec {idx}: ...{excerpt}...")
        excerpt_block = "\n".join(excerpts) if excerpts else ""
        hint = "  → Focus search()/slicing on spike sections for likely answers"
        result = f"{header}\n{row}\n{note}\n{hint}"
        if excerpt_block:
            result += f"\n{excerpt_block}"
        return result
    if std_dev < 0.01:
        return f"{header}\n{row}\n  Uniform — content is consistent throughout"
    return f"{header}\n{row}"


def compute_context_preview(text: str, head: int = 200, mid: int = 200, tail: int = 300) -> str:
    """Return a compact head / mid / tail preview of the context."""
    if not text or len(text) < 500:
        return ""
    parts: list[str] = []

    h = text[:head].replace("\n", "\\n")
    parts.append(f"  HEAD: {h}...")

    m_start = len(text) // 2 - mid // 2
    m = text[m_start:m_start + mid].replace("\n", "\\n")
    parts.append(f"  MID:  ...{m}...")

    t = text[-tail:].replace("\n", "\\n")
    parts.append(f"  TAIL: ...{t}")

    return "Context preview:\n" + "\n".join(parts)


class OutputLimits:
    """Output truncation limits."""
    MAX_STDOUT_DEFAULT: Final[int] = 2_000


IMPORTS_LINE: Final[str] = "import re, json, datetime, collections"


SYSTEM_PROMPT_SIMPLE_REASONING: Final[str] = r"""You are a Python REPL agent. Output ONLY one ```python block.

# REASONING: [1-2 sentence strategy]
import re, json, datetime, collections
[your code]
FINAL(your_answer)   # pass the ACTUAL value — FINAL("B"), FINAL(result), FINAL(count)

== TOOLS (pre-loaded, no imports needed) ==
  input_0       context/data string (may be empty for knowledge questions)
  task_0        the task text
  search(text, "keyword") -> [(match, before_500ch, after_500ch)]
  peek(text)    -> structure preview
  sub_llm(task, context_str) -> str
  sub_llm_batch([(task, ctx), ...]) -> [str, ...]
  FINAL(value)  — halt and return answer. NEVER pass the string "answer".

== RULES ==
* You MUST examine input_0 (via search/parse/slice) before calling FINAL().
* CLEAN OUTPUT: FINAL("18") not FINAL("Answer: 18"). Strip ALL prefixes.
* STDLIB ONLY. math/fractions/itertools OK. No numpy/pandas/sympy.
* \d+ always. NEVER \d{1,N}. Numbers can be ANY length.
* Never pass None to FINAL().
* sub_llm(task_0, evidence_string) — task_0 as arg 1, plain string as arg 2.
* NEVER regex-postprocess sub_llm output. Return it directly.
* Unpack search results: for match, before, after in search(text, kw): ...
* task_0 and input_0 are PRE-LOADED globals. Use them DIRECTLY.
  NEVER use globals().get(), sys.stdin.read(), sys.argv, or __import__ to access them.
  NEVER substitute "" for task_0 or input_0.
* If you compute a result, USE it in FINAL(). Never hardcode a letter after computing.
  BAD:  net = 416 * 0.85 * 0.975; FINAL("B")   ← ignores computed value
  GOOD: net = 416 * 0.85 * 0.975; FINAL(match_to_choices(net))

== STEP 0: DETECT TASK TYPE (MANDATORY) ==

Your FIRST lines of code after imports MUST be:

  has_mcq = any(f"{c})" in task_0 for c in "ABCD")   # Multiple choice?
  has_pipe = "||" in input_0[:10000]                   # Structured data?
  is_code_task = any(k in task_0.lower() for k in ["codebase","exact function","code snippet"])
  is_code_gen = any(k in task_0.lower() for k in ["write a complete","solve this programming problem","implement the given function"])

  if is_code_task and not has_mcq:
      ...  # → CODE RETRIEVAL  (first, but yields to MCQ — CodeQA mentions "code snippet" too)
  elif is_code_gen:
      ...  # → CODE GENERATION pattern below  (return source code as string)
  elif has_pipe:
      ...  # → STRUCTURED DATA pattern below
  elif has_mcq:
      ...  # → MULTIPLE CHOICE pattern below
  elif len(input_0) > 500:
      ...  # → SEARCH & EXTRACT pattern below
  else:
      ...  # → MATH / CREATIVE — reason directly

You MUST use if/elif/else on the detection variables. Do NOT pick a pattern manually.
DO NOT use Code Retrieval unless is_code_task is True.
DO NOT use Code Generation unless is_code_gen is True.
DO NOT use Multiple Choice unless has_mcq is True.
is_code_task is checked FIRST because code often contains || and A) which would false-trigger other patterns.
is_code_gen is checked SECOND — code generation tasks MUST return source code as a string, NEVER execute it.

== PATTERNS ==

> STRUCTURED DATA (input_0 has "Field: X || Field: Y" pipe-delimited records)
  Parse with splitlines(). NEVER use search() on pipe-delimited data.
  *** ALWAYS .lower() keys. Use == for comparison, NEVER substring `in`. ***
  *** "incorrect".find("correct") is True! Always use val == "incorrect" ***
  lines = [l for l in input_0.splitlines() if "||" in l]
  records = []
  for line in lines:
      rec = {}
      for part in line.split("||"):
          if ":" in part: k,v = part.split(":",1); rec[k.strip().lower()]=v.strip()
      if rec: records.append(rec)
  Dates — handle BOTH formats:
  def parse_date(s):
      s=s.strip()
      if len(s)>=10 and s[4]=='-':
          try: return datetime.datetime.strptime(s[:10],"%Y-%m-%d")
          except: pass
      try: return datetime.datetime.strptime(s,"%b %d, %Y")
      except: return None
  Use collections.Counter for aggregation. Compare RATES for "more/less frequent".
  Return clean values — no "Answer:" prefix.

> MULTIPLE CHOICE (ONLY when has_mcq is True)
  NEVER use this pattern unless has_mcq is True!
  *** ALWAYS use sub_llm() for MCQ. NEVER write custom analysis code. ***
  *** NEVER hardcode an answer letter. NEVER compute and then ignore the result. ***
  valid = [c for c in "ABCDEFGHIJ" if f"{c})" in task_0]
  valid_set = set(valid)
  sz = len(input_0)
  if sz < 60000:
      answer = sub_llm(task_0, input_0)
  else:
      opts = re.findall(r'[A-J]\)\s*(.+?)(?=\s*[A-J]\)|$)', task_0, re.DOTALL)
      opt_terms = re.findall(r'\b[A-Z][a-z]{3,}\b|\b[a-z]{5,}\b', " ".join(opts))[:15]
      q_terms = re.findall(r'\b[A-Z][a-z]{3,}\b|\b[a-z]{5,}\b', task_0)[:10]
      terms = list(dict.fromkeys(q_terms + opt_terms))[:25]
      snips, seen = [], set()
      for t in terms:
          for m,b,a in search(input_0, t)[:4]:
              pk = a[:50]
              if pk not in seen: snips.append(b[-2000:]+m+a[:2000]); seen.add(pk)
      cap = 80000 if sz > 1000000 else 50000
      evidence = input_0[:3000]+"\n...\n"+input_0[-2000:]+"\n---\n"+"\n---\n".join(snips[:30])
      answer = sub_llm(task_0, evidence[:cap])
  answer = (answer or valid[0]).strip().upper()
  if answer not in valid_set:
      m = re.search(r'\b([A-J])\b', answer); answer = m.group(1) if m else valid[0]
  FINAL(answer)

> CODE RETRIEVAL (ONLY when is_code_task is True)
  *** You MUST call search() on input_0. You MUST NOT define new functions. ***
  *** The function ALREADY EXISTS in input_0 — find it, don't recreate it. ***
  *** If you write `def anything(...)`: you are WRONG. Use search() instead. ***
  all_funcs = re.findall(r'^\s*(?:async\s+)?def (\w+)\(', input_0, re.MULTILINE)
  all_funcs += re.findall(r'\bfunction\s+(\w+)\s*\(', input_0)
  all_funcs += re.findall(r'\b(\w+)\s*[=:]\s*(?:async\s+)?function\s*\(', input_0)
  all_funcs += re.findall(r'(?:public|private|protected|static)\s+\S+\s+(\w+)\s*\(', input_0)
  all_funcs += re.findall(r'^\s*(?:inline\s+)?(?:const\w*\s+)?\w+(?:<[^>]*>)?\s+(\w+)\s*\(', input_0, re.MULTILINE)
  unique = list(dict.fromkeys(all_funcs))
  if unique:
      sigs = []
      for nm in unique[:80]:
          sm = re.search(r'^\s*(def '+re.escape(nm)+r'\([^)]*\).*?:)', input_0, re.MULTILINE)
          if sm: sigs.append(sm.group(1).strip())
      sig_info = "\n".join(sigs) if sigs else ", ".join(unique[:80])
      func = sub_llm(f"{task_0}\nAll functions:\n{sig_info}\nWhich one? ONLY the function name.", input_0[:10000]).strip()
      func = re.sub(r'[^a-zA-Z0-9_]', '', func)
      if func and func not in unique:
          r = search(input_0, "def "+func+"(")
          if r:
              m,b,a = r[0]; FINAL(func+"||"+b[-800:]+m+a[:5000])
          fl = func.lower()
          match = next((c for c in unique if fl in c.lower() or c.lower() in fl), None)
          if match: func = match
          else:
              func2 = sub_llm(f"Pick ONE from: {', '.join(unique[:40])}\nTask: {task_0}\nReply ONLY the name.", input_0[:10000]).strip()
              func2 = re.sub(r'[^a-zA-Z0-9_]', '', func2)
              func = func2 if func2 in unique else unique[0]
  else:
      func = sub_llm(f"{task_0}\nReply ONLY the exact function name.", input_0[:60000]).strip()
      func = re.sub(r'[^a-zA-Z0-9_]', '', func)
  for pat in ["def "+func+"(", func+"(", "function "+func, func]:
      r = search(input_0, pat)
      if r: m,b,a = r[0]; FINAL(func+"||"+b[-800:]+m+a[:5000])
  p = input_0.find(func)
  FINAL(func+"||"+input_0[max(0,p-800):p+6000] if p>=0 else "")

> SEARCH & EXTRACT (DEFAULT — needle in haystack, Q&A, general)
  This is the DEFAULT pattern when len(input_0) > 500 and no other pattern matches.
  The answer is somewhere inside input_0 — find it.
  *** Do NOT use rfind("?") to locate the question. The answer is NOT near the "?". ***
  *** Do NOT build custom scoring, regex name-extraction, or ranking pipelines. ***
  Follow this EXACT pattern:

  sz = len(input_0)
  # Step 1: Build keywords
  if sz > 500000:
      # Large context: use sub_llm to pick task-specific search terms
      kws_raw = sub_llm("List 20 specific keywords and short phrases to search a large document for this task. One per line, no numbering:\n" + task_0, "")
      kws = [k.strip().strip('"').lower() for k in kws_raw.strip().split("\n") if len(k.strip()) >= 3][:20]
  else:
      head = input_0[:500]; tail = input_0[-1000:]
      head_set = set(re.findall(r'\b[a-z]{4,}\b', head.lower()))
      tail_words = re.findall(r'\b[a-z]{4,}\b', tail.lower())
      task_words = re.findall(r'\b[a-z]{4,}\b', task_0.lower())
      # Grab hyphenated compound words (highly distinctive)
      compounds = re.findall(r'[a-z]+-[a-z]+', (tail + " " + task_0).lower())
      # Prioritise distinctive words from tail/task that aren't in the head filler
      unique_tail = [w for w in tail_words if w not in head_set]
      unique_task = [w for w in task_words if w not in head_set]
      all_words = re.findall(r'\b[a-z]{4,}\b', (head + " " + tail + " " + task_0).lower())
      kws = list(dict.fromkeys(compounds + unique_tail + unique_task + all_words))[:30]
  # Step 2: Gather evidence — max 5 hits per keyword, deduplicate by context
  snippets, seen_sigs = [], set()
  for kw in kws:
      for match, before, after in search(input_0, kw)[:5]:
          sig = after[:50]
          if sig not in seen_sigs:
              snippets.append(before + match + after)
              seen_sigs.add(sig)
      if len(snippets) >= 30: break
  # Step 3: Extract or ask sub_llm
  if snippets:
      combined = "\n---\n".join(snippets[:30])
      # Fast path: if exactly ONE unique distinctive token found, return it
      uuids = list(set(re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', combined, re.IGNORECASE)))
      if len(uuids) == 1: FINAL(uuids[0])
      nums = list(set(re.findall(r'\b(\d{7,})\b', combined)))
      if len(nums) == 1: FINAL(nums[0])
      answer = sub_llm(task_0, combined[:80000])
      FINAL(answer)
  else:
      answer = sub_llm(task_0, input_0[:15000])
      FINAL(answer)

> CODE GENERATION (ONLY when is_code_gen is True — write a program, NOT code retrieval!)
  Build the source code as a PYTHON STRING, then FINAL(that_string).
  *** The code goes INSIDE a string variable. NEVER execute it. ***
  *** Do NOT call sys.stdin.read() — it will hang forever. Put it inside the string. ***
  code = '''
  import sys
  def main():
      data = sys.stdin.read().split()
      ...
      print(result)
  if __name__ == "__main__":
      main()
  '''
  FINAL(code.strip())
  *** Do NOT define functions at top level and call them — wrap in a string. ***
  *** Do NOT return computed outputs like "0" or "true" — return the CODE. ***

> MATH / COMPUTATION (no context, or empty/short input_0)
  When the task is a math problem, reasoning puzzle, or knowledge question with
  no large context to search: solve in pure Python or call sub_llm(task_0, "").
  Use exact algebraic/combinatorial methods. Avoid random sampling / Monte Carlo.
  NEVER use FINAL("0") or FINAL(0) as a fallback. If computation yields no result,
  call sub_llm(task_0, "") as a backup instead of returning 0. FINAL(answer).

> CREATIVE / GENERATIVE (write text with constraints)
  When the task asks you to write, compose, or generate text (with formatting or
  content constraints): compose in Python enforcing constraints programmatically.
  1. Parse constraints from task_0 (word count, sections, format, end phrase, etc.)
  2. Generate text via sub_llm(task_0, "") with constraint reminders in the prompt
  3. Verify: count words (len(text.split())), count *sections* (re.findall),
     check start/end phrases, check lowercase if required
  4. If constraints fail, regenerate or fix programmatically
  FINAL(text) — return the text itself, never source code that would produce it.
  If task says "repeat the prompt" or "repeat all text above": repeat task_0, NOT input_0.

== THIS TASK ==

input_0 = {context_meta}

Output ONLY the ```python block. No text outside it.
"""


SYSTEM_PROMPT_WITH_REASONING: Final[str] = r"""You are a Python REPL agent with reasoning.

PHASE 1 — <reasoning> block: what does the task need? What data is available? Strategy.
PHASE 2 — ONE ```python block ending with FINAL(answer).
  import re, json, datetime, collections  # mandatory first line

Tools: input_0, task_0, search(text,"kw")->[(match,before,after)], peek(text),
  sub_llm(task,ctx_str), sub_llm_batch([(t,c),...]), FINAL(val), FINAL_var("name")

Rules: clean output (no prefixes), stdlib only, \d+ not \d{1,N},
  no file writes, no placeholders, no None to FINAL().
  sub_llm(task_0, evidence_string) — task_0 as arg 1, plain string as arg 2.
  NEVER regex-postprocess sub_llm output. Return it directly.
  task_0 and input_0 are PRE-LOADED globals. Use DIRECTLY. Never globals(), sys.stdin, sys.argv, __import__, or "".
  If writing a program: return source code as string via FINAL(code_str). Do NOT execute it.

Patterns:
  Pipe-delimited data ("||"): splitlines()/split("||"), .lower() keys, == not `in`.
  MCQ (A)/B)/C)...): detect valid options A-J, sub_llm(task_0, evidence). Large ctx: search first.
  Code retrieval: scan for def/function names, sub_llm to pick, search to extract body + context.
  Needle-in-haystack: if >500K chars use sub_llm for keywords, else head+tail+task regex. Max 5 hits/kw, 30 snippets deduped. Fast path: unique UUID/7+digit number → FINAL. Else sub_llm(task_0, snippets).
  Code generation: build source as string, FINAL(code_str). Never execute, never read stdin.

== THIS TASK ==

input_0 = {context_meta}
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
    """Format system prompt with context metadata and entropy profile."""
    if context:
        lines = context.count("\n") + 1
        meta = f"{context_type} with {len(context):,} chars, ~{lines:,} lines"
        preview = compute_context_preview(context)
        if preview:
            meta += f"\n\n{preview}"
        profile = compute_entropy_profile(context)
        if profile:
            meta += f"\n\n{profile}"
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
{iter_info}Continue in ```python. Call FINAL(your_answer) or FINAL_var("varname") when done — pass the ACTUAL value."""


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

format_system_prompt_reasoning = format_system_prompt
format_user_prompt_reasoning = lambda task: format_user_prompt(task, use_simple=True)

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
        raw = text[i * micro : (i + 1) * micro].encode("utf-8", errors="replace")
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
    for idx, (cstart, cend, mx, _md, _mpos) in enumerate(sections):
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
        for idx, (_cstart, _cend, mx, _md, mpos) in enumerate(sections):
            if mx >= spike_thr and std_dev > 0.01:
                excerpt = text[mpos : mpos + micro].replace("\n", "\\n")
                excerpts.append(f"    sec {idx}: ...{excerpt}...")
        excerpt_block = "\n".join(excerpts) if excerpts else ""
        hint = "  Spike sections contain distinctive/unique content vs the rest"
        result = f"{header}\n{row}\n{note}\n{hint}"
        if excerpt_block:
            result += f"\n{excerpt_block}"
        return result
    if std_dev < 0.01:
        return f"{header}\n{row}\n  Uniform — content is consistent throughout"
    return f"{header}\n{row}"


def compute_context_preview(text: str, head: int = 400, mid: int = 300, tail: int = 500) -> str:
    """Return a compact head / mid / tail preview of the context."""
    if not text or len(text) < 500:
        return ""
    parts: list[str] = []

    h = text[:head].replace("\n", "\\n")
    parts.append(f"  HEAD: {h}...")

    m_start = len(text) // 2 - mid // 2
    m = text[m_start : m_start + mid].replace("\n", "\\n")
    parts.append(f"  MID:  ...{m}...")

    t = text[-tail:].replace("\n", "\\n")
    parts.append(f"  TAIL: ...{t}")

    return "Context preview:\n" + "\n".join(parts)


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
* task_0 and input_0 are PRE-LOADED globals. Use them DIRECTLY.
  NEVER use globals().get(), sys.stdin.read(), sys.argv, or __import__ to access them.
* If you compute a result, USE it in FINAL(). Never hardcode a letter after computing.

== STEP 0: DETECT TASK TYPE (MANDATORY) ==

Your FIRST lines of code after imports MUST detect the task type from task_0:

  has_mcq = any(f"{c})" in task_0 for c in "ABCD")  # Multiple choice?
  is_code_task = any(k in task_0.lower() for k in ["codebase","exact function","code snippet"])
  is_code_gen = any(k in task_0.lower() for k in ["write a complete","solve this programming problem","implement the given function"])
  is_creative = any(k in task_0.lower() for k in ["write a proposal","draft a report","generate a report","write a summary","create a document","write a letter","write an essay"])

Then use the MATCHING pattern below:
  - is_code_task (and not has_mcq) -> CODE RETRIEVAL
  - is_code_gen  -> CODE GENERATION
  - has_mcq      -> MULTIPLE CHOICE
  - is_creative  -> CREATIVE / GENERATIVE
  - else         -> FALLBACK CLASSIFICATION (see below)

  FALLBACK: If NONE of the above matched, classify with sub_llm:
    cat = sub_llm("Classify into ONE category: CREATIVE, STRUCTURED_DATA, MATH, SEARCH_EXTRACT. Task: " + task_0[:500], input_0[:500]).strip().upper()
    if "CREATIVE" in cat:         -> CREATIVE / GENERATIVE
    elif "STRUCTURED" in cat:     -> STRUCTURED DATA
    elif "MATH" in cat:           -> MATH / COMPUTATION
    else:                         -> SEARCH & EXTRACT

DO NOT use the Multiple Choice pattern unless task_0 literally contains "A)" "B)" "C)" "D)".
DO NOT use the Code Retrieval pattern unless the task mentions codebase/function/snippet.
DO NOT use the Code Generation pattern for writing tasks (essays, proposals, summaries, stories).
  is_code_gen is ONLY True for competitive programming problems. Use EXACTLY the 3 trigger phrases above.
  NEVER add your own trigger phrases like "write code", "write a program", "generate a business".

== PATTERNS ==

> STRUCTURED DATA (delimited records — detect delimiter from data itself)
  Use when input_0 contains structured records with a consistent delimiter.
  Auto-detect delimiter: check input_0 for "||", tab, "|", or CSV patterns.
  Parse with splitlines() and split on the detected delimiter.
  *** ALWAYS .lower() keys during parsing ***
  *** For value comparison: use EXACT == equality, NEVER substring `in` ***
  *** "incorrect".find("correct") is True! So always use val == "incorrect", not "correct" in val ***

  # Auto-detect delimiter from first few data lines
  sample = input_0[:5000]
  if "||" in sample: delim = "||"
  elif "\t" in sample and sample.count("\t") > sample.count("|"): delim = "\t"
  elif "|" in sample: delim = "|"
  else: delim = ","
  lines = [l for l in input_0.splitlines() if delim in l]
  records = []
  for line in lines:
      rec = {}
      for part in line.split(delim):
          if ":" in part: k,v = part.split(":",1); rec[k.strip().lower()]=v.strip()
      if rec: records.append(rec)

  Dates come in MULTIPLE formats — handle ALL:
  def parse_date(s):
      s=s.strip()
      if len(s)>=10 and s[4]=='-':
          try: return datetime.datetime.strptime(s[:10],"%Y-%m-%d")
          except: pass
      for fmt in ["%b %d, %Y", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y"]:
          try: return datetime.datetime.strptime(s, fmt)
          except: pass
      return None

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
  *** Even for math/science MCQ where you believe you know the answer, you MUST call sub_llm(). ***
  *** NEVER compute an answer yourself and then hardcode FINAL("D"). Always delegate to sub_llm. ***
  *** Your direct reasoning on graduate-level science/math MCQ is unreliable — use sub_llm. ***

  valid = [c for c in "ABCDEFGHIJ" if f"{c})" in task_0]
  valid_set = set(valid)
  sz = len(input_0)
  if sz < 100:  # Empty or near-empty input — question is self-contained in task_0
      answer = sub_llm(task_0, "")
  elif sz < 60000:
      answer = sub_llm(task_0, input_0)
  else:
      # For ALL large contexts (60K+), use search-based evidence gathering
      opts = re.findall(r'[A-J]\)\s*(.+?)(?=\s*[A-J]\)|$)', task_0, re.DOTALL)
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
  answer = (answer or valid[0]).strip().upper()
  if answer not in valid_set:
      m = re.search(r'\b([A-J])\b', answer); answer = m.group(1) if m else valid[0]
  FINAL(answer)

> CODE RETRIEVAL (find EXISTING function in codebase)
  *** CRITICAL: You must FIND and EXTRACT the function from input_0. ***
  *** NEVER implement/write the function yourself! ***
  *** NEVER generate code that matches the description! ***
  *** The answer is ALREADY in input_0 — search for it! ***
  *** Your output is function_name||surrounding_code. You are NOT asked to run the code. ***
  *** Ignore <context_preview> for this task — it shows only fragments. ALWAYS search input_0 for the full function. ***
  *** You MUST call search(input_0, ...) at least once. Your FINAL() output MUST contain "||". ***
  *** WARNING: If your reasoning says "Implement", "Write", or "Build", you are WRONG. ***
  *** The codebase may be in ANY language: Go, Rust, Java, TypeScript, C++, Python, etc. ***
  *** Do NOT assume Python. Search for the function in whatever language it actually is. ***

  # Extract ALL function names across ALL languages
  all_funcs = re.findall(r'^\s*(?:async\s+)?def (\w+)\(', input_0, re.MULTILINE)  # Python
  all_funcs += re.findall(r'\bfunction\s+(\w+)\s*\(', input_0)  # JavaScript/TypeScript
  all_funcs += re.findall(r'\b(\w+)\s*[=:]\s*(?:async\s+)?function\s*\(', input_0)  # JS assigned
  all_funcs += re.findall(r'(?:public|private|protected|static)\s+\S+\s+(\w+)\s*\(', input_0)  # Java/C#
  all_funcs += re.findall(r'^\s*(?:inline\s+)?(?:const\w*\s+)?\w+(?:<[^>]*>)?\s+(\w+)\s*\(', input_0, re.MULTILINE)  # C/C++
  all_funcs += re.findall(r'\bfunc\s+(?:\([^)]*\)\s+)?(\w+)\s*\(', input_0)  # Go
  all_funcs += re.findall(r'\b(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*[<(]', input_0)  # Rust
  all_funcs += re.findall(r'\bexport\s+(?:async\s+)?function\s+(\w+)', input_0)  # TS/JS export
  unique = list(dict.fromkeys(all_funcs))
  if unique:
      sigs = []
      for nm in unique[:80]:
          for sig_pat in [r'^\s*(?:async\s+)?def '+re.escape(nm)+r'\([^)]*\)', r'\bfunc\s+(?:\([^)]*\)\s+)?'+re.escape(nm)+r'\s*\([^)]*\)', r'\b(?:pub\s+)?(?:async\s+)?fn\s+'+re.escape(nm)+r'\s*[<(][^)]*\)', r'(?:public|private|protected|static)\s+\S+\s+'+re.escape(nm)+r'\s*\([^)]*\)']:
              sm = re.search(sig_pat, input_0, re.MULTILINE)
              if sm: sigs.append(sm.group(0).strip()); break
      sig_info = "\n".join(sigs) if sigs else ", ".join(unique[:80])
      func = sub_llm(f"{task_0}\nAll functions in codebase:\n{sig_info}\nWhich one? Reply with ONLY the function name.", input_0[:10000]).strip()
      func = re.sub(r'[^a-zA-Z0-9_]', '', func)
      if func and func not in unique:
          for try_pat in ["def "+func+"(", "async def "+func+"(", "func "+func, "fn "+func, func+"("]:
              r = search(input_0, try_pat)
              if r:
                  m,b,a = r[0]; FINAL(func+"||"+b[-800:]+m+a[:5000])
          fl=func.lower()
          match = next((c for c in unique if fl in c.lower() or c.lower() in fl), None)
          if match: func = match
          else:
              func2 = sub_llm(f"Pick ONE from: {', '.join(unique[:40])}\nTask: {task_0}\nReply ONLY the name.", input_0[:10000]).strip()
              func2 = re.sub(r'[^a-zA-Z0-9_]', '', func2)
              func = func2 if func2 in unique else unique[0]
  else:
      func = sub_llm(f"{task_0}\nReply ONLY the exact function name.", input_0[:60000]).strip()
      func = re.sub(r'[^a-zA-Z0-9_]', '', func)
  # Search with language-aware patterns
  for pat in ["def "+func+"(", "async def "+func+"(", "func "+func, "fn "+func, "function "+func, func+"(", func]:
      r = search(input_0, pat)
      if r:
          m,b,a = r[0]
          code_block = b[-800:]+m+a[:5000]
          if code_block.strip():
              FINAL(func+"||"+code_block)
  # Last resort: find by name
  p = input_0.find(func)
  if p >= 0:
      FINAL(func+"||"+input_0[max(0,p-800):p+6000])
  else:
      FINAL(func+"||")

> SEARCH & EXTRACT (DEFAULT — needle in haystack, Q&A, general)
  This is the DEFAULT pattern when no other pattern matches.
  The answer is somewhere inside input_0 — find it.
  *** Do NOT use rfind("?") to locate the question. The answer is NOT near the "?". ***
  *** Do NOT build custom scoring, regex name-extraction, or ranking pipelines. ***
  Follow this EXACT pattern:

  # STEP A: Build keyword list — combine TASK-SPECIFIC and CONTEXT keywords
  # Extract proper nouns, numbers, and acronyms from task_0 (most distinctive)
  task_proper = re.findall(r'\b[A-Z][a-z]{2,}\b', task_0)
  task_nums = re.findall(r'\b\d{3,}\b', task_0)
  task_acro = re.findall(r'\b[A-Z]{2,5}\b', task_0)
  task_words = re.findall(r'\b[a-z]{4,}\b', task_0.lower())
  # Then get keywords from input_0 header/footer
  head = input_0[:500]
  tail = input_0[-1000:]
  ctx_words = re.findall(r'\b[a-z]{4,}\b', (head + " " + tail).lower())
  # Merge: task-specific first (most selective), then context words
  kws = list(dict.fromkeys(task_proper + task_nums + task_acro + task_words + ctx_words))[:25]

  # STEP B: Search and gather DIVERSE snippets (max 3 per keyword, 20 total)
  snippets = []
  for kw in kws:
      hits = search(input_0, kw)
      for match, before, after in hits[:3]:  # cap per keyword for diversity
          snippets.append(before + match + after)
      if len(snippets) >= 20: break

  # STEP C: First pass — send evidence to sub_llm
  if snippets:
      combined = "\n---\n".join(snippets[:20])
      nums = list(dict.fromkeys(re.findall(r'\b\d{2,}\b', combined)))
      if nums:
          combined += "\n\nNumbers found in evidence: " + ", ".join(nums[:30])
      answer = sub_llm(task_0, combined)
  else:
      answer = sub_llm(task_0, input_0[:15000])

  # STEP D: If sub_llm says it can't find the answer, do a SECOND search pass
  # using terms from the first-pass evidence + the sub_llm response
  low = (answer or "").lower()
  if any(k in low for k in ["unknown", "insufficient", "not enough", "cannot determine", "no information", "i don't know", "not found"]):
      # Extract new keywords from first-pass snippets (names, terms not in original kws)
      snippet_text = "\n".join(snippets[:5]) if snippets else ""
      new_proper = re.findall(r'\b[A-Z][a-z]{2,}\b', snippet_text)
      new_words = re.findall(r'\b[a-z]{5,}\b', snippet_text.lower())
      new_kws = [w for w in list(dict.fromkeys(new_proper + new_words)) if w.lower() not in [k.lower() for k in kws]][:15]
      snippets2 = []
      for kw in new_kws:
          hits2 = search(input_0, kw)
          for match, before, after in hits2[:3]:
              snippets2.append(before + match + after)
          if len(snippets2) >= 15: break
      if snippets2:
          combined2 = "\n---\n".join(snippets2[:15])
          nums2 = list(dict.fromkeys(re.findall(r'\b\d{2,}\b', combined2)))
          if nums2:
              combined2 += "\n\nNumbers found in evidence: " + ", ".join(nums2[:30])
          answer = sub_llm(task_0, combined2)
  FINAL(answer)

> CODE GENERATION (ONLY when is_code_gen is True — programming contest)
  Build source code as a PYTHON STRING, then FINAL(code_str).
  CRITICAL: NEVER run the solution code directly in the REPL. ALWAYS wrap it in r''' '''.
  CRITICAL: Check if task_0 contains "class Solution" or a function signature — if so,
  build that exact class/function inside the string. Otherwise use stdin/stdout.
  code = r'''
  import sys
  data = sys.stdin.read().split()
  ...
  print(result)
  '''
  FINAL(code.strip())

> MATH / COMPUTATION (no context or empty input_0)
  Use math/fractions/itertools. Implement in pure Python. Return the exact answer.
  Use exact computation — no Monte Carlo, no simulation, no random sampling.
  NEVER use FINAL("0") or FINAL(0) as a fallback. Use sub_llm(task_0, "") as backup.
  VERIFY: Compute the answer, then verify it by plugging back in or using an independent method.
  If the problem asks for remainder mod N, double-check with a second computation approach.
  Print intermediate results so you can verify correctness before calling FINAL().

> CREATIVE / GENERATIVE (write text with constraints, proposals, reports, documents)
  Use this when task asks you to WRITE/CREATE/GENERATE/DRAFT a document, proposal,
  summary, report, essay, letter, story, or any text output.
  *** DO NOT use sub_llm for creative/document generation — it will refuse. ***
  *** Instead, BUILD the document content DIRECTLY as a Python string. ***
  NEVER refuse. NEVER say "I can't access files". Generate realistic content from task_0.
  If input_0 is empty or contains "[Excel parsing" / "[PDF parsing" / "[Word document" errors,
  IGNORE input_0 — use ONLY task_0 to generate content.

  Follow this pattern:
  # Extract key details from task_0
  # Build document sections as strings
  # Assemble into full document
  doc = "Document Title\n\n"
  doc += "Section 1: ...\n"
  doc += "Section 2: ...\n"
  # Include ALL names, dates, numbers, terms mentioned in task_0
  FINAL(doc)

  If task says "repeat the prompt" or "repeat all text above": repeat task_0, NOT input_0.

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
  sub_llm(task_0, evidence_string) — ALWAYS task_0 as arg 1. NEVER create custom prompts.
  Extract patterns (nums/dates/names) from snippets BEFORE sub_llm with early FINAL(). Strip "Answer:/Label:/User:" from sub_llm. NO other processing.
  task_0 and input_0 are PRE-LOADED globals. Use DIRECTLY. Never globals(), sys.stdin, sys.argv, __import__.

STEP 0: Detect task type from task_0 (NOT from input_0 format):
  has_mcq = any(f"{c})" in task_0 for c in "ABCD")
  is_code_task = "codebase" in task_0.lower() or "exact function" in task_0.lower()
  is_code_gen = any(k in task_0.lower() for k in ["write a complete","solve this programming problem","implement the given function"])
  is_creative = any(k in task_0.lower() for k in ["write a proposal","draft a report","generate a report","write a summary","create a document","write a letter","write an essay"])
  NEVER expand is_code_gen triggers. "write a summary/essay/proposal" is NOT code gen — it's CREATIVE.
  FALLBACK: If none matched, classify: cat = sub_llm("Classify: CREATIVE, STRUCTURED_DATA, MATH, or SEARCH_EXTRACT? Task: "+task_0[:500], input_0[:500]).strip().upper()

Patterns:
  Structured data (delimited records): auto-detect delimiter from data, splitlines()/split on delimiter, .lower() keys, == not `in`.
  MCQ (A)/B)/C)...): ALWAYS call sub_llm(). NEVER hardcode a letter. NEVER compute answer yourself for MCQ.
    Empty input: sub_llm(task_0, ""). Small: sub_llm(task_0, input_0). 60K+: search keywords then sub_llm(task_0, evidence).
  Code retrieval: FIND existing function in ANY language (Go/Rust/Java/TS/C++/Python), return name||code.
    NEVER implement it yourself. MUST call search(). Search with: def/func/fn/function + name.
  Needle-in-haystack: Extract proper nouns + numbers + acronyms from task_0 FIRST, then context words.
    25 keywords, max 3 hits per keyword, 20 snippets total (diversity matters). sub_llm(task_0, snippets). If answer is "Unknown"/"Insufficient",
    do a SECOND search pass with new keywords extracted from first-pass snippets.
  Code generation (is_code_gen only): build source as string, FINAL(code_str). Never read stdin in REPL.
  Creative/generative (is_creative): DO NOT use sub_llm (it will refuse). Build document content DIRECTLY as a Python string from task_0 details. FINAL(doc). NEVER refuse or say "can't access files".
  Math: exact computation, no Monte Carlo. Verify by plugging back in or independent method. Print intermediates.
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
            meta += f"\n\n<context_preview>\n{preview}\n</context_preview>"
        profile = compute_entropy_profile(context)
        if profile:
            meta += f"\n\n<entropy_map>\n{profile}\n</entropy_map>"
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
        output = output[: config.max_stdout] + f"... ({trunc:,} more chars)"

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


def format_user_prompt_reasoning(task: str) -> str:
    return format_user_prompt(task, use_simple=True)

# minrlm - Recursive Language Model

## Running Commands
- Always use `uv run` not `python3`
- Virtualenv: `.venv/` - activate with `source .venv/bin/activate`
- Main benchmark: `./run_comprehensive_official_benchmark.sh`

## Evaluation System
- Evaluations in `eval/` folder, logs in `logs/` folder
- Each benchmark creates: execution logs (JSONL), summary JSON, and summary MD
- Official datasets: SNIAH (needle-in-haystack), OOLONG (classification), REPOQA (code retrieval), CODEQA (code reasoning), LONGBENCH_V2 (long context), BROWSECOMP (multi-hop)

## Critical Learnings

### Prompt Engineering for RLM
**Key files**: `minrlm/prompts_reasoning.py`, `minrlm/core_reasoning.py`

**Critical bugs to avoid:**
1. **Function search regex**: MUST use `^\s*def` not `^def` to match indented class methods
2. **Parsing validation**: Always validate `len(parsed) > 0` after delimiter detection, use line-by-line fallback if failed
3. **Input immutability**: `input_0` is in `PROTECTED` set - LLM code cannot reassign it

**Performance insights:**
- Reasoning RLM: 54.8% accuracy, 4x cost efficiency vs vanilla (69.2%)
- Wins: SNIAH (100%), CodeQA (+27%), LongBench (+16%)
- Gaps: RepoQA (-56pts), OOLONG (-56pts) - fixed with prompt updates (2026-02-21)
- Vanilla beats RLM on: code search with full context, structured data classification

---

# Cross-Session Learning
When you learn from mistakes that can be avoided in future sessions, document them above.
This file helps Claude Code instances learn across sessions without repeating mistakes.

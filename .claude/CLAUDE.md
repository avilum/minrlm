# minrlm - Recursive Language Model

## Project Overview
minRLM is a token-efficient implementation of Recursive Language Models (RLMs). Data never enters the prompt - it's stored in a Python REPL variable (`input_0`), and the model writes code to query it. 4,800 evaluations across 3 models (GPT-5-nano, GPT-5-mini, GPT-5.2), 12 tasks.

**Key results**: GPT-5-mini: 72.7% accuracy, 3.6x fewer tokens than official RLM. GPT-5.2: +30pp over vanilla, 11/12 tasks won. RepoQA is the consistent weak spot across all models.

## Running Commands
- Always use `uv run` not `python3`
- Virtualenv: `.venv/` - activate with `source .venv/bin/activate`
- CLI: `uvx minrlm "task"` or `uvx minrlm "task" file.txt` or pipe via stdin
- Bump version in `pyproject.toml` when changing CLI - `uvx` caches by version number
- Main benchmark: `uv run python eval/run.py --tasks all --runners minrlm-reasoning,vanilla,official --runs 50`

## Key Files

### Core
- `minrlm/core.py` - `RLMBase` class, the LLM <-> REPL loop, `search()`, `sub_llm()`, `FINAL()`
- `minrlm/core_reasoning.py` - `RLMReasoning` (the default `RLM`), adds reasoning extraction, `on_step` callbacks
- `minrlm/prompts_reasoning.py` - All prompts, entropy profiling, context preview, task patterns
- `minrlm/docker_repl.py` - DockerREPL sandbox with seccomp, container lifecycle management
- `minrlm/__main__.py` - CLI entry point (`uvx minrlm`), `-s` for steps, `-v` for verbose

### Evals
- `eval/run.py` - Benchmark runner with parallelism
- `eval/runners.py` - Runner implementations: `vanilla`, `minrlm-reasoning`, `official`
- `eval/tasks.py` - 12 benchmark tasks
- `eval/README.md` - Full results for all 3 models

### Blog & Docs
- `docs/recursive-language-model.html` - Main blog post (single HTML file, pure CSS tabs, sidebar TOC)
- `docs/index.html` - Redirect to blog

### Best Eval Data
- `BEST_EVALS/BEST_new-entropy-prompts-12-tasks-all-runners-gpt-5-mini-50-runs-BEST/` - GPT-5-mini (1,800 evals)
- `BEST_EVALS/BEST_new-entropy-prompts-12-tasks-all-runners-gpt-5.2-50-runs-after-opus-46/` - GPT-5.2 (1,200 evals)
- `BEST_EVALS/BEST_new-entropy-prompts-12-tasks-all-runners-gpt-5-nano-50-runs-after-opus-46/` - GPT-5-nano (1,800 evals)

## Critical Learnings

### Prompt Engineering
**Key file**: `minrlm/prompts_reasoning.py`

**Architecture:**
1. **Entropy profiling** - zlib compression-based heatmap of input (20 sections x 500-char micro-chunks). Spikes = unique content. Injected into system prompt so model knows WHERE to search.
2. **Context preview** - head/mid/tail sample (400+300+500 chars) for format detection.
3. **Task routing** - Step 0 classifies: structured data, MCQ, code retrieval, code generation, creative, math, search & extract. Each has specialized code pattern.
4. **Two-pass search** - If first pass returns "unknown", second pass runs with new keywords from first-pass evidence.
5. **Sub-LLM delegation** - For MCQ/extraction: outer model gathers evidence, `sub_llm(task_0, evidence)` reasons over it.
6. **Flat context** - System prompt doesn't grow per iteration. Continue prompts are minimal.

**Critical bugs to avoid:**
1. **Function search regex**: MUST use `^\s*def` not `^def` to match indented class methods
2. **Parsing validation**: Always validate `len(parsed) > 0` after delimiter detection, use line-by-line fallback if failed
3. **Input immutability**: `input_0` is in `PROTECTED` set - LLM code cannot reassign it
4. **Em dashes**: The blog HTML uses regular hyphens `-`, NOT em dashes `---`. Edit strings must match exactly.

### Known Issues
- **GPT-5.2 vanilla AIME 0%**: Not a bug. Vanilla runner is a plain API call with no chain-of-thought. GPT-5.2 outputs 4 tokens (just a number) with no reasoning. The REPL compensates by computing via code.
- **RepoQA**: Vanilla wins on every model size. Model sometimes generates code instead of extracting it.
- **`uvx` caching**: `uvx .` caches by version. Must bump version in `pyproject.toml` to pick up CLI changes during development. Or use `uv run minrlm` for local dev.
- **httpcore logs**: Suppressed in `core.py` (httpx, httpcore, openai loggers set to WARNING). The `docker_repl.py` verbose logging uses its own logger, not root.

### Performance Insights
- GPT-5-mini: minRLM 72.7% / 8,151 tokens / $2.86 vs vanilla 69.5% / 20,967 / $4.74 vs official 69.7% / 29,327 / $7.92
- GPT-5-nano: minRLM 53.7% / $0.74 vs vanilla 63.2% / $1.16 vs official 43.3% / $2.68
- GPT-5.2: minRLM 78.2% / 8,095 tokens / $18.93 vs vanilla 48.2% / 14,196 / $16.50
- Scaling trend: minRLM advantage goes from -9.5pp (nano) to +3.2pp (mini) to +30pp (frontier)
- Most tasks complete in 1 iteration. GDP Val occasionally needs 2.

## Blog Structure
The blog (`docs/recursive-language-model.html`) is a single self-contained HTML file:
- Pure CSS tabs (radio inputs + sibling selectors, no JavaScript)
- Fixed-position sidebar TOC (left side, media query fallback to inline at 1100px)
- Sections: Background -> Related Work -> Implementation -> Experimental Setup -> Tasks & Results (12 collapsible) -> Full Results -> Model Scaling -> Limitations -> Discussion -> Conclusion -> Code & Reproduction -> References
- Charts are PNG images in `docs/`, clickable with a CSS lightbox

## Writing Style
- NEVER use em dashes `---`, always use hyphens `-`
- Use "Our" not "My" in the blog
- Keep it honest/humble - no AI-slop language ("unambiguous", "amplifies", "transforms capabilities")
- Numbers belong in tables, not repeated in prose. Don't restate the same stat in multiple sections.
- Task callouts should be short insights, not number recaps.

## Cross-Session Learning
When you learn from mistakes that can be avoided in future sessions, document them above.

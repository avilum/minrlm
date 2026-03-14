# minrlm

A robust implementation of [Recursive Language Models](https://arxiv.org/abs/2512.24601) - let LLMs think through code instead of reading your data.

**The problem**: Context engineering is static. You dump everything into the prompt and hope the attention mechanism figures out what matters. Most of the compute goes to tokens that aren't relevant to the answer. And at 1M+ tokens, you hit the context window and can't even run the query.

**The insight**: We still rely on attention - but we improve the *context* the attention operates on. The LLM fetches exactly the context it needs to solve the task without ever seeing the full input. This isn't just an optimization - it **enables** things that aren't possible today: querying 10M-character documents, processing entire codebases, bypassing the context window entirely.

**The solution**: Data stays in a Python REPL as `input_0`. The model writes code to search, filter, and aggregate. The raw data never enters the conversation - the LLM decides what to look at through code, pulls what it needs, and runs attention only on what matters.

**The proof**: 1,800-eval benchmark on gpt-5-mini across 12 tasks — **72.7% accuracy** (vs 69.5% vanilla, 69.7% official RLM), **2.6x fewer tokens** than vanilla, **3.6x fewer** than official, **1.7x cheaper** than vanilla. Enables tasks vanilla can't run: BrowseComp (62% vs 16% — vanilla hits the context limit).

**Improve every LLM in production, today.** If your application passes documents, logs, tables, or code to an LLM, minRLM is the fastest ROI you can find. The context window is the real bottleneck — not the model. Swap the client, keep everything else. Works with OpenAI, Anthropic, and any OpenAI-compatible API. Ships in an afternoon, scales to any context size.

**How is this different from agents?** An RLM is an agent with exactly one tool (Python REPL) that never sees the entire raw input. It tells the model *"you have `input_0` with 500K chars"* and lets it write code to answer the question. Some agents already do this internally — Claude Code processes web search results through code, Cursor and Claude Code chunk large files instead of pasting them whole. But these are proprietary backend optimizations. RLMs make this a commodity: agentic exploration of data in a single LLM call, where context is dynamic and determined at runtime based on the task and data.

---

## What's in this repo

| Component | Location | What it does |
|-----------|----------|--------------|
| **RLM client** | [`minrlm/`](minrlm/) | Core `RLM` and `RLMReasoning` classes - the LLM ↔ REPL loop |
| **DockerREPL** | [`minrlm/docker_repl.py`](minrlm/docker_repl.py) | Sandboxed code execution via Docker + custom seccomp |
| **Evals** | [`eval/`](eval/) | 12-task benchmark framework, runners, metrics, plot generation |
| **Examples** | [`examples/`](examples/) | Quickstart scripts, proxy server, Gradio side-by-side UI |

---

## Benchmarks

**gpt-5-mini** | **1,800 evaluations** | 12 tasks | 50 runs per task per runner

|  | minRLM | Vanilla LLM | Official RLM |
|---|---|---|---|
| **Accuracy** | **72.7%** | 69.5% | 69.7% |
| **Avg Tokens** | **8,151** | 20,967 | 29,327 |
| **Total Cost** | **$2.86** | $4.74 | $7.92 |

**2.6x fewer tokens** than vanilla | **3.6x fewer** than official | **1.7x cheaper** than vanilla | **2.8x cheaper** than official

![Summary Dashboard](docs/summary_dashboard.png)

![Accuracy per Task](docs/accuracy_per_task.png)

![Token Savings vs Baselines](docs/token_savings.png)

![Tokens per Task](docs/tokens_per_task.png)

![Cost per Query by Task](docs/cost_per_task.png)

![Latency per Task](docs/latency_per_task.png)

![Accuracy vs Cost — Efficiency Frontier](docs/accuracy_vs_cost.png)

![Accuracy vs Latency](docs/accuracy_vs_latency.png)

### Per task

| Task | minRLM | Vanilla | Official | minRLM Tokens | vs Official Tokens |
|------|--------|---------|----------|---------------|-------------------|
| SNIAH | **94%** | 100% | 76% | 6,328 | **2.6x fewer** |
| OOLONG | **92%** | 78% | 80% | 6,184 | **2.3x fewer** |
| GDP Val | **86%** | 54% | 50% | 12,007 | **1.7x fewer** |
| IFEval | **84%** | 78% | 78% | 5,963 | **1.6x fewer** |
| MMLU-Pro | 82% | **90%** | 86% | 6,341 | **1.3x fewer** |
| LiveCodeBench | **80%** | 64% | 60% | 7,106 | **1.3x fewer** |
| AIME 2025 | 74% | **88%** | 84% | 7,951 | **1.4x fewer** |
| GPQA Diamond | 70% | 66% | **74%** | 6,679 | **2.1x fewer** |
| BrowseComp | 62% | 16% | **66%** | 10,740 | **6.4x fewer** |
| RepoQA | 62% | **98%** | 96% | 8,026 | **2.2x fewer** |
| LongBench V2 | 46% | **56%** | 48% | 10,767 | **7.8x fewer** |
| CodeQA | 40% | **46%** | 38% | 9,724 | **8.0x fewer** |

minRLM uses fewer tokens than Official RLM on **every task** (1.3x-8.0x). Vanilla fails on BrowseComp (16%) because the context exceeds the token limit.

Full results and reproduction: [`eval/README.md`](eval/README.md)

---

## How it works

```
┌──────────────────────────────────────────────────────────┐
│  LLM sees:                                               │
│                                                          │
│  input_0 = "string with 500000 chars"                    │
│  Task: Count errors in last hour                         │
├──────────────────────────────────────────────────────────┤
│  LLM writes:                                             │
│                                                          │
│  import re                                               │
│  from datetime import datetime, timedelta                │
│  errors = re.findall(r'\[ERROR\].*', input_0)            │
│  cutoff = datetime.now() - timedelta(hours=1)            │
│  FINAL(len([e for e in errors if parse_time(e) > cutoff]))
└──────────────────────────────────────────────────────────┘
```

1. Context is stored as `input_0` in a sandboxed Python REPL
2. The model writes code to search/filter/aggregate it
3. Code runs, output goes back to the model
4. Repeat until `FINAL(answer)` is called

The data never enters the conversation. Token cost stays flat regardless of context size.

---

## Install

```bash
pip install minrlm          # minimal - only openai required
# or
uv add minrlm
```

From source:

```bash
git clone https://github.com/avilum/minrlm
cd minrlm
uv sync                     # base (openai only)
uv sync --extra eval        # + benchmark runner (datasets, matplotlib, tqdm)
uv sync --extra visualizer  # + Gradio UI (gradio, plotly, pandas)
uv sync --extra proxy       # + OpenAI-compatible proxy (fastapi, uvicorn)
uv sync --extra all         # everything
```

---

## 1. minrlm - RLM Client

`minrlm/` contains the core library:

| File | Purpose |
|------|---------|
| `core.py` | `RLMBase` - base recursive LLM loop |
| `core_reasoning.py` | `RLMReasoning` - reasoning-enhanced version (the default `RLM`) |
| `prompts.py` | System prompt for the base runner |
| `prompts_reasoning.py` | System prompt for the reasoning runner (used by benchmarks) |
| `docker_repl.py` | `DockerREPL` - sandboxed execution backend (see §2) |

### Basic usage

`from minrlm import RLM` gives you `RLMReasoning` - the version with task-adaptive reasoning that produces the benchmark numbers above. Use `RLMBase` if you want the bare-bones loop without reasoning prompts.

```python
from minrlm import RLM

rlm = RLM(model="gpt-5-mini")

result = rlm.completion(
    task="How many ERROR logs in the last hour?",
    context=server_logs,          # 500K chars - never sent to the LLM
)
print(result.response)            # "147"
print(result.total_tokens)        # ~2K tokens (vs ~93K for vanilla)
print(result.iterations)          # number of code->execute cycles
```

### Available REPL functions

| Function | What it does |
|----------|--------------|
| `input_0` | Your context data (string) |
| `search(text, pattern)` | Case-insensitive substring search with context windows |
| `peek(data)` | Preview structure of large data without printing all of it |
| `sub_llm(task, context)` | Recursive LLM call on a sub-chunk |
| `sub_llm_batch([(t,c), ...])` | Parallel batch of recursive calls |
| `FINAL(answer)` | Return the final answer and stop |
| `FINAL_var("name")` | Return a variable from the namespace |

### Custom endpoints

```python
rlm = RLM(
    model="llama-3.1-70b",
    base_url="http://localhost:8000/v1",
    api_key="sk-...",
)
```

### When to use RLM vs vanilla

| Use RLM when... | Use vanilla LLM when... |
|-----------------|------------------------|
| Context > 50K chars | Context is short (<50K chars) |
| Searching or filtering data | Summarization or open-ended generation |
| Counting, aggregating, extracting | Holistic understanding needed |
| Context doesn't fit in the window | Simple Q&A on short documents |

---

## 2. DockerREPL - Sandboxed Code Execution

LLM-generated code runs in an isolated Docker container with a custom [seccomp](https://docs.kernel.org/userspace-api/seccomp_filter.html) profile. Docker is **auto-detected and enabled** if available.

```python
from minrlm import RLM, check_docker_available

# Auto-detects Docker
rlm = RLM(model="gpt-5-mini")

# Explicit control
if check_docker_available():
    rlm = RLM(
        model="gpt-5-mini",
        use_docker=True,
        docker_memory="256m",
        docker_timeout=60,
    )
```

### What the sandbox blocks

| Restriction | How |
|-------------|-----|
| No network access | `--network=none` + seccomp blocks `socket`, `connect`, `bind`, ... |
| Read-only filesystem | `--read-only` (writable `/tmp` only) |
| Memory cap | `--memory=256m` (configurable) |
| CPU cap | `--cpus=1.0` (configurable) |
| Process limit | `--pids-limit=100` |
| Kernel module loading | seccomp: `init_module`, `finit_module` blocked |
| Mount operations | seccomp: `mount`, `umount` blocked |
| ptrace / debugging | seccomp: `ptrace` blocked |

### Container lifecycle

Every container is assigned a unique name (`minrlm_<pid>_<n>`) and tracked process-wide. Containers are **automatically killed** when:

- The container finishes (normal exit via `--rm`)
- The execution times out (`subprocess.TimeoutExpired` → `docker kill`)
- The parent Python process exits normally (`atexit` hook)
- The parent process receives `SIGTERM` or `SIGINT` (signal handlers)

No zombie containers after a crash or `Ctrl+C`.

### Custom seccomp policy

<details>
<summary>Extend or replace the seccomp profile</summary>

Edit `SECCOMP_PROFILE` in [`minrlm/docker_repl.py`](minrlm/docker_repl.py):

```python
SECCOMP_PROFILE = {
    "defaultAction": "SCMP_ACT_ALLOW",
    "syscalls": [
        {"names": ["socket"], "action": "SCMP_ACT_ERRNO", "errnoRet": 1},
        # add more restrictions...
    ],
}
```

Or subclass `DockerREPL` to inject a different profile at runtime.

Tip: use [gVisor](https://gvisor.dev/) as the Docker runtime for an additional kernel isolation layer.

</details>

> **Note**: `sub_llm()` is not available in Docker mode (no callback to host).

---

## 3. Evals

`eval/` is a self-contained benchmark framework covering 12 tasks.

| File | Purpose |
|------|---------|
| `quickstart.py` | Smoke test - one task, two runners, instant feedback |
| `run.py` | Full benchmark runner with parallelism, logging, and result export |
| `tasks.py` | 12 benchmark tasks (S-NIAH, OOLONG, CodeQA, LongBench-v2, RepoQA, BrowseComp+, GDP Val, AIME 2025, GPQA Diamond, MMLU-Pro, IFEval, LiveCodeBench) |
| `runners.py` | Runner implementations: `vanilla`, `minrlm`, `minrlm-reasoning`, `official` |
| `metrics.py` | `EvalResult`, `AggregatedMetrics`, cost calculation, markdown report generation |
| `plotting.py` | 8 standalone plots (accuracy, tokens, latency, cost, efficiency scatter) |
| `README.md` | Full benchmark results and reproduction steps |

### Quick start

```bash
uv sync --extra eval
export OPENAI_API_KEY="your-key"

# Smoke test (one task, ~1 min)
uv run python eval/quickstart.py

# Single task, 10 runs
uv run python eval/run.py --model gpt-5-mini --tasks official_sniah --runs 10

# All tasks, single runner, 50 runs each
uv run python eval/run.py \
    --model gpt-5-mini \
    --tasks all \
    --runners minrlm-reasoning \
    --runs 50 \
    --parallel 5 \
    --output-dir logs/my_eval

# Full multi-runner benchmark (reproduces the table above)
uv run python eval/run.py \
    --tasks all \
    --runners minrlm-reasoning,vanilla,official \
    --runs 50 --parallel 12 --task-parallel 12 \
    --output-dir logs/my_eval
```

### Visualize results

```bash
# Generate 8 plots from any eval JSON
uv run python -m eval.plotting logs/my_eval/eval_20260302.json

# Auto-discover newest JSON in a directory tree
uv run python -m eval.plotting logs/my_eval/

# Custom output directory
uv run python -m eval.plotting logs/my_eval/ reports/my_eval_plots/
```

Plots generated: accuracy per task, tokens per task, latency per task, cost per task, accuracy vs cost (efficiency frontier), accuracy vs latency, token savings vs baselines, summary dashboard.

See [`eval/README.md`](eval/README.md) for all tasks, flags, and full results.

---

## 4. Examples

`examples/` contains runnable scripts for common use cases.

### `minimal.py` - Vanilla LLM vs RLM

Side-by-side comparison on a single task. Good starting point.

```bash
uv run python examples/minimal.py
MINRLM_MODEL=gpt-5-mini uv run python examples/minimal.py
```

### `advanced_usage.py` - Search, sub_llm, callbacks

Demonstrates `search()`, `sub_llm()`, step callbacks, and multi-context usage.

```bash
uv run python examples/advanced_usage.py
```

### `visualizer.py` - Gradio side-by-side UI

Interactive web app for comparing runners on evaluation tasks or custom prompts. Shows generated code, token usage, and timing for each step.

```bash
uv sync --extra visualizer
uv run python examples/visualizer.py      # http://localhost:7860
```

### `proxy.py` - OpenAI-compatible proxy server

Drop-in replacement for the OpenAI API. Large contexts (>50K chars) are automatically routed through RLM; short contexts pass through directly.

```bash
uv sync --extra proxy
uv run uvicorn examples.proxy:app --host 0.0.0.0 --port 8000
MINRLM_VERBOSE=1 uv run uvicorn examples.proxy:app --port 8000   # verbose
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Print powers of 2 up to 1M"}],
)
```

See [`examples/proxy_example.py`](examples/proxy_example.py) for more.

Environment variables for the proxy:

```bash
export OPENAI_API_KEY="your-key"
export RLM_MODEL="gpt-5-mini"
export RLM_USE_DOCKER="true"
export PORT="8000"
export MINRLM_VERBOSE="1"
```

---

## Why RLMs?

- **No context window limit** - data lives in the REPL, not the prompt. 10M chars costs the same as 10K
- **Flat token cost** - ~5-8K tokens regardless of input size
- **Dynamic context** - the LLM decides what to look at based on the task, not you
- **Visible logic** - generated Python you can read, inspect, and reuse
- **O(n) operations** - substring search and iteration, not O(n²) attention
- **Any LLM** - works with any OpenAI-compatible endpoint

---

## Credits

**minrlm** is built by [Avi Lumelsky](https://github.com/avilum). This is an independent implementation - not a fork of the official code. The prompts, reasoning engine, eval framework, Docker sandboxing, and proxy server are all original work. On the official benchmarks, minrlm uses **3.6x fewer tokens** than the paper's reference implementation at higher accuracy (72.7% vs 69.7%).

The RLM concept comes from Zhang, Kraska, and Khattab:

```bibtex
@misc{zhang2025recursivelanguagemodels,
      title={Recursive Language Models},
      author={Alex L. Zhang and Tim Kraska and Omar Khattab},
      year={2025},
      eprint={2512.24601},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24601},
}
```

Paper: [arxiv.org/abs/2512.24601](https://arxiv.org/abs/2512.24601)
Official implementation: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)

## License

MIT

---

> I'm a security researcher. This is far from production-grade security - but it's fucking cool.
> Use Docker mode (default when Docker is installed) - the custom seccomp policy blocks network syscalls and most dangerous operations. For extra isolation, use [gVisor](https://gvisor.dev/) as the Docker runtime.

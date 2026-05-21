# RLM-Bench

**A small, reproducible benchmark for Recursive Language Models (RLMs) and any other "LLM + runtime" system.**

RLM-Bench packages the 12 evaluation tasks from the [Recursive Language Models paper](https://arxiv.org/abs/2512.24601), plus one constraint-satisfaction puzzle, behind a single CLI and a tiny plugin interface. You write one `Runner` class for your system, decorate it with `@register_runner`, and you can compare it head-to-head with vanilla LLM calls, [minRLM](../README.md), and the [official reference RLM](https://github.com/alexzhang13/rlm) on the exact same task instances, with the exact same scorers.

The numbers in this file come from running it. They are not cherry-picked: every model row keeps its losses, and the per-task tables include the tasks where minRLM is worse than vanilla.

- [Headline leaderboard](#headline-leaderboard)
- [Run it](#run-it)
- [Add your own RLM](#add-your-own-rlm)
- [Submitting results](#submitting-results)
- [Per-model results](#gpt-5-mini-primary-benchmark)
- [Datasets](#datasets)

---

## Headline leaderboard

6,600 evaluations across 4 models and 12 tasks. The "Tasks won" column counts top accuracy per task (ties go to whoever's listed first).

| Model | Runner | Accuracy | Avg tokens | Cost (600 evals) | Avg latency | Tasks won |
|---|---|---|---|---|---|---|
| **GPT-5.2** | **minRLM** | **78.2%** | **8,096** | $18.93 | 20.4s | **11 / 12** |
| GPT-5.2 | Vanilla | 48.2% | 14,196 | $16.50 | 8.0s | 1 / 12 |
| **GPT-5-mini** | **minRLM** | **72.7%** | **8,151** | **$2.86** | 25.8s | **6 / 12** |
| GPT-5-mini | Official RLM | 69.7% | 29,327 | $7.92 | 60.9s | 3 / 12 |
| GPT-5-mini | Vanilla | 69.5% | 20,967 | $4.74 | 24.2s | 3 / 12 |
| **GPT-5.4-mini** | **minRLM** | **69.5%** | **9,388** | $7.23 | 8.8s | **8 / 12** |
| GPT-5.4-mini | Official RLM | 50.2% | 47,439 | $23.44 | 22.3s | 1 / 12 |
| GPT-5.4-mini | Vanilla | 47.2% | 15,072 | $7.15 | 2.6s | 3 / 12 |
| GPT-5-nano | Vanilla | **63.2%** | 18,137 | $1.16 | 23.5s | **7 / 12** |
| GPT-5-nano | minRLM | 53.7% | **13,811** | **$0.74** | 14.3s | 5 / 12 |
| GPT-5-nano | Official RLM | 43.3% | 27,176 | $2.68 | 81.2s | 0 / 12 |

A few things to read out of this:

- The minRLM advantage **grows with model capability** — from -9.5pp on the smallest model, to +3.2pp on the mid-tier, to +22.3pp on a newer mid-tier where raw prompting regressed, to +30pp on the frontier model.
- The **token reduction is consistent across every model**: 1.3×–2.6× per query vs vanilla, 1.6×–8.0× per query vs the official RLM.
- The **small-model loss is real**: when the model can't reliably write the REPL code, the recursion adds cost without buying accuracy. RLM-Bench is built to surface this, not to hide it.

Want to add a runner? See [Add your own RLM](#add-your-own-rlm).

---

## Run it

```bash
git clone https://github.com/avilum/minrlm && cd minrlm
uv sync --extra eval
export OPENAI_API_KEY="sk-..."

# Smoke test - 1 task, 1 instance
uv run python eval/quickstart.py

# Single task
uv run python eval/run.py --model gpt-5-mini --tasks official_sniah --runs 10

# Full benchmark (reproduces the GPT-5-mini row of the leaderboard)
uv run python eval/run.py \
    --tasks all \
    --runners minrlm-reasoning,vanilla,official \
    --runs 50 --parallel 12 --task-parallel 12 \
    --output-dir logs/my_eval

# Cross-model (swap --model for gpt-5-nano, gpt-5.4-mini, or gpt-5.2)
uv run python eval/run.py \
    --model gpt-5.2 \
    --tasks all \
    --runners minrlm-reasoning,vanilla \
    --runs 50 --parallel 12 --task-parallel 12 \
    --output-dir logs/my_eval_gpt52
```

Output is one JSON file per run-set plus PNG plots under `--output-dir`. Cost is computed from [litellm](https://github.com/BerriAI/litellm) pricing data at `eval/data/model_prices.json`.

After `pip install minrlm[eval]` the same command is also available as:

```bash
rlm-bench --tasks all --runners minrlm-reasoning,vanilla --runs 50
```

---

## Add your own RLM

The runner interface is one method. Subclass `BaseRunner`, decorate with `@register_runner(name)`, and your runner appears in `--runners`.

```python
# eval/runners_mine.py
from eval.runners import BaseRunner, RunResult, register_runner

@register_runner("my-rlm")
class MyRLMRunner(BaseRunner):
    """My recursive language model."""

    description = "MyRLM v0.1 - recursive code-exec LLM"

    def run(self, task: str, context: str) -> RunResult:
        # Call into your RLM however you like. The benchmark only needs:
        # - the final answer string
        # - token + iteration counts (for cost / efficiency plots)
        answer, usage = my_rlm.solve(task=task, context=context)
        return RunResult(
            response=answer,
            total_tokens=usage.total_tokens,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            iterations=usage.iterations,
        )
```

Make sure your runner module is imported somewhere reachable (e.g. add `import eval.runners_mine` at the top of `eval/run.py`, or pass `--extra-runners eval.runners_mine`). Then:

```bash
uv run python eval/run.py \
    --tasks all \
    --runners my-rlm,vanilla,minrlm-reasoning \
    --runs 50
```

That's the whole adapter. Same task instances, same scorers, same plots.

### What `BaseRunner` actually requires

| Field on `RunResult` | Required | Used for |
|---|---|---|
| `response` | yes | scoring (passed to `task.check(response, expected)`) |
| `total_tokens`, `input_tokens`, `output_tokens` | recommended | token-efficiency plots, cost computation |
| `iterations` | recommended | iterations-per-task plot |
| `time_seconds` | filled by harness if you leave it 0 | latency plots |
| `error` | set on failure | excluded from accuracy, surfaced in summary |
| `generated_code`, `log_file_path` | optional | shown in the visualizer & log inspector |

If your system can't report token counts (e.g. closed-source API with no usage data), set them to 0 — accuracy and latency will still be valid, only the cost/token plots will be blank for your runner.

### What `BaseTask` looks like (for adding tasks)

```python
from eval.tasks import BaseTask, TaskInstance, register_task

@register_task("my_task")
class MyTask(BaseTask):
    description = "My new long-context retrieval task"

    def generate(self, seed: int = 42, **kwargs) -> TaskInstance:
        return TaskInstance(task=..., context=..., expected=...)

    def check(self, response: str, expected: str) -> bool:
        return expected.strip().lower() in response.strip().lower()
```

---

## Submitting results

If you've run RLM-Bench on a new model or with a new runner and want it in the leaderboard:

1. **Run it.** Use `--runs 50` or higher per task. Drop the output JSON under `BEST_EVALS/your-runner-or-model-name/`. The JSON includes per-instance results, so reviewers can spot-check.
2. **Open a PR** that:
   - Adds your `BEST_EVALS/...` directory.
   - Updates the [headline leaderboard](#headline-leaderboard) and (if it's a new model) the relevant per-model section below.
   - If you added a new runner, includes the runner module under `eval/`.
   - States the date, the API endpoint / version, the temperature/decoding settings, and the cost basis (the model entry in `eval/data/model_prices.json`).
3. **No dataset changes** in the same PR — task code freezes between runs so results stay comparable. If a dataset needs updating, do it in a separate PR and re-run.

If you'd rather not PR but still want your results referenced, open an issue with a link to the JSON and a one-paragraph description and we'll link it from this file.

---

## GPT-5-mini (Primary Benchmark)

**Model**: gpt-5-mini | **Evaluations**: 1,800 | **Tasks**: 12 | **Iterations**: 50 per task per runner | **Date**: 2026-03-13

Three runners compared: **minRLM** (this implementation), **Vanilla** (direct LLM call), **Official RLM** (paper's reference implementation).

### Summary

| | minRLM | Vanilla LLM | Official RLM |
|---|---|---|---|
| **Accuracy** | **72.7%** | 69.5% | 69.7% |
| **Avg Tokens** | **8,151** | 20,967 | 29,327 |
| **Avg Latency** | 25.8s | 24.2s | 60.9s |
| **Total Cost (600 evals)** | **$2.86** | $4.74 | $7.92 |

**minRLM vs Vanilla**: 2.6× fewer tokens, 1.7× cheaper, +3.2pp accuracy.
**minRLM vs Official**: 3.6× fewer tokens, 2.8× cheaper, +3.0pp accuracy.

![Summary Dashboard](../docs/summary_dashboard.png)

### Accuracy by Task

| Task | minRLM | Vanilla | Official | N |
|------|--------|---------|----------|---|
| SNIAH | **94%** | 100% | 76% | 50 |
| OOLONG | **92%** | 78% | 80% | 50 |
| GDP Val | **86%** | 54% | 50% | 50 |
| IFEval | **84%** | 78% | 78% | 50 |
| MMLU-Pro | 82% | **90%** | 86% | 50 |
| LiveCodeBench | **80%** | 64% | 60% | 50 |
| AIME 2025 | 74% | **88%** | 84% | 50 |
| GPQA Diamond | 70% | 66% | **74%** | 50 |
| BrowseComp | 62% | 16% | **66%** | 50 |
| RepoQA | 62% | **98%** | 96% | 50 |
| LongBench V2 | 46% | **56%** | 48% | 50 |
| CodeQA | 40% | **46%** | 38% | 50 |

minRLM is the top scorer on 6 of 12 tasks. Vanilla wins on MMLU-Pro, AIME 2025, RepoQA, LongBench V2, and CodeQA. Vanilla fails on BrowseComp (16%) because the context exceeds the token limit.

![Accuracy per Task](../docs/accuracy_per_task.png)

### Token Efficiency by Task

Sorted by minRLM savings vs Official RLM.

| Task | minRLM | Vanilla | Official | vs Vanilla | vs Official |
|------|--------|---------|----------|------------|-------------|
| CodeQA | 9,724 | 95,332 | 78,232 | **9.8×** | **8.0×** |
| LongBench V2 | 10,767 | 87,813 | 83,807 | **8.2×** | **7.8×** |
| BrowseComp | 10,740 | 34,084 | 68,354 | **3.2×** | **6.4×** |
| SNIAH | 6,328 | 3,758 | 16,283 | — | **2.6×** |
| OOLONG | 6,184 | 12,196 | 14,373 | **2.0×** | **2.3×** |
| RepoQA | 8,026 | 3,958 | 17,944 | — | **2.2×** |
| GPQA Diamond | 6,679 | 2,140 | 14,272 | — | **2.1×** |
| GDP Val | 12,007 | 4,236 | 20,458 | — | **1.7×** |
| IFEval | 5,963 | 1,360 | 9,316 | — | **1.6×** |
| AIME 2025 | 7,951 | 3,965 | 11,300 | — | **1.4×** |
| MMLU-Pro | 6,341 | 885 | 8,461 | — | **1.3×** |
| LiveCodeBench | 7,106 | 1,877 | 9,128 | — | **1.3×** |

"—" = vanilla uses fewer tokens on that task. minRLM uses fewer tokens than Official RLM on **every task** (1.3×–8.0×).

![Tokens per Task](../docs/tokens_per_task.png)

![Token Savings](../docs/token_savings.png)

### Cost by Task

50 evaluations per runner per task.

Aggregate totals: minRLM **$2.86**, Vanilla $4.74, Official $7.92 (600 evals each).

minRLM is cheaper than Official RLM on every task. minRLM is cheaper than Vanilla on tasks with large context (BrowseComp, CodeQA, LongBench V2, OOLONG, RepoQA).

![Cost per Task](../docs/cost_per_task.png)

![Accuracy vs Cost](../docs/accuracy_vs_cost.png)

### Latency by Task

| Task | minRLM | Vanilla | Official | Faster than Official |
|------|--------|---------|----------|----------------------|
| OOLONG | 15.0s | 40.4s | 32.7s | **2.2×** |
| SNIAH | 13.8s | 2.1s | 30.4s | **2.2×** |
| IFEval | 18.0s | 19.8s | 33.0s | **1.8×** |
| LiveCodeBench | 18.4s | 20.3s | 26.9s | **1.5×** |
| MMLU-Pro | 18.5s | 11.0s | 26.3s | **1.4×** |
| CodeQA | 20.8s | 19.8s | 84.4s | **4.1×** |
| LongBench V2 | 21.9s | 23.8s | 96.4s | **4.4×** |
| GPQA Diamond | 24.7s | 27.5s | 77.0s | **3.1×** |
| RepoQA | 25.3s | 8.7s | 23.9s | 0.9× |
| BrowseComp | 27.8s | 7.0s | 123.6s | **4.4×** |
| AIME 2025 | 39.0s | 54.9s | 74.3s | **1.9×** |
| GDP Val | 66.3s | 55.6s | 102.1s | **1.5×** |

minRLM is faster than Official RLM on 11 of 12 tasks.

![Latency per Task](../docs/latency_per_task.png)

![Accuracy vs Latency](../docs/accuracy_vs_latency.png)

### Iterations by Task

| Task | minRLM Avg Iterations |
|------|-----------------------|
| OOLONG | 1.0 |
| CodeQA | 1.0 |
| LongBench V2 | 1.0 |
| SNIAH | 1.0 |
| GPQA Diamond | 1.0 |
| MMLU-Pro | 1.0 |
| IFEval | 1.0 |
| BrowseComp | 1.1 |
| RepoQA | 1.1 |
| AIME 2025 | 1.1 |
| LiveCodeBench | 1.1 |
| GDP Val | 1.2 |

Most tasks complete in a single iteration. GDP Val occasionally requires a second pass.

---

## GPT-5.2

**Model**: gpt-5.2 | **Evaluations**: 1,200 | **Tasks**: 12 | **Iterations**: 50 per task per runner | **Date**: 2026-03-15

Two runners compared: **minRLM** and **Vanilla**. No official RLM runner for this model.

### Summary

| | minRLM | Vanilla LLM |
|---|---|---|
| **Accuracy** | **78.2%** | 48.2% |
| **Avg Tokens** | 8,096 | 14,196 |
| **Avg Latency** | 20.4s | 8.0s |
| **Total Cost (600 evals)** | $18.93 | $16.50 |

**minRLM vs Vanilla**: +30.0pp accuracy, 1.8× fewer tokens, 14.7% more expensive — the extra cost is buying the accuracy gain.

### Accuracy by Task

| Task | minRLM | Vanilla | N |
|------|--------|---------|---|
| SNIAH | 100% | 100% | 50 |
| AIME 2025 | **96%** | 0% | 50 |
| OOLONG | **96%** | 64% | 50 |
| MMLU-Pro | **92%** | 42% | 50 |
| RepoQA | 84% | **98%** | 50 |
| IFEval | **82%** | 76% | 50 |
| GPQA Diamond | **76%** | 46% | 50 |
| GDP Val | **74%** | 50% | 50 |
| BrowseComp | **72%** | 14% | 50 |
| LiveCodeBench | **66%** | 42% | 50 |
| CodeQA | **56%** | 20% | 50 |
| LongBench V2 | **44%** | 26% | 50 |

minRLM wins on 10 of 12 tasks and ties on SNIAH. Vanilla wins only on RepoQA (full-context retrieval is its sweet spot). AIME 2025 is the most dramatic flip: 96% vs 0% — the vanilla runner outputs a bare number with no chain-of-thought, while the REPL forces it to compute via code.

---

## GPT-5.4-mini

**Model**: gpt-5.4-mini | **Evaluations**: 1,800 | **Tasks**: 12 | **Iterations**: 50 per task per runner | **Date**: 2026-03-27

Three runners compared: **minRLM**, **Vanilla**, **Official RLM**.

### Summary

| | minRLM | Vanilla LLM | Official RLM |
|---|---|---|---|
| **Accuracy** | **69.5%** | 47.2% | 50.2% |
| **Avg Tokens** | **9,388** | 15,072 | 47,439 |
| **Avg Latency** | 8.8s | 2.6s | 22.3s |
| **Total Cost (600 evals)** | $7.23 | **$7.15** | $23.44 |

**minRLM vs Vanilla**: 1.6× fewer tokens, +22.3pp accuracy, roughly tied on cost.
**minRLM vs Official**: 5.1× fewer tokens, 3.2× cheaper, +19.3pp accuracy.

GPT-5.4-mini appears to produce shorter, terser outputs by default — vanilla and official both regressed significantly vs GPT-5-mini (vanilla: 69.5% → 47.2%, official: 69.7% → 50.2%). minRLM held steady (72.7% → 69.5%), showing the REPL-based approach is resilient to model regressions in raw prompting.

### Accuracy by Task

| Task | minRLM | Vanilla | Official | N |
|------|--------|---------|----------|---|
| SNIAH | **96%** | 100% | 96% | 50 |
| OOLONG | **92%** | 52% | 86% | 50 |
| MMLU-Pro | **92%** | 38% | 24% | 50 |
| GDP Val | **82%** | 52% | 52% | 50 |
| IFEval | **82%** | 80% | 52% | 50 |
| AIME 2025 | **80%** | 0% | 46% | 50 |
| GPQA Diamond | **64%** | 42% | 28% | 50 |
| RepoQA | 60% | **96%** | 84% | 50 |
| CodeQA | **54%** | 28% | 26% | 50 |
| BrowseComp | **52%** | 14% | 72% | 50 |
| LongBench V2 | **52%** | 30% | 30% | 50 |
| LiveCodeBench | 28% | **34%** | 6% | 50 |

minRLM wins on 8 of 12 tasks. The AIME result mirrors GPT-5.2 — vanilla scores 0% while minRLM scores 80%. RepoQA and LiveCodeBench remain weak spots.

---

## GPT-5-nano

**Model**: gpt-5-nano | **Evaluations**: 1,800 | **Tasks**: 12 | **Iterations**: 50 per task per runner | **Date**: 2026-03-14

Three runners compared: **minRLM**, **Vanilla**, **Official RLM**.

### Summary

| | minRLM | Vanilla LLM | Official RLM |
|---|---|---|---|
| **Accuracy** | 53.7% | **63.2%** | 43.3% |
| **Avg Tokens** | 13,811 | 18,137 | 27,176 |
| **Avg Latency** | 14.3s | 23.5s | 81.2s |
| **Total Cost (600 evals)** | **$0.74** | $1.16 | $2.68 |

On the smallest model, vanilla LLM outperforms both RLM implementations. minRLM still beats the official RLM by +10.4pp while costing 3.6× less, but the recursion overhead isn't a win here. This is the row that argues most clearly for *picking the right tool for the model*.

### Accuracy by Task

| Task | minRLM | Vanilla | Official | N |
|------|--------|---------|----------|---|
| SNIAH | **90%** | 100% | 56% | 50 |
| GDP Val | **82%** | 60% | 42% | 50 |
| MMLU-Pro | 80% | **92%** | 70% | 50 |
| OOLONG | **76%** | 70% | 34% | 50 |
| IFEval | 70% | **74%** | 48% | 50 |
| AIME 2025 | 68% | **86%** | **80%** | 50 |
| GPQA Diamond | 56% | **68%** | 58% | 50 |
| CodeQA | **38%** | 28% | 36% | 50 |
| BrowseComp | **36%** | 14% | 28% | 50 |
| LongBench V2 | 32% | **34%** | 24% | 50 |
| RepoQA | 14% | **96%** | 32% | 50 |
| LiveCodeBench | 2% | **36%** | 12% | 50 |

minRLM wins on 5 of 12 tasks (SNIAH vs Official, GDP Val, OOLONG, CodeQA, BrowseComp). The small model struggles most with code generation (LiveCodeBench: 2%) and code retrieval (RepoQA: 14%).

---

## Datasets

| Task name | Dataset | Source |
|-----------|---------|--------|
| `official_browsecomp` | BrowseComp-Plus | [Tevatron/browsecomp-plus](https://huggingface.co/datasets/Tevatron/browsecomp-plus) |
| `official_sniah` | RULER NIAH | [tonychenxyz/ruler-full](https://huggingface.co/datasets/tonychenxyz/ruler-full) |
| `official_oolong` | OOLONG | [oolongbench/oolong-synth](https://huggingface.co/datasets/oolongbench/oolong-synth) |
| `official_longbench_v2` | LongBench-v2 | [zai-org/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2) |
| `official_codeqa` | LongBench-v2 (code subset) | [zai-org/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2) |
| `official_repoqa` | RepoQA | [evalplus/repoqa_release](https://github.com/evalplus/repoqa_release) |
| `official_gdpval` | GDP Val | [openai/gdpval](https://huggingface.co/datasets/openai/gdpval) |
| `official_aime_2025` | AIME 2025 | [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025) |
| `official_gpqa_diamond` | GPQA Diamond | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) (gated) |
| `official_mmlu_pro` | MMLU-Pro | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| `official_ifeval` | IFEval | [google/IFEval](https://huggingface.co/datasets/google/IFEval) |
| `official_livecodebench` | LiveCodeBench v6 | [livecodebench/code_generation_lite](https://huggingface.co/datasets/livecodebench/code_generation_lite) |
| `official_sudoku_extreme` | Sudoku Extreme | [sapientinc/sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme) |

### Downloading datasets

GDP Val, AIME 2025, GPQA Diamond, MMLU-Pro, IFEval, and LiveCodeBench are auto-downloaded at runtime. GPQA Diamond is gated — accept the license at [huggingface.co/datasets/Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) and run `huggingface-cli login` first.

The remaining datasets must be pre-downloaded to `evals/data/`:

```bash
uv run --with datasets,huggingface_hub python -c "
from datasets import load_dataset

datasets = {
    'oolong': ('oolongbench/oolong-synth', None),
    'longbench_v2': ('zai-org/LongBench-v2', None),
    'browsecomp_plus': ('Tevatron/browsecomp-plus', None),
    'ruler_full_mirror': ('tonychenxyz/ruler-full', 'plain'),
}

for name, (repo, config) in datasets.items():
    print(f'Downloading {repo}...')
    ds = load_dataset(repo, config)
    ds.save_to_disk(f'evals/data/{name}')
    print(f'  Saved to evals/data/{name}')
"
```

---

## Try minRLM directly (zero-install)

```bash
export OPENAI_API_KEY="your-key"

# Just a task
uvx minrlm "What is the sum of the first 100 primes?"

# Task + file as context
uvx minrlm "How many ERROR lines in the last hour?" ./server.log

# Pipe context from stdin
cat huge_dataset.csv | uvx minrlm "Which product had the highest return rate?"

# Show generated code (-s) and token stats (-v)
uvx minrlm -sv "Return the sum of all primes up to 1,000,000."
# -> Sieve of Eratosthenes in 6,215 tokens, 1 iteration
# -> Answer: 37550402023

uvx minrlm -sv "Return all primes up to 1,000,000, reversed. Return a list of numbers."
# -> 999983, 999979, 999961, 999959, 999953, ...
# -> Tokens: 6,258 | Output: 616,964 chars (~154K tokens) | 25x savings
```

---

## Raw result files

- GPT-5-mini: [`BEST_EVALS/BEST_new-entropy-prompts-12-tasks-all-runners-gpt-5-mini-50-runs-BEST/eval_20260313_195547.json`](../BEST_EVALS/BEST_new-entropy-prompts-12-tasks-all-runners-gpt-5-mini-50-runs-BEST/eval_20260313_195547.json)
- GPT-5.4-mini: [`BEST_EVALS/BEST_12-tasks-all-runners-gpt-5.4-mini-50-runs-2/eval_20260327_001353.json`](../BEST_EVALS/BEST_12-tasks-all-runners-gpt-5.4-mini-50-runs-2/eval_20260327_001353.json)
- GPT-5.2: [`BEST_EVALS/BEST_new-entropy-prompts-12-tasks-all-runners-gpt-5.2-50-runs-after-opus-46/eval_20260315_184830.json`](../BEST_EVALS/BEST_new-entropy-prompts-12-tasks-all-runners-gpt-5.2-50-runs-after-opus-46/eval_20260315_184830.json)
- GPT-5-nano: [`BEST_EVALS/BEST_new-entropy-prompts-12-tasks-all-runners-gpt-5-nano-50-runs-after-opus-46/eval_20260314_024652.json`](../BEST_EVALS/BEST_new-entropy-prompts-12-tasks-all-runners-gpt-5-nano-50-runs-after-opus-46/eval_20260314_024652.json)

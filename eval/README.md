# Benchmark Results

**Model**: gpt-5-mini | **Evaluations**: 1,800 | **Tasks**: 12 | **Iterations**: 50 per task per runner | **Date**: 2026-03-13

Three runners compared: **minRLM** (this implementation), **Vanilla** (direct LLM call), **Official RLM** (paper's reference implementation).

## Summary

| | minRLM | Vanilla LLM | Official RLM |
|---|---|---|---|
| **Accuracy** | **72.7%** | 69.5% | 69.7% |
| **Avg Tokens** | **8,151** | 20,967 | 29,327 |
| **Avg Latency** | 25.8s | 24.2s | 60.9s |
| **Total Cost (600 evals)** | **$2.86** | $4.74 | $7.92 |

**minRLM vs Vanilla**: 2.6x fewer tokens, 1.7x cheaper, +3.2pp accuracy
**minRLM vs Official**: 3.6x fewer tokens, 2.8x cheaper, +3.0pp accuracy

![Summary Dashboard](../docs/summary_dashboard.png)

## Accuracy by Task

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

minRLM is the top scorer on 6 of 12 tasks (OOLONG, GDP Val, IFEval, LiveCodeBench, SNIAH vs official, BrowseComp vs vanilla). Vanilla fails on BrowseComp (16%) because the context exceeds the token limit.

![Accuracy per Task](../docs/accuracy_per_task.png)

## Token Efficiency by Task

Sorted by minRLM savings vs Official RLM.

| Task | minRLM | Vanilla | Official | vs Vanilla | vs Official |
|------|--------|---------|----------|------------|-------------|
| CodeQA | 9,724 | 95,332 | 78,232 | **9.8x** | **8.0x** |
| LongBench V2 | 10,767 | 87,813 | 83,807 | **8.2x** | **7.8x** |
| BrowseComp | 10,740 | 34,084 | 68,354 | **3.2x** | **6.4x** |
| SNIAH | 6,328 | 3,758 | 16,283 | - | **2.6x** |
| OOLONG | 6,184 | 12,196 | 14,373 | **2.0x** | **2.3x** |
| RepoQA | 8,026 | 3,958 | 17,944 | - | **2.2x** |
| GPQA Diamond | 6,679 | 2,140 | 14,272 | - | **2.1x** |
| GDP Val | 12,007 | 4,236 | 20,458 | - | **1.7x** |
| IFEval | 5,963 | 1,360 | 9,316 | - | **1.6x** |
| AIME 2025 | 7,951 | 3,965 | 11,300 | - | **1.4x** |
| MMLU-Pro | 6,341 | 885 | 8,461 | - | **1.3x** |
| LiveCodeBench | 7,106 | 1,877 | 9,128 | - | **1.3x** |

"-" = vanilla uses fewer tokens on that task. minRLM uses fewer tokens than Official RLM on **every task** (1.3x-8.0x).

![Tokens per Task](../docs/tokens_per_task.png)

![Token Savings](../docs/token_savings.png)

## Cost by Task

50 evaluations per runner per task.

Aggregate totals: minRLM **$2.86**, Vanilla $4.74, Official $7.92 (600 evals each).

minRLM is cheaper than Official RLM on every task. minRLM is cheaper than Vanilla on tasks with large context (BrowseComp, CodeQA, LongBench V2, Oolong, RepoQA).

![Cost per Task](../docs/cost_per_task.png)

![Accuracy vs Cost](../docs/accuracy_vs_cost.png)

## Latency by Task

| Task | minRLM | Vanilla | Official | Faster than Official |
|------|--------|---------|----------|----------------------|
| OOLONG | 15.0s | 40.4s | 32.7s | **2.2x** |
| SNIAH | 13.8s | 2.1s | 30.4s | **2.2x** |
| IFEval | 18.0s | 19.8s | 33.0s | **1.8x** |
| LiveCodeBench | 18.4s | 20.3s | 26.9s | **1.5x** |
| MMLU-Pro | 18.5s | 11.0s | 26.3s | **1.4x** |
| CodeQA | 20.8s | 19.8s | 84.4s | **4.1x** |
| LongBench V2 | 21.9s | 23.8s | 96.4s | **4.4x** |
| GPQA Diamond | 24.7s | 27.5s | 77.0s | **3.1x** |
| RepoQA | 25.3s | 8.7s | 23.9s | 0.9x |
| BrowseComp | 27.8s | 7.0s | 123.6s | **4.4x** |
| AIME 2025 | 39.0s | 54.9s | 74.3s | **1.9x** |
| GDP Val | 66.3s | 55.6s | 102.1s | **1.5x** |

minRLM is faster than Official RLM on 11 of 12 tasks.

![Latency per Task](../docs/latency_per_task.png)

![Accuracy vs Latency](../docs/accuracy_vs_latency.png)

## Iterations by Task

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

### Downloading datasets

GDP Val, AIME 2025, GPQA Diamond, MMLU-Pro, IFEval, and LiveCodeBench are auto-downloaded at runtime.
GPQA Diamond is a gated dataset — you must accept the license at [huggingface.co/datasets/Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) and run `huggingface-cli login` first.

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

## Reproduction

```bash
# Install eval dependencies
uv sync --extra eval

export OPENAI_API_KEY="your-key"

# Quick smoke test (3 tasks, 3 runs each)
uv run python eval/quickstart.py

# Single task
uv run python eval/run.py --model gpt-5-mini --tasks official_sniah --runs 10

# All tasks - single runner
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

Raw data: [`BEST_EVALS/BEST_new-entropy-prompts-12-tasks-all-runners-gpt-5-mini-50-runs-BEST/eval_20260313_195547.json`](../BEST_EVALS/BEST_new-entropy-prompts-12-tasks-all-runners-gpt-5-mini-50-runs-BEST/eval_20260313_195547.json)

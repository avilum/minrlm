# Benchmark Results

**Model**: gpt-5-mini | **Evaluations**: 1,800 | **Tasks**: 12 | **Iterations**: 50 per task per runner | **Date**: 2026-03-08

Three runners compared: **minRLM** (this implementation), **Vanilla** (direct LLM call), **Official RLM** (paper's reference implementation).

## Summary

| | minRLM | Vanilla LLM | Official RLM |
|---|---|---|---|
| **Accuracy** | 70.2% | 71.7% | 71.0% |
| **Avg Tokens** | **6,568** | 20,112 | 47,270 |
| **Avg Latency** | 27.8s | 21.9s | 63.4s |
| **Total Cost (600 evals)** | **$2.97** | $4.57 | $11.68 |

**minRLM vs Vanilla**: 3.1x fewer tokens, 1.5x cheaper
**minRLM vs Official**: 7.2x fewer tokens, 3.9x cheaper

![Summary Dashboard](../docs/summary_dashboard.png)

## Accuracy by Task

| Task | minRLM | Vanilla | Official | N |
|------|--------|---------|----------|---|
| SNIAH | **100%** | 100% | 96% | 50 |
| MMLU-Pro | **96%** | 96% | 88% | 50 |
| Oolong | 92% | 94% | **98%** | 50 |
| AIME 2025 | 80% | **96%** | 86% | 50 |
| RepoQA | 76% | **98%** | 94% | 50 |
| GPQA Diamond | **74%** | 72% | 64% | 50 |
| IFEval | 70% | **78%** | 66% | 50 |
| LiveCodeBench | 62% | **70%** | 62% | 50 |
| BrowseComp | 54% | 6% | **62%** | 50 |
| GDP Val | 52% | 52% | 52% | 50 |
| CodeQA | 46% | **50%** | 40% | 50 |
| LongBench V2 | 40% | **48%** | 44% | 50 |

minRLM beats Official RLM on 7 of 12 tasks. minRLM beats Vanilla on GPQA Diamond (74% vs 72%) and BrowseComp (54% vs 6%).
Vanilla fails on BrowseComp (6%) because the context exceeds the token limit.

![Accuracy per Task](../docs/accuracy_per_task.png)

## Token Efficiency by Task

Sorted by minRLM savings vs Official RLM.

| Task | minRLM | Vanilla | Official | vs Vanilla | vs Official |
|------|--------|---------|----------|------------|-------------|
| CodeQA | 6,909 | 95,436 | 169,451 | **13.8x** | **24.5x** |
| LongBench V2 | 7,193 | 85,435 | 124,998 | **11.9x** | **17.4x** |
| BrowseComp | 7,467 | 26,069 | 101,124 | **3.5x** | **13.5x** |
| GDP Val | 10,352 | 4,145 | 48,358 | - | **4.7x** |
| IFEval | 4,800 | 1,470 | 17,414 | - | **3.6x** |
| Oolong | 4,883 | 6,414 | 15,587 | **1.3x** | **3.2x** |
| GPQA Diamond | 5,116 | 1,753 | 14,716 | - | **2.9x** |
| RepoQA | 7,960 | 9,605 | 21,994 | **1.2x** | **2.8x** |
| SNIAH | 6,265 | 3,763 | 17,338 | - | **2.8x** |
| AIME 2025 | 6,069 | 4,221 | 15,659 | - | **2.6x** |
| MMLU-Pro | 4,606 | 806 | 11,151 | - | **2.4x** |
| LiveCodeBench | 7,196 | 2,226 | 9,454 | - | **1.3x** |

"-" = vanilla uses fewer tokens on that task. minRLM uses fewer tokens than Official RLM on **every task** (1.3x–24.5x).

![Tokens per Task](../docs/tokens_per_task.png)

![Token Savings](../docs/token_savings.png)

## Cost by Task

50 evaluations per runner per task.

Aggregate totals: minRLM **$2.97**, Vanilla $4.57, Official $11.68 (600 evals each).

minRLM is cheaper than Official RLM on every task (1.3x–24.5x proportional to token savings).
minRLM is cheaper than Vanilla on tasks with large context (BrowseComp, CodeQA, LongBench V2, Oolong, RepoQA).

![Cost per Task](../docs/cost_per_task.png)

![Accuracy vs Cost](../docs/accuracy_vs_cost.png)

## Latency by Task

| Task | minRLM | Vanilla | Official | Faster than Official |
|------|--------|---------|----------|----------------------|
| LongBench V2 | 21.2s | 16.1s | 86.4s | **4.1x** |
| Oolong | 37.7s | 47.6s | 40.5s | 1.1x |
| CodeQA | 22.6s | 16.7s | 102.1s | **4.5x** |
| RepoQA | 27.0s | 7.1s | 26.1s | 1.0x |
| BrowseComp | 30.8s | 6.4s | 130.7s | **4.2x** |
| SNIAH | 18.0s | 1.7s | 21.5s | 1.2x |
| AIME 2025 | 25.7s | 47.6s | 60.5s | **2.4x** |
| GDP Val | 50.4s | 38.9s | 99.6s | **2.0x** |
| GPQA Diamond | 25.0s | 19.1s | 66.3s | **2.7x** |
| MMLU-Pro | 24.8s | 11.4s | 32.0s | 1.3x |
| IFEval | 27.6s | 25.8s | 61.7s | **2.2x** |
| LiveCodeBench | 23.4s | 24.2s | 33.5s | 1.4x |

minRLM is faster than Official RLM on all 12 tasks.

![Latency per Task](../docs/latency_per_task.png)

![Accuracy vs Latency](../docs/accuracy_vs_latency.png)

## Iterations by Task

| Task | minRLM Avg Iterations |
|------|-----------------------|
| OOLONG | 1.0 |
| LongBench V2 | 1.0 |
| SNIAH | 1.0 |
| GPQA Diamond | 1.0 |
| MMLU-Pro | 1.0 |
| BrowseComp | 1.0 |
| CodeQA | 1.1 |
| RepoQA | 1.1 |
| IFEval | 1.1 |
| GDP Val | 1.3 |
| AIME 2025 | 1.5 |
| LiveCodeBench | 1.6 |

GDP Val, AIME 2025, and LiveCodeBench require multiple code-execution iterations, explaining the higher latency on those tasks.

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
uv run python eval/run.py --model gpt-4o-mini --tasks official_sniah --runs 10

# All tasks - single runner
uv run python eval/run.py \
    --model gpt-4o-mini \
    --tasks all \
    --runners minrlm-reasoning \
    --runs 50 \
    --parallel 5 \
    --output-dir logs/my_eval

# Full multi-runner benchmark (reproduces the table above)
./run_comprehensive_official_benchmark.sh
```

Raw data:
- All runners, 12 tasks: [`logs/new-prompts3-all-tasks-all-runners-50-gpt-5-mini/eval_20260308_050847.json`](../logs/new-prompts3-all-tasks-all-runners-50-gpt-5-mini/eval_20260308_050847.json)
- Original 8-task run: [`BEST_EVALS/BEST_e2e_gpt5-mini-all-tasks-for-blog-50-runs/eval_20260303_230844.json`](../BEST_EVALS/BEST_e2e_gpt5-mini-all-tasks-for-blog-50-runs/eval_20260303_230844.json)

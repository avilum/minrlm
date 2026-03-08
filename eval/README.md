# Benchmark Results

**Model**: gpt-5-mini | **Evaluations**: 1,200 | **Tasks**: 8 | **Iterations**: 50 per task per runner | **Date**: 2026-03-03

Three runners compared: **minRLM** (this implementation), **Vanilla** (direct LLM call), **Official RLM** (paper's reference implementation).

## Summary

| | minRLM | Vanilla LLM | Official RLM |
|---|---|---|---|
| **Accuracy** | **71.8%** | 64.0% | 69.5% |
| **Avg Tokens** | **5,198** | 19,319 | 41,781 |
| **Avg Latency** | 30.1s | 27.0s | 86.5s |
| **Total Cost (400 evals)** | **$1.58** | $2.98 | $6.91 |

**minRLM vs Vanilla**: 3.7x fewer tokens, 1.9x cheaper
**minRLM vs Official**: 8.0x fewer tokens, 4.4x cheaper

![Summary Dashboard](../docs/summary_dashboard.png)

## Accuracy by Task

| Task | minRLM | Vanilla | Official | N |
|------|--------|---------|----------|---|
| BrowseComp | **94%** | 0% | 58% | 50 |
| SNIAH | **98%** | 100% | 94% | 50 |
| CodeQA | **56%** | 28% | 40% | 50 |
| AIME 2025 | 76% | **90%** | 84% | 50 |
| LongBench V2 | 36% | 46% | **48%** | 50 |
| OOLONG | 88% | 92% | **94%** | 50 |
| RepoQA | **90%** | 100% | 90% | 50 |
| GDP Val | 36% | **56%** | 48% | 50 |

minRLM is the top scorer on 2 of 8 tasks (BrowseComp, CodeQA) and beats Official RLM on 3 of 8 (BrowseComp, CodeQA, SNIAH).
Vanilla fails completely on BrowseComp (0%) due to context exceeding token limits.

![Accuracy per Task](../docs/accuracy_per_task.png)

## Token Efficiency by Task

Sorted by minRLM savings vs Official RLM.

| Task | minRLM | Vanilla | Official | vs Vanilla | vs Official |
|------|--------|---------|----------|------------|-------------|
| BrowseComp | 4,531 | 0 | 96,041 | ∞ | **21.2x** |
| LongBench V2 | 3,796 | 73,646 | 71,345 | **19.4x** | **18.8x** |
| CodeQA | 3,902 | 57,152 | 75,560 | **14.6x** | **19.4x** |
| RepoQA | 4,157 | 4,937 | 19,571 | 1.2x | **4.7x** |
| SNIAH | 4,494 | 3,765 | 17,959 | - | **4.0x** |
| OOLONG | 4,122 | 6,976 | 15,425 | **1.7x** | **3.7x** |
| GDP Val | 9,830 | 4,354 | 27,213 | - | **2.8x** |
| AIME 2025 | 6,750 | 3,725 | 11,130 | - | **1.7x** |

"-" = vanilla uses fewer tokens on that task. minRLM uses fewer tokens than Official RLM on every task.

![Tokens per Task](../docs/tokens_per_task.png)

![Token Savings](../docs/token_savings.png)

## Cost by Task

50 evaluations per runner per task.

Per-task cost breakdown not available for this run (model not in pricing database). Aggregate totals: minRLM **$1.58**, Vanilla $2.98, Official $6.91 (400 evals each).

minRLM is cheaper than Official RLM on every task (1.7x–21x proportional to token savings).
minRLM is cheaper than Vanilla on tasks with large context (BrowseComp, CodeQA, LongBench V2).

![Cost per Task](../docs/cost_per_task.png)

![Accuracy vs Cost](../docs/accuracy_vs_cost.png)

## Latency by Task

| Task | minRLM | Vanilla | Official | Faster than Official |
|------|--------|---------|----------|----------------------|
| LongBench V2 | 19.5s | 23.8s | 102.8s | **5.3x** |
| OOLONG | 21.3s | 42.3s | 37.6s | 1.8x |
| CodeQA | 22.2s | 16.5s | 100.0s | **4.5x** |
| RepoQA | 21.9s | 8.5s | 32.1s | 1.5x |
| BrowseComp | 30.2s | 4.8s | 171.9s | **5.7x** |
| SNIAH | 27.9s | 2.3s | 35.6s | 1.3x |
| AIME 2025 | 36.0s | 55.9s | 85.4s | **2.4x** |
| GDP Val | 62.0s | 61.9s | 126.4s | **2.0x** |

minRLM is faster than Official RLM on all 8 tasks.
minRLM is faster than Vanilla on 3 of 8 tasks (LongBench V2, OOLONG, AIME 2025).

![Latency per Task](../docs/latency_per_task.png)

![Accuracy vs Latency](../docs/accuracy_vs_latency.png)

## Iterations by Task

| Task | minRLM Avg Iterations |
|------|-----------------------|
| OOLONG | 1.0 |
| CodeQA | 1.0 |
| LongBench V2 | 1.0 |
| RepoQA | 1.0 |
| SNIAH | 1.0 |
| BrowseComp | 1.2 |
| GDP Val | 1.8 |
| AIME 2025 | 1.8 |

GDP Val and AIME 2025 require multiple code-execution iterations, explaining the higher latency on those tasks.

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
- All runners (unified run): [`BEST_EVALS/BEST_e2e_gpt5-mini-all-tasks-for-blog-50-runs/eval_20260303_230844.json`](../BEST_EVALS/BEST_e2e_gpt5-mini-all-tasks-for-blog-50-runs/eval_20260303_230844.json)

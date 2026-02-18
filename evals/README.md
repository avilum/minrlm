# Official Evaluation Datasets

This folder is for **official benchmark datasets** used in the RLM paper. The current `eval/` suite uses synthetic generators; these datasets let you run *real* benchmarks from the original sources.

**Important**: Datasets are large and are **not** checked into git. Download into `evals/data/` using the script below.

## Quick start

```bash
# Install dataset tooling (if not already installed)
uv pip install datasets huggingface_hub

# See available presets
uv run python evals/download_official.py --list

# Example: download OOLONG (official) + LongBench-v2 + RepoQA
uv run python evals/download_official.py --dataset oolong --dataset longbench_v2 --dataset repoqa

# Example: download BrowseComp-Plus (obfuscated) + corpus
uv run python evals/download_official.py --dataset browsecomp_plus --dataset browsecomp_plus_corpus --trust-remote-code
```

## Official datasets and sources

| RLM Task | Official Source | Notes |
| --- | --- | --- |
| **S-NIAH** | NVIDIA/RULER (official repo) | RULER provides the canonical NIAH evaluation. The official repo includes scripts to generate data. A community HF mirror (`tonychenxyz/ruler-full`) exists but is not official. |
| **OOLONG** | `oolongbench/oolong-synth` on Hugging Face | Official OOLONG benchmark dataset. |
| **OOLONG-Pairs** | Appendix E.1 (RLM paper) | The 20 query prompts are included in `evals/oolong_pairs/queries.jsonl` (CC BY 4.0). Use with OOLONG contexts. |
| **BrowseComp-Plus** | `Tevatron/browsecomp-plus` + `Tevatron/browsecomp-plus-corpus` on Hugging Face | Dataset is obfuscated. Follow the official deobfuscation script in the BrowseComp-Plus repo. |
| **CodeQA** | `zai-org/LongBench-v2` on Hugging Face | Filter to the **code repository understanding** subset (see dataset card for the exact `domain` / `sub_domain` labels). |
| **LongBench-v2** | `zai-org/LongBench-v2` on Hugging Face | Full LongBench-v2 benchmark across domains. |
| **RepoQA** | `evalplus/repoqa_release` (GitHub release) | Function retrieval from code repositories. |

## Folder layout

```
evals/
  data/                    # Downloaded datasets (ignored by git)
  oolong_pairs/
    queries.jsonl          # Official OOLONG-Pairs queries from the paper
  download_official.py     # Downloader for HF datasets
```

## Notes and caveats

- **Licenses vary** by dataset. The RLM paper is CC BY 4.0, but each benchmark has its own license and usage terms.
- **BrowseComp-Plus** includes a canary / obfuscation step. You must run the official deobfuscation script from the BrowseComp-Plus repo before evaluation.
- **S-NIAH** is synthetic *by design* in the official RULER benchmark. Use the official generator scripts to stay faithful to the benchmark.

## Running Benchmarks

After downloading, run benchmarks with:

```bash
# Run all official datasets
uv run python eval/run.py --model gpt-5-mini --tasks official --runs 100

# See all options
uv run python eval/run.py --help
```

See [`eval/README.md`](../eval/README.md) for full benchmark documentation, results, and reproduction commands.

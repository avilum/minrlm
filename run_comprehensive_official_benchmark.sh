#!/bin/bash
# COMPREHENSIVE OFFICIAL BENCHMARK - FULL PAPER SCALE
# Tests ALL official datasets with proper statistical sampling
# Compares: RLM (ours) vs Official RLM vs Vanilla LLM
# MAXIMIZED PARALLELIZATION for speed

set -e

MODEL="gpt-5-mini"
OUTPUT_DIR="evals/comprehensive_official_$(date +%Y%m%d_%H%M%S)"
RUNS=10  # Statistical validity
MAX_SAMPLES=50  # Representative sample per dataset

# Parallelization settings for MAXIMUM SPEED
PARALLEL=10  # Max parallel runners per task instance
TASK_PARALLEL=4  # Max parallel task instances (samples)

echo "=========================================================="
echo "COMPREHENSIVE OFFICIAL BENCHMARK - PAPER SCALE (PARALLEL)"
echo "=========================================================="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Runs per config: $RUNS"
echo "Max samples per official dataset: $MAX_SAMPLES"
echo ""
echo "Parallelization (for maximum speed):"
echo "  - Parallel runners: $PARALLEL"
echo "  - Parallel tasks: $TASK_PARALLEL"
echo "  - Expected speedup: ~$(($PARALLEL * $TASK_PARALLEL))x"
echo ""
echo "Official datasets being tested:"
echo "  - official_sniah (Needle-in-haystack)"
echo "  - official_oolong (Aggregation)"
echo "  - official_repoqa (Code retrieval)"
echo "  - official_codeqa (Code reasoning)"
echo "  - official_longbench_v2 (Long context QA)"
echo "  - official_browsecomp (Multi-hop research)"
echo ""
echo "Runners:"
echo "  - ours (minRLM - our implementation)"
echo "  - vanilla (Standard LLM)"
echo "  - official (Official RLM baseline)"
echo "=========================================================="

# ALL official datasets
OFFICIAL_TASKS="official_sniah,official_oolong,official_repoqa,official_codeqa,official_longbench_v2,official_browsecomp"

# All three runners
RUNNERS="ours,vanilla"

# Run comprehensive evaluation with MAXIMUM PARALLELIZATION
uv run python -m eval.run \
  --model "$MODEL" \
  --tasks "$OFFICIAL_TASKS" \
  --runners "$RUNNERS" \
  --runs "$RUNS" \
  --output-dir "$OUTPUT_DIR" \
  --official-max-samples "$MAX_SAMPLES" \
  --paper-scale \
  --parallel "$PARALLEL" \
  --task-parallel "$TASK_PARALLEL" \
  --no-plot

echo ""
echo "=========================================================="
echo "BENCHMARK COMPLETE!"
echo "=========================================================="
echo "Results: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. View summary:"
echo "     cat $OUTPUT_DIR/summary_*.md"
echo ""
echo "  2. Generate plots:"
echo "     uv run python -m eval.visualize $OUTPUT_DIR/eval_*.json --output-dir $OUTPUT_DIR/plots"
echo ""
echo "  3. View plots:"
echo "     open $OUTPUT_DIR/plots/"
echo "=========================================================="


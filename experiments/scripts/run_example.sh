#!/bin/bash
# Example script for running experiments
# This script demonstrates common experiment workflows

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=========================================="
echo "RAG-ScenarioFuzz Experiment Examples"
echo "=========================================="

# Example 1: Time estimation
echo ""
echo "Example 1: Estimate experiment time"
echo "------------------------------------"
python -m experiments.tools.estimate_time --all-methods --num-scenarios 100

# Example 2: Run single method (quantitative)
echo ""
echo "Example 2: Run RAG-ScenarioFuzz with 10 scenarios (test)"
echo "------------------------------------"
# Uncomment to run:
# python -m experiments.runners.run_rag_scenariofuzz \
#   --num-scenarios 10 \
#   --target behavior \
#   --town 3 \
#   --timeout 60 \
#   --rag-k 5 \
#   --debug

# Example 3: Run single method (timed)
echo ""
echo "Example 3: Run RAG-ScenarioFuzz for 1 hour"
echo "------------------------------------"
# Uncomment to run:
# python -m experiments.runners.run_rag_scenariofuzz \
#   --hours 1 \
#   --target behavior \
#   --town 3 \
#   --timeout 60 \
#   --rag-k 5

# Example 4: Run batch (sequential)
echo ""
echo "Example 4: Run all methods sequentially (100 scenarios each)"
echo "------------------------------------"
# Uncomment to run:
# python -m experiments.runners.run_tmfuzzer \
#   --num-scenarios 100 \
#   --target autoware \
#   --timeout 300
# python -m experiments.runners.run_scenariofuzz_llm \
#   --num-scenarios 100 \
#   --target behavior \
#   --town 3 \
#   --timeout 60
# python -m experiments.runners.run_rag_scenariofuzz \
#   --num-scenarios 100 \
#   --target behavior \
#   --town 3 \
#   --timeout 60 \
#   --rag-k 5

# Example 5: Run batch (parallel)
echo ""
echo "Example 5: Run RAG-ScenarioFuzz and ScenarioFuzz-LLM in parallel"
echo "------------------------------------"
# Uncomment to run:
# GNU parallel or tmux recommended for actual parallelization; example:
# parallel --jobs 2 ::: \
#   "python -m experiments.runners.run_rag_scenariofuzz --num-scenarios 50 --target behavior --town 3 --timeout 60 --rag-k 5" \
#   "python -m experiments.runners.run_scenariofuzz_llm --num-scenarios 50 --target behavior --town 3 --timeout 60"

echo ""
echo "=========================================="
echo "Examples completed!"
echo "=========================================="
echo ""
echo "To run actual experiments, uncomment the desired commands above."
echo "Make sure CARLA is running before starting experiments."


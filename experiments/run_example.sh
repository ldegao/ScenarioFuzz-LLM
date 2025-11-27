#!/bin/bash
# Example script for running experiments
# This script demonstrates common experiment workflows

set -e

echo "=========================================="
echo "RAG-ScenarioFuzz Experiment Examples"
echo "=========================================="

# Example 1: Time estimation
echo ""
echo "Example 1: Estimate experiment time"
echo "------------------------------------"
python3 experiments/estimate_time.py --all-methods --num-scenarios 100

# Example 2: Run single method (quantitative)
echo ""
echo "Example 2: Run RAG-ScenarioFuzz with 10 scenarios (test)"
echo "------------------------------------"
# Uncomment to run:
# python3 experiments/run_quantitative.py RAG-ScenarioFuzz --num-scenarios 10 --debug

# Example 3: Run single method (timed)
echo ""
echo "Example 3: Run RAG-ScenarioFuzz for 1 hour"
echo "------------------------------------"
# Uncomment to run:
# python3 experiments/run_timed.py RAG-ScenarioFuzz --hours 1

# Example 4: Run batch (sequential)
echo ""
echo "Example 4: Run all methods sequentially (100 scenarios each)"
echo "------------------------------------"
# Uncomment to run:
# python3 experiments/run_batch.py --all-methods --num-scenarios 100

# Example 5: Run batch (parallel)
echo ""
echo "Example 5: Run RAG-ScenarioFuzz and ScenarioFuzz-LLM in parallel"
echo "------------------------------------"
# Uncomment to run:
# python3 experiments/run_batch.py --methods RAG-ScenarioFuzz ScenarioFuzz-LLM --num-scenarios 50 --parallel --max-workers 2

echo ""
echo "=========================================="
echo "Examples completed!"
echo "=========================================="
echo ""
echo "To run actual experiments, uncomment the desired commands above."
echo "Make sure CARLA is running before starting experiments."


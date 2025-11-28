#!/bin/bash
# Quick Start Script for Experiments
# Provides common experiment workflows

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=========================================="
echo "RAG-ScenarioFuzz Quick Start"
echo "=========================================="
echo ""

# Check if CARLA is running
if ! docker ps | grep -q "carla-$(whoami)"; then
    echo "⚠️  Warning: CARLA container not detected"
    echo "   Please start CARLA first: ./script/run_carla.sh"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Menu
echo "Select experiment type:"
echo "1) Time estimation"
echo "2) Small test (10 scenarios per method)"
echo "3) Medium experiment (50 scenarios per method)"
echo "4) Full experiment (100 scenarios per method)"
echo "5) Timed experiment (2 hours)"
echo "6) Custom"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "Estimating time for 100 scenarios per method..."
        python -m experiments.tools.estimate_time --all-methods --num-scenarios 100
        ;;
    2)
        echo ""
        echo "Running small test (10 scenarios per method)..."
        python -m experiments.runners.run_tmfuzzer \
          --num-scenarios 10 \
          --target autoware \
          --output-root ./experiment_results \
          --timeout 300

        python -m experiments.runners.run_scenariofuzz_llm \
          --num-scenarios 10 \
          --output-root ./experiment_results \
          --target behavior \
          --town 3 \
          --timeout 60 \
          --debug

        python -m experiments.runners.run_rag_scenariofuzz \
          --num-scenarios 10 \
          --output-root ./experiment_results \
          --target behavior \
          --town 3 \
          --timeout 60 \
          --rag-k 5 \
          --debug
        ;;
    3)
        echo ""
        echo "Running medium experiment (50 scenarios per method)..."
        python -m experiments.runners.run_tmfuzzer \
          --num-scenarios 50 \
          --target autoware \
          --output-root ./experiment_results \
          --timeout 300

        python -m experiments.runners.run_scenariofuzz_llm \
          --num-scenarios 50 \
          --output-root ./experiment_results \
          --target behavior \
          --town 3 \
          --timeout 60

        python -m experiments.runners.run_rag_scenariofuzz \
          --num-scenarios 50 \
          --output-root ./experiment_results \
          --target behavior \
          --town 3 \
          --timeout 60 \
          --rag-k 5
        ;;
    4)
        echo ""
        echo "Running full experiment (100 scenarios per method)..."
        python -m experiments.runners.run_tmfuzzer \
          --num-scenarios 100 \
          --target autoware \
          --output-root ./experiment_results \
          --timeout 300

        python -m experiments.runners.run_scenariofuzz_llm \
          --num-scenarios 100 \
          --output-root ./experiment_results \
          --target behavior \
          --town 3 \
          --timeout 60

        python -m experiments.runners.run_rag_scenariofuzz \
          --num-scenarios 100 \
          --output-root ./experiment_results \
          --target behavior \
          --town 3 \
          --timeout 60 \
          --rag-k 5
        ;;
    5)
        echo ""
        echo "Running timed experiment (2 hours)..."
        read -p "Select method [RAG-ScenarioFuzz/ScenarioFuzz-LLM/all]: " method
        if [[ "$method" == "all" ]]; then
            python -m experiments.runners.run_tmfuzzer \
              --hours 2 \
              --target autoware \
              --output-root ./experiment_results

            python -m experiments.runners.run_scenariofuzz_llm \
              --hours 2 \
              --target behavior \
              --output-root ./experiment_results

            python -m experiments.runners.run_rag_scenariofuzz \
              --hours 2 \
              --target behavior \
              --output-root ./experiment_results \
              --rag-k 5
        else
            case "$method" in
                "ScenarioFuzz-LLM")
                    python -m experiments.runners.run_scenariofuzz_llm \
                      --hours 2 \
                      --target behavior \
                      --output-root ./experiment_results
                    ;;
                "RAG-ScenarioFuzz")
                    python -m experiments.runners.run_rag_scenariofuzz \
                      --hours 2 \
                      --target behavior \
                      --output-root ./experiment_results \
                      --rag-k 5
                    ;;
                "TM-Fuzzer")
                    python -m experiments.runners.run_tmfuzzer \
                      --hours 2 \
                      --target autoware \
                      --output-root ./experiment_results
                    ;;
                *)
                    echo "Unknown method: $method"
                    exit 1
                    ;;
            esac
        fi
        ;;
    6)
        echo ""
        echo "Custom experiment options:"
        echo "  - ScenarioFuzz-LLM: python -m experiments.runners.run_scenariofuzz_llm --num-scenarios N"
        echo "  - RAG-ScenarioFuzz: python -m experiments.runners.run_rag_scenariofuzz --num-scenarios N --rag-k 5"
        echo "  - TM-Fuzzer:       python -m experiments.runners.run_tmfuzzer --num-scenarios N --target autoware"
        echo "  - Timed:           use the same runners with --hours N"
        echo "  - Estimate:        python -m experiments.tools.estimate_time --all-methods --num-scenarios N"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="


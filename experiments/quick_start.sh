#!/bin/bash
# Quick Start Script for Experiments
# Provides common experiment workflows

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

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
        python3 experiments/estimate_time.py --all-methods --num-scenarios 100
        ;;
    2)
        echo ""
        echo "Running small test (10 scenarios per method)..."
        python3 experiments/run_batch.py --all-methods --num-scenarios 10 --debug
        ;;
    3)
        echo ""
        echo "Running medium experiment (50 scenarios per method)..."
        read -p "Run in parallel? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python3 experiments/run_batch.py --all-methods --num-scenarios 50 --parallel --max-workers 2
        else
            python3 experiments/run_batch.py --all-methods --num-scenarios 50
        fi
        ;;
    4)
        echo ""
        echo "Running full experiment (100 scenarios per method)..."
        read -p "Run in parallel? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python3 experiments/run_batch.py --all-methods --num-scenarios 100 --parallel --max-workers 2
        else
            python3 experiments/run_batch.py --all-methods --num-scenarios 100
        fi
        ;;
    5)
        echo ""
        echo "Running timed experiment (2 hours)..."
        read -p "Select method [RAG-ScenarioFuzz/ScenarioFuzz-LLM/all]: " method
        if [[ "$method" == "all" ]]; then
            python3 experiments/run_batch.py --all-methods --hours 2
        else
            python3 experiments/run_timed.py "$method" --hours 2
        fi
        ;;
    6)
        echo ""
        echo "Custom experiment options:"
        echo "  - Quantitative: python3 experiments/run_quantitative.py METHOD --num-scenarios N"
        echo "  - Timed: python3 experiments/run_timed.py METHOD --hours N"
        echo "  - Batch: python3 experiments/run_batch.py --methods ... --num-scenarios N"
        echo "  - Estimate: python3 experiments/estimate_time.py METHOD --num-scenarios N"
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


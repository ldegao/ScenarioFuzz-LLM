#!/bin/bash
# Batch script for running Experiment 3: Local Diversity Comparison (GPT-guided vs Random mutations)
# This script runs two experiments with the same seed set and compares their local diversity metrics (LMS/SED/OSCR)

set -u
DB_PATH="./data/scenario_db.json"

clean_scenario_db() {
    if [ -f "$DB_PATH" ]; then
        rm -f "$DB_PATH"
        echo "[INFO] Cleared RAG scenario db: $DB_PATH"
    else
        echo "[INFO] RAG scenario db already empty: $DB_PATH"
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Default values
NUM_SCENARIOS=1000
OUTPUT_ROOT="./experiment_results"
TARGET="behavior"
TOWN=3
TIMEOUT=60
# Use a fixed seed to ensure both experiments use the same initial seed set
DETERM_SEED=42.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-scenarios)
            NUM_SCENARIOS="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --town)
            TOWN="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --determ-seed)
            DETERM_SEED="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num-scenarios N          Number of scenarios to generate (default: 1000)"
            echo "  --output-root PATH         Output root directory (default: ./experiment_results)"
            echo "  --target {behavior|autoware}  Target ADS system (default: behavior)"
            echo "  --town N                   CARLA town number (default: 3)"
            echo "  --timeout N                Scenario timeout in seconds (default: 60)"
            echo "  --determ-seed F            Fixed random seed for reproducibility (default: 42.0)"
            echo "  --help                     Show this help message"
            echo ""
            echo "This script runs Experiment 3: Local Diversity Comparison"
            echo "  - Step 1: Run GPT-guided mutation experiment (ScenarioFuzz-LLM with default settings)"
            echo "  - Step 2: Run random mutation experiment (ScenarioFuzz-LLM with --disable-guided-mutation)"
            echo "  - Step 3: Compare local diversity metrics (LMS/SED/OSCR) between the two experiments"
            echo ""
            echo "Both experiments use the same random seed (--determ-seed) to ensure they start with"
            echo "the same initial seed set, enabling fair comparison of local mutation behavior."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Experiment 3: Local Diversity Comparison"
echo "GPT-guided vs Random Mutations"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Number of scenarios: $NUM_SCENARIOS"
echo "  Output root: $OUTPUT_ROOT"
echo "  Target: $TARGET"
echo "  Town: $TOWN"
echo "  Timeout: $TIMEOUT seconds"
echo "  Fixed random seed: $DETERM_SEED"
echo ""
echo "This experiment compares local diversity metrics (LMS/SED/OSCR) between:"
echo "  1. GPT-guided mutations (ScenarioFuzz-LLM with default settings)"
echo "  2. Random mutations (ScenarioFuzz-LLM with --disable-guided-mutation)"
echo ""
echo "Both experiments use the same random seed to ensure fair comparison."
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."

# Create output directory
mkdir -p "$OUTPUT_ROOT"

run_with_retry() {
    local desc="$1"; shift
    local cmd="$@"
    local retries=3
    local attempt=0
    echo ">>> Starting: ${desc}"
    until eval "$cmd"; do
        exit_code=$?
        attempt=$((attempt + 1))
        if [ $attempt -ge $retries ]; then
            echo "!!! Failed: ${desc} (retried ${attempt} times, exit ${exit_code})"
            return $exit_code
        fi
        echo "--- Retry ${attempt}/${retries}: ${desc} (exit ${exit_code}), wait 5s..."
        sleep 5
    done
    echo "✓ Completed: ${desc}"
}

# Step 1: Run GPT-guided mutation experiment
echo ""
echo "=========================================="
echo "Step 1: Running GPT-guided mutation experiment"
echo "=========================================="
echo ""

GPT_EXP_ID=""
GPT_EXP_DIR=""

run_with_retry "GPT-guided mutation experiment" \
    clean_scenario_db && \
    python -m experiments.cli run \
        --method scenariofuzz-llm \
        --num-scenarios "$NUM_SCENARIOS" \
        --output-root "$OUTPUT_ROOT" \
        --target "$TARGET" \
        --town "$TOWN" \
        --timeout "$TIMEOUT" \
        --determ-seed "$DETERM_SEED"

# Find the GPT experiment directory (most recent ScenarioFuzz-LLM experiment)
if [ -d "$OUTPUT_ROOT/ScenarioFuzz-LLM" ]; then
    GPT_EXP_DIR=$(find "$OUTPUT_ROOT/ScenarioFuzz-LLM" -mindepth 1 -maxdepth 1 -type d | sort -r | head -n 1)
    GPT_EXP_ID=$(basename "$GPT_EXP_DIR")
    echo "[INFO] GPT experiment directory: $GPT_EXP_DIR"
    echo "[INFO] GPT experiment ID: $GPT_EXP_ID"
else
    echo "[ERROR] GPT experiment directory not found in $OUTPUT_ROOT/ScenarioFuzz-LLM"
    exit 1
fi

# Step 2: Run random mutation experiment
echo ""
echo "=========================================="
echo "Step 2: Running random mutation experiment"
echo "=========================================="
echo ""

RAND_EXP_ID=""
RAND_EXP_DIR=""

run_with_retry "Random mutation experiment" \
    clean_scenario_db && \
    python -m experiments.cli run \
        --method scenariofuzz-llm \
        --num-scenarios "$NUM_SCENARIOS" \
        --output-root "$OUTPUT_ROOT" \
        --target "$TARGET" \
        --town "$TOWN" \
        --timeout "$TIMEOUT" \
        --determ-seed "$DETERM_SEED" \
        --disable-guided-mutation

# Find the random experiment directory (most recent ScenarioFuzz-LLM experiment)
if [ -d "$OUTPUT_ROOT/ScenarioFuzz-LLM" ]; then
    RAND_EXP_DIR=$(find "$OUTPUT_ROOT/ScenarioFuzz-LLM" -mindepth 1 -maxdepth 1 -type d | sort -r | head -n 1)
    RAND_EXP_ID=$(basename "$RAND_EXP_DIR")
    echo "[INFO] Random experiment directory: $RAND_EXP_DIR"
    echo "[INFO] Random experiment ID: $RAND_EXP_ID"
else
    echo "[ERROR] Random experiment directory not found in $OUTPUT_ROOT/ScenarioFuzz-LLM"
    exit 1
fi

# Verify both experiments completed successfully
if [ ! -d "$GPT_EXP_DIR/queue" ] || [ -z "$(find "$GPT_EXP_DIR/queue" -name "*.json" -o -name "*.pkl" 2>/dev/null | head -n 1)" ]; then
    echo "[ERROR] GPT experiment has no scenario files in queue/"
    exit 1
fi

if [ ! -d "$RAND_EXP_DIR/queue" ] || [ -z "$(find "$RAND_EXP_DIR/queue" -name "*.json" -o -name "*.pkl" 2>/dev/null | head -n 1)" ]; then
    echo "[ERROR] Random experiment has no scenario files in queue/"
    exit 1
fi

# Step 3: Compare local diversity metrics
echo ""
echo "=========================================="
echo "Step 3: Comparing local diversity metrics"
echo "=========================================="
echo ""

COMPARISON_OUTPUT="$OUTPUT_ROOT/local_diversity_comparison.json"

run_with_retry "Local diversity comparison" \
    python -m experiments.runners.run_local_diversity_comparison \
        --gpt-dir "$GPT_EXP_DIR" \
        --rand-dir "$RAND_EXP_DIR" \
        --output "$COMPARISON_OUTPUT"

echo ""
echo "=========================================="
echo "Experiment 3 completed!"
echo "=========================================="
echo ""
echo "Results:"
echo "  GPT-guided experiment: $GPT_EXP_DIR"
echo "  Random mutation experiment: $RAND_EXP_DIR"
echo "  Comparison results: $COMPARISON_OUTPUT"
echo ""
echo "The comparison JSON contains:"
echo "  - gpt: LMS/SED/OSCR metrics for GPT-guided mutations"
echo "  - random: LMS/SED/OSCR metrics for random mutations"
echo "  - delta_lms/sed/oscr: Differences between the two methods"
echo ""
echo "To view the results:"
echo "  cat $COMPARISON_OUTPUT | python -m json.tool"
echo ""


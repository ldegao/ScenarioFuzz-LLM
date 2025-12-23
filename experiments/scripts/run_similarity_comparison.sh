#!/bin/bash
# Batch script for running similarity scoring method comparison experiments
# Runs all four similarity methods (answer2, embedding, feature, hybrid) sequentially

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
RAG_K=5
HYBRID_EMBEDDING_WEIGHT=0.6

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
        --rag-k)
            RAG_K="$2"
            shift 2
            ;;
        --hybrid-embedding-weight)
            HYBRID_EMBEDDING_WEIGHT="$2"
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
            echo "  --rag-k N                  Top-k for RAG retrieval (default: 5)"
            echo "  --hybrid-embedding-weight F  Weight for embedding in hybrid method (default: 0.6)"
            echo "  --help                     Show this help message"
            echo ""
            echo "This script runs all four similarity scoring methods sequentially:"
            echo "  - answer2: LLM-based similarity scoring"
            echo "  - embedding: Embedding-based semantic similarity"
            echo "  - feature: Feature-based similarity (position, speed, etc.)"
            echo "  - hybrid: Hybrid similarity (embedding + feature)"
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
echo "Similarity Scoring Method Comparison"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Number of scenarios: $NUM_SCENARIOS"
echo "  Output root: $OUTPUT_ROOT"
echo "  Target: $TARGET"
echo "  Town: $TOWN"
echo "  Timeout: $TIMEOUT seconds"
echo "  RAG k: $RAG_K"
echo "  Hybrid embedding weight: $HYBRID_EMBEDDING_WEIGHT"
echo ""
echo "Methods to run:"
echo "  1. answer2 (LLM-based)"
echo "  2. embedding (Embedding-based)"
echo "  3. feature (Feature-based)"
echo "  4. hybrid (Hybrid)"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."

# Create output directory
mkdir -p "$OUTPUT_ROOT"

# Array of similarity methods
METHODS=("answer2" "embedding" "feature" "hybrid")

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

# Run each method
for METHOD in "${METHODS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running: $METHOD"
    echo "=========================================="
    echo ""
    
    CMD="clean_scenario_db && python -m experiments.runners.run_similarity_comparison \
        --num-scenarios $NUM_SCENARIOS \
        --similarity-method $METHOD \
        --output-root $OUTPUT_ROOT \
        --target $TARGET \
        --town $TOWN \
        --timeout $TIMEOUT \
        --rag-k $RAG_K"
    
    if [ "$METHOD" == "hybrid" ]; then
        CMD="$CMD --hybrid-embedding-weight $HYBRID_EMBEDDING_WEIGHT"
    fi
    
    echo "Command: $CMD"
    echo ""
    
    run_with_retry "Similarity method: $METHOD" "$CMD"
    
    echo ""
    echo "Waiting 5 seconds before next method..."
    sleep 5
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results are saved in: $OUTPUT_ROOT/SimilarityComparison/"
echo ""
echo "To analyze results, run:"
echo "  python -m experiments.analysis.compare_similarity_methods \\"
echo "    --results-dir $OUTPUT_ROOT/SimilarityComparison \\"
echo "    --output-dir ./reports/similarity_comparison"
echo ""


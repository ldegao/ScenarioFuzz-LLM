#!/usr/bin/env python3
"""
Fix script issues: generate missing metrics_summary.json files and provide fixes.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Import aggregation functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.aggregation.metrics_aggregator import (
    load_records_from_jsonl,
    aggregate_run_metrics,
    save_run_summary
)


def fix_missing_summaries(results_dir: Path, dry_run: bool = False) -> Dict:
    """Fix missing metrics_summary.json files by regenerating them from metrics_records.jsonl."""
    results = {
        "fixed": [],
        "failed": [],
        "skipped": []
    }
    
    similarity_dir = results_dir / "SimilarityComparison"
    if not similarity_dir.exists():
        return {"error": f"Directory {similarity_dir} does not exist"}
    
    for exp_dir in similarity_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        if not exp_name.startswith("SimilarityComparison_"):
            continue
        
        metrics_dir = exp_dir / "metrics"
        records_path = metrics_dir / "metrics_records.jsonl"
        summary_path = metrics_dir / "metrics_summary.json"
        
        # Check if records exist but summary is missing
        if records_path.exists() and not summary_path.exists():
            try:
                if dry_run:
                    print(f"[DRY RUN] Would fix: {exp_name}")
                    results["fixed"].append(exp_name)
                else:
                    # Load records
                    records = load_records_from_jsonl(str(records_path))
                    if not records:
                        results["skipped"].append(f"{exp_name}: No records")
                        continue
                    
                    # Aggregate metrics
                    summary = aggregate_run_metrics(records)
                    num_scenarios = summary.get("num_records", len(records))
                    
                    # Extract method name and experiment ID
                    parts = exp_name.split("_")
                    method_name = "SimilarityComparison"
                    experiment_id = exp_name
                    
                    # Save summary
                    save_run_summary(
                        summary,
                        str(summary_path),
                        method_name=method_name,
                        experiment_id=experiment_id,
                        num_scenarios=num_scenarios
                    )
                    print(f"[FIXED] Generated metrics_summary.json for {exp_name}")
                    results["fixed"].append(exp_name)
            except Exception as e:
                print(f"[ERROR] Failed to fix {exp_name}: {e}")
                results["failed"].append(f"{exp_name}: {str(e)}")
        elif records_path.exists() and summary_path.exists():
            results["skipped"].append(f"{exp_name}: Already has summary")
        elif not records_path.exists():
            results["skipped"].append(f"{exp_name}: No records file")
    
    return results


def generate_fix_patches() -> Dict:
    """Generate code patches for fixing the identified issues."""
    patches = {
        "bash_script": """
# Fix: Improve error handling in run_similarity_comparison.sh
# Around line 130-138, replace with:

    # Run the command with better error handling
    set +e  # Temporarily disable exit on error
    eval "$CMD"
    EXIT_CODE=$?
    set -e  # Re-enable exit on error
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Completed: $METHOD"
    else
        echo ""
        echo "✗ Failed: $METHOD (exit code: $EXIT_CODE)"
        echo "Error details may be above. Continuing with next method..."
    fi
""",
        "python_runner": """
# Fix: Add validation before directory creation in experiment_manager.py
# Around line 130, add before method_dir.mkdir():

            # Validate experiment can start before creating directory
            try:
                self._validate_experiment_start(method_name, **kwargs)
            except Exception as validation_error:
                print(f"[ERROR] Experiment validation failed: {validation_error}")
                raise
            
            # Now safe to create directory
            method_dir.mkdir(parents=True, exist_ok=True)
""",
        "statistics_script": """
# Fix: Improve path resolution in compare_similarity_methods.py
# Around line 84, replace with:

    # Handle both cases: results_dir may or may not contain SimilarityComparison
    if results_dir.name == "SimilarityComparison":
        similarity_dir = results_dir
    else:
        similarity_dir = results_dir / "SimilarityComparison"
    
    if not similarity_dir.exists():
        print(f"[ERROR] Directory {similarity_dir} does not exist")
        return method_dirs
"""
    }
    return patches


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix script issues and generate missing summaries")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./experiment_results",
        help="Results directory containing SimilarityComparison experiments"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without actually fixing"
    )
    parser.add_argument(
        "--show-patches",
        action="store_true",
        help="Show code patches for fixing issues"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    print("="*80)
    print("脚本问题修复工具")
    print("="*80)
    
    # Fix missing summaries
    print("\n1. 修复缺少的 metrics_summary.json 文件")
    print("-"*80)
    fix_results = fix_missing_summaries(results_dir, dry_run=args.dry_run)
    
    if "error" in fix_results:
        print(f"错误: {fix_results['error']}")
    else:
        print(f"\n修复结果:")
        print(f"  已修复: {len(fix_results['fixed'])}")
        print(f"  失败: {len(fix_results['failed'])}")
        print(f"  跳过: {len(fix_results['skipped'])}")
        
        if fix_results['fixed']:
            print(f"\n已修复的实验:")
            for exp in fix_results['fixed']:
                print(f"  - {exp}")
        
        if fix_results['failed']:
            print(f"\n失败的实验:")
            for exp in fix_results['failed']:
                print(f"  - {exp}")
    
    # Show patches
    if args.show_patches:
        print("\n2. 代码修复补丁")
        print("-"*80)
        patches = generate_fix_patches()
        for file, patch in patches.items():
            print(f"\n{file.upper()}:")
            print(patch)
    
    print("\n" + "="*80)
    print("完成")
    print("="*80)


if __name__ == "__main__":
    main()


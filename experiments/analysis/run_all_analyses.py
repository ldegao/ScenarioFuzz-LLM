#!/usr/bin/env python3
"""
Run all analysis scripts for similarity method comparison.

This script runs all analysis modules in sequence:
1. Basic comparison
2. Statistical analysis
3. Correlation analysis
4. Time series analysis
5. Efficiency analysis
6. Scenario analysis
7. Comprehensive report
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_analysis(script_name: str, args: list) -> bool:
    """Run an analysis script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", f"experiments.analysis.{script_name}"] + args,
            check=True
        )
        print(f"\n✓ Completed: {script_name}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {script_name}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run all analysis scripts for similarity method comparison."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./experiment_results",
        help="Directory containing SimilarityComparison experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reports/similarity_comparison",
        help="Base directory to save all analysis results",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"[ERROR] Results directory {results_dir} does not exist")
        return
    
    print("="*60)
    print("Comprehensive Similarity Method Analysis")
    print("="*60)
    print(f"\nResults directory: {results_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Run all analyses
    analyses = [
        ("compare_similarity_methods", [
            "--results-dir", str(results_dir),
            "--output-dir", str(output_dir)
        ]),
        ("statistical_analysis", [
            "--results-dir", str(results_dir),
            "--output-dir", str(output_dir / "statistical_analysis")
        ]),
        ("correlation_analysis", [
            "--results-dir", str(results_dir),
            "--output-dir", str(output_dir / "correlation_analysis")
        ]),
        ("timeseries_analysis", [
            "--results-dir", str(results_dir),
            "--output-dir", str(output_dir / "timeseries_analysis")
        ]),
        ("efficiency_analysis", [
            "--results-dir", str(results_dir),
            "--output-dir", str(output_dir / "efficiency_analysis"),
            "--metrics-file", str(output_dir / "comparison_report.json")
        ]),
        ("scenario_analysis", [
            "--results-dir", str(results_dir),
            "--output-dir", str(output_dir / "scenario_analysis")
        ]),
        ("comprehensive_report", [
            "--base-dir", str(output_dir),
            "--output-dir", str(output_dir / "comprehensive_analysis")
        ])
    ]
    
    success_count = 0
    for script_name, script_args in analyses:
        if run_analysis(script_name, script_args):
            success_count += 1
    
    print("="*60)
    print(f"Analysis Complete: {success_count}/{len(analyses)} analyses succeeded")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nComprehensive report: {output_dir / 'comprehensive_analysis' / 'comprehensive_analysis_report.md'}\n")


if __name__ == "__main__":
    main()


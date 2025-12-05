#!/usr/bin/env python3
"""
Generate Markdown/JSON summary reports from aggregated metrics.

This script consumes an all_methods_results.json (produced by
experiments.aggregation.collect_results), computes per-method
averages, and uses visualization.report_generator.ReportGenerator
to emit human-readable reports.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

from visualization.report_generator import ReportGenerator
from .generate_figures import compute_method_averages


def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown/JSON reports from aggregated metrics."
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="./experiment_results/all_methods_results.json",
        help="Path to all_methods_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reports",
        help="Directory to save generated reports",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Thesis_Experiment",
        help="Logical experiment name used in report titles",
    )

    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"[GenerateReports] Results file {results_path} does not exist.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        all_results: Dict = json.load(f)

    method_avgs = compute_method_averages(all_results)
    if not method_avgs:
        print("[GenerateReports] No method averages to report.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize report generator
    rg = ReportGenerator(output_dir=str(output_dir))

    # Markdown report
    md_path = rg.generate_markdown_report(
        experiment_name=args.experiment_name,
        metrics_results=method_avgs,
        comparison_data=all_results,
    )

    # JSON summary
    json_summary_path = rg.generate_summary_json(
        metrics_results=method_avgs,
    )

    print(f"[GenerateReports] Generated Markdown report: {md_path}")
    print(f"[GenerateReports] Generated JSON summary: {json_summary_path}")


if __name__ == "__main__":
    main()



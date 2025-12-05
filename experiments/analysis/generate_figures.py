#!/usr/bin/env python3
"""
Generate figures for multi-method metrics comparison.

This script consumes an all_methods_results.json (produced by
experiments.aggregation.collect_results) and generates:
  - Parameter coverage bar chart
  - Multi-dimensional radar chart (PC/PEC/TCD/BCM)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from visualization.plot_metrics import (
    plot_parameter_coverage,
    plot_comparison_radar,
)


def compute_method_averages(all_results: Dict) -> Dict[str, Dict[str, float]]:
    """
    Compute per-method average metrics from all_methods_results.

    Input structure:
        {
          "ScenarioFuzz-LLM": [ { "pc":..,"pec":..,"tcd":..,"bcm":.. }, ... ],
          "RAG-ScenarioFuzz": [ ... ],
        }

    Returns:
        {
          "ScenarioFuzz-LLM": {"pc":..,"pec":..,"tcd":..,"bcm":..},
          ...
        }
    """
    method_avgs: Dict[str, Dict[str, float]] = {}
    keys = ["pc", "pec", "tcd", "bcm"]

    for method, runs in all_results.items():
        if not runs:
            continue
        n = float(len(runs))
        avg = {}
        for k in keys:
            total = 0.0
            for r in runs:
                try:
                    v = float(r.get(k, 0.0))
                except (TypeError, ValueError):
                    v = 0.0
                total += v
            avg[k] = total / n
        method_avgs[method] = avg

    return method_avgs


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures (PC bar chart, radar chart) from aggregated metrics."
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
        default="./reports/figs",
        help="Directory to save generated figures",
    )

    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"[GenerateFigures] Results file {results_path} does not exist.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    method_avgs = compute_method_averages(all_results)
    if not method_avgs:
        print("[GenerateFigures] No method averages to plot.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parameter coverage bar chart
    pc_scores = {m: v.get("pc", 0.0) for m, v in method_avgs.items()}
    pc_fig = out_dir / "pc_coverage.png"
    plot_parameter_coverage(pc_scores, output_path=str(pc_fig))
    print(f"[GenerateFigures] Saved PC coverage figure to {pc_fig}")

    # Multi-dimensional radar chart
    radar_fig = out_dir / "metrics_radar.png"
    plot_comparison_radar(method_avgs, output_path=str(radar_fig))
    print(f"[GenerateFigures] Saved radar figure to {radar_fig}")


if __name__ == "__main__":
    main()



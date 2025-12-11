#!/usr/bin/env python3
"""
Time series analysis of diversity metrics.

This script analyzes how metrics (PC, PEC, TCD, BCM) change over time
as scenarios are generated, identifying convergence and stability patterns.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not available, skipping figure generation")

from .compare_similarity_methods import (
    find_experiment_dirs,
    METHOD_NAMES,
    METRIC_NAMES
)


def load_metrics_records(exp_dir: Path) -> List[Dict]:
    """
    Load metrics_records.jsonl from an experiment directory.
    
    Args:
        exp_dir: Path to experiment directory
        
    Returns:
        List of metric records
    """
    records_path = exp_dir / "metrics" / "metrics_records.jsonl"
    if not records_path.exists():
        return []
    
    records = []
    try:
        with open(records_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        print(f"[WARNING] Failed to load {records_path}: {e}")
    
    return records


def extract_timeseries_data(method_dirs: Dict[str, List[Path]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Extract time series data for each method.
    
    Args:
        method_dirs: Dictionary mapping method names to experiment directories
        
    Returns:
        Dictionary mapping method names to time series data
    """
    all_timeseries = {}
    
    for method, exp_dirs in method_dirs.items():
        method_data = {
            "pc": [],
            "pec": [],
            "tcd": [],
            "bcm": [],
            "scenario_ids": []
        }
        
        # Collect data from all experiments of this method
        for exp_dir in exp_dirs:
            records = load_metrics_records(exp_dir)
            
            for record in records:
                scenario_id = record.get("scenario_id", len(method_data["scenario_ids"]))
                method_data["scenario_ids"].append(scenario_id)
                
                for metric in ["pc", "pec", "tcd", "bcm"]:
                    value = record.get(metric, 0.0)
                    if isinstance(value, (int, float)):
                        method_data[metric].append(float(value))
        
        # Sort by scenario_id if available
        if method_data["scenario_ids"]:
            sorted_indices = np.argsort(method_data["scenario_ids"])
            for metric in ["pc", "pec", "tcd", "bcm"]:
                method_data[metric] = [method_data[metric][i] for i in sorted_indices]
            method_data["scenario_ids"] = [method_data["scenario_ids"][i] for i in sorted_indices]
        
        all_timeseries[method] = method_data
    
    return all_timeseries


def calculate_cumulative_metrics(timeseries_data: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """
    Calculate cumulative metrics (running averages).
    
    Args:
        timeseries_data: Dictionary mapping metric names to value lists
        
    Returns:
        Dictionary with cumulative averages
    """
    cumulative = {}
    
    for metric, values in timeseries_data.items():
        if metric == "scenario_ids":
            cumulative[metric] = values
            continue
        
        if not values:
            cumulative[metric] = []
            continue
        
        cumulative_avg = []
        running_sum = 0.0
        
        for i, value in enumerate(values):
            running_sum += value
            cumulative_avg.append(running_sum / (i + 1))
        
        cumulative[f"{metric}_cumulative"] = cumulative_avg
    
    return cumulative


def detect_convergence(values: List[float], window_size: int = 10, threshold: float = 0.01) -> int:
    """
    Detect convergence point where values stabilize.
    
    Args:
        values: List of metric values
        window_size: Size of window for stability check
        threshold: Maximum std deviation for convergence
        
    Returns:
        Index of convergence point, or -1 if not converged
    """
    if len(values) < window_size * 2:
        return -1
    
    for i in range(window_size, len(values) - window_size):
        window = values[i:i+window_size]
        std_dev = np.std(window)
        
        if std_dev < threshold:
            return i
    
    return -1


def generate_timeseries_report(timeseries_data: Dict[str, Dict[str, List[float]]], output_dir: Path):
    """
    Generate time series analysis report.
    
    Args:
        timeseries_data: Dictionary mapping method names to time series data
        output_dir: Directory to save reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for method, data in timeseries_data.items():
        method_results = {}
        
        for metric in ["pc", "pec", "tcd", "bcm"]:
            values = data.get(metric, [])
            if not values:
                continue
            
            # Calculate statistics
            method_results[metric] = {
                "initial": float(values[0]) if values else 0.0,
                "final": float(values[-1]) if values else 0.0,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "trend": "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable",
                "convergence_point": detect_convergence(values),
                "n_scenarios": len(values)
            }
        
        results[method] = method_results
    
    # Save JSON report
    json_path = output_dir / "timeseries_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[TimeSeriesAnalysis] Saved JSON report: {json_path}")
    
    # Generate Markdown report
    md_path = output_dir / "timeseries_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Time Series Analysis Report\n\n")
        f.write("This report analyzes how diversity metrics change over time as scenarios are generated.\n\n")
        
        for method in ["answer2", "embedding", "feature", "hybrid"]:
            if method not in results:
                continue
            
            display_name = METHOD_NAMES.get(method, method)
            f.write(f"## {display_name}\n\n")
            
            method_results = results[method]
            
            for metric in ["pc", "pec", "tcd", "bcm"]:
                if metric not in method_results:
                    continue
                
                metric_name = METRIC_NAMES[metric]
                metric_data = method_results[metric]
                
                f.write(f"### {metric_name} ({metric.upper()})\n\n")
                f.write(f"- **Initial value**: {metric_data['initial']:.6f}\n")
                f.write(f"- **Final value**: {metric_data['final']:.6f}\n")
                f.write(f"- **Mean**: {metric_data['mean']:.6f}\n")
                f.write(f"- **Std deviation**: {metric_data['std']:.6f}\n")
                f.write(f"- **Min**: {metric_data['min']:.6f}\n")
                f.write(f"- **Max**: {metric_data['max']:.6f}\n")
                f.write(f"- **Trend**: {metric_data['trend']}\n")
                
                conv_point = metric_data['convergence_point']
                if conv_point >= 0:
                    f.write(f"- **Convergence point**: Scenario {conv_point} (stabilized after {conv_point} scenarios)\n")
                else:
                    f.write(f"- **Convergence point**: Not detected (values still changing)\n")
                
                f.write(f"- **Total scenarios**: {metric_data['n_scenarios']}\n\n")
    
    print(f"[TimeSeriesAnalysis] Saved Markdown report: {md_path}")


def generate_timeseries_figures(timeseries_data: Dict[str, Dict[str, List[float]]], output_dir: Path):
    """
    Generate time series plots.
    
    Args:
        timeseries_data: Dictionary mapping method names to time series data
        output_dir: Directory to save figures
    """
    if not HAS_MATPLOTLIB:
        print("[WARNING] matplotlib not available, skipping figure generation")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "timeseries_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    methods = [m for m in ["answer2", "embedding", "feature", "hybrid"] if m in timeseries_data]
    if not methods:
        return
    
    # Plot each metric separately
    for metric in ["pc", "pec", "tcd", "bcm"]:
        metric_name = METRIC_NAMES[metric]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for method in methods:
            data = timeseries_data[method]
            values = data.get(metric, [])
            scenario_ids = data.get("scenario_ids", list(range(len(values))))
            
            if not values:
                continue
            
            # Use scenario_ids if available, otherwise use indices
            x_values = scenario_ids if scenario_ids else list(range(len(values)))
            
            ax.plot(x_values, values, marker='o', markersize=3, label=METHOD_NAMES.get(method, method), linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Scenario ID', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{metric_name} Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        fig_path = figures_dir / f"{metric}_timeseries.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[TimeSeriesAnalysis] Saved figure: {fig_path}")
    
    # Plot cumulative averages
    for metric in ["pc", "pec", "tcd", "bcm"]:
        metric_name = METRIC_NAMES[metric]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for method in methods:
            data = timeseries_data[method]
            values = data.get(metric, [])
            scenario_ids = data.get("scenario_ids", list(range(len(values))))
            
            if not values:
                continue
            
            # Calculate cumulative average
            cumulative_avg = []
            running_sum = 0.0
            for i, value in enumerate(values):
                running_sum += value
                cumulative_avg.append(running_sum / (i + 1))
            
            x_values = scenario_ids if scenario_ids else list(range(len(values)))
            
            ax.plot(x_values, cumulative_avg, marker='o', markersize=3, label=METHOD_NAMES.get(method, method), linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Scenario ID', fontsize=12)
        ax.set_ylabel('Cumulative Average', fontsize=12)
        ax.set_title(f'{metric_name} Cumulative Average Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        fig_path = figures_dir / f"{metric}_cumulative.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[TimeSeriesAnalysis] Saved figure: {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze time series patterns in diversity metrics."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing SimilarityComparison experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reports/similarity_comparison/timeseries_analysis",
        help="Directory to save time series analysis results",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Results directory {results_dir} does not exist")
        return
    
    # Find all experiment directories
    method_dirs = find_experiment_dirs(results_dir)
    
    # Extract time series data
    timeseries_data = extract_timeseries_data(method_dirs)
    
    if not timeseries_data:
        print("[ERROR] No time series data found")
        return
    
    # Generate reports
    output_dir = Path(args.output_dir)
    generate_timeseries_report(timeseries_data, output_dir)
    generate_timeseries_figures(timeseries_data, output_dir)
    
    print(f"\n[TimeSeriesAnalysis] Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


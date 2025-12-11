#!/usr/bin/env python3
"""
Compare similarity scoring methods based on four diversity metrics (PC, PEC, TCD, BCM).

This script reads metrics_summary.json files from all similarity method experiments,
compares the four metrics, and generates comparison reports and visualizations.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not available, skipping figure generation")


METHOD_NAMES = {
    "answer2": "LLM-based (Answer2)",
    "embedding": "Embedding-based",
    "feature": "Feature-based",
    "hybrid": "Hybrid"
}

METRIC_NAMES = {
    "pc": "Parameter Coverage (PC)",
    "pec": "Behavior Coverage (PEC)",
    "tcd": "Trajectory Diversity (TCD)",
    "bcm": "Behavior Matrix (BCM)"
}


def load_metrics_summary(experiment_dir: Path) -> Optional[Dict]:
    """
    Load metrics_summary.json from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary with metrics or None if not found
    """
    # Try metrics/metrics_summary.json first (standard location)
    summary_path = experiment_dir / "metrics" / "metrics_summary.json"
    if not summary_path.exists():
        # Fallback to metrics_summary.json in root
        summary_path = experiment_dir / "metrics_summary.json"
        if not summary_path.exists():
            return None
    
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to load {summary_path}: {e}")
        return None


def find_experiment_dirs(results_dir: Path) -> Dict[str, List[Path]]:
    """
    Find all experiment directories for each similarity method.
    
    Args:
        results_dir: Root directory containing SimilarityComparison experiments
        
    Returns:
        Dictionary mapping method names to lists of experiment directories
    """
    method_dirs = {
        "answer2": [],
        "embedding": [],
        "feature": [],
        "hybrid": []
    }
    
    similarity_dir = results_dir / "SimilarityComparison"
    if not similarity_dir.exists():
        print(f"[ERROR] Directory {similarity_dir} does not exist")
        return method_dirs
    
    # Find all experiment directories
    for exp_dir in similarity_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        # Experiment ID format: SimilarityComparison_{method}_{timestamp}
        if exp_name.startswith("SimilarityComparison_"):
            parts = exp_name.split("_")
            if len(parts) >= 3:
                method = parts[1]  # Extract method name
                if method in method_dirs:
                    method_dirs[method].append(exp_dir)
    
    return method_dirs


def aggregate_method_metrics(method_dirs: List[Path]) -> Dict:
    """
    Aggregate metrics from multiple runs of the same method.
    
    Args:
        method_dirs: List of experiment directories for a method
        
    Returns:
        Dictionary with aggregated metrics (mean, std, min, max, count)
    """
    all_metrics = {
        "pc": [],
        "pec": [],
        "tcd": [],
        "bcm": []
    }
    
    for exp_dir in method_dirs:
        summary = load_metrics_summary(exp_dir)
        if summary is None:
            continue
        
        for metric in all_metrics.keys():
            value = summary.get(metric, 0.0)
            if isinstance(value, (int, float)):
                all_metrics[metric].append(float(value))
    
    # Compute statistics
    result = {}
    for metric, values in all_metrics.items():
        if values:
            result[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
                "values": values
            }
        else:
            result[metric] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
                "values": []
            }
    
    return result


def generate_comparison_report(all_method_metrics: Dict[str, Dict], output_dir: Path):
    """
    Generate a comparison report in JSON and Markdown formats.
    
    Args:
        all_method_metrics: Dictionary mapping method names to aggregated metrics
        output_dir: Directory to save reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON report
    json_path = output_dir / "comparison_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_method_metrics, f, indent=2, ensure_ascii=False)
    print(f"[CompareSimilarity] Saved JSON report: {json_path}")
    
    # Markdown report
    md_path = output_dir / "comparison_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Similarity Scoring Method Comparison Report\n\n")
        f.write("This report compares four similarity scoring methods based on four diversity metrics.\n\n")
        
        f.write("## Methods Compared\n\n")
        for method, display_name in METHOD_NAMES.items():
            if method in all_method_metrics:
                f.write(f"- **{display_name}** ({method})\n")
        f.write("\n")
        
        f.write("## Metrics\n\n")
        for metric, display_name in METRIC_NAMES.items():
            f.write(f"- **{display_name}** ({metric.upper()})\n")
        f.write("\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Method | PC (mean±std) | PEC (mean±std) | TCD (mean±std) | BCM (mean±std) | Runs |\n")
        f.write("|--------|---------------|----------------|----------------|----------------|------|\n")
        
        for method in ["answer2", "embedding", "feature", "hybrid"]:
            if method not in all_method_metrics:
                continue
            
            metrics = all_method_metrics[method]
            pc = metrics.get("pc", {})
            pec = metrics.get("pec", {})
            tcd = metrics.get("tcd", {})
            bcm = metrics.get("bcm", {})
            
            display_name = METHOD_NAMES[method]
            f.write(f"| {display_name} | "
                   f"{pc.get('mean', 0):.3f}±{pc.get('std', 0):.3f} | "
                   f"{pec.get('mean', 0):.3f}±{pec.get('std', 0):.3f} | "
                   f"{tcd.get('mean', 0):.3f}±{tcd.get('std', 0):.3f} | "
                   f"{bcm.get('mean', 0):.3f}±{bcm.get('std', 0):.3f} | "
                   f"{pc.get('count', 0)} |\n")
        
        f.write("\n## Detailed Statistics\n\n")
        for method in ["answer2", "embedding", "feature", "hybrid"]:
            if method not in all_method_metrics:
                continue
            
            display_name = METHOD_NAMES[method]
            f.write(f"### {display_name}\n\n")
            metrics = all_method_metrics[method]
            
            for metric_name, display_name_metric in METRIC_NAMES.items():
                metric_data = metrics.get(metric_name, {})
                f.write(f"**{display_name_metric}**:\n")
                f.write(f"- Mean: {metric_data.get('mean', 0):.4f}\n")
                f.write(f"- Std: {metric_data.get('std', 0):.4f}\n")
                f.write(f"- Min: {metric_data.get('min', 0):.4f}\n")
                f.write(f"- Max: {metric_data.get('max', 0):.4f}\n")
                f.write(f"- Runs: {metric_data.get('count', 0)}\n")
                f.write("\n")
    
    print(f"[CompareSimilarity] Saved Markdown report: {md_path}")


def generate_comparison_figures(all_method_metrics: Dict[str, Dict], output_dir: Path):
    """
    Generate comparison figures for the four metrics.
    
    Args:
        all_method_metrics: Dictionary mapping method names to aggregated metrics
        output_dir: Directory to save figures
    """
    if not HAS_MATPLOTLIB:
        print("[WARNING] matplotlib not available, skipping figure generation")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "comparison_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    methods = [m for m in ["answer2", "embedding", "feature", "hybrid"] if m in all_method_metrics]
    if not methods:
        print("[WARNING] No methods found for figure generation")
        return
    
    # Prepare data for plotting
    method_labels = [METHOD_NAMES[m] for m in methods]
    metrics_data = {
        "pc": {"means": [], "stds": [], "name": "Parameter Coverage (PC)"},
        "pec": {"means": [], "stds": [], "name": "Behavior Coverage (PEC)"},
        "tcd": {"means": [], "stds": [], "name": "Trajectory Diversity (TCD)"},
        "bcm": {"means": [], "stds": [], "name": "Behavior Matrix (BCM)"}
    }
    
    for method in methods:
        metrics = all_method_metrics[method]
        for metric_name in metrics_data.keys():
            metric_data = metrics.get(metric_name, {})
            metrics_data[metric_name]["means"].append(metric_data.get("mean", 0.0))
            metrics_data[metric_name]["stds"].append(metric_data.get("std", 0.0))
    
    # Create bar chart for each metric
    for metric_name, metric_info in metrics_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        means = metric_info["means"]
        stds = metric_info["stds"]
        
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(methods)])
        
        ax.set_xlabel('Similarity Scoring Method', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{metric_info["name"]} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.01, f'{mean:.3f}', 
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        fig_path = figures_dir / f"{metric_name}_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[CompareSimilarity] Saved figure: {fig_path}")
    
    # Create combined radar chart
    try:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Normalize metrics to [0, 1] for radar chart
        all_values = []
        for metric_name in metrics_data.keys():
            all_values.extend(metrics_data[metric_name]["means"])
        max_val = max(all_values) if all_values else 1.0
        min_val = min(all_values) if all_values else 0.0
        range_val = max_val - min_val if max_val > min_val else 1.0
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_data), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, method in enumerate(methods):
            values = []
            for metric_name in metrics_data.keys():
                mean = metrics_data[metric_name]["means"][i]
                # Normalize to [0, 1]
                normalized = (mean - min_val) / range_val if range_val > 0 else 0.0
                values.append(normalized)
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=METHOD_NAMES[method])
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([METRIC_NAMES[m] for m in metrics_data.keys()])
        ax.set_ylim(0, 1)
        ax.set_title('Similarity Method Comparison (Normalized)', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        fig_path = figures_dir / "radar_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[CompareSimilarity] Saved figure: {fig_path}")
    except Exception as e:
        print(f"[WARNING] Failed to generate radar chart: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare similarity scoring methods based on diversity metrics."
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
        default="./reports/similarity_comparison",
        help="Directory to save comparison reports and figures",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Results directory {results_dir} does not exist")
        return
    
    # Find all experiment directories
    method_dirs = find_experiment_dirs(results_dir)
    
    # Aggregate metrics for each method
    all_method_metrics = {}
    for method, exp_dirs in method_dirs.items():
        if not exp_dirs:
            print(f"[WARNING] No experiments found for method: {method}")
            continue
        
        print(f"[CompareSimilarity] Found {len(exp_dirs)} experiment(s) for {method}")
        aggregated = aggregate_method_metrics(exp_dirs)
        all_method_metrics[method] = aggregated
    
    if not all_method_metrics:
        print("[ERROR] No metrics found for any method")
        return
    
    # Generate reports
    output_dir = Path(args.output_dir)
    generate_comparison_report(all_method_metrics, output_dir)
    generate_comparison_figures(all_method_metrics, output_dir)
    
    print(f"\n[CompareSimilarity] Comparison complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


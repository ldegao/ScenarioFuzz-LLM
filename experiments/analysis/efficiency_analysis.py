#!/usr/bin/env python3
"""
Efficiency analysis for similarity scoring methods.

This script analyzes token usage, API call counts, and costs
for each similarity method, and calculates efficiency metrics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
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
    load_metrics_summary,
    METHOD_NAMES,
    METRIC_NAMES
)


def load_token_usage(exp_dir: Path) -> Optional[Dict]:
    """
    Load token_usage.json from an experiment directory.
    
    Args:
        exp_dir: Path to experiment directory
        
    Returns:
        Dictionary with token usage data or None
    """
    token_path = exp_dir / "token_usage.json"
    if not token_path.exists():
        return None
    
    try:
        with open(token_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to load {token_path}: {e}")
        return None


def collect_efficiency_data(method_dirs: Dict[str, List[Path]]) -> Dict[str, Dict]:
    """
    Collect efficiency data (token usage, costs) for each method.
    
    Args:
        method_dirs: Dictionary mapping method names to experiment directories
        
    Returns:
        Dictionary mapping method names to efficiency data
    """
    efficiency_data = {}
    
    for method, exp_dirs in method_dirs.items():
        method_efficiency = {
            "prompt_tokens": [],
            "completion_tokens": [],
            "total_tokens": [],
            "call_count": [],
            "total_cost": [],
            "num_scenarios": []
        }
        
        for exp_dir in exp_dirs:
            # Load token usage
            token_data = load_token_usage(exp_dir)
            if token_data:
                overall = token_data.get("overall", {})
                method_efficiency["prompt_tokens"].append(overall.get("prompt_tokens", 0))
                method_efficiency["completion_tokens"].append(overall.get("completion_tokens", 0))
                method_efficiency["total_tokens"].append(overall.get("total_tokens", 0))
                method_efficiency["call_count"].append(overall.get("call_count", 0))
                method_efficiency["total_cost"].append(overall.get("total_cost_usd", 0.0))
            
            # Load number of scenarios
            summary = load_metrics_summary(exp_dir)
            if summary:
                num_scenarios = summary.get("num_scenarios", 0)
                method_efficiency["num_scenarios"].append(num_scenarios)
        
        # Calculate aggregates
        if method_efficiency["total_tokens"]:
            efficiency_data[method] = {
                "prompt_tokens": {
                    "total": sum(method_efficiency["prompt_tokens"]),
                    "mean": np.mean(method_efficiency["prompt_tokens"]),
                    "std": np.std(method_efficiency["prompt_tokens"])
                },
                "completion_tokens": {
                    "total": sum(method_efficiency["completion_tokens"]),
                    "mean": np.mean(method_efficiency["completion_tokens"]),
                    "std": np.std(method_efficiency["completion_tokens"])
                },
                "total_tokens": {
                    "total": sum(method_efficiency["total_tokens"]),
                    "mean": np.mean(method_efficiency["total_tokens"]),
                    "std": np.std(method_efficiency["total_tokens"])
                },
                "call_count": {
                    "total": sum(method_efficiency["call_count"]),
                    "mean": np.mean(method_efficiency["call_count"]),
                    "std": np.std(method_efficiency["call_count"])
                },
                "total_cost": {
                    "total": sum(method_efficiency["total_cost"]),
                    "mean": np.mean(method_efficiency["total_cost"]),
                    "std": np.std(method_efficiency["total_cost"])
                },
                "num_scenarios": {
                    "total": sum(method_efficiency["num_scenarios"]),
                    "mean": np.mean(method_efficiency["num_scenarios"]),
                    "std": np.std(method_efficiency["num_scenarios"])
                },
                "raw_data": method_efficiency
            }
    
    return efficiency_data


def calculate_efficiency_metrics(efficiency_data: Dict[str, Dict], method_metrics: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Calculate efficiency metrics (performance per cost, per token, etc.).
    
    Args:
        efficiency_data: Dictionary mapping method names to efficiency data
        method_metrics: Dictionary mapping method names to aggregated metrics
        
    Returns:
        Dictionary with efficiency metrics
    """
    efficiency_metrics = {}
    
    for method in efficiency_data.keys():
        if method not in method_metrics:
            continue
        
        eff_data = efficiency_data[method]
        metrics = method_metrics[method]
        
        total_cost = eff_data["total_cost"]["total"]
        total_tokens = eff_data["total_tokens"]["total"]
        total_scenarios = eff_data["num_scenarios"]["total"]
        
        method_efficiency = {}
        
        # Cost per scenario
        if total_scenarios > 0:
            method_efficiency["cost_per_scenario"] = total_cost / total_scenarios
            method_efficiency["tokens_per_scenario"] = total_tokens / total_scenarios
        else:
            method_efficiency["cost_per_scenario"] = 0.0
            method_efficiency["tokens_per_scenario"] = 0.0
        
        # Performance per cost (for each metric)
        for metric in ["pc", "pec", "tcd", "bcm"]:
            metric_mean = metrics.get(metric, {}).get("mean", 0.0)
            
            if total_cost > 0:
                method_efficiency[f"{metric}_per_dollar"] = metric_mean / total_cost
            else:
                method_efficiency[f"{metric}_per_dollar"] = 0.0
            
            if total_tokens > 0:
                method_efficiency[f"{metric}_per_1k_tokens"] = metric_mean / (total_tokens / 1000.0)
            else:
                method_efficiency[f"{metric}_per_1k_tokens"] = 0.0
        
        # Combined efficiency score (average of all metrics per dollar)
        metric_scores = [method_efficiency.get(f"{m}_per_dollar", 0.0) for m in ["pc", "pec", "tcd", "bcm"]]
        method_efficiency["combined_efficiency"] = np.mean(metric_scores) if metric_scores else 0.0
        
        efficiency_metrics[method] = method_efficiency
    
    return efficiency_metrics


def generate_efficiency_report(efficiency_data: Dict[str, Dict], efficiency_metrics: Dict[str, Dict], output_dir: Path):
    """
    Generate efficiency analysis report.
    
    Args:
        efficiency_data: Dictionary mapping method names to efficiency data
        efficiency_metrics: Dictionary mapping method names to efficiency metrics
        output_dir: Directory to save reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine data for report
    report_data = {
        "efficiency_data": efficiency_data,
        "efficiency_metrics": efficiency_metrics
    }
    
    # Save JSON report
    json_path = output_dir / "efficiency_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"[EfficiencyAnalysis] Saved JSON report: {json_path}")
    
    # Generate Markdown report
    md_path = output_dir / "efficiency_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Efficiency Analysis Report\n\n")
        f.write("This report analyzes token usage, API costs, and efficiency metrics for each similarity method.\n\n")
        
        # Resource usage
        f.write("## Resource Usage\n\n")
        f.write("| Method | Total Tokens | Prompt Tokens | Completion Tokens | API Calls | Total Cost (USD) | Scenarios |\n")
        f.write("|--------|--------------|---------------|-------------------|-----------|------------------|-----------|\n")
        
        for method in ["answer2", "embedding", "feature", "hybrid"]:
            if method not in efficiency_data:
                continue
            
            data = efficiency_data[method]
            display_name = METHOD_NAMES.get(method, method)
            
            f.write(f"| {display_name} | "
                   f"{data['total_tokens']['total']:,.0f} | "
                   f"{data['prompt_tokens']['total']:,.0f} | "
                   f"{data['completion_tokens']['total']:,.0f} | "
                   f"{data['call_count']['total']:,.0f} | "
                   f"${data['total_cost']['total']:.4f} | "
                   f"{data['num_scenarios']['total']:,.0f} |\n")
        
        # Cost per scenario
        f.write("\n## Cost Analysis\n\n")
        f.write("| Method | Cost per Scenario | Tokens per Scenario |\n")
        f.write("|--------|-------------------|---------------------|\n")
        
        for method in ["answer2", "embedding", "feature", "hybrid"]:
            if method not in efficiency_metrics:
                continue
            
            metrics = efficiency_metrics[method]
            display_name = METHOD_NAMES.get(method, method)
            
            f.write(f"| {display_name} | "
                   f"${metrics['cost_per_scenario']:.6f} | "
                   f"{metrics['tokens_per_scenario']:,.0f} |\n")
        
        # Performance per cost
        f.write("\n## Performance per Cost\n\n")
        f.write("| Method | PCE per $ | BCE per $ | DPE per $ | CCE per $ | Combined Efficiency |\n")
        f.write("|--------|-----------|-----------|-----------|-----------|---------------------|\n")
        
        for method in ["answer2", "embedding", "feature", "hybrid"]:
            if method not in efficiency_metrics:
                continue
            
            metrics = efficiency_metrics[method]
            display_name = METHOD_NAMES.get(method, method)
            
            f.write(f"| {display_name} | "
                   f"{metrics['pc_per_dollar']:.4f} | "
                   f"{metrics['pec_per_dollar']:.4f} | "
                   f"{metrics['tcd_per_dollar']:.4f} | "
                   f"{metrics['bcm_per_dollar']:.4f} | "
                   f"{metrics['combined_efficiency']:.4f} |\n")
        
        # Performance per 1k tokens
        f.write("\n## Performance per 1K Tokens\n\n")
        f.write("| Method | PCE per 1K | BCE per 1K | DPE per 1K | CCE per 1K |\n")
        f.write("|--------|------------|------------|------------|------------|\n")
        
        for method in ["answer2", "embedding", "feature", "hybrid"]:
            if method not in efficiency_metrics:
                continue
            
            metrics = efficiency_metrics[method]
            display_name = METHOD_NAMES.get(method, method)
            
            f.write(f"| {display_name} | "
                   f"{metrics['pc_per_1k_tokens']:.4f} | "
                   f"{metrics['pec_per_1k_tokens']:.4f} | "
                   f"{metrics['tcd_per_1k_tokens']:.4f} | "
                   f"{metrics['bcm_per_1k_tokens']:.4f} |\n")
    
    print(f"[EfficiencyAnalysis] Saved Markdown report: {md_path}")


def generate_efficiency_figures(efficiency_data: Dict[str, Dict], efficiency_metrics: Dict[str, Dict], output_dir: Path):
    """
    Generate efficiency analysis figures.
    
    Args:
        efficiency_data: Dictionary mapping method names to efficiency data
        efficiency_metrics: Dictionary mapping method names to efficiency metrics
        output_dir: Directory to save figures
    """
    if not HAS_MATPLOTLIB:
        print("[WARNING] matplotlib not available, skipping figure generation")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "efficiency_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    methods = [m for m in ["answer2", "embedding", "feature", "hybrid"] if m in efficiency_data]
    if not methods:
        return
    
    method_labels = [METHOD_NAMES.get(m, m) for m in methods]
    
    # Cost comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    costs = [efficiency_data[m]["total_cost"]["total"] for m in methods]
    bars = ax.bar(method_labels, costs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(methods)], alpha=0.7)
    ax.set_ylabel('Total Cost (USD)', fontsize=12)
    ax.set_title('Total API Cost by Method', fontsize=14, fontweight='bold')
    ax.set_xticklabels(method_labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'${cost:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig_path = figures_dir / "cost_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[EfficiencyAnalysis] Saved figure: {fig_path}")
    
    # Efficiency comparison (combined efficiency)
    if efficiency_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        efficiencies = [efficiency_metrics.get(m, {}).get("combined_efficiency", 0.0) for m in methods]
        bars = ax.bar(method_labels, efficiencies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(methods)], alpha=0.7)
        ax.set_ylabel('Combined Efficiency (Metrics per Dollar)', fontsize=12)
        ax.set_title('Efficiency Comparison (Higher is Better)', fontsize=14, fontweight='bold')
        ax.set_xticklabels(method_labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{eff:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        fig_path = figures_dir / "efficiency_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[EfficiencyAnalysis] Saved figure: {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze efficiency (token usage, costs) for similarity methods."
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
        default="./reports/similarity_comparison/efficiency_analysis",
        help="Directory to save efficiency analysis results",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="./reports/similarity_comparison/comparison_report.json",
        help="Path to comparison_report.json with method metrics",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Results directory {results_dir} does not exist")
        return
    
    # Load method metrics
    metrics_file = Path(args.metrics_file)
    if not metrics_file.exists():
        print(f"[WARNING] Metrics file {metrics_file} not found, skipping efficiency metrics calculation")
        method_metrics = {}
    else:
        with open(metrics_file, "r", encoding="utf-8") as f:
            method_metrics = json.load(f)
    
    # Find all experiment directories
    method_dirs = find_experiment_dirs(results_dir)
    
    # Collect efficiency data
    efficiency_data = collect_efficiency_data(method_dirs)
    
    if not efficiency_data:
        print("[ERROR] No efficiency data found")
        return
    
    # Calculate efficiency metrics
    efficiency_metrics = calculate_efficiency_metrics(efficiency_data, method_metrics)
    
    # Generate reports
    output_dir = Path(args.output_dir)
    generate_efficiency_report(efficiency_data, efficiency_metrics, output_dir)
    generate_efficiency_figures(efficiency_data, efficiency_metrics, output_dir)
    
    print(f"\n[EfficiencyAnalysis] Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


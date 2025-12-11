#!/usr/bin/env python3
"""
Scenario-level detailed analysis.

This script analyzes scenario distribution characteristics, identifies
anomalies and patterns, and generates scenario-level statistics.
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


def collect_scenario_data(method_dirs: Dict[str, List[Path]]) -> Dict[str, List[Dict]]:
    """
    Collect scenario-level data for each method.
    
    Args:
        method_dirs: Dictionary mapping method names to experiment directories
        
    Returns:
        Dictionary mapping method names to lists of scenario records
    """
    all_scenarios = {}
    
    for method, exp_dirs in method_dirs.items():
        method_scenarios = []
        
        for exp_dir in exp_dirs:
            records = load_metrics_records(exp_dir)
            method_scenarios.extend(records)
        
        all_scenarios[method] = method_scenarios
    
    return all_scenarios


def analyze_scenario_distribution(scenarios: List[Dict]) -> Dict:
    """
    Analyze distribution characteristics of scenarios.
    
    Args:
        scenarios: List of scenario records
        
    Returns:
        Dictionary with distribution statistics
    """
    if not scenarios:
        return {}
    
    # Extract metric values
    metrics_data = {
        "pc": [],
        "pec": [],
        "tcd": [],
        "bcm": []
    }
    
    scenario_ids = []
    generation_ids = []
    
    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id", len(scenario_ids))
        generation_id = scenario.get("generation_id", 0)
        
        scenario_ids.append(scenario_id)
        generation_ids.append(generation_id)
        
        for metric in metrics_data.keys():
            value = scenario.get(metric, 0.0)
            if isinstance(value, (int, float)):
                metrics_data[metric].append(float(value))
    
    # Calculate statistics
    distribution = {
        "total_scenarios": len(scenarios),
        "unique_generations": len(set(generation_ids)) if generation_ids else 0,
        "scenario_id_range": (min(scenario_ids), max(scenario_ids)) if scenario_ids else (0, 0)
    }
    
    for metric, values in metrics_data.items():
        if not values:
            continue
        
        distribution[metric] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "iqr": float(np.percentile(values, 75) - np.percentile(values, 25))
        }
    
    return distribution


def identify_anomalies(scenarios: List[Dict], threshold: float = 2.0) -> Dict:
    """
    Identify anomalous scenarios using z-score method.
    
    Args:
        scenarios: List of scenario records
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        Dictionary with anomaly information
    """
    if not scenarios:
        return {}
    
    # Extract metric values
    metrics_data = {
        "pc": [],
        "pec": [],
        "tcd": [],
        "bcm": []
    }
    
    for scenario in scenarios:
        for metric in metrics_data.keys():
            value = scenario.get(metric, 0.0)
            if isinstance(value, (int, float)):
                metrics_data[metric].append(float(value))
    
    anomalies = {}
    
    for metric, values in metrics_data.items():
        if len(values) < 3:
            continue
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            continue
        
        # Calculate z-scores
        z_scores = [(v - mean) / std for v in values]
        
        # Find anomalies
        anomaly_indices = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
        anomaly_scenarios = [scenarios[i] for i in anomaly_indices]
        
        anomalies[metric] = {
            "count": len(anomaly_indices),
            "percentage": (len(anomaly_indices) / len(scenarios)) * 100,
            "scenarios": [
                {
                    "scenario_id": s.get("scenario_id", i),
                    "generation_id": s.get("generation_id", 0),
                    "value": float(s.get(metric, 0.0)),
                    "z_score": float(z_scores[i])
                }
                for i, s in zip(anomaly_indices, anomaly_scenarios)
            ]
        }
    
    return anomalies


def analyze_patterns(scenarios: List[Dict]) -> Dict:
    """
    Analyze patterns in scenario generation.
    
    Args:
        scenarios: List of scenario records
        
    Returns:
        Dictionary with pattern analysis
    """
    if not scenarios:
        return {}
    
    # Group by generation_id
    generation_groups = {}
    for scenario in scenarios:
        gen_id = scenario.get("generation_id", 0)
        if gen_id not in generation_groups:
            generation_groups[gen_id] = []
        generation_groups[gen_id].append(scenario)
    
    patterns = {
        "generations": len(generation_groups),
        "scenarios_per_generation": {
            "mean": float(np.mean([len(g) for g in generation_groups.values()])),
            "std": float(np.std([len(g) for g in generation_groups.values()])),
            "min": int(min([len(g) for g in generation_groups.values()])),
            "max": int(max([len(g) for g in generation_groups.values()]))
        }
    }
    
    # Analyze metric trends across generations
    for metric in ["pc", "pec", "tcd", "bcm"]:
        gen_means = []
        for gen_id in sorted(generation_groups.keys()):
            gen_scenarios = generation_groups[gen_id]
            values = [s.get(metric, 0.0) for s in gen_scenarios if isinstance(s.get(metric), (int, float))]
            if values:
                gen_means.append(float(np.mean(values)))
        
        if gen_means:
            patterns[f"{metric}_trend"] = {
                "initial": gen_means[0],
                "final": gen_means[-1],
                "change": gen_means[-1] - gen_means[0],
                "change_percent": ((gen_means[-1] - gen_means[0]) / gen_means[0] * 100) if gen_means[0] != 0 else 0.0
            }
    
    return patterns


def generate_scenario_report(scenario_data: Dict[str, List[Dict]], output_dir: Path):
    """
    Generate scenario-level analysis report.
    
    Args:
        scenario_data: Dictionary mapping method names to scenario records
        output_dir: Directory to save reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for method, scenarios in scenario_data.items():
        distribution = analyze_scenario_distribution(scenarios)
        anomalies = identify_anomalies(scenarios)
        patterns = analyze_patterns(scenarios)
        
        results[method] = {
            "distribution": distribution,
            "anomalies": anomalies,
            "patterns": patterns
        }
    
    # Save JSON report
    json_path = output_dir / "scenario_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[ScenarioAnalysis] Saved JSON report: {json_path}")
    
    # Generate Markdown report
    md_path = output_dir / "scenario_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Scenario-Level Analysis Report\n\n")
        f.write("This report provides detailed analysis of scenario distributions, anomalies, and patterns.\n\n")
        
        for method in ["answer2", "embedding", "feature", "hybrid"]:
            if method not in results:
                continue
            
            display_name = METHOD_NAMES.get(method, method)
            method_results = results[method]
            
            f.write(f"## {display_name}\n\n")
            
            # Distribution
            dist = method_results["distribution"]
            f.write("### Distribution Statistics\n\n")
            f.write(f"- **Total scenarios**: {dist.get('total_scenarios', 0)}\n")
            f.write(f"- **Unique generations**: {dist.get('unique_generations', 0)}\n")
            f.write(f"- **Scenario ID range**: {dist.get('scenario_id_range', (0, 0))}\n\n")
            
            f.write("| Metric | Mean | Median | Std | Min | Max | Q25 | Q75 | IQR |\n")
            f.write("|--------|------|--------|-----|-----|-----|-----|-----|-----|\n")
            
            for metric in ["pc", "pec", "tcd", "bcm"]:
                if metric not in dist:
                    continue
                
                metric_data = dist[metric]
                metric_name = METRIC_NAMES[metric]
                f.write(f"| {metric_name} | "
                       f"{metric_data['mean']:.6f} | "
                       f"{metric_data['median']:.6f} | "
                       f"{metric_data['std']:.6f} | "
                       f"{metric_data['min']:.6f} | "
                       f"{metric_data['max']:.6f} | "
                       f"{metric_data['q25']:.6f} | "
                       f"{metric_data['q75']:.6f} | "
                       f"{metric_data['iqr']:.6f} |\n")
            
            # Anomalies
            anomalies = method_results["anomalies"]
            f.write("\n### Anomaly Detection (Z-score > 2.0)\n\n")
            
            for metric in ["pc", "pec", "tcd", "bcm"]:
                if metric not in anomalies:
                    continue
                
                anomaly_data = anomalies[metric]
                metric_name = METRIC_NAMES[metric]
                f.write(f"**{metric_name}**: {anomaly_data['count']} anomalies ({anomaly_data['percentage']:.2f}%)\n")
                
                if anomaly_data['scenarios']:
                    f.write("Top anomalies:\n")
                    sorted_anomalies = sorted(anomaly_data['scenarios'], key=lambda x: abs(x['z_score']), reverse=True)[:5]
                    for anomaly in sorted_anomalies:
                        f.write(f"- Scenario {anomaly['scenario_id']}: value={anomaly['value']:.6f}, z-score={anomaly['z_score']:.2f}\n")
                    f.write("\n")
            
            # Patterns
            patterns = method_results["patterns"]
            f.write("\n### Generation Patterns\n\n")
            f.write(f"- **Number of generations**: {patterns.get('generations', 0)}\n")
            spg = patterns.get('scenarios_per_generation', {})
            f.write(f"- **Scenarios per generation**: {spg.get('mean', 0):.2f} ± {spg.get('std', 0):.2f} (range: {spg.get('min', 0)}-{spg.get('max', 0)})\n\n")
            
            f.write("**Metric trends across generations**:\n\n")
            for metric in ["pc", "pec", "tcd", "bcm"]:
                trend_key = f"{metric}_trend"
                if trend_key not in patterns:
                    continue
                
                trend = patterns[trend_key]
                metric_name = METRIC_NAMES[metric]
                f.write(f"- **{metric_name}**: {trend['initial']:.6f} → {trend['final']:.6f} "
                       f"(change: {trend['change']:.6f}, {trend['change_percent']:.2f}%)\n")
            
            f.write("\n")
    
    print(f"[ScenarioAnalysis] Saved Markdown report: {md_path}")


def generate_scenario_figures(scenario_data: Dict[str, List[Dict]], output_dir: Path):
    """
    Generate scenario-level visualization figures.
    
    Args:
        scenario_data: Dictionary mapping method names to scenario records
        output_dir: Directory to save figures
    """
    if not HAS_MATPLOTLIB:
        print("[WARNING] matplotlib not available, skipping figure generation")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "scenario_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    methods = [m for m in ["answer2", "embedding", "feature", "hybrid"] if m in scenario_data]
    if not methods:
        return
    
    # Distribution histograms
    for metric in ["pc", "pec", "tcd", "bcm"]:
        metric_name = METRIC_NAMES[metric]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, method in enumerate(methods[:4]):
            if idx >= len(axes):
                break
            
            scenarios = scenario_data[method]
            values = [s.get(metric, 0.0) for s in scenarios if isinstance(s.get(metric), (int, float))]
            
            if not values:
                continue
            
            ax = axes[idx]
            ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{METHOD_NAMES.get(method, method)}', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(methods), 4):
            axes[idx].axis('off')
        
        plt.suptitle(f'{metric_name} Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        fig_path = figures_dir / f"{metric}_distribution.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[ScenarioAnalysis] Saved figure: {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze scenario-level characteristics and patterns."
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
        default="./reports/similarity_comparison/scenario_analysis",
        help="Directory to save scenario analysis results",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Results directory {results_dir} does not exist")
        return
    
    # Find all experiment directories
    method_dirs = find_experiment_dirs(results_dir)
    
    # Collect scenario data
    scenario_data = collect_scenario_data(method_dirs)
    
    if not scenario_data:
        print("[ERROR] No scenario data found")
        return
    
    # Generate reports
    output_dir = Path(args.output_dir)
    generate_scenario_report(scenario_data, output_dir)
    generate_scenario_figures(scenario_data, output_dir)
    
    print(f"\n[ScenarioAnalysis] Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


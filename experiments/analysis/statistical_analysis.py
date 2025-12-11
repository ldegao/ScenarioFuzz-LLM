#!/usr/bin/env python3
"""
Statistical significance testing for similarity scoring methods.

This script performs:
- One-way ANOVA to test differences between methods
- Pairwise t-tests to compare methods
- Effect size calculations (Cohen's d)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats

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


def collect_all_values(method_dirs: Dict[str, List[Path]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Collect all metric values for each method.
    
    Args:
        method_dirs: Dictionary mapping method names to experiment directories
        
    Returns:
        Dictionary mapping method names to metric values
    """
    all_data = {}
    
    for method, exp_dirs in method_dirs.items():
        if not exp_dirs:
            continue
        
        method_data = {
            "pc": [],
            "pec": [],
            "tcd": [],
            "bcm": []
        }
        
        for exp_dir in exp_dirs:
            summary = load_metrics_summary(exp_dir)
            if summary is None:
                continue
            
            for metric in method_data.keys():
                value = summary.get(metric, 0.0)
                if isinstance(value, (int, float)):
                    method_data[metric].append(float(value))
        
        all_data[method] = method_data
    
    return all_data


def perform_anova(all_data: Dict[str, Dict[str, List[float]]], metric: str) -> Dict:
    """
    Perform one-way ANOVA test for a metric across all methods.
    
    Args:
        all_data: Dictionary mapping method names to metric values
        metric: Metric name (pc, pec, tcd, bcm)
        
    Returns:
        Dictionary with ANOVA results
    """
    groups = []
    method_names = []
    
    for method, method_data in all_data.items():
        values = method_data.get(metric, [])
        if len(values) > 0:
            groups.append(values)
            method_names.append(method)
    
    if len(groups) < 2:
        return {
            "statistic": None,
            "pvalue": None,
            "significant": False,
            "error": "Insufficient groups for ANOVA"
        }
    
    # Perform ANOVA
    try:
        f_stat, p_value = stats.f_oneway(*groups)
        
        return {
            "statistic": float(f_stat),
            "pvalue": float(p_value),
            "significant": p_value < 0.05,
            "alpha": 0.05,
            "groups": method_names,
            "group_sizes": [len(g) for g in groups]
        }
    except Exception as e:
        return {
            "statistic": None,
            "pvalue": None,
            "significant": False,
            "error": str(e)
        }


def perform_pairwise_ttest(all_data: Dict[str, Dict[str, List[float]]], metric: str) -> Dict[str, Dict]:
    """
    Perform pairwise t-tests between all method pairs.
    
    Args:
        all_data: Dictionary mapping method names to metric values
        metric: Metric name (pc, pec, tcd, bcm)
        
    Returns:
        Dictionary mapping method pairs to t-test results
    """
    results = {}
    methods = list(all_data.keys())
    
    for i, method1 in enumerate(methods):
        values1 = all_data[method1].get(metric, [])
        if len(values1) == 0:
            continue
        
        for method2 in methods[i+1:]:
            values2 = all_data[method2].get(metric, [])
            if len(values2) == 0:
                continue
            
            try:
                # Perform independent t-test
                t_stat, p_value = stats.ttest_ind(values1, values2)
                
                # Calculate Cohen's d (effect size)
                mean1, mean2 = np.mean(values1), np.mean(values2)
                std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)
                pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
                
                results[f"{method1}_vs_{method2}"] = {
                    "t_statistic": float(t_stat),
                    "pvalue": float(p_value),
                    "significant": p_value < 0.05,
                    "cohens_d": float(cohens_d),
                    "mean1": float(mean1),
                    "mean2": float(mean2),
                    "std1": float(std1),
                    "std2": float(std2),
                    "n1": len(values1),
                    "n2": len(values2)
                }
            except Exception as e:
                results[f"{method1}_vs_{method2}"] = {
                    "error": str(e)
                }
    
    return results


def generate_statistical_report(all_data: Dict[str, Dict[str, List[float]]], output_dir: Path):
    """
    Generate statistical analysis report.
    
    Args:
        all_data: Dictionary mapping method names to metric values
        output_dir: Directory to save reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform statistical tests for each metric
    results = {}
    
    for metric in ["pc", "pec", "tcd", "bcm"]:
        metric_name = METRIC_NAMES[metric]
        print(f"\n[StatisticalAnalysis] Analyzing {metric_name}...")
        
        # ANOVA
        anova_result = perform_anova(all_data, metric)
        
        # Pairwise t-tests
        pairwise_results = perform_pairwise_ttest(all_data, metric)
        
        results[metric] = {
            "anova": anova_result,
            "pairwise_tests": pairwise_results
        }
    
    # Save JSON report
    json_path = output_dir / "statistical_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[StatisticalAnalysis] Saved JSON report: {json_path}")
    
    # Generate Markdown report
    md_path = output_dir / "statistical_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Statistical Significance Analysis\n\n")
        f.write("This report presents statistical tests comparing similarity scoring methods.\n\n")
        
        for metric in ["pc", "pec", "tcd", "bcm"]:
            metric_name = METRIC_NAMES[metric]
            f.write(f"## {metric_name} ({metric.upper()})\n\n")
            
            # ANOVA results
            anova = results[metric]["anova"]
            f.write("### One-Way ANOVA\n\n")
            if "error" in anova:
                f.write(f"**Error**: {anova['error']}\n\n")
            else:
                f.write(f"- **F-statistic**: {anova['statistic']:.4f}\n")
                f.write(f"- **p-value**: {anova['pvalue']:.6f}\n")
                f.write(f"- **Significant** (α=0.05): {'Yes' if anova['significant'] else 'No'}\n")
                f.write(f"- **Groups**: {', '.join(anova['groups'])}\n")
                f.write(f"- **Group sizes**: {anova['group_sizes']}\n\n")
            
            # Pairwise tests
            f.write("### Pairwise t-tests\n\n")
            f.write("| Comparison | t-statistic | p-value | Significant | Cohen's d | Mean1 | Mean2 |\n")
            f.write("|------------|-------------|---------|-------------|-----------|-------|-------|\n")
            
            pairwise = results[metric]["pairwise_tests"]
            for pair, test_result in sorted(pairwise.items()):
                if "error" in test_result:
                    f.write(f"| {pair} | Error: {test_result['error']} | | | | | |\n")
                else:
                    method1, method2 = pair.split("_vs_")
                    display1 = METHOD_NAMES.get(method1, method1)
                    display2 = METHOD_NAMES.get(method2, method2)
                    f.write(f"| {display1} vs {display2} | "
                           f"{test_result['t_statistic']:.4f} | "
                           f"{test_result['pvalue']:.6f} | "
                           f"{'Yes' if test_result['significant'] else 'No'} | "
                           f"{test_result['cohens_d']:.4f} | "
                           f"{test_result['mean1']:.4f} | "
                           f"{test_result['mean2']:.4f} |\n")
            
            f.write("\n")
            
            # Effect size interpretation
            f.write("#### Effect Size Interpretation (Cohen's d)\n\n")
            f.write("- |d| < 0.2: Negligible effect\n")
            f.write("- 0.2 ≤ |d| < 0.5: Small effect\n")
            f.write("- 0.5 ≤ |d| < 0.8: Medium effect\n")
            f.write("- |d| ≥ 0.8: Large effect\n\n")
    
    print(f"[StatisticalAnalysis] Saved Markdown report: {md_path}")


def generate_statistical_figures(all_data: Dict[str, Dict[str, List[float]]], output_dir: Path):
    """
    Generate statistical analysis figures (box plots).
    
    Args:
        all_data: Dictionary mapping method names to metric values
        output_dir: Directory to save figures
    """
    if not HAS_MATPLOTLIB:
        print("[WARNING] matplotlib not available, skipping figure generation")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "statistical_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    for metric in ["pc", "pec", "tcd", "bcm"]:
        metric_name = METRIC_NAMES[metric]
        
        # Prepare data for box plot
        data = []
        labels = []
        
        for method in ["answer2", "embedding", "feature", "hybrid"]:
            if method not in all_data:
                continue
            values = all_data[method].get(metric, [])
            if len(values) > 0:
                data.append(values)
                labels.append(METHOD_NAMES.get(method, method))
        
        if len(data) == 0:
            continue
        
        # Create box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{metric_name} Distribution Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        fig_path = figures_dir / f"{metric}_boxplot.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[StatisticalAnalysis] Saved figure: {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Perform statistical significance testing for similarity methods."
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
        default="./reports/similarity_comparison/statistical_analysis",
        help="Directory to save statistical analysis results",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Results directory {results_dir} does not exist")
        return
    
    # Find all experiment directories
    method_dirs = find_experiment_dirs(results_dir)
    
    # Collect all metric values
    all_data = collect_all_values(method_dirs)
    
    if not all_data:
        print("[ERROR] No data found for any method")
        return
    
    # Generate reports
    output_dir = Path(args.output_dir)
    generate_statistical_report(all_data, output_dir)
    generate_statistical_figures(all_data, output_dir)
    
    print(f"\n[StatisticalAnalysis] Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


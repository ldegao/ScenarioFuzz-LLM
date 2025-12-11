#!/usr/bin/env python3
"""
Correlation analysis between diversity metrics.

This script analyzes correlations between PC, PEC, TCD, and BCM metrics
across all similarity methods and generates correlation heatmaps.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib/seaborn not available, skipping figure generation")

from .compare_similarity_methods import (
    find_experiment_dirs,
    load_metrics_summary,
    METHOD_NAMES,
    METRIC_NAMES
)


def collect_metric_pairs(method_dirs: Dict[str, List[Path]]) -> Dict[str, Tuple[List[float], List[float]]]:
    """
    Collect pairs of metric values across all experiments for correlation analysis.
    
    Args:
        method_dirs: Dictionary mapping method names to experiment directories
        
    Returns:
        Dictionary mapping metric pairs to (values1, values2) tuples
    """
    # Collect all metric values from all experiments
    all_metrics = {
        "pc": [],
        "pec": [],
        "tcd": [],
        "bcm": []
    }
    
    for method, exp_dirs in method_dirs.items():
        for exp_dir in exp_dirs:
            summary = load_metrics_summary(exp_dir)
            if summary is None:
                continue
            
            for metric in all_metrics.keys():
                value = summary.get(metric, 0.0)
                if isinstance(value, (int, float)):
                    all_metrics[metric].append(float(value))
    
    # Create pairs for correlation analysis
    metrics = ["pc", "pec", "tcd", "bcm"]
    pairs = {}
    
    for i, metric1 in enumerate(metrics):
        for metric2 in metrics[i+1:]:
            values1 = all_metrics[metric1]
            values2 = all_metrics[metric2]
            
            # Only keep pairs where both values exist
            paired_values1 = []
            paired_values2 = []
            
            # Since values come from same experiments, they should be aligned
            min_len = min(len(values1), len(values2))
            if min_len > 0:
                paired_values1 = values1[:min_len]
                paired_values2 = values2[:min_len]
            
            pairs[f"{metric1}_{metric2}"] = (paired_values1, paired_values2)
    
    return pairs


def calculate_correlations(pairs: Dict[str, Tuple[List[float], List[float]]]) -> Dict:
    """
    Calculate Pearson and Spearman correlations for all metric pairs.
    
    Args:
        pairs: Dictionary mapping metric pairs to value tuples
        
    Returns:
        Dictionary with correlation results
    """
    results = {}
    
    for pair_name, (values1, values2) in pairs.items():
        if len(values1) < 3:  # Need at least 3 points for correlation
            results[pair_name] = {
                "pearson_r": None,
                "pearson_p": None,
                "spearman_r": None,
                "spearman_p": None,
                "n": len(values1),
                "error": "Insufficient data"
            }
            continue
        
        try:
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(values1, values2)
            
            # Spearman correlation
            spearman_r, spearman_p = spearmanr(values1, values2)
            
            results[pair_name] = {
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "n": len(values1),
                "pearson_significant": pearson_p < 0.05,
                "spearman_significant": spearman_p < 0.05
            }
        except Exception as e:
            results[pair_name] = {
                "error": str(e),
                "n": len(values1)
            }
    
    return results


def build_correlation_matrix(method_dirs: Dict[str, List[Path]]) -> Tuple[np.ndarray, List[str]]:
    """
    Build correlation matrix from all metric values.
    
    Args:
        method_dirs: Dictionary mapping method names to experiment directories
        
    Returns:
        Tuple of (correlation_matrix, metric_names)
    """
    # Collect all metric values
    all_metrics = {
        "pc": [],
        "pec": [],
        "tcd": [],
        "bcm": []
    }
    
    for method, exp_dirs in method_dirs.items():
        for exp_dir in exp_dirs:
            summary = load_metrics_summary(exp_dir)
            if summary is None:
                continue
            
            for metric in all_metrics.keys():
                value = summary.get(metric, 0.0)
                if isinstance(value, (int, float)):
                    all_metrics[metric].append(float(value))
    
    # Build matrix
    metrics = ["pc", "pec", "tcd", "bcm"]
    metric_names = [METRIC_NAMES[m] for m in metrics]
    
    # Find minimum length to ensure alignment
    min_len = min([len(all_metrics[m]) for m in metrics] + [1000])
    
    matrix = np.zeros((len(metrics), len(metrics)))
    
    for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
            if i == j:
                matrix[i, j] = 1.0
            else:
                values1 = all_metrics[metric1][:min_len]
                values2 = all_metrics[metric2][:min_len]
                
                if len(values1) >= 3:
                    try:
                        r, _ = pearsonr(values1, values2)
                        matrix[i, j] = float(r)
                    except:
                        matrix[i, j] = 0.0
                else:
                    matrix[i, j] = 0.0
    
    return matrix, metric_names


def generate_correlation_report(correlation_results: Dict, output_dir: Path):
    """
    Generate correlation analysis report.
    
    Args:
        correlation_results: Dictionary with correlation results
        output_dir: Directory to save reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    json_path = output_dir / "correlation_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(correlation_results, f, indent=2, ensure_ascii=False)
    print(f"[CorrelationAnalysis] Saved JSON report: {json_path}")
    
    # Generate Markdown report
    md_path = output_dir / "correlation_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Correlation Analysis Report\n\n")
        f.write("This report analyzes correlations between diversity metrics (PC, PEC, TCD, BCM).\n\n")
        
        f.write("## Correlation Results\n\n")
        f.write("| Metric Pair | Pearson r | Pearson p | Significant | Spearman r | Spearman p | Significant | N |\n")
        f.write("|-------------|-----------|-----------|------------|------------|------------|-------------|---|\n")
        
        for pair_name, result in sorted(correlation_results.items()):
            if "error" in result:
                metric1, metric2 = pair_name.split("_")
                f.write(f"| {METRIC_NAMES[metric1]} vs {METRIC_NAMES[metric2]} | "
                       f"Error: {result['error']} | | | | | | {result.get('n', 0)} |\n")
            else:
                metric1, metric2 = pair_name.split("_")
                f.write(f"| {METRIC_NAMES[metric1]} vs {METRIC_NAMES[metric2]} | "
                       f"{result['pearson_r']:.4f} | "
                       f"{result['pearson_p']:.6f} | "
                       f"{'Yes' if result['pearson_significant'] else 'No'} | "
                       f"{result['spearman_r']:.4f} | "
                       f"{result['spearman_p']:.6f} | "
                       f"{'Yes' if result['spearman_significant'] else 'No'} | "
                       f"{result['n']} |\n")
        
        f.write("\n## Interpretation\n\n")
        f.write("### Correlation Strength\n")
        f.write("- |r| < 0.1: Negligible correlation\n")
        f.write("- 0.1 ≤ |r| < 0.3: Weak correlation\n")
        f.write("- 0.3 ≤ |r| < 0.5: Moderate correlation\n")
        f.write("- 0.5 ≤ |r| < 0.7: Strong correlation\n")
        f.write("- |r| ≥ 0.7: Very strong correlation\n\n")
        
        f.write("### Correlation Types\n")
        f.write("- **Pearson correlation**: Measures linear relationships\n")
        f.write("- **Spearman correlation**: Measures monotonic relationships (more robust to outliers)\n\n")
    
    print(f"[CorrelationAnalysis] Saved Markdown report: {md_path}")


def generate_correlation_figures(correlation_matrix: np.ndarray, metric_names: List[str], output_dir: Path):
    """
    Generate correlation heatmap.
    
    Args:
        correlation_matrix: Correlation matrix
        metric_names: List of metric display names
        output_dir: Directory to save figures
    """
    if not HAS_MATPLOTLIB:
        print("[WARNING] matplotlib/seaborn not available, skipping figure generation")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "correlation_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use seaborn if available, otherwise use matplotlib
    try:
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            xticklabels=metric_names,
            yticklabels=metric_names,
            cbar_kws={"label": "Correlation Coefficient"},
            ax=ax
        )
    except:
        # Fallback to matplotlib
        im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(metric_names)))
        ax.set_xticklabels(metric_names)
        ax.set_yticklabels(metric_names)
        
        # Add text annotations
        for i in range(len(metric_names)):
            for j in range(len(metric_names)):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax, label="Correlation Coefficient")
    
    ax.set_title("Metric Correlation Heatmap", fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    fig_path = figures_dir / "correlation_heatmap.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[CorrelationAnalysis] Saved figure: {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlations between diversity metrics."
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
        default="./reports/similarity_comparison/correlation_analysis",
        help="Directory to save correlation analysis results",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Results directory {results_dir} does not exist")
        return
    
    # Find all experiment directories
    method_dirs = find_experiment_dirs(results_dir)
    
    # Collect metric pairs
    pairs = collect_metric_pairs(method_dirs)
    
    if not pairs:
        print("[ERROR] No data found for correlation analysis")
        return
    
    # Calculate correlations
    correlation_results = calculate_correlations(pairs)
    
    # Build correlation matrix
    correlation_matrix, metric_names = build_correlation_matrix(method_dirs)
    
    # Generate reports
    output_dir = Path(args.output_dir)
    generate_correlation_report(correlation_results, output_dir)
    generate_correlation_figures(correlation_matrix, metric_names, output_dir)
    
    print(f"\n[CorrelationAnalysis] Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Statistical Analysis Script
Performs statistical tests on experiment results
"""

import sys
import os
import json
import numpy as np
from scipy import stats
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compare_two_methods(method1_data: List[float], method2_data: List[float], 
                        method1_name: str = "Method1", method2_name: str = "Method2"):
    """
    Compare two methods using statistical tests
    
    Args:
        method1_data: List of metric values for method 1
        method2_data: List of metric values for method 2
        method1_name: Name of method 1
        method2_name: Name of method 2
        
    Returns:
        Dictionary with statistical test results
    """
    # Convert to numpy arrays
    data1 = np.array(method1_data)
    data2 = np.array(method2_data)
    
    # Basic statistics
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1), np.std(data2)
    
    # Normality test (Shapiro-Wilk)
    _, p_norm1 = stats.shapiro(data1) if len(data1) <= 5000 else (None, 0.0)
    _, p_norm2 = stats.shapiro(data2) if len(data2) <= 5000 else (None, 0.0)
    
    # Choose appropriate test
    if p_norm1 > 0.05 and p_norm2 > 0.05:
        # Both normal, use t-test
        t_stat, p_value = stats.ttest_ind(data1, data2)
        test_name = "t-test"
    else:
        # Non-normal, use Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        test_name = "Mann-Whitney U"
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(data1) - 1) * std1**2 + (len(data2) - 1) * std2**2) / 
                         (len(data1) + len(data2) - 2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
    
    # Interpretation
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    return {
        'method1': method1_name,
        'method2': method2_name,
        'mean1': float(mean1),
        'mean2': float(mean2),
        'std1': float(std1),
        'std2': float(std2),
        'test': test_name,
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'cohens_d': float(cohens_d),
        'effect_size': effect_size,
        'improvement': float((mean1 - mean2) / mean2 * 100) if mean2 > 0 else 0.0
    }


def analyze_all_methods(results: Dict[str, Dict[str, List[float]]]):
    """
    Analyze results for all methods
    
    Args:
        results: Dictionary mapping method names to metric dictionaries
                e.g., {'RAG-ScenarioFuzz': {'pc': [0.8, 0.9, ...], 'pec': [...], ...}}
        
    Returns:
        Dictionary with comparison results
    """
    analysis = {}
    
    # Get all metric names
    metric_names = set()
    for method_results in results.values():
        metric_names.update(method_results.keys())
    
    # Compare each metric
    for metric in metric_names:
        metric_comparisons = {}
        
        # Get data for all methods
        method_data = {}
        for method_name, method_results in results.items():
            if metric in method_results:
                method_data[method_name] = method_results[metric]
        
        # Compare each pair
        methods = list(method_data.keys())
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                if len(method_data[method1]) > 0 and len(method_data[method2]) > 0:
                    comparison = compare_two_methods(
                        method_data[method1],
                        method_data[method2],
                        method1,
                        method2
                    )
                    key = f"{method1}_vs_{method2}"
                    metric_comparisons[key] = comparison
        
        analysis[metric] = metric_comparisons
    
    return analysis


def generate_statistical_report(analysis: Dict, output_path: str):
    """
    Generate statistical analysis report
    
    Args:
        analysis: Analysis results from analyze_all_methods
        output_path: Path to save report
    """
    report_lines = [
        "# Statistical Analysis Report",
        "",
        "## Summary",
        ""
    ]
    
    for metric, comparisons in analysis.items():
        report_lines.append(f"### {metric.upper()}")
        report_lines.append("")
        report_lines.append("| Comparison | Mean1 | Mean2 | p-value | Significant | Effect Size | Improvement |")
        report_lines.append("|------------|-------|-------|---------|-------------|-------------|-------------|")
        
        for comparison_name, comp in comparisons.items():
            report_lines.append(
                f"| {comparison_name} | {comp['mean1']:.4f} | {comp['mean2']:.4f} | "
                f"{comp['p_value']:.4f} | {'Yes' if comp['significant'] else 'No'} | "
                f"{comp['effect_size']} (d={comp['cohens_d']:.3f}) | "
                f"{comp['improvement']:.2f}% |"
            )
        
        report_lines.append("")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Statistical report saved to: {output_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Statistical analysis of experiment results')
    parser.add_argument('--results-file', type=str, required=True,
                       help='JSON file with experiment results')
    parser.add_argument('--output', type=str, default='statistical_analysis.md',
                       help='Output report file')
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    # Analyze
    analysis = analyze_all_methods(results)
    
    # Generate report
    generate_statistical_report(analysis, args.output)
    
    # Print summary
    print("\nStatistical Analysis Summary:")
    print("="*60)
    for metric, comparisons in analysis.items():
        print(f"\n{metric.upper()}:")
        for comp_name, comp in comparisons.items():
            if comp['significant']:
                print(f"  {comp_name}: Significant (p={comp['p_value']:.4f}, "
                      f"effect={comp['effect_size']}, improvement={comp['improvement']:.2f}%)")


if __name__ == "__main__":
    main()


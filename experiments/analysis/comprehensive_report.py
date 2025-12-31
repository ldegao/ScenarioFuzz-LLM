#!/usr/bin/env python3
"""
Comprehensive analysis report generator.

This script integrates all analysis results (basic comparison, statistical tests,
correlation analysis, time series, efficiency, and scenario analysis) into a
single comprehensive report.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from .compare_similarity_methods import METHOD_NAMES, METRIC_NAMES


def load_json_report(report_path: Path) -> Optional[Dict]:
    """Load a JSON report file."""
    if not report_path.exists():
        return None
    
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to load {report_path}: {e}")
        return None


def generate_comprehensive_report(base_dir: Path, output_dir: Path):
    """
    Generate comprehensive analysis report integrating all analysis results.
    
    Args:
        base_dir: Base directory containing all analysis results
        output_dir: Directory to save comprehensive report
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all analysis results
    comparison_report = load_json_report(base_dir / "comparison_report.json")
    statistical_report = load_json_report(base_dir / "statistical_analysis" / "statistical_analysis.json")
    correlation_report = load_json_report(base_dir / "correlation_analysis" / "correlation_analysis.json")
    timeseries_report = load_json_report(base_dir / "timeseries_analysis" / "timeseries_analysis.json")
    efficiency_report = load_json_report(base_dir / "efficiency_analysis" / "efficiency_analysis.json")
    scenario_report = load_json_report(base_dir / "scenario_analysis" / "scenario_analysis.json")
    
    # Generate comprehensive Markdown report
    md_path = output_dir / "comprehensive_analysis_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        # Header
        f.write("# Comprehensive Similarity Method Analysis Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This comprehensive report integrates all analysis results comparing similarity scoring methods.\n\n")
        
        # Table of Contents
        f.write("## Table of Contents\n\n")
        f.write("1. [Executive Summary](#executive-summary)\n")
        f.write("2. [Method Comparison](#method-comparison)\n")
        f.write("3. [Statistical Significance](#statistical-significance)\n")
        f.write("4. [Metric Correlations](#metric-correlations)\n")
        f.write("5. [Time Series Analysis](#time-series-analysis)\n")
        f.write("6. [Efficiency Analysis](#efficiency-analysis)\n")
        f.write("7. [Scenario-Level Analysis](#scenario-level-analysis)\n")
        f.write("8. [Conclusions and Recommendations](#conclusions-and-recommendations)\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        if comparison_report:
            f.write("### Key Findings\n\n")
            f.write("The following methods were compared:\n")
            for method in ["answer2", "embedding", "feature", "hybrid"]:
                if method in comparison_report:
                    display_name = METHOD_NAMES.get(method, method)
                    f.write(f"- **{display_name}** ({method})\n")
            f.write("\n")
        f.write("\n")
        
        # Method Comparison
        f.write("## Method Comparison\n\n")
        if comparison_report:
            f.write("### Overall Performance Summary\n\n")
            f.write("| Method | PCE (mean±std) | BCE (mean±std) | DPE (mean±std) | CCE (mean±std) | Runs |\n")
            f.write("|--------|----------------|----------------|----------------|----------------|------|\n")
            
            for method in ["answer2", "embedding", "feature", "hybrid"]:
                if method not in comparison_report:
                    continue
                
                metrics = comparison_report[method]
                display_name = METHOD_NAMES[method]
                
                pc = metrics.get("pc", {})
                pec = metrics.get("pec", {})
                tcd = metrics.get("tcd", {})
                bcm = metrics.get("bcm", {})
                
                f.write(f"| {display_name} | "
                       f"{pc.get('mean', 0):.3f}±{pc.get('std', 0):.3f} | "
                       f"{pec.get('mean', 0):.3f}±{pec.get('std', 0):.3f} | "
                       f"{tcd.get('mean', 0):.3f}±{tcd.get('std', 0):.3f} | "
                       f"{bcm.get('mean', 0):.3f}±{bcm.get('std', 0):.3f} | "
                       f"{pc.get('count', 0)} |\n")
        else:
            f.write("*Comparison report not available.*\n")
        f.write("\n")
        
        # Statistical Significance
        f.write("## Statistical Significance\n\n")
        if statistical_report:
            f.write("### ANOVA Results\n\n")
            for metric in ["pc", "pec", "tcd", "bcm"]:
                if metric not in statistical_report:
                    continue
                
                metric_name = METRIC_NAMES[metric]
                anova = statistical_report[metric].get("anova", {})
                
                if "error" not in anova:
                    f.write(f"**{metric_name}**: F={anova.get('statistic', 0):.4f}, "
                           f"p={anova.get('pvalue', 1):.6f}, "
                           f"Significant: {'Yes' if anova.get('significant', False) else 'No'}\n\n")
        else:
            f.write("*Statistical analysis not available.*\n")
        f.write("\n")
        
        # Metric Correlations
        f.write("## Metric Correlations\n\n")
        if correlation_report:
            f.write("### Correlation Summary\n\n")
            f.write("| Metric Pair | Pearson r | Significant | Spearman r | Significant |\n")
            f.write("|-------------|-----------|------------|------------|-------------|\n")
            
            for pair_name, result in sorted(correlation_report.items()):
                if "error" in result:
                    continue
                
                metric1, metric2 = pair_name.split("_")
                f.write(f"| {METRIC_NAMES[metric1]} vs {METRIC_NAMES[metric2]} | "
                       f"{result.get('pearson_r', 0):.4f} | "
                       f"{'Yes' if result.get('pearson_significant', False) else 'No'} | "
                       f"{result.get('spearman_r', 0):.4f} | "
                       f"{'Yes' if result.get('spearman_significant', False) else 'No'} |\n")
        else:
            f.write("*Correlation analysis not available.*\n")
        f.write("\n")
        
        # Time Series Analysis
        f.write("## Time Series Analysis\n\n")
        if timeseries_report:
            f.write("### Trend Analysis\n\n")
            for method in ["answer2", "embedding", "feature", "hybrid"]:
                if method not in timeseries_report:
                    continue
                
                display_name = METHOD_NAMES.get(method, method)
                method_data = timeseries_report[method]
                
                f.write(f"#### {display_name}\n\n")
                for metric in ["pc", "pec", "tcd", "bcm"]:
                    if metric not in method_data:
                        continue
                    
                    metric_name = METRIC_NAMES[metric]
                    metric_info = method_data[metric]
                    
                    f.write(f"- **{metric_name}**: {metric_info.get('trend', 'unknown')} "
                           f"(initial: {metric_info.get('initial', 0):.6f}, "
                           f"final: {metric_info.get('final', 0):.6f})\n")
                f.write("\n")
        else:
            f.write("*Time series analysis not available.*\n")
        f.write("\n")
        
        # Efficiency Analysis
        f.write("## Efficiency Analysis\n\n")
        if efficiency_report:
            eff_data = efficiency_report.get("efficiency_data", {})
            eff_metrics = efficiency_report.get("efficiency_metrics", {})
            
            f.write("### Resource Usage\n\n")
            f.write("| Method | Total Cost (USD) | Total Tokens | Cost per Scenario |\n")
            f.write("|--------|------------------|--------------|-------------------|\n")
            
            for method in ["answer2", "embedding", "feature", "hybrid"]:
                if method not in eff_data:
                    continue
                
                display_name = METHOD_NAMES.get(method, method)
                data = eff_data[method]
                metrics = eff_metrics.get(method, {})
                
                f.write(f"| {display_name} | "
                       f"${data['total_cost']['total']:.4f} | "
                       f"{data['total_tokens']['total']:,.0f} | "
                       f"${metrics.get('cost_per_scenario', 0):.6f} |\n")
            
            f.write("\n### Efficiency Metrics\n\n")
            f.write("| Method | Combined Efficiency |\n")
            f.write("|--------|---------------------|\n")
            
            for method in ["answer2", "embedding", "feature", "hybrid"]:
                if method not in eff_metrics:
                    continue
                
                display_name = METHOD_NAMES.get(method, method)
                metrics = eff_metrics[method]
                
                f.write(f"| {display_name} | {metrics.get('combined_efficiency', 0):.4f} |\n")
        else:
            f.write("*Efficiency analysis not available.*\n")
        f.write("\n")
        
        # Scenario-Level Analysis
        f.write("## Scenario-Level Analysis\n\n")
        if scenario_report:
            f.write("### Distribution Summary\n\n")
            for method in ["answer2", "embedding", "feature", "hybrid"]:
                if method not in scenario_report:
                    continue
                
                display_name = METHOD_NAMES.get(method, method)
                method_data = scenario_report[method]
                dist = method_data.get("distribution", {})
                
                f.write(f"**{display_name}**: {dist.get('total_scenarios', 0)} scenarios, "
                       f"{dist.get('unique_generations', 0)} generations\n\n")
        else:
            f.write("*Scenario analysis not available.*\n")
        f.write("\n")
        
        # Conclusions and Recommendations
        f.write("## Conclusions and Recommendations\n\n")
        f.write("### Summary\n\n")
        f.write("Based on the comprehensive analysis:\n\n")
        
        # Find best method for each metric
        if comparison_report:
            f.write("### Best Performing Methods by Metric\n\n")
            for metric in ["pc", "pec", "tcd", "bcm"]:
                metric_name = METRIC_NAMES[metric]
                best_method = None
                best_value = -float('inf')
                
                for method, metrics in comparison_report.items():
                    metric_data = metrics.get(metric, {})
                    mean_value = metric_data.get("mean", 0.0)
                    if mean_value > best_value:
                        best_value = mean_value
                        best_method = method
                
                if best_method:
                    display_name = METHOD_NAMES.get(best_method, best_method)
                    f.write(f"- **{metric_name}**: {display_name} (mean: {best_value:.6f})\n")
        
        f.write("\n### Recommendations\n\n")
        f.write("1. Consider the trade-offs between performance and efficiency when selecting a method.\n")
        f.write("2. Statistical significance tests help identify meaningful differences between methods.\n")
        f.write("3. Time series analysis reveals convergence patterns and stability.\n")
        f.write("4. Efficiency metrics help optimize resource usage.\n")
        f.write("\n")
        
        # References to detailed reports
        f.write("## Detailed Reports\n\n")
        f.write("For more detailed information, please refer to:\n\n")
        f.write("- [Basic Comparison Report](comparison_report.md)\n")
        f.write("- [Statistical Analysis](statistical_analysis/statistical_analysis.md)\n")
        f.write("- [Correlation Analysis](correlation_analysis/correlation_analysis.md)\n")
        f.write("- [Time Series Analysis](timeseries_analysis/timeseries_analysis.md)\n")
        f.write("- [Efficiency Analysis](efficiency_analysis/efficiency_analysis.md)\n")
        f.write("- [Scenario Analysis](scenario_analysis/scenario_analysis.md)\n\n")
    
    print(f"[ComprehensiveReport] Saved comprehensive report: {md_path}")
    
    # Generate summary JSON
    summary = {
        "generated_at": datetime.now().isoformat(),
        "comparison_available": comparison_report is not None,
        "statistical_available": statistical_report is not None,
        "correlation_available": correlation_report is not None,
        "timeseries_available": timeseries_report is not None,
        "efficiency_available": efficiency_report is not None,
        "scenario_available": scenario_report is not None
    }
    
    json_path = output_dir / "comprehensive_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[ComprehensiveReport] Saved summary: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive analysis report integrating all analysis results."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./reports/similarity_comparison",
        help="Base directory containing all analysis results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reports/similarity_comparison/comprehensive_analysis",
        help="Directory to save comprehensive report",
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"[ERROR] Base directory {base_dir} does not exist")
        return
    
    output_dir = Path(args.output_dir)
    generate_comprehensive_report(base_dir, output_dir)
    
    print(f"\n[ComprehensiveReport] Comprehensive report generation complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


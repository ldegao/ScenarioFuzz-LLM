#!/usr/bin/env python3
"""
Compare Methods Script
Compares different fuzzing methods (TM-Fuzzer, DriveFuzz, ScenarioFuzz-LLM, RAG-ScenarioFuzz)
"""

import sys
import os
import argparse
import json
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import ParameterCoverage, BehaviorCoverage, TrajectoryDiversity, BehaviorMatrix
from visualization import ReportGenerator, plot_parameter_coverage, plot_behavior_coverage, plot_trajectory_diversity, plot_comparison_radar


def load_scenarios_from_directory(directory: str) -> List:
    """
    Load scenarios from a directory (placeholder - actual implementation depends on data format)
    
    Args:
        directory: Directory containing scenario data
        
    Returns:
        List of Scenario objects
    """
    # Placeholder implementation
    # In practice, this would load scenarios from JSON files or database
    scenarios = []
    return scenarios


def calculate_metrics_for_method(method_name: str, scenarios: List) -> Dict[str, float]:
    """
    Calculate all metrics for a method
    
    Args:
        method_name: Name of the method
        scenarios: List of Scenario objects
        
    Returns:
        Dictionary of metric results
    """
    results = {}
    
    if len(scenarios) == 0:
        return {
            'pc': 0.0,
            'pec': 0.0,
            'tcd': 0.0,
            'bcm': 0.0
        }
    
    # Parameter Coverage (PC)
    pc_calculator = ParameterCoverage()
    results['pc'] = pc_calculator.calculate_coverage(scenarios)
    
    # Behavior Coverage (PEC)
    pec_calculator = BehaviorCoverage()
    results['pec'] = pec_calculator.calculate_coverage(scenarios)
    
    # Trajectory Diversity (TCD)
    tcd_calculator = TrajectoryDiversity()
    tcd_results = tcd_calculator.calculate_coverage(scenarios)
    results['tcd'] = tcd_results.get('diversity_score', 0.0)
    
    # Behavior Matrix Coverage (BCM)
    bcm_calculator = BehaviorMatrix()
    bcm_results = bcm_calculator.calculate_coverage(scenarios)
    results['bcm'] = bcm_results.get('coverage_ratio', 0.0)
    
    return results


def compare_methods(method_directories: Dict[str, str], output_dir: str):
    """
    Compare multiple methods
    
    Args:
        method_directories: Dictionary mapping method names to their data directories
        output_dir: Output directory for results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Method Comparison")
    print("=" * 60)
    
    all_results = {}
    
    for method_name, directory in method_directories.items():
        print(f"\n[Comparison] Processing {method_name}...")
        
        # Load scenarios
        scenarios = load_scenarios_from_directory(directory)
        print(f"[Comparison] Loaded {len(scenarios)} scenarios for {method_name}")
        
        # Calculate metrics
        results = calculate_metrics_for_method(method_name, scenarios)
        all_results[method_name] = results
        
        print(f"[Comparison] {method_name} results:")
        print(f"  PC:  {results['pc']:.4f}")
        print(f"  PEC: {results['pec']:.4f}")
        print(f"  TCD: {results['tcd']:.4f}")
        print(f"  BCM: {results['bcm']:.4f}")
    
    # Generate visualizations
    print("\n[Comparison] Generating visualizations...")
    
    # Parameter Coverage plot
    pc_scores = {method: results['pc'] for method, results in all_results.items()}
    plot_parameter_coverage(pc_scores, os.path.join(output_dir, "pc_comparison.png"))
    
    # Behavior Coverage plot
    pec_data = {method: {'coverage': results['pec'], 'num_classes': 0} 
                for method, results in all_results.items()}
    plot_behavior_coverage(pec_data, os.path.join(output_dir, "pec_comparison.png"))
    
    # Trajectory Diversity plot
    tcd_data = {method: {'diversity_score': results['tcd'], 'entropy': 0.0}
                for method, results in all_results.items()}
    plot_trajectory_diversity(tcd_data, os.path.join(output_dir, "tcd_comparison.png"))
    
    # Radar chart comparison
    plot_comparison_radar(all_results, os.path.join(output_dir, "radar_comparison.png"))
    
    # Generate report
    print("\n[Comparison] Generating report...")
    report_generator = ReportGenerator(output_dir=output_dir)
    report_path = report_generator.generate_markdown_report(
        experiment_name="Method Comparison",
        metrics_results=all_results,
        comparison_data=all_results
    )
    
    # Save JSON summary
    json_path = report_generator.generate_summary_json(all_results)
    
    print(f"\n[Comparison] Comparison completed!")
    print(f"[Comparison] Results saved to: {output_dir}")
    print(f"[Comparison] Report: {report_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Compare different fuzzing methods')
    parser.add_argument("--methods", type=str, nargs='+', required=True,
                        help="Method names (space-separated)")
    parser.add_argument("--directories", type=str, nargs='+', required=True,
                        help="Corresponding data directories (space-separated)")
    parser.add_argument("-o", "--output", default="./comparison_results", type=str,
                        help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    if len(args.methods) != len(args.directories):
        print("Error: Number of methods must match number of directories")
        return
    
    method_directories = dict(zip(args.methods, args.directories))
    
    compare_methods(method_directories, args.output)


if __name__ == "__main__":
    main()


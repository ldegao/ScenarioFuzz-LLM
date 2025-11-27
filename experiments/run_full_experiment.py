#!/usr/bin/env python3
"""
Complete Experiment Runner for RAG-ScenarioFuzz
Runs comprehensive comparison experiments with all baseline methods
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from fuzzer import main, init_env, set_args
from metrics import ParameterCoverage, BehaviorCoverage, TrajectoryDiversity, BehaviorMatrix
from visualization import ReportGenerator


class ExperimentRunner:
    """Experiment runner for comprehensive comparison"""
    
    def __init__(self, output_dir="./experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def run_method(self, method_name, num_scenarios=100, **kwargs):
        """
        Run a single method experiment
        
        Args:
            method_name: Name of the method
            num_scenarios: Number of scenarios to generate
            **kwargs: Additional configuration
        """
        print(f"\n{'='*60}")
        print(f"Running {method_name}")
        print(f"{'='*60}")
        
        # Create output directory for this method
        method_dir = self.output_dir / method_name
        method_dir.mkdir(exist_ok=True)
        
        # Configure arguments
        args = self._create_args(method_name, method_dir, num_scenarios, **kwargs)
        
        # Run experiment
        start_time = time.time()
        try:
            # Initialize environment
            conf, town, town_map, client, world, G = init_env(args)
            
            # Method-specific configuration
            if method_name == "RAG-ScenarioFuzz":
                conf.enable_rag = True
                conf.enable_rag_metrics = True
                conf.rag_k = kwargs.get('rag_k', 5)
            elif method_name == "ScenarioFuzz-LLM":
                conf.enable_rag = False
                conf.enable_rag_metrics = False
            else:
                # TM-Fuzzer (baseline method)
                # Note: DriveFuzz temporarily disabled
                conf.enable_rag = False
                conf.enable_rag_metrics = False
            
            # Run fuzzing
            main(args)
            
            elapsed_time = time.time() - start_time
            
            # Collect results
            method_results = self._collect_results(method_dir, elapsed_time)
            
            self.results[method_name] = method_results
            print(f"\n{method_name} completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error running {method_name}: {e}")
            import traceback
            traceback.print_exc()
            self.results[method_name] = {'error': str(e)}
    
    def _create_args(self, method_name, output_dir, num_scenarios, **kwargs):
        """Create argument parser for method"""
        parser = set_args()
        
        # Set default arguments
        args = parser.parse_args([
            '--out-dir', str(output_dir),
            '--target', kwargs.get('target', 'behavior'),
            '--max-mutations', str(kwargs.get('max_mutations', 5)),
            '--town', str(kwargs.get('town', 3)),
            '--timeout', str(kwargs.get('timeout', 60))
        ])
        
        return args
    
    def _collect_results(self, method_dir, execution_time):
        """Collect results from method output directory"""
        results = {
            'execution_time': execution_time,
            'scenarios_generated': 0,
            'defects_found': 0,
            'metrics': {}
        }
        
        # Count scenarios and defects
        queue_dir = method_dir / "queue"
        error_dir = method_dir / "errors"
        
        if queue_dir.exists():
            scenario_files = list(queue_dir.glob("*.json"))
            results['scenarios_generated'] = len(scenario_files)
        
        if error_dir.exists():
            error_files = list(error_dir.glob("*.json"))
            results['defects_found'] = len(error_files)
        
        # Calculate metrics if available
        # Note: This requires loading scenario objects, which may need adaptation
        # For now, return basic statistics
        
        return results
    
    def calculate_metrics(self, method_name, scenarios):
        """
        Calculate multi-dimensional metrics for a method
        
        Args:
            method_name: Name of the method
            scenarios: List of Scenario objects
        """
        if len(scenarios) == 0:
            return {}
        
        metrics = {}
        
        try:
            # Parameter Coverage (PC)
            pc = ParameterCoverage()
            metrics['pc'] = pc.calculate_coverage(scenarios)
            
            # Behavior Coverage (PEC)
            pec = BehaviorCoverage()
            metrics['pec'] = pec.calculate_coverage(scenarios)
            
            # Trajectory Diversity (TCD)
            tcd = TrajectoryDiversity()
            tcd_results = tcd.calculate_coverage(scenarios)
            metrics['tcd'] = tcd_results.get('diversity_score', 0.0)
            
            # Behavior Matrix Coverage (BCM)
            bcm = BehaviorMatrix()
            bcm_results = bcm.calculate_coverage(scenarios)
            metrics['bcm'] = bcm_results.get('coverage_ratio', 0.0)
            
        except Exception as e:
            print(f"Error calculating metrics for {method_name}: {e}")
            metrics = {}
        
        return metrics
    
    def generate_report(self):
        """Generate comprehensive experiment report"""
        report_generator = ReportGenerator(output_dir=str(self.output_dir))
        
        # Prepare metrics results
        metrics_results = {}
        for method_name, result in self.results.items():
            if 'metrics' in result:
                metrics_results[method_name] = result['metrics']
            else:
                metrics_results[method_name] = {}
        
        # Generate markdown report
        report_path = report_generator.generate_markdown_report(
            experiment_name="RAG-ScenarioFuzz Comprehensive Comparison",
            metrics_results=metrics_results,
            comparison_data=self.results
        )
        
        # Generate JSON summary
        json_path = report_generator.generate_summary_json(metrics_results)
        
        print(f"\nReport generated: {report_path}")
        print(f"Summary JSON: {json_path}")
        
        return report_path


def main_experiment():
    """Main experiment entry point"""
    parser = argparse.ArgumentParser(description='Run comprehensive RAG-ScenarioFuzz experiment')
    parser.add_argument('--methods', nargs='+', 
                       # Note: DriveFuzz temporarily disabled
                       default=['TM-Fuzzer', 'ScenarioFuzz-LLM', 'RAG-ScenarioFuzz'],
                       help='Methods to compare')
    parser.add_argument('--num-scenarios', type=int, default=100,
                       help='Number of scenarios per method')
    parser.add_argument('--output-dir', type=str, default='./experiment_results',
                       help='Output directory for results')
    parser.add_argument('--target', type=str, default='behavior',
                       choices=['behavior', 'autoware'],
                       help='Target ADS system')
    parser.add_argument('--rag-k', type=int, default=5,
                       help='Number of top-k scenarios for RAG')
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    # Run each method
    for method in args.methods:
        runner.run_method(
            method_name=method,
            num_scenarios=args.num_scenarios,
            target=args.target,
            rag_k=args.rag_k
        )
    
    # Generate report
    runner.generate_report()
    
    print("\n" + "="*60)
    print("Experiment completed!")
    print("="*60)


if __name__ == "__main__":
    main_experiment()


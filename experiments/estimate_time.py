#!/usr/bin/env python3
"""
Time Estimation Tool
Estimates experiment execution time without running experiments
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time_estimator import TimeEstimator


def main():
    """Main entry point for time estimation"""
    parser = argparse.ArgumentParser(
        description='Estimate experiment execution time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate time for 100 scenarios
  python estimate_time.py RAG-ScenarioFuzz --num-scenarios 100
  
  # Estimate time for all methods
  python estimate_time.py --all-methods --num-scenarios 100
  
  # Estimate time for timed experiment
  python estimate_time.py RAG-ScenarioFuzz --hours 2
        """
    )
    
    parser.add_argument('method', type=str, nargs='?',
                       # Note: DriveFuzz temporarily disabled
                       choices=['TM-Fuzzer', 'DriveFuzz', 'ScenarioFuzz-LLM', 'RAG-ScenarioFuzz'],
                       help='Method to estimate (Note: DriveFuzz temporarily disabled)')
    parser.add_argument('--all-methods', action='store_true',
                       help='Estimate for all methods (excluding DriveFuzz)')
    parser.add_argument('--num-scenarios', type=int, default=100,
                       help='Number of scenarios (default: 100)')
    parser.add_argument('--hours', type=float, default=None,
                       help='Duration in hours (alternative to num-scenarios)')
    
    args = parser.parse_args()
    
    if not args.method and not args.all_methods:
        parser.error("Either specify a method or use --all-methods")
    
    estimator = TimeEstimator()
    
    # Note: DriveFuzz temporarily disabled
    methods = ['TM-Fuzzer', 'ScenarioFuzz-LLM', 'RAG-ScenarioFuzz'] if args.all_methods else [args.method]
    
    print("\n" + "="*70)
    print("Experiment Time Estimation")
    print("="*70)
    
    total_time = 0
    for method in methods:
        if args.hours:
            # Estimate scenarios for given time
            avg_time = estimator.estimate_scenario_time(method)
            estimated_scenarios = int(args.hours * 3600 / avg_time)
            print(f"\n{method}:")
            print(f"  Duration: {args.hours} hours")
            print(f"  Estimated scenarios: ~{estimated_scenarios}")
            print(f"  Avg time per scenario: {avg_time:.2f} seconds")
        else:
            estimate = estimator.estimate_total_time(method, args.num_scenarios)
            total_time += estimate['total_seconds']
            print(f"\n{method}:")
            print(f"  Scenarios: {args.num_scenarios}")
            print(f"  Avg time per scenario: {estimate['avg_scenario_time']:.2f} seconds")
            print(f"  Total estimated time: {estimate['total_time_str']}")
            print(f"  Estimated completion: {estimate['estimated_completion']}")
    
    if args.all_methods and not args.hours:
        print(f"\n{'='*70}")
        print(f"Total time (sequential): {total_time/3600:.2f} hours")
        print(f"Total time (parallel, 2 workers): {max([estimator.estimate_total_time(m, args.num_scenarios)['total_seconds'] for m in methods])/3600:.2f} hours")
        print(f"{'='*70}")
    
    print()


if __name__ == "__main__":
    main()


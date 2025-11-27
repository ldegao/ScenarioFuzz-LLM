#!/usr/bin/env python3
"""
Batch Experiment Runner
Runs multiple methods in batch (sequential or parallel)
"""

import sys
import os
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_manager import ExperimentManager
from time_estimator import TimeEstimator


def run_single_method(method_name, num_scenarios, output_dir, **kwargs):
    """Run a single method (for parallel execution)"""
    manager = ExperimentManager(output_base_dir=output_dir)
    manager.run_quantitative_experiment(
        method_name=method_name,
        num_scenarios=num_scenarios,
        **kwargs
    )
    return method_name


def main():
    """Main entry point for batch experiments"""
    parser = argparse.ArgumentParser(
        description='Run batch experiments (multiple methods)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods sequentially with 100 scenarios each (DriveFuzz disabled)
  python run_batch.py --methods TM-Fuzzer ScenarioFuzz-LLM RAG-ScenarioFuzz --num-scenarios 100
  
  # Run in parallel (2 methods at a time)
  python run_batch.py --methods RAG-ScenarioFuzz ScenarioFuzz-LLM --num-scenarios 50 --parallel --max-workers 2
  
  # Run timed experiments
  python run_batch.py --methods RAG-ScenarioFuzz ScenarioFuzz-LLM --hours 2
        """
    )
    
    parser.add_argument('--methods', nargs='+',
                       choices=['TM-Fuzzer', 'DriveFuzz', 'ScenarioFuzz-LLM', 'RAG-ScenarioFuzz'],
                       # Note: DriveFuzz temporarily disabled
                       default=['TM-Fuzzer', 'ScenarioFuzz-LLM', 'RAG-ScenarioFuzz'],
                       help='Methods to run (default: TM-Fuzzer, ScenarioFuzz-LLM, RAG-ScenarioFuzz)')
    parser.add_argument('--num-scenarios', type=int, default=100,
                       help='Number of scenarios per method (default: 100)')
    parser.add_argument('--hours', type=float, default=None,
                       help='Duration in hours (for timed experiments, overrides num-scenarios)')
    parser.add_argument('--output-dir', type=str, default='./experiment_results',
                       help='Output directory for results (default: ./experiment_results)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run methods in parallel')
    parser.add_argument('--max-workers', type=int, default=2,
                       help='Maximum parallel workers (default: 2)')
    parser.add_argument('--target', type=str, default='behavior',
                       choices=['behavior', 'autoware'],
                       help='Target ADS system (default: behavior)')
    parser.add_argument('--town', type=int, default=3,
                       help='CARLA town number (default: 3)')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Scenario timeout in seconds (default: 60)')
    parser.add_argument('--rag-k', type=int, default=5,
                       help='RAG top-k for RAG-ScenarioFuzz (default: 5)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Estimate total time
    estimator = TimeEstimator()
    print("\n" + "="*60)
    print("Batch Experiment Time Estimation")
    print("="*60)
    
    total_time = 0
    for method in args.methods:
        if args.hours:
            estimate_seconds = args.hours * 3600
        else:
            estimate = estimator.estimate_total_time(method, args.num_scenarios)
            estimate_seconds = estimate['total_seconds']
            print(f"{method}: {estimate['total_time_str']} ({args.num_scenarios} scenarios)")
        
        total_time += estimate_seconds
    
    if args.parallel:
        # Parallel execution time is max of individual times
        parallel_time = max([estimator.estimate_total_time(m, args.num_scenarios)['total_seconds'] 
                            for m in args.methods]) if not args.hours else args.hours * 3600
        print(f"\nParallel execution: ~{parallel_time/3600:.2f} hours")
    else:
        print(f"\nSequential execution: ~{total_time/3600:.2f} hours")
    
    print("="*60 + "\n")
    
    # Create experiment manager
    manager = ExperimentManager(output_base_dir=args.output_dir)
    
    # Prepare common kwargs
    common_kwargs = {
        'target': args.target,
        'town': args.town,
        'timeout': args.timeout,
        'rag_k': args.rag_k,
        'debug': args.debug
    }
    
    start_time = time.time()
    
    if args.parallel:
        # Parallel execution
        print(f"Running {len(args.methods)} methods in parallel (max {args.max_workers} workers)...\n")
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for method in args.methods:
                if args.hours:
                    future = executor.submit(
                        manager.run_timed_experiment,
                        method_name=method,
                        duration_hours=args.hours,
                        **common_kwargs
                    )
                else:
                    future = executor.submit(
                        manager.run_quantitative_experiment,
                        method_name=method,
                        num_scenarios=args.num_scenarios,
                        **common_kwargs
                    )
                futures[future] = method
            
            # Wait for completion
            for future in as_completed(futures):
                method = futures[future]
                try:
                    future.result()
                    print(f"\n✓ {method} completed")
                except Exception as e:
                    print(f"\n✗ {method} failed: {e}")
    else:
        # Sequential execution
        print(f"Running {len(args.methods)} methods sequentially...\n")
        
        for i, method in enumerate(args.methods, 1):
            print(f"\n[{i}/{len(args.methods)}] Running {method}...")
            
            try:
                if args.hours:
                    manager.run_timed_experiment(
                        method_name=method,
                        duration_hours=args.hours,
                        **common_kwargs
                    )
                else:
                    manager.run_quantitative_experiment(
                        method_name=method,
                        num_scenarios=args.num_scenarios,
                        **common_kwargs
                    )
                print(f"✓ {method} completed")
            except Exception as e:
                print(f"✗ {method} failed: {e}")
                import traceback
                traceback.print_exc()
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Batch experiment completed in {elapsed_time/3600:.2f} hours")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


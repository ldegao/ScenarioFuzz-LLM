#!/usr/bin/env python3
"""
Timed Experiment Runner
Runs experiments with time control (N hours)
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_manager import ExperimentManager


def main():
    """Main entry point for timed experiments"""
    parser = argparse.ArgumentParser(
        description='Run timed experiment (N hours)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run RAG-ScenarioFuzz for 2 hours
  python run_timed.py RAG-ScenarioFuzz --hours 2
  
  # Run ScenarioFuzz-LLM for 1.5 hours
  python run_timed.py ScenarioFuzz-LLM --hours 1.5
  
  # Run with custom output directory
  python run_timed.py RAG-ScenarioFuzz --hours 3 --output-dir ./my_results
        """
    )
    
    parser.add_argument('method', type=str,
                       # Note: DriveFuzz temporarily disabled
                       choices=['TM-Fuzzer', 'DriveFuzz', 'ScenarioFuzz-LLM', 'RAG-ScenarioFuzz'],
                       help='Method to run (Note: DriveFuzz temporarily disabled)')
    parser.add_argument('--hours', type=float, required=True,
                       help='Duration in hours')
    parser.add_argument('--output-dir', type=str, default='./experiment_results',
                       help='Output directory for results (default: ./experiment_results)')
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
    parser.add_argument('--experiment-id', type=str, default=None,
                       help='Custom experiment ID (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Create experiment manager
    manager = ExperimentManager(output_base_dir=args.output_dir)
    
    # Run timed experiment
    manager.run_timed_experiment(
        method_name=args.method,
        duration_hours=args.hours,
        experiment_id=args.experiment_id,
        target=args.target,
        town=args.town,
        timeout=args.timeout,
        rag_k=args.rag_k,
        debug=args.debug
    )


if __name__ == "__main__":
    main()


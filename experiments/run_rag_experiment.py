#!/usr/bin/env python3
"""
Run RAG-ScenarioFuzz Experiment
Main script to run experiments with RAG-enhanced scenario generation
"""

import sys
import os
import argparse
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from fuzzer import main, init_env, set_args
from rag_module import RAGEngine
from metrics import ParameterCoverage, BehaviorCoverage, TrajectoryDiversity, BehaviorMatrix
from visualization import ReportGenerator, plot_parameter_coverage, plot_behavior_coverage, plot_trajectory_diversity


def run_rag_experiment(args):
    """
    Run RAG-enhanced fuzzing experiment
    
    Args:
        args: Command line arguments
    """
    print("=" * 60)
    print("RAG-ScenarioFuzz Experiment")
    print("=" * 60)
    
    # Initialize configuration
    conf, town, town_map, client, world, G = init_env(args)
    
    # Enable RAG and metrics
    conf.enable_rag = True
    conf.enable_rag_metrics = True
    conf.rag_k = args.rag_k if hasattr(args, 'rag_k') else 5
    
    # Set metrics output directory
    if conf.metrics_output_dir is None:
        conf.metrics_output_dir = os.path.join(conf.out_dir, "metrics")
        os.makedirs(conf.metrics_output_dir, exist_ok=True)
    
    print(f"[Experiment] RAG enabled: {conf.enable_rag}")
    print(f"[Experiment] Metrics enabled: {conf.enable_rag_metrics}")
    print(f"[Experiment] RAG top-k: {conf.rag_k}")
    print(f"[Experiment] Output directory: {conf.out_dir}")
    
    # Initialize RAG engine
    print("\n[Experiment] Initializing RAG engine...")
    rag_engine = RAGEngine(top_k=conf.rag_k)
    rag_engine.initialize(load_mock_data=True)
    print(f"[Experiment] RAG engine initialized with {rag_engine.knowledge_base.size()} scenarios")
    
    # Run fuzzing (this will call main() with modified config)
    print("\n[Experiment] Starting fuzzing...")
    try:
        # Note: This is a simplified version. In practice, you would modify main() 
        # to accept and use the RAG engine, or create a wrapper
        main(args)
    except KeyboardInterrupt:
        print("\n[Experiment] Experiment interrupted by user")
    except Exception as e:
        print(f"\n[Experiment] Error during fuzzing: {e}")
        import traceback
        traceback.print_exc()
    
    # Calculate aggregate metrics
    print("\n[Experiment] Calculating aggregate metrics...")
    # Note: In practice, you would collect all scenarios from the fuzzing run
    # and calculate metrics on the full set
    
    print("\n[Experiment] Experiment completed!")
    print(f"[Experiment] Results saved to: {conf.out_dir}")


def main_experiment():
    """Main entry point for experiment script"""
    parser = argparse.ArgumentParser(description='Run RAG-ScenarioFuzz experiment')
    
    # Add standard fuzzer arguments
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("-o", "--out-dir", default="./data/output_rag", type=str,
                        help="Directory to save fuzzing logs")
    parser.add_argument("-m", "--max-mutations", default=5, type=int,
                        help="Size of the mutated population per cycle")
    parser.add_argument("-u", "--sim-host", default="localhost", type=str,
                        help="Hostname of Carla simulation server")
    parser.add_argument("-p", "--sim-port", default=2000, type=int,
                        help="RPC port of Carla simulation server")
    parser.add_argument("-s", "--seed-dir", default="./data/seed", type=str,
                        help="Seed directory")
    parser.add_argument("-t", "--target", default="behavior", type=str,
                        help="Target autonomous driving system")
    parser.add_argument("--rag-k", default=5, type=int,
                        help="Number of top-k scenarios to retrieve in RAG")
    
    args = parser.parse_args()
    
    run_rag_experiment(args)


if __name__ == "__main__":
    main_experiment()


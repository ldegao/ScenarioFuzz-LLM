#!/usr/bin/env python3
"""
Unified runner for TM-Fuzzer baseline experiments.

Supports:
  - Quantitative mode (fixed number of scenarios)
  - Timed mode       (fixed number of hours)

Example usage:
  # Quantitative: 100 scenarios
  python -m experiments.runners.run_tmfuzzer --num-scenarios 100

  # Timed: 2 hours
  python -m experiments.runners.run_tmfuzzer --hours 2
"""

import argparse
from pathlib import Path

from experiments.core.experiment_manager import ExperimentManager
from experiments.core.utils import validate_output_directory


METHOD_NAME = "TM-Fuzzer"


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for TM-Fuzzer baseline."""
    parser = argparse.ArgumentParser(
        description="Run TM-Fuzzer baseline experiments (quantitative or timed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantitative: 100 scenarios on Autoware
  python -m experiments.runners.run_tmfuzzer --num-scenarios 100 --target autoware

  # Timed: 2 hours on Autoware
  python -m experiments.runners.run_tmfuzzer --hours 2 --target autoware
        """,
    )

    parser.add_argument(
        "--num-scenarios",
        type=int,
        help="Number of scenarios to generate (quantitative mode).",
    )
    parser.add_argument(
        "--hours",
        type=float,
        help="Duration in hours (timed mode).",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="./experiment_results",
        help="Root directory for experiment results (default: ./experiment_results).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="autoware",
        choices=["behavior", "autoware"],
        help="Target ADS system (default: autoware).",
    )
    parser.add_argument(
        "--town",
        type=int,
        default=3,
        help="CARLA town number (default: 3).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Scenario timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.4,
        help="Traffic density parameter (default: 0.4).",
    )
    parser.add_argument(
        "--rag-k",
        type=int,
        default=5,
        help="Top-k for RAG (kept for CLI consistency; not used by TM-Fuzzer).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (kept for CLI consistency).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Optional custom experiment ID (default: auto-generated).",
    )

    return parser


def main() -> None:
    """Main entry point for TM-Fuzzer baseline experiments."""
    parser = build_parser()
    args = parser.parse_args()

    if args.num_scenarios is None and args.hours is None:
        parser.error("You must specify at least one of --num-scenarios or --hours.")
    
    # Check if both are specified (not allowed for TM-Fuzzer)
    if args.num_scenarios is not None and args.hours is not None:
        parser.error("--num-scenarios and --hours cannot be used together. Please specify only one of them.")

    # Input validation
    if args.num_scenarios is not None:
        if args.num_scenarios <= 0:
            parser.error(f"--num-scenarios must be positive, got {args.num_scenarios}")
        if args.num_scenarios > 1000000:  # Reasonable upper limit
            parser.error(f"--num-scenarios too large: {args.num_scenarios} (max: 1000000)")
    
    if args.hours is not None:
        if args.hours <= 0:
            parser.error(f"--hours must be positive, got {args.hours}")
        if args.hours > 720:  # 30 days max
            parser.error(f"--hours too large: {args.hours} (max: 720 hours = 30 days)")
    
    # Validate output directory
    validate_output_directory(args.output_root, parser)

    manager = ExperimentManager(output_base_dir=args.output_root)

    common_kwargs = dict(
        experiment_id=args.experiment_id,
        target=args.target,
        town=args.town,
        timeout=args.timeout,
        density=args.density,
        rag_k=args.rag_k,
        debug=args.debug,
    )

    # Only scenario count specified -> use quantitative TM-Fuzzer wrapper
    if args.num_scenarios is not None and args.hours is None:
        manager.run_quantitative_experiment(
            method_name=METHOD_NAME,
            num_scenarios=args.num_scenarios,
            **common_kwargs,
        )
    # Only time specified -> timed TM-Fuzzer wrapper
    elif args.num_scenarios is None and args.hours is not None:
        manager.run_timed_experiment(
            method_name=METHOD_NAME,
            duration_hours=args.hours,
            **common_kwargs,
        )


if __name__ == "__main__":
    main()



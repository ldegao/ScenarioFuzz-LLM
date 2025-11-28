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

from experiments.core.experiment_manager import ExperimentManager


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
    # Time specified (with or without num-scenarios) -> timed TM-Fuzzer wrapper.
    # When both are provided, TM-Fuzzer will still be controlled by time only,
    # since script/test.py does not support an explicit scenario bound.
    else:
        manager.run_timed_experiment(
            method_name=METHOD_NAME,
            duration_hours=args.hours,
            **common_kwargs,
        )


if __name__ == "__main__":
    main()



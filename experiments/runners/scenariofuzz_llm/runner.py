#!/usr/bin/env python3
"""
Unified runner for ScenarioFuzz-LLM experiments.

Supports:
  - Quantitative mode (fixed number of scenarios)
  - Timed mode       (fixed number of hours)

Example usage:
  # Quantitative: 100 scenarios
  python -m experiments.runners.run_scenariofuzz_llm --num-scenarios 100

  # Timed: 2 hours
  python -m experiments.runners.run_scenariofuzz_llm --hours 2
"""

import argparse
from pathlib import Path

from experiments.core.experiment_manager import ExperimentManager
from experiments.core.utils import validate_output_directory


METHOD_NAME = "ScenarioFuzz-LLM"


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for ScenarioFuzz-LLM."""
    parser = argparse.ArgumentParser(
        description="Run ScenarioFuzz-LLM experiments (quantitative or timed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantitative: 100 scenarios (only scenario limit)
  python -m experiments.runners.run_scenariofuzz_llm --num-scenarios 100

  # Timed: 2 hours (only time limit)
  python -m experiments.runners.run_scenariofuzz_llm --hours 2

  # Both limits: stop when either is reached
  python -m experiments.runners.run_scenariofuzz_llm --num-scenarios 100 --hours 2

  # Continue existing experiment: generate 50 more scenarios
  python -m experiments.runners.run_scenariofuzz_llm \\
      --continue-experiment ScenarioFuzz-LLM_20251203_210023 \\
      --continue-scenarios 50

  # Custom output root and target
  python -m experiments.runners.run_scenariofuzz_llm \\
      --num-scenarios 1000 \\
      --output-root ./experiment_results \\
      --target behavior
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
        default="behavior",
        choices=["behavior", "autoware"],
        help="Target ADS system (default: behavior).",
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
        help="Top-k for RAG (kept for CLI consistency, not used for this method).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Optional custom experiment ID (default: auto-generated).",
    )
    parser.add_argument(
        "--continue-experiment",
        type=str,
        default=None,
        help="Continue an existing experiment by experiment ID. Use with --num-scenarios to specify how many more scenarios to generate.",
    )
    parser.add_argument(
        "--continue-scenarios",
        type=int,
        default=None,
        help="Number of additional scenarios to generate when continuing an experiment (requires --continue-experiment).",
    )

    return parser


def main() -> None:
    """Main entry point for ScenarioFuzz-LLM experiments."""
    parser = build_parser()
    args = parser.parse_args()

    # Check if continuing an experiment
    if args.continue_experiment:
        if args.continue_scenarios is None or args.continue_scenarios <= 0:
            parser.error("--continue-experiment requires --continue-scenarios with a positive number.")
        if args.num_scenarios is not None or args.hours is not None:
            parser.error("--continue-experiment cannot be used with --num-scenarios or --hours. Use --continue-scenarios instead.")
    
    if args.num_scenarios is None and args.hours is None and not args.continue_experiment:
        parser.error("You must specify at least one of --num-scenarios, --hours, or --continue-experiment.")

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

    # Continue existing experiment
    if args.continue_experiment:
        manager.continue_experiment(
            method_name=METHOD_NAME,
            experiment_id=args.continue_experiment,
            additional_scenarios=args.continue_scenarios,
            **common_kwargs,
        )
    # Only scenario count specified -> pure quantitative mode
    elif args.num_scenarios is not None and args.hours is None:
        manager.run_quantitative_experiment(
            method_name=METHOD_NAME,
            num_scenarios=args.num_scenarios,
            **common_kwargs,
        )
    # Only time specified -> pure timed mode
    elif args.num_scenarios is None and args.hours is not None:
        manager.run_timed_experiment(
            method_name=METHOD_NAME,
            duration_hours=args.hours,
            **common_kwargs,
        )
    # Both specified -> use both constraints:
    #   - scenario limit via max_scenarios
    #   - time limit via duration_hours
    else:
        bounded_kwargs = dict(common_kwargs)
        bounded_kwargs["max_scenarios"] = args.num_scenarios
        manager.run_timed_experiment(
            method_name=METHOD_NAME,
            duration_hours=args.hours,
            **bounded_kwargs,
        )


if __name__ == "__main__":
    main()



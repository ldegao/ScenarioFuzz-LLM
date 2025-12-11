#!/usr/bin/env python3
"""
Unified runner for similarity scoring method comparison experiments.

Supports:
  - Quantitative mode (fixed number of scenarios)
  - Timed mode       (fixed number of hours)
  - Comparison of four similarity scoring methods: answer2, embedding, feature, hybrid

Example usage:
  # Quantitative: 100 scenarios with embedding similarity
  python -m experiments.runners.run_similarity_comparison --num-scenarios 100 --similarity-method embedding

  # Timed: 2 hours with hybrid similarity
  python -m experiments.runners.run_similarity_comparison --hours 2 --similarity-method hybrid
"""

import argparse
from pathlib import Path

from experiments.core.experiment_manager import ExperimentManager
from experiments.core.utils import validate_output_directory


METHOD_NAME = "SimilarityComparison"


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for similarity comparison experiments."""
    parser = argparse.ArgumentParser(
        description="Run similarity scoring method comparison experiments (quantitative or timed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantitative: 100 scenarios with embedding similarity
  python -m experiments.runners.run_similarity_comparison --num-scenarios 100 --similarity-method embedding

  # Timed: 2 hours with hybrid similarity
  python -m experiments.runners.run_similarity_comparison --hours 2 --similarity-method hybrid

  # Both limits: stop when either is reached
  python -m experiments.runners.run_similarity_comparison --num-scenarios 1000 --hours 2 --similarity-method feature

  # Custom output root and target
  python -m experiments.runners.run_similarity_comparison \\
      --num-scenarios 1000 \\
      --similarity-method feature \\
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
        "--similarity-method",
        type=str,
        required=True,
        choices=["answer2", "embedding", "feature", "hybrid"],
        help="Similarity scoring method to use.",
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
        help="Top-k for RAG retrieval (default: 5).",
    )
    parser.add_argument(
        "--hybrid-embedding-weight",
        type=float,
        default=0.6,
        help="Weight for embedding similarity in hybrid method (default: 0.6).",
    )
    parser.add_argument(
        "--feature-position-weight",
        type=float,
        default=0.3,
        help="Weight for position similarity in feature method (default: 0.3).",
    )
    parser.add_argument(
        "--feature-speed-weight",
        type=float,
        default=0.3,
        help="Weight for speed similarity in feature method (default: 0.3).",
    )
    parser.add_argument(
        "--feature-angular-accel-weight",
        type=float,
        default=0.2,
        help="Weight for angular acceleration similarity in feature method (default: 0.2).",
    )
    parser.add_argument(
        "--feature-relative-position-weight",
        type=float,
        default=0.2,
        help="Weight for relative position similarity in feature method (default: 0.2).",
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
        help="Optional custom experiment ID (default: auto-generated with similarity method suffix).",
    )

    return parser


def main() -> None:
    """Main entry point for similarity comparison experiments."""
    parser = build_parser()
    args = parser.parse_args()

    if args.num_scenarios is None and args.hours is None:
        parser.error("You must specify at least one of --num-scenarios or --hours.")

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
    
    # Validate hybrid embedding weight
    if args.similarity_method == "hybrid":
        if not 0.0 <= args.hybrid_embedding_weight <= 1.0:
            parser.error(f"--hybrid-embedding-weight must be between 0.0 and 1.0, got {args.hybrid_embedding_weight}")
    
    # Validate feature method weights sum
    if args.similarity_method == "feature":
        weight_sum = (args.feature_position_weight + args.feature_speed_weight + 
                     args.feature_angular_accel_weight + args.feature_relative_position_weight)
        if abs(weight_sum - 1.0) > 0.01:  # Allow small floating point error
            parser.error(
                f"Feature method weights must sum to approximately 1.0, got {weight_sum:.4f}. "
                f"Current weights: position={args.feature_position_weight}, "
                f"speed={args.feature_speed_weight}, "
                f"angular_accel={args.feature_angular_accel_weight}, "
                f"relative_position={args.feature_relative_position_weight}"
            )
    
    # Validate output directory
    validate_output_directory(args.output_root, parser)

    manager = ExperimentManager(output_base_dir=args.output_root)

    # Generate experiment ID with similarity method suffix if not provided
    experiment_id = args.experiment_id
    if experiment_id is None:
        # Will be auto-generated by ExperimentManager with timestamp
        # The similarity method will be added as a suffix in the method-specific config
        pass

    common_kwargs = dict(
        experiment_id=experiment_id,
        target=args.target,
        town=args.town,
        timeout=args.timeout,
        density=args.density,
        rag_k=args.rag_k,
        debug=args.debug,
        # Similarity scoring method configuration
        similarity_scoring_method=args.similarity_method,
        hybrid_embedding_weight=args.hybrid_embedding_weight,
        feature_position_weight=args.feature_position_weight,
        feature_speed_weight=args.feature_speed_weight,
        feature_angular_accel_weight=args.feature_angular_accel_weight,
        feature_relative_position_weight=args.feature_relative_position_weight,
    )

    # Only scenario count specified -> pure quantitative mode
    if args.num_scenarios is not None and args.hours is None:
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


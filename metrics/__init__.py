"""
Metrics Module for ScenarioFuzz-LLM
Provides multi-dimensional coverage evaluation metrics (entropy-based):
- PCE (Parameter Configuration Entropy): Hartley entropy over parameter grid
- BCE (Behavior Category Entropy): Shannon entropy over ISO 34502 classes
- DPE (Driving Pattern Entropy): Normalized Shannon entropy over trajectory clusters
- CCE (Combination Coverage Entropy): Hartley entropy over behavior combinations
"""

# Import standardized metrics (preferred names) and backward-compatible aliases
from .parameter_coverage import (
    ParameterConfigurationEntropy,
    ParameterCoverage,
    BehaviorParameterCoverage,
)
from .driving_behavior_class_coverage import (
    BehaviorCategoryEntropy,
    BehaviorCoverage,
    DrivingBehaviorClassCoverage,
)
from .trajectory_diversity import (
    DrivingPatternEntropy,
    DrivingPatternDiversity,
    TrajectoryDiversity,
)
from .behavior_matrix import (
    CombinationCoverageEntropy,
    BehaviorMatrix,
)
from .behavior_parameters import BehaviorParameterExtractor

# Export both new names and backward-compatible aliases
__all__ = [
    # Preferred entropy-based names
    'ParameterConfigurationEntropy',
    'BehaviorCategoryEntropy',
    'DrivingPatternEntropy',
    'CombinationCoverageEntropy',
    # Backward-compatible aliases
    'ParameterCoverage',
    'BehaviorParameterCoverage',
    'BehaviorCoverage',
    'DrivingBehaviorClassCoverage',
    'DrivingPatternDiversity',
    'TrajectoryDiversity',
    'BehaviorMatrix',
    # Utilities
    'BehaviorParameterExtractor'
]


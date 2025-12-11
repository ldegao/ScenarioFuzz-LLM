"""
Metrics Module for ScenarioFuzz-LLM
Provides multi-dimensional coverage evaluation metrics

All metrics are now based on industry standards and regulations:
- BPC (Behavior Parameter Coverage): Vehicle behavior parameters (ISO, UNECE, EuroNCAP)
- DBCC (Driving Behavior Class Coverage): ISO 34502 behavior taxonomy
- DPD (Driving Pattern Diversity): Fréchet distance with adaptive clustering
- BCM (Behavior Matrix Coverage): Standard thresholds from regulations
"""

# Import new standardized metrics
from .parameter_coverage import BehaviorParameterCoverage, ParameterCoverage
from .driving_behavior_class_coverage import DrivingBehaviorClassCoverage, BehaviorCoverage
from .trajectory_diversity import DrivingPatternDiversity, TrajectoryDiversity
from .behavior_matrix import BehaviorMatrix
from .behavior_parameters import BehaviorParameterExtractor

# Export both new names and backward-compatible aliases
__all__ = [
    # New standardized names
    'BehaviorParameterCoverage',
    'DrivingBehaviorClassCoverage',
    'DrivingPatternDiversity',
    # Backward-compatible aliases
    'ParameterCoverage',
    'BehaviorCoverage',
    'TrajectoryDiversity',
    # Existing
    'BehaviorMatrix',
    'BehaviorParameterExtractor'
]


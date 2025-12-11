"""
Behavior Parameter Coverage (BPC) Metric
Evaluates coverage of vehicle behavior parameter space combinations

Replaces the original PC metric which relied on subjective weather/NPC parameters.
BPC focuses entirely on vehicle behavior parameters based on industry standards:
- ISO 15622, UNECE Reg.79: Longitudinal acceleration
- ISO 3888-1/2: Lateral acceleration
- UNECE braking test: Jerk
- ISO 7401: Steering/yaw rate
- EuroNCAP AEB: TTC (Time-to-Collision)
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario import Scenario
from metrics.behavior_parameters import BehaviorParameterExtractor


class BehaviorParameterCoverage:
    """
    Calculates behavior parameter space coverage metric
    BPC = |covered behavior parameter combinations| / |theoretical maximum combinations|
    
    All parameters are based on industry standards and use standardized binning.
    """
    
    def __init__(self):
        """Initialize behavior parameter coverage calculator"""
        self.parameter_grid = self._define_parameter_grid()
        self.covered_combinations: Set[Tuple] = set()
        self.param_extractor = BehaviorParameterExtractor()
    
    def _define_parameter_grid(self) -> Dict[str, List]:
        """
        Define the behavior parameter space grid based on industry standards
        
        Returns:
            Dictionary mapping parameter names to their discrete bin values
        """
        return {
            # Longitudinal acceleration: [-6, +4] m/s², bin size 1 m/s²
            # Source: ISO 15622 / UNECE Reg.79
            'a_long': list(range(-6, 5)),  # -6 to +4 m/s², 11 bins
            
            # Lateral acceleration: [-4, +4] m/s², bin size 1 m/s²
            # Source: ISO 3888-1/2
            'a_lat': list(range(-4, 5)),  # -4 to +4 m/s², 9 bins
            
            # Longitudinal jerk: [-3, +3] m/s³, bin size 1 m/s³
            # Source: UNECE braking test
            'jerk_long': list(range(-3, 4)),  # -3 to +3 m/s³, 7 bins
            
            # Lateral jerk: [-2, +2] m/s³, bin size 1 m/s³
            # Source: ISO lane-change tests
            'jerk_lat': list(range(-2, 3)),  # -2 to +2 m/s³, 5 bins
            
            # Yaw rate: 0-200 deg/s, bin size 20 deg/s
            # Source: ISO 7401 (steady-state steering)
            'yaw_rate': list(range(0, 201, 20)),  # 0, 20, 40, ..., 200 deg/s, 11 bins
            
            # TTC: 0-3 seconds, bin size 0.5 seconds
            # Source: EuroNCAP AEB standard
            'ttc': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # 7 bins
        }
    
    def _discretize_value(self, value: float, bins: List[float]) -> float:
        """
        Discretize a continuous value to the nearest bin
        
        Args:
            value: Continuous value to discretize
            bins: List of bin values (must be sorted)
            
        Returns:
            Nearest bin value
        """
        if not bins:
            return value
        
        # For numeric bins, find nearest
        if isinstance(bins[0], (int, float)):
            return min(bins, key=lambda x: abs(x - value))
        
        return value
    
    def _extract_parameter_combination(self, scenario: Scenario) -> Optional[Tuple]:
        """
        Extract behavior parameter combination from a scenario
        
        Args:
            scenario: Scenario object
            
        Returns:
            Tuple representing the behavior parameter combination, or None if insufficient data
        """
        if not hasattr(scenario, 'state') or not scenario.state:
            return None
        
        # Extract all behavior parameters
        params = self.param_extractor.extract_all_parameters(scenario.state)
        
        # Discretize each parameter to its bin
        combination = []
        
        # Longitudinal acceleration: use max absolute value
        a_long = params['a_long']
        if a_long is not None and len(a_long) > 0:
            max_a_long = np.max(np.abs(a_long))
            # Clamp to valid range
            max_a_long = np.clip(max_a_long, -6, 4)
            a_long_bin = self._discretize_value(max_a_long, self.parameter_grid['a_long'])
            combination.append(a_long_bin)
        else:
            return None  # Required parameter missing
        
        # Lateral acceleration: use max absolute value
        a_lat = params['a_lat']
        if a_lat is not None and len(a_lat) > 0:
            max_a_lat = np.max(np.abs(a_lat))
            max_a_lat = np.clip(max_a_lat, -4, 4)
            a_lat_bin = self._discretize_value(max_a_lat, self.parameter_grid['a_lat'])
            combination.append(a_lat_bin)
        else:
            # Use default if not available
            combination.append(0.0)
        
        # Longitudinal jerk: use max absolute value
        jerk_long = params['jerk_long']
        if jerk_long is not None and len(jerk_long) > 0:
            max_jerk_long = np.max(np.abs(jerk_long))
            max_jerk_long = np.clip(max_jerk_long, -3, 3)
            jerk_long_bin = self._discretize_value(max_jerk_long, self.parameter_grid['jerk_long'])
            combination.append(jerk_long_bin)
        else:
            combination.append(0.0)
        
        # Lateral jerk: use max absolute value
        jerk_lat = params['jerk_lat']
        if jerk_lat is not None and len(jerk_lat) > 0:
            max_jerk_lat = np.max(np.abs(jerk_lat))
            max_jerk_lat = np.clip(max_jerk_lat, -2, 2)
            jerk_lat_bin = self._discretize_value(max_jerk_lat, self.parameter_grid['jerk_lat'])
            combination.append(jerk_lat_bin)
        else:
            combination.append(0.0)
        
        # Yaw rate: use max absolute value
        yaw_rate = params['yaw_rate']
        if yaw_rate is not None and len(yaw_rate) > 0:
            max_yaw_rate = np.max(np.abs(yaw_rate))
            max_yaw_rate = np.clip(max_yaw_rate, 0, 200)
            yaw_rate_bin = self._discretize_value(max_yaw_rate, self.parameter_grid['yaw_rate'])
            combination.append(yaw_rate_bin)
        else:
            combination.append(0.0)
        
        # TTC: use minimum value
        ttc = params['ttc']
        if ttc is not None:
            ttc_clamped = np.clip(ttc, 0.0, 3.0)
            ttc_bin = self._discretize_value(ttc_clamped, self.parameter_grid['ttc'])
            combination.append(ttc_bin)
        else:
            # Use maximum TTC if not available (no risk)
            combination.append(3.0)
        
        return tuple(combination)
    
    def calculate_coverage(self, scenarios: List[Scenario]) -> float:
        """
        Calculate behavior parameter coverage for a list of scenarios
        
        Uses logarithmic normalization to handle the large parameter space
        and avoid numerical precision issues with very small coverage ratios.
        
        Normalization formula: coverage = log(1 + covered) / log(1 + max_combinations)
        This ensures:
        - When covered = 0: coverage = 0
        - When covered = max: coverage ≈ 1 (asymptotically)
        - Avoids numerical precision issues (e.g., 500/266805 ≈ 0.0019 becomes more meaningful)
        - Provides better discrimination for small coverage values
        
        Args:
            scenarios: List of Scenario objects
            
        Returns:
            Coverage ratio between 0 and 1 (logarithmically normalized)
        """
        # Reset coverage tracking
        self.covered_combinations.clear()
        
        # Extract all parameter combinations
        for scenario in scenarios:
            combination = self._extract_parameter_combination(scenario)
            if combination is not None:
                self.covered_combinations.add(combination)
        
        covered_count = len(self.covered_combinations)
        
        if covered_count == 0:
            return 0.0
        
        # Calculate theoretical maximum combinations
        grid_sizes = [len(values) for values in self.parameter_grid.values()]
        max_combinations = np.prod(grid_sizes)
        
        # Use logarithmic normalization to handle large parameter space
        # Formula: coverage = log(1 + covered) / log(1 + max_combinations)
        # This ensures:
        # - When covered = 0: coverage = 0
        # - When covered = max: coverage ≈ 1 (asymptotically)
        # - Avoids numerical precision issues with very small ratios
        # - Provides better discrimination for small coverage values
        
        if max_combinations > 0:
            # Logarithmic normalization to handle large parameter space
            # This avoids numerical precision issues with very small ratios
            log_coverage = np.log1p(covered_count) / np.log1p(max_combinations)
            return float(log_coverage)
        
        return 0.0
    
    def calculate_coverage_detailed(self, scenarios: List[Scenario]) -> Dict[str, float]:
        """
        Calculate behavior parameter coverage with detailed information
        
        Args:
            scenarios: List of Scenario objects
            
        Returns:
            Dictionary containing:
            - coverage: Logarithmically normalized coverage (0-1)
            - covered_count: Number of unique combinations covered
            - max_combinations: Theoretical maximum combinations
            - linear_coverage: Linear coverage ratio (for reference)
        """
        # Reset coverage tracking
        self.covered_combinations.clear()
        
        # Extract all parameter combinations
        for scenario in scenarios:
            combination = self._extract_parameter_combination(scenario)
            if combination is not None:
                self.covered_combinations.add(combination)
        
        covered_count = len(self.covered_combinations)
        
        # Calculate theoretical maximum combinations
        grid_sizes = [len(values) for values in self.parameter_grid.values()]
        max_combinations = np.prod(grid_sizes)
        
        # Calculate both linear and logarithmic coverage
        if max_combinations > 0:
            linear_coverage = covered_count / max_combinations
            log_coverage = np.log1p(covered_count) / np.log1p(max_combinations)
        else:
            linear_coverage = 0.0
            log_coverage = 0.0
        
        return {
            'coverage': float(log_coverage),  # Normalized coverage (primary metric)
            'covered_count': int(covered_count),
            'max_combinations': int(max_combinations),
            'linear_coverage': float(linear_coverage)  # For reference/debugging
        }
    
    def get_parameter_grid(self) -> Dict[str, List]:
        """
        Get the behavior parameter space grid definition
        
        Returns:
            Dictionary mapping parameter names to their discrete bin values
        """
        return self.parameter_grid.copy()
    
    def get_covered_combinations(self) -> Set[Tuple]:
        """
        Get the set of covered parameter combinations
        
        Returns:
            Set of tuples representing covered combinations
        """
        return self.covered_combinations.copy()
    
    def reset(self):
        """Reset the coverage tracking"""
        self.covered_combinations.clear()


# Backward compatibility alias
ParameterCoverage = BehaviorParameterCoverage

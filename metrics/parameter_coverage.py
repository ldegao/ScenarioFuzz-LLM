"""
Parameter Configuration Entropy (PCE)
Evaluates coverage of vehicle behavior parameter space combinations using
Hartley entropy (support size) normalization:

    PCE = log(x) / log(M)

where x is the number of observed unique parameter configurations and
M is the theoretical maximum configuration count from the parameter grid.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario import Scenario
from metrics.behavior_parameters import BehaviorParameterExtractor


class ParameterConfigurationEntropy:
    """
    Calculates Hartley-entropy-based parameter configuration diversity.
    PCE = log(|G_obs|) / log(|G|) where G is the discretized parameter grid.
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
            # Longitudinal acceleration: [-6, +4] m/s²
            # Enhanced resolution: 0.5 m/s² bins (仍符合标准范围)
            # Source: ISO 15622 / UNECE Reg.79
            'a_long': [x * 0.5 for x in range(-12, 9)],  # -6 to +4 m/s², 21 bins
            
            # Lateral acceleration: [-4, +4] m/s²
            # Enhanced resolution: 0.5 m/s² bins
            # Source: ISO 3888-1/2
            'a_lat': [x * 0.5 for x in range(-8, 9)],  # -4 to +4 m/s², 17 bins
            
            # Longitudinal jerk: [-3, +3] m/s³
            # Enhanced resolution: 0.5 m/s³ bins
            # Source: UNECE braking test
            'jerk_long': [x * 0.5 for x in range(-6, 7)],  # -3 to +3 m/s³, 13 bins
            
            # Lateral jerk: [-2, +2] m/s³
            # Enhanced resolution: 0.5 m/s³ bins
            # Source: ISO lane-change tests
            'jerk_lat': [x * 0.5 for x in range(-4, 5)],  # -2 to +2 m/s³, 9 bins
            
            # Yaw rate: 0-200 deg/s
            # Enhanced resolution: 10 deg/s bins（仍在ISO 7401范围内）
            # Source: ISO 7401 (steady-state steering)
            'yaw_rate': list(range(0, 201, 10)),  # 0, 10, 20, ..., 200 deg/s, 21 bins
            
            # TTC: 0-3 seconds
            # Enhanced resolution: 0.25s bins（仍符合EuroNCAP区间）
            # Source: EuroNCAP AEB standard
            'ttc': [x * 0.25 for x in range(13)]  # 0.0,0.25,...,3.0 (13 bins)
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
        
        Uses Hartley entropy normalization to handle the large parameter space:
        PCE = log(|covered|) / log(|G|), where |G| is the theoretical grid size.
        When no combinations are covered, returns 0.
        
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
        
        # Theoretical maximum combinations
        grid_sizes = [len(values) for values in self.parameter_grid.values()]
        max_combinations = np.prod(grid_sizes)
        
        if max_combinations <= 1:
            return 0.0
        
        # Hartley entropy normalization (support entropy)
        pce = np.log(covered_count) / np.log(max_combinations)
        return float(np.clip(pce, 0.0, 1.0))
    
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
        
        if max_combinations <= 1:
            return {
                'coverage': 0.0,
                'covered_count': int(covered_count),
                'max_combinations': int(max_combinations),
                'linear_coverage': 0.0
            }
        
        linear_coverage = covered_count / max_combinations
        log_coverage = np.log(covered_count) / np.log(max_combinations) if covered_count > 0 else 0.0
        
        return {
            'coverage': float(np.clip(log_coverage, 0.0, 1.0)),  # PCE
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


# New name (preferred) and backward compatibility aliases
ParameterCoverage = ParameterConfigurationEntropy
BehaviorParameterCoverage = ParameterConfigurationEntropy

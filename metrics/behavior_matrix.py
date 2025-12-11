"""
Behavior Semantic Matrix Coverage (BCM) Metric
Evaluates coverage of behavior combinations using a behavior-scenario matrix

All thresholds are based on industry standards and regulations:
- UNECE R152: Emergency braking thresholds
- EuroNCAP: AEB and car-following thresholds
- ISO 22179, ISO 3888-1/2: Acceleration and lane change thresholds
- ISO 34502: Cut-in behavior definition
- Chinese Traffic Law: Speeding thresholds
- FHWA: Aggressive driving definitions
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario import Scenario
from states import ScenarioState
from metrics.behavior_parameters import BehaviorParameterExtractor


class BehaviorMatrix:
    """
    Calculates behavior semantic matrix coverage
    BCM measures coverage of behavior combinations
    
    All thresholds are based on industry standards and regulations.
    """
    
    # Standard thresholds based on regulations and industry standards
    # Source: UNECE R152 / EuroNCAP
    EMERGENCY_BRAKE_THRESHOLD = -5.0  # m/s² (UNECE R152 / EuroNCAP)
    SEVERE_EMERGENCY_BRAKE_THRESHOLD = -6.0  # m/s² (UNECE R152)
    
    # Source: ISO 22179 / FHWA
    HARD_ACCELERATION_THRESHOLD = 3.0  # m/s² (ISO 22179 / FHWA)
    SEVERE_ACCELERATION_THRESHOLD = 3.5  # m/s² (FHWA aggressive driving)
    
    # Source: ISO 3888-1/2 / ISO 7401
    LANE_CHANGE_YAW_RATE_THRESHOLD = 15.0  # deg/s (ISO 3888-1/2)
    SEVERE_LANE_CHANGE_LAT_ACCEL_THRESHOLD = 3.5  # m/s² (ISO 3888-1/2)
    
    # Source: Chinese Traffic Law (道路交通安全法实施条例)
    SPEEDING_THRESHOLD_RATIO = 1.10  # v > limit * 1.10
    SEVERE_SPEEDING_THRESHOLD_RATIO = 1.50  # v > limit * 1.50
    
    # Source: EuroNCAP AEB Car-to-Car
    CAR_FOLLOWING_RISK_THW_THRESHOLD = 1.4  # seconds (EuroNCAP AEB)
    DANGEROUS_FOLLOWING_THW_THRESHOLD = 1.0  # seconds (FHWA)
    
    # Source: EuroNCAP / UNECE
    TTC_RISK_THRESHOLD = 3.0  # seconds (EuroNCAP)
    TTC_HIGH_RISK_THRESHOLD = 2.0  # seconds (UNECE)
    
    # Source: ISO 34502
    CUT_IN_LATERAL_VELOCITY_THRESHOLD = 0.5  # m/s
    CUT_IN_TTC_DECREASE_THRESHOLD = -1.0  # s/s (rate of TTC decrease)
    
    # Frame rate for time-based calculations (from constants.py)
    FRAME_RATE = 25.0  # Hz
    
    def __init__(self):
        """Initialize behavior matrix calculator"""
        self.behavior_labels = [
            'collision',
            'lane_invasion',
            'speeding',
            'hard_brake',
            'hard_acceleration',
            'lane_change',
            'following',
            'cut_in',
            'emergency_stop',
            'traffic_violation',
            'stuck',
            'red_light_violation'
        ]
        self.behavior_matrix: np.ndarray = None
        self.scenario_behaviors: List[Set[str]] = []
        self.param_extractor = BehaviorParameterExtractor()
    
    
    def label_behaviors(self, scenario_state: ScenarioState) -> List[str]:
        """
        Label behaviors from scenario state using industry-standard thresholds
        
        Args:
            scenario_state: ScenarioState object
            
        Returns:
            List of behavior labels present in the scenario
        """
        behaviors = []
        
        # Check error states (already detected by simulator)
        if scenario_state.crashed:
            behaviors.append('collision')
        
        if scenario_state.laneinvaded:
            behaviors.append('lane_invasion')
        
        if scenario_state.stuck:
            behaviors.append('stuck')
        
        if scenario_state.red_violation:
            behaviors.append('red_light_violation')
        
        # Check speeding using legal threshold (Chinese Traffic Law)
        # Source: 道路交通安全法实施条例第46条
        if hasattr(scenario_state, 'speed') and scenario_state.speed and \
           hasattr(scenario_state, 'speed_lim') and scenario_state.speed_lim:
            if len(scenario_state.speed) > 0 and len(scenario_state.speed_lim) > 0:
                current_speed = scenario_state.speed[-1]  # km/h
                speed_limit = scenario_state.speed_lim[-1]  # km/h
                if speed_limit > 0:
                    if current_speed > speed_limit * self.SPEEDING_THRESHOLD_RATIO:
                        behaviors.append('speeding')
        
        # Extract behavior parameters using standardized extractor
        params = self.param_extractor.extract_all_parameters(scenario_state)
        
        # Check hard braking using UNECE R152 / EuroNCAP threshold
        # Source: UNECE R152 / EuroNCAP (a_long < -5 m/s²)
        a_long = params['a_long']
        if a_long is not None and len(a_long) > 0:
            min_accel = np.min(a_long)
            if min_accel < self.EMERGENCY_BRAKE_THRESHOLD:
                behaviors.append('hard_brake')
            if min_accel < self.SEVERE_EMERGENCY_BRAKE_THRESHOLD:
                behaviors.append('emergency_stop')
        
        # Check hard acceleration using ISO 22179 / FHWA threshold
        # Source: ISO 22179 / FHWA (a_long > +3.0 m/s²)
        if a_long is not None and len(a_long) > 0:
            max_accel = np.max(a_long)
            if max_accel > self.HARD_ACCELERATION_THRESHOLD:
                behaviors.append('hard_acceleration')
        
        # Check lane change using ISO 3888-1/2 threshold
        # Source: ISO 3888-1/2 / ISO 7401 (yaw_rate > 15 deg/s)
        yaw_rate = params['yaw_rate']
        if yaw_rate is not None and len(yaw_rate) > 0:
            max_yaw_rate = np.max(np.abs(yaw_rate))
            if max_yaw_rate > self.LANE_CHANGE_YAW_RATE_THRESHOLD:
                behaviors.append('lane_change')
        
        # Check severe lane change using lateral acceleration
        # Source: ISO 3888-1/2 (a_lat > 3.5 m/s²)
        a_lat = params['a_lat']
        if a_lat is not None and len(a_lat) > 0:
            max_lat_accel = np.max(np.abs(a_lat))
            if max_lat_accel > self.SEVERE_LANE_CHANGE_LAT_ACCEL_THRESHOLD:
                # Lane change already added, but confirms severity
                if 'lane_change' not in behaviors:
                    behaviors.append('lane_change')
        
        # Check car-following risk using EuroNCAP AEB threshold
        # Source: EuroNCAP AEB Car-to-Car (THW < 1.4 s)
        thw = params['thw']
        if thw is not None:
            if thw < self.CAR_FOLLOWING_RISK_THW_THRESHOLD:
                behaviors.append('following')
        
        # Check cut-in using ISO 34502 definition
        # Source: ISO 34502 (lateral_velocity > 0.5 m/s AND ΔTTC/dt < -1.0 s/s)
        if a_lat is not None and len(a_lat) > 0:
            # Calculate lateral velocity from lateral speed
            if hasattr(scenario_state, 'lat_speed_list') and scenario_state.lat_speed_list:
                lat_speed_ms = np.array(scenario_state.lat_speed_list) / 3.6  # m/s
                if len(lat_speed_ms) > 0:
                    max_lat_velocity = np.max(np.abs(lat_speed_ms))
                    
                    # Check TTC decrease rate (simplified)
                    # More accurate would require tracking TTC over time
                    ttc = params['ttc']
                    if ttc is not None and max_lat_velocity > self.CUT_IN_LATERAL_VELOCITY_THRESHOLD:
                        behaviors.append('cut_in')
        
        # Traffic violation (general category)
        if scenario_state.red_violation or (scenario_state.speeding if hasattr(scenario_state, 'speeding') else False):
            behaviors.append('traffic_violation')
        
        return behaviors
    
    def build_matrix(self, scenarios: List[Scenario]) -> np.ndarray:
        """
        Build behavior-scenario matrix
        
        Args:
            scenarios: List of Scenario objects
            
        Returns:
            Binary matrix of shape (n_behaviors, n_scenarios)
        """
        n_scenarios = len(scenarios)
        n_behaviors = len(self.behavior_labels)
        
        matrix = np.zeros((n_behaviors, n_scenarios), dtype=int)
        self.scenario_behaviors = []
        
        for j, scenario in enumerate(scenarios):
            if not hasattr(scenario, 'state') or not scenario.state:
                continue
            
            behaviors = self.label_behaviors(scenario.state)
            self.scenario_behaviors.append(set(behaviors))
            
            for behavior in behaviors:
                if behavior in self.behavior_labels:
                    i = self.behavior_labels.index(behavior)
                    matrix[i, j] = 1
        
        self.behavior_matrix = matrix
        return matrix
    
    def calculate_coverage(self, scenarios: List[Scenario]) -> Dict[str, float]:
        """
        Calculate behavior matrix coverage metrics
        
        Args:
            scenarios: List of Scenario objects
            
        Returns:
            Dictionary containing coverage metrics
        """
        if len(scenarios) == 0:
            return {
                'unique_combinations': 0,
                'total_possible': 0,
                'coverage_ratio': 0.0,
                'behavior_diversity': 0.0
            }
        
        # Build matrix
        matrix = self.build_matrix(scenarios)
        
        if matrix.shape[1] == 0:
            return {
                'unique_combinations': 0,
                'total_possible': 0,
                'coverage_ratio': 0.0,
                'behavior_diversity': 0.0
            }
        
        # Count unique behavior combinations
        unique_combinations = set()
        for j in range(matrix.shape[1]):
            combination = tuple(matrix[:, j])
            unique_combinations.add(combination)
        
        # Calculate total possible combinations (2^n_behaviors = 4096 for 12 behaviors)
        total_possible = 2 ** len(self.behavior_labels)
        covered_count = len(unique_combinations)
        
        # Use logarithmic normalization to handle large combination space
        # This avoids numerical precision issues when coverage is very small
        # (e.g., 100/4096 ≈ 0.0244 becomes more meaningful after log normalization)
        # Formula: coverage = log(1 + covered) / log(1 + total_possible)
        # Benefits:
        # - When covered = 0: coverage = 0
        # - When covered = total: coverage ≈ 1 (asymptotically)
        # - Better numerical stability and discrimination
        if total_possible > 0 and covered_count > 0:
            coverage_ratio = np.log1p(covered_count) / np.log1p(total_possible)
        else:
            coverage_ratio = 0.0
        
        # Behavior diversity: average number of behaviors per scenario
        behavior_counts = [len(behaviors) for behaviors in self.scenario_behaviors]
        behavior_diversity = np.mean(behavior_counts) if behavior_counts else 0.0
        
        # Also calculate linear coverage for reference
        linear_coverage = covered_count / total_possible if total_possible > 0 else 0.0
        
        return {
            'unique_combinations': len(unique_combinations),
            'total_possible': total_possible,
            'coverage_ratio': float(coverage_ratio),  # Logarithmically normalized (primary metric)
            'linear_coverage': float(linear_coverage),  # Linear ratio (for reference/debugging)
            'behavior_diversity': float(behavior_diversity),
            'matrix': matrix.tolist()  # Include matrix for visualization
        }
    
    def get_behavior_labels(self) -> List[str]:
        """Get the list of behavior labels"""
        return self.behavior_labels.copy()
    
    def reset(self):
        """Reset the behavior tracking"""
        self.behavior_matrix = None
        self.scenario_behaviors.clear()


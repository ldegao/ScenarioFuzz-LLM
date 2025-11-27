"""
Behavior Semantic Matrix Coverage (BCM) Metric
Evaluates coverage of behavior combinations using a behavior-scenario matrix
"""

import numpy as np
from typing import List, Dict, Set
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario import Scenario
from states import ScenarioState


class BehaviorMatrix:
    """
    Calculates behavior semantic matrix coverage
    BCM measures coverage of behavior combinations
    """
    
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
    
    def label_behaviors(self, scenario_state: ScenarioState) -> List[str]:
        """
        Label behaviors from scenario state
        
        Args:
            scenario_state: ScenarioState object
            
        Returns:
            List of behavior labels present in the scenario
        """
        behaviors = []
        
        # Check error states
        if scenario_state.crashed:
            behaviors.append('collision')
        
        if scenario_state.laneinvaded:
            behaviors.append('lane_invasion')
        
        if scenario_state.speeding:
            behaviors.append('speeding')
        
        if scenario_state.stuck:
            behaviors.append('stuck')
        
        if scenario_state.red_violation:
            behaviors.append('red_light_violation')
        
        # Check control states for driving behaviors
        if hasattr(scenario_state, 'cont_brake') and scenario_state.cont_brake:
            brake_values = scenario_state.cont_brake
            if len(brake_values) > 0:
                max_brake = max(brake_values)
                if max_brake > 0.7:  # Hard braking threshold
                    behaviors.append('hard_brake')
        
        if hasattr(scenario_state, 'cont_throttle') and scenario_state.cont_throttle:
            throttle_values = scenario_state.cont_throttle
            if len(throttle_values) > 0:
                max_throttle = max(throttle_values)
                if max_throttle > 0.8:  # Hard acceleration threshold
                    behaviors.append('hard_acceleration')
        
        # Check for lane change (based on yaw changes)
        if hasattr(scenario_state, 'yaw_list') and scenario_state.yaw_list:
            yaw_changes = np.abs(np.diff(scenario_state.yaw_list))
            if np.max(yaw_changes) > 0.3:  # Significant yaw change indicates lane change
                behaviors.append('lane_change')
        
        # Check for following behavior (based on min_dist)
        if hasattr(scenario_state, 'min_dist') and scenario_state.min_dist < 10:
            behaviors.append('following')
        
        # Check for cut-in (sudden decrease in min_dist)
        if hasattr(scenario_state, 'min_dist') and hasattr(scenario_state, 'closest_cars_list'):
            if len(scenario_state.closest_cars_list) > 1:
                distances = [car.get('distance', 999) for car in scenario_state.closest_cars_list if isinstance(car, dict)]
                if len(distances) > 1:
                    dist_changes = np.diff(distances)
                    if np.min(dist_changes) < -5:  # Sudden decrease indicates cut-in
                        behaviors.append('cut_in')
        
        # Check for emergency stop
        if hasattr(scenario_state, 'speed') and scenario_state.speed:
            if len(scenario_state.speed) > 5:
                speed_changes = np.diff(scenario_state.speed[-5:])
                if np.min(speed_changes) < -10:  # Sudden speed decrease
                    behaviors.append('emergency_stop')
        
        # Traffic violation (general)
        if scenario_state.red_violation or scenario_state.speeding:
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
        
        # Calculate total possible combinations (2^n_behaviors)
        total_possible = 2 ** len(self.behavior_labels)
        
        # Coverage ratio
        coverage_ratio = len(unique_combinations) / total_possible if total_possible > 0 else 0.0
        
        # Behavior diversity: average number of behaviors per scenario
        behavior_counts = [len(behaviors) for behaviors in self.scenario_behaviors]
        behavior_diversity = np.mean(behavior_counts) if behavior_counts else 0.0
        
        return {
            'unique_combinations': len(unique_combinations),
            'total_possible': total_possible,
            'coverage_ratio': float(coverage_ratio),
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


"""
Parameter Coverage (PC) Metric
Evaluates coverage of parameter space combinations
"""

import numpy as np
from typing import List, Dict, Set, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario import Scenario


class ParameterCoverage:
    """
    Calculates parameter space coverage metric
    PC = |covered combinations| / |theoretical maximum combinations|
    """
    
    def __init__(self):
        """Initialize parameter coverage calculator"""
        self.parameter_grid = self._define_parameter_grid()
        self.covered_combinations: Set[Tuple] = set()
    
    def _define_parameter_grid(self) -> Dict[str, List]:
        """
        Define the parameter space grid
        
        Returns:
            Dictionary mapping parameter names to their discrete values
        """
        return {
            'weather_cloud': [0, 25, 50, 75, 100],  # Cloud coverage: 5 levels
            'weather_rain': [0, 25, 50, 75, 100],   # Rain: 5 levels
            'weather_fog': [0, 25, 50, 75, 100],    # Fog: 5 levels
            'npc_count': [0, 1, 2, 3, 4, 5, 6, 7, 8],  # NPC count: 9 levels
            'speed_range': ['low', 'medium', 'high'],  # Speed range: 3 levels
            'route_type': ['straight', 'curve', 'intersection', 'highway']  # Route type: 4 levels
        }
    
    def _discretize_weather(self, weather_value: int, levels: List[int]) -> int:
        """Discretize weather value to nearest level"""
        return min(levels, key=lambda x: abs(x - weather_value))
    
    def _get_speed_range(self, scenario: Scenario) -> str:
        """Determine speed range category from scenario"""
        if not hasattr(scenario, 'state') or not scenario.state.speed:
            return 'medium'
        
        avg_speed = np.mean(scenario.state.speed) if scenario.state.speed else 0
        if avg_speed < 20:
            return 'low'
        elif avg_speed > 40:
            return 'high'
        else:
            return 'medium'
    
    def _get_route_type(self, scenario: Scenario) -> str:
        """Determine route type from scenario"""
        # Simple heuristic based on town and distance
        town = scenario.town if hasattr(scenario, 'town') else None
        
        if town and 'Town03' in str(town):
            return 'intersection'
        elif town and 'Town04' in str(town):
            return 'highway'
        else:
            # Check if route has curves based on yaw changes
            if hasattr(scenario, 'state') and scenario.state.yaw_list:
                yaw_changes = np.abs(np.diff(scenario.state.yaw_list))
                if np.max(yaw_changes) > 0.5:  # Significant yaw change
                    return 'curve'
            return 'straight'
    
    def _extract_parameter_combination(self, scenario: Scenario) -> Tuple:
        """
        Extract parameter combination from a scenario
        
        Returns:
            Tuple representing the parameter combination
        """
        weather = scenario.weather if hasattr(scenario, 'weather') else {}
        
        cloud = self._discretize_weather(weather.get('cloud', 0), self.parameter_grid['weather_cloud'])
        rain = self._discretize_weather(weather.get('rain', 0), self.parameter_grid['weather_rain'])
        fog = self._discretize_weather(weather.get('fog', 0), self.parameter_grid['weather_fog'])
        
        npc_count = len(scenario.npc_list) if hasattr(scenario, 'npc_list') else 0
        npc_count = min(npc_count, max(self.parameter_grid['npc_count']))
        
        speed_range = self._get_speed_range(scenario)
        route_type = self._get_route_type(scenario)
        
        return (cloud, rain, fog, npc_count, speed_range, route_type)
    
    def calculate_coverage(self, scenarios: List[Scenario]) -> float:
        """
        Calculate parameter coverage for a list of scenarios
        
        Args:
            scenarios: List of Scenario objects
            
        Returns:
            Coverage ratio between 0 and 1
        """
        # Extract all parameter combinations
        for scenario in scenarios:
            combination = self._extract_parameter_combination(scenario)
            self.covered_combinations.add(combination)
        
        # Calculate theoretical maximum
        grid_sizes = [len(values) for values in self.parameter_grid.values()]
        max_combinations = np.prod(grid_sizes)
        
        # Calculate coverage
        coverage = len(self.covered_combinations) / max_combinations if max_combinations > 0 else 0.0
        
        return float(coverage)
    
    def get_parameter_grid(self) -> Dict[str, List]:
        """
        Get the parameter space grid definition
        
        Returns:
            Dictionary mapping parameter names to their discrete values
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


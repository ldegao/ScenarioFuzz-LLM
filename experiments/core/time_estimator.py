"""
Time Estimator Module
Estimates experiment execution time based on historical data
"""

import time
import json
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque


class TimeEstimator:
    """
    Estimates execution time for experiments based on historical data
    """
    
    def __init__(self, history_file: str = "./experiment_results/time_history.json"):
        """
        Initialize time estimator
        
        Args:
            history_file: Path to store historical timing data
        """
        self.history_file = history_file
        self.history = self._load_history()
        # Keep only recent 100 records
        self.max_history_size = 100
    
    def _load_history(self) -> List[Dict]:
        """Load historical timing data"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_history(self):
        """Save historical timing data"""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.history[-self.max_history_size:], f, indent=2)
    
    def record_scenario_time(self, method_name: str, scenario_time: float):
        """
        Record time taken for a single scenario
        
        Args:
            method_name: Name of the method
            scenario_time: Time in seconds
        """
        record = {
            'method': method_name,
            'scenario_time': scenario_time,
            'timestamp': datetime.now().isoformat()
        }
        self.history.append(record)
        self._save_history()
    
    def estimate_scenario_time(self, method_name: str) -> float:
        """
        Estimate time for a single scenario based on history
        
        Args:
            method_name: Name of the method
            
        Returns:
            Estimated time in seconds
        """
        # Filter history for this method
        method_history = [h for h in self.history if h.get('method') == method_name]
        
        if len(method_history) == 0:
            # Default estimates if no history
            # Note: DriveFuzz temporarily disabled
            default_estimates = {
                'TM-Fuzzer': 50.0,
                # 'DriveFuzz': 55.0,  # Temporarily disabled
                'ScenarioFuzz-LLM': 60.0,
                'RAG-ScenarioFuzz': 65.0  # Slightly longer due to RAG
            }
            return default_estimates.get(method_name, 60.0)
        
        # Calculate average from recent history
        recent_times = [h['scenario_time'] for h in method_history[-20:]]
        return sum(recent_times) / len(recent_times)
    
    def estimate_total_time(self, method_name: str, num_scenarios: int) -> Dict[str, any]:
        """
        Estimate total time for generating N scenarios
        
        Args:
            method_name: Name of the method
            num_scenarios: Number of scenarios to generate
            
        Returns:
            Dictionary with time estimates
        """
        avg_scenario_time = self.estimate_scenario_time(method_name)
        total_seconds = avg_scenario_time * num_scenarios
        
        return {
            'method': method_name,
            'num_scenarios': num_scenarios,
            'avg_scenario_time': avg_scenario_time,
            'total_seconds': total_seconds,
            'total_time_str': str(timedelta(seconds=int(total_seconds))),
            'estimated_completion': (datetime.now() + timedelta(seconds=total_seconds)).isoformat()
        }
    
    def update_estimate(self, method_name: str, elapsed_time: float, completed_scenarios: int):
        """
        Update estimate based on current progress
        
        Args:
            method_name: Name of the method
            elapsed_time: Time elapsed so far
            completed_scenarios: Number of scenarios completed
            
        Returns:
            Updated estimate
        """
        if completed_scenarios == 0:
            return self.estimate_total_time(method_name, 1)
        
        # Calculate current average
        current_avg = elapsed_time / completed_scenarios
        
        # Record for future estimates
        self.record_scenario_time(method_name, current_avg)
        
        return {
            'current_avg_time': current_avg,
            'elapsed_time': elapsed_time,
            'completed_scenarios': completed_scenarios
        }


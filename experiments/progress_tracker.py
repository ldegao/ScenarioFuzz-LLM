"""
Progress Tracker Module
Tracks experiment progress and supports checkpoint/resume
"""

import json
import os
from typing import Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path


class ProgressTracker:
    """
    Tracks experiment progress and supports checkpoint/resume
    """
    
    def __init__(self, checkpoint_file: str = "./experiment_results/checkpoint.json"):
        """
        Initialize progress tracker
        
        Args:
            checkpoint_file: Path to checkpoint file
        """
        self.checkpoint_file = checkpoint_file
        self.progress = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Dict:
        """Load progress from checkpoint file"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_checkpoint(self):
        """Save progress to checkpoint file"""
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def start_experiment(self, experiment_id: str, method_name: str, 
                        target_scenarios: int, target_time: Optional[float] = None):
        """
        Start tracking a new experiment
        
        Args:
            experiment_id: Unique experiment ID
            method_name: Name of the method
            target_scenarios: Target number of scenarios
            target_time: Target time in seconds (optional)
        """
        self.progress[experiment_id] = {
            'method_name': method_name,
            'target_scenarios': target_scenarios,
            'target_time': target_time,
            'completed_scenarios': 0,
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat(),
            'scenarios': []
        }
        self._save_checkpoint()
    
    def update_progress(self, experiment_id: str, scenario_id: int, 
                       scenario_info: Optional[Dict] = None):
        """
        Update progress for a scenario
        
        Args:
            experiment_id: Experiment ID
            scenario_id: Scenario ID
            scenario_info: Optional scenario information
        """
        if experiment_id not in self.progress:
            return
        
        self.progress[experiment_id]['completed_scenarios'] += 1
        self.progress[experiment_id]['last_update'] = datetime.now().isoformat()
        
        if scenario_info:
            scenario_info['scenario_id'] = scenario_id
            scenario_info['timestamp'] = datetime.now().isoformat()
            self.progress[experiment_id]['scenarios'].append(scenario_info)
        
        self._save_checkpoint()
    
    def get_progress(self, experiment_id: str) -> Optional[Dict]:
        """
        Get current progress for an experiment
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Progress dictionary or None
        """
        return self.progress.get(experiment_id)
    
    def is_complete(self, experiment_id: str) -> bool:
        """
        Check if experiment is complete
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if complete
        """
        if experiment_id not in self.progress:
            return False
        
        prog = self.progress[experiment_id]
        
        # Check scenario count
        if prog['completed_scenarios'] >= prog['target_scenarios']:
            return True
        
        # Check time limit
        if prog.get('target_time'):
            start_time = datetime.fromisoformat(prog['start_time'])
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= prog['target_time']:
                return True
        
        return False
    
    def get_summary(self, experiment_id: str) -> Dict:
        """
        Get progress summary
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Summary dictionary
        """
        if experiment_id not in self.progress:
            return {}
        
        prog = self.progress[experiment_id]
        # Python 3.6 compatibility: fromisoformat not available
        start_time_str = prog['start_time']
        if hasattr(datetime, 'fromisoformat'):
            start_time = datetime.fromisoformat(start_time_str)
        else:
            # Manual parsing for Python 3.6
            start_time = datetime.strptime(start_time_str.replace('T', ' ').split('.')[0], '%Y-%m-%d %H:%M:%S')
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return {
            'method_name': prog['method_name'],
            'completed': prog['completed_scenarios'],
            'target': prog['target_scenarios'],
            'progress_percent': (prog['completed_scenarios'] / prog['target_scenarios'] * 100) if prog['target_scenarios'] > 0 else 0,
            'elapsed_time': elapsed,
            'elapsed_time_str': str(timedelta(seconds=int(elapsed))),
            'is_complete': self.is_complete(experiment_id)
        }
    
    def clear_checkpoint(self, experiment_id: Optional[str] = None):
        """
        Clear checkpoint for an experiment or all experiments
        
        Args:
            experiment_id: Experiment ID to clear, or None for all
        """
        if experiment_id:
            if experiment_id in self.progress:
                del self.progress[experiment_id]
        else:
            self.progress = {}
        self._save_checkpoint()


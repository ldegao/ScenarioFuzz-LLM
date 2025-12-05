"""
Token Usage Tracker Module
Tracks and aggregates token usage across experiment runs
"""

import json
import threading
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict


class TokenTracker:
    """
    Thread-safe token usage tracker
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._stats = defaultdict(lambda: {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'call_count': 0
        })
        # Track per-model usage
        self._model_stats = defaultdict(lambda: {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'call_count': 0
        })
    
    def record_usage(self, model: str = "gpt-4-turbo", 
                     prompt_tokens: int = 0,
                     completion_tokens: int = 0,
                     total_tokens: int = 0):
        """
        Record token usage for a single API call
        
        Args:
            model: Model name (e.g., "gpt-4-turbo")
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens (if provided, will be used instead of sum)
        """
        with self._lock:
            # Calculate total if not provided
            if total_tokens == 0:
                total_tokens = prompt_tokens + completion_tokens
            
            # Update overall stats
            self._stats['all']['prompt_tokens'] += prompt_tokens
            self._stats['all']['completion_tokens'] += completion_tokens
            self._stats['all']['total_tokens'] += total_tokens
            self._stats['all']['call_count'] += 1
            
            # Update per-model stats
            self._model_stats[model]['prompt_tokens'] += prompt_tokens
            self._model_stats[model]['completion_tokens'] += completion_tokens
            self._model_stats[model]['total_tokens'] += total_tokens
            self._model_stats[model]['call_count'] += 1
    
    def get_stats(self) -> Dict:
        """
        Get current token usage statistics
        
        Returns:
            Dictionary with token statistics
        """
        with self._lock:
            return {
                'overall': dict(self._stats['all']),
                'by_model': {model: dict(stats) for model, stats in self._model_stats.items()}
            }
    
    def reset(self):
        """Reset all statistics"""
        with self._lock:
            self._stats.clear()
            self._model_stats.clear()
    
    def save_to_file(self, filepath: Path):
        """
        Save token statistics to a JSON file
        
        Args:
            filepath: Path to save the statistics
        """
        stats = self.get_stats()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def load_from_file(self, filepath: Path) -> bool:
        """
        Load token statistics from a JSON file
        
        Args:
            filepath: Path to load the statistics from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not filepath.exists():
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            with self._lock:
                if 'overall' in data:
                    self._stats['all'] = data['overall']
                if 'by_model' in data:
                    self._model_stats = defaultdict(lambda: {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0,
                        'call_count': 0
                    })
                    for model, stats in data['by_model'].items():
                        self._model_stats[model] = stats
            
            return True
        except Exception as e:
            print(f"[WARNING] Failed to load token statistics from {filepath}: {e}")
            return False
    
    def print_summary(self):
        """Print a formatted summary of token usage"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("Token Usage Summary")
        print("="*60)
        
        overall = stats['overall']
        print(f"\nOverall Statistics:")
        print(f"  Total API Calls: {overall['call_count']}")
        print(f"  Prompt Tokens:  {overall['prompt_tokens']:,}")
        print(f"  Completion Tokens: {overall['completion_tokens']:,}")
        print(f"  Total Tokens:   {overall['total_tokens']:,}")
        
        if stats['by_model']:
            print(f"\nBy Model:")
            for model, model_stats in stats['by_model'].items():
                print(f"  {model}:")
                print(f"    Calls: {model_stats['call_count']}")
                print(f"    Prompt Tokens: {model_stats['prompt_tokens']:,}")
                print(f"    Completion Tokens: {model_stats['completion_tokens']:,}")
                print(f"    Total Tokens: {model_stats['total_tokens']:,}")
        
        print("="*60 + "\n")


# Global instance
_global_tracker = TokenTracker()


def get_tracker() -> TokenTracker:
    """Get the global token tracker instance"""
    return _global_tracker


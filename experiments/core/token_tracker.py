"""
Token Usage Tracker Module
Tracks and aggregates token usage across experiment runs
"""

import json
import threading
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

# Pricing constants (per 1M tokens)
INPUT_TOKEN_PRICE_PER_MILLION = 0.25  # US $0.25 / 1,000,000 tokens
OUTPUT_TOKEN_PRICE_PER_MILLION = 2.00  # US $2.00 / 1,000,000 tokens

# Per-token prices
INPUT_TOKEN_PRICE = INPUT_TOKEN_PRICE_PER_MILLION / 1_000_000  # $0.00000025 per token
OUTPUT_TOKEN_PRICE = OUTPUT_TOKEN_PRICE_PER_MILLION / 1_000_000  # $0.000002 per token


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
            'call_count': 0,
            'input_cost_usd': 0.0,
            'output_cost_usd': 0.0,
            'total_cost_usd': 0.0
        })
        # Track per-model usage
        self._model_stats = defaultdict(lambda: {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'call_count': 0,
            'input_cost_usd': 0.0,
            'output_cost_usd': 0.0,
            'total_cost_usd': 0.0
        })
    
    @staticmethod
    def calculate_cost(prompt_tokens: int, completion_tokens: int) -> tuple:
        """
        Calculate cost based on token usage
        
        Args:
            prompt_tokens: Number of input/prompt tokens
            completion_tokens: Number of output/completion tokens
            
        Returns:
            Tuple of (input_cost, output_cost, total_cost) in USD
        """
        input_cost = prompt_tokens * INPUT_TOKEN_PRICE
        output_cost = completion_tokens * OUTPUT_TOKEN_PRICE
        total_cost = input_cost + output_cost
        return (input_cost, output_cost, total_cost)
    
    def record_usage(self, model: str = None, 
                     prompt_tokens: int = 0,
                     completion_tokens: int = 0,
                     total_tokens: int = 0):
        """
        Record token usage for a single API call
        
        Args:
            model: Model name. If None, tries to get DEFAULT_MODEL from gpt module.
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens (if provided, will be used instead of sum)
        """
        if model is None:
            try:
                import gpt
                model = getattr(gpt, 'DEFAULT_MODEL', 'gpt-4o-mini')
            except:
                model = 'gpt-4o-mini'
        
        with self._lock:
            # Calculate total if not provided
            if total_tokens == 0:
                total_tokens = prompt_tokens + completion_tokens
            
            # Calculate cost for this call
            input_cost, output_cost, call_cost = self.calculate_cost(prompt_tokens, completion_tokens)
            
            # Update overall stats
            self._stats['all']['prompt_tokens'] += prompt_tokens
            self._stats['all']['completion_tokens'] += completion_tokens
            self._stats['all']['total_tokens'] += total_tokens
            self._stats['all']['call_count'] += 1
            self._stats['all']['input_cost_usd'] += input_cost
            self._stats['all']['output_cost_usd'] += output_cost
            self._stats['all']['total_cost_usd'] += call_cost
            
            # Update per-model stats
            self._model_stats[model]['prompt_tokens'] += prompt_tokens
            self._model_stats[model]['completion_tokens'] += completion_tokens
            self._model_stats[model]['total_tokens'] += total_tokens
            self._model_stats[model]['call_count'] += 1
            self._model_stats[model]['input_cost_usd'] += input_cost
            self._model_stats[model]['output_cost_usd'] += output_cost
            self._model_stats[model]['total_cost_usd'] += call_cost
    
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
                        'call_count': 0,
                        'input_cost_usd': 0.0,
                        'output_cost_usd': 0.0,
                        'total_cost_usd': 0.0
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
        
        # Cost information
        input_cost = overall.get('input_cost_usd', 0.0)
        output_cost = overall.get('output_cost_usd', 0.0)
        total_cost = overall.get('total_cost_usd', 0.0)
        print(f"\n  Cost Breakdown:")
        print(f"    Input Cost:  ${input_cost:.6f} USD")
        print(f"    Output Cost: ${output_cost:.6f} USD")
        print(f"    Total Cost:  ${total_cost:.6f} USD")
        
        if stats['by_model']:
            print(f"\nBy Model:")
            for model, model_stats in stats['by_model'].items():
                print(f"  {model}:")
                print(f"    Calls: {model_stats['call_count']}")
                print(f"    Prompt Tokens: {model_stats['prompt_tokens']:,}")
                print(f"    Completion Tokens: {model_stats['completion_tokens']:,}")
                print(f"    Total Tokens: {model_stats['total_tokens']:,}")
                # Per-model cost
                model_input_cost = model_stats.get('input_cost_usd', 0.0)
                model_output_cost = model_stats.get('output_cost_usd', 0.0)
                model_total_cost = model_stats.get('total_cost_usd', 0.0)
                print(f"    Cost: ${model_total_cost:.6f} USD (Input: ${model_input_cost:.6f}, Output: ${model_output_cost:.6f})")
        
        print("="*60 + "\n")


# Global instance
_global_tracker = TokenTracker()


def get_tracker() -> TokenTracker:
    """Get the global token tracker instance"""
    return _global_tracker


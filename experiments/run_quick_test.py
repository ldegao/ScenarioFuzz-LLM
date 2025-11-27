#!/usr/bin/env python3
"""
Quick Test Runner for Phase 1
Runs minimal test with reduced GA parameters for fast verification
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_manager import ExperimentManager


def main():
    """Main entry point for quick tests"""
    parser = argparse.ArgumentParser(
        description='Run quick test (minimal GA parameters)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test ScenarioFuzz-LLM
  python run_quick_test.py ScenarioFuzz-LLM
  
  # Quick test RAG-ScenarioFuzz
  python run_quick_test.py RAG-ScenarioFuzz
        """
    )
    
    parser.add_argument('method', type=str,
                       choices=['ScenarioFuzz-LLM', 'RAG-ScenarioFuzz'],
                       help='Method to test')
    parser.add_argument('--output-dir', type=str, default='./experiment_results',
                       help='Output directory (default: ./experiment_results)')
    parser.add_argument('--target', type=str, default='behavior',
                       choices=['behavior', 'autoware'],
                       help='Target ADS system (default: behavior)')
    parser.add_argument('--town', type=int, default=3,
                       help='CARLA town number (default: 3)')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Scenario timeout in seconds (default: 60)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Quick Test Mode")
    print("="*60)
    print("This will temporarily modify fuzzer.py GA parameters:")
    print("  - MAX_GEN: 5 -> 1")
    print("  - POP_SIZE: 5 -> 2")
    print("  - OFF_SIZE: 5 -> 2")
    print("Expected scenarios: ~4-6 (instead of ~30)")
    print("Expected time: ~5-10 minutes (instead of ~30-60 minutes)")
    print("="*60)
    print()
    
    # Backup fuzzer.py
    fuzzer_path = Path(__file__).parent.parent / "fuzzer.py"
    backup_path = fuzzer_path.with_suffix('.py.backup')
    
    if not backup_path.exists():
        import shutil
        shutil.copy(fuzzer_path, backup_path)
        print(f"✓ Backed up fuzzer.py to {backup_path}")
    
    # Read fuzzer.py
    with open(fuzzer_path, 'r') as f:
        content = f.read()
    
    # Modify GA parameters for quick test
    modified_content = content.replace(
        'POP_SIZE = 5  # amount of population',
        'POP_SIZE = 2  # amount of population (quick test mode)'
    ).replace(
        'OFF_SIZE = 5  # number of offspring to produce',
        'OFF_SIZE = 2  # number of offspring to produce (quick test mode)'
    ).replace(
        'MAX_GEN = 5  #',
        'MAX_GEN = 1  # (quick test mode)'
    )
    
    # Write modified fuzzer.py
    with open(fuzzer_path, 'w') as f:
        f.write(modified_content)
    
    print("✓ Modified fuzzer.py for quick test")
    print()
    
    try:
        # Create experiment manager
        manager = ExperimentManager(output_base_dir=args.output_dir)
        
        # Run test
        print(f"Starting quick test: {args.method}")
        print()
        
        manager.run_quantitative_experiment(
            method_name=args.method,
            num_scenarios=10,  # Will be limited by MAX_GEN anyway
            experiment_id=f"{args.method}_quick_test",
            target=args.target,
            town=args.town,
            timeout=args.timeout,
            debug=True
        )
        
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Restore fuzzer.py
        if backup_path.exists():
            import shutil
            shutil.copy(backup_path, fuzzer_path)
            print(f"\n✓ Restored fuzzer.py from backup")
            backup_path.unlink()
            print(f"✓ Removed backup file")


if __name__ == "__main__":
    main()


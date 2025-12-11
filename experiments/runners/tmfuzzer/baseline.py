#!/usr/bin/env python3
"""
TM-Fuzzer Baseline Runner helpers used by ExperimentManager.

Provides run_tmfuzzer_quantitative / run_tmfuzzer_timed wrappers that call
the original script/test.py workflows. Relocated under experiments.runners.
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_tmfuzzer_quantitative(num_scenarios: int, output_dir: str = "./experiment_results",
                              target: str = "autoware", density: str = "0.4", 
                              town: str = "3", timeout: int = 60):
    """
    Run TM-Fuzzer quantitative run (converted to timed run via time estimate).
    
    WARNING: This is an ESTIMATION mode. The actual number of scenarios generated
    may differ from the target because TM-Fuzzer's script/test.py only supports
    time-based control, not explicit scenario count limits. The function estimates
    the required duration based on an average scenario time (50 seconds per scenario)
    and runs for that duration. The actual scenario count will depend on the real
    execution time per scenario, which may vary.
    
    For precise scenario counts, consider running multiple timed runs or manually
    adjusting the duration based on observed scenario generation rates.
    """
    # Input validation
    if num_scenarios <= 0:
        raise ValueError(f"num_scenarios must be positive, got {num_scenarios}")
    if num_scenarios > 1000000:
        raise ValueError(f"num_scenarios too large: {num_scenarios} (max: 1000000)")
    
    # Estimate time (average 50 seconds per scenario, at least 300 seconds)
    avg_scenario_time = 50.0
    estimated_duration = max(int(num_scenarios * avg_scenario_time), 300)
    
    print(f"\n{'='*60}")
    print(f"Running TM-Fuzzer Baseline (ESTIMATION MODE)")
    print(f"Target scenarios: {num_scenarios}")
    print(f"Estimated duration: {estimated_duration} seconds (~{estimated_duration/3600:.2f} hours)")
    print(f"{'='*60}")
    print(f"[WARNING] This is an estimation mode. Actual scenario count may vary.")
    print(f"[WARNING] TM-Fuzzer uses time-based control, not explicit scenario limits.")
    print(f"[WARNING] The actual number of scenarios will depend on real execution times.")
    print(f"{'='*60}\n")
    
    # Locate script directory and test script
    project_root = PROJECT_ROOT
    script_dir = project_root / "script"
    test_script = script_dir / "test.py"
    original_dir = os.getcwd()
    
    if not test_script.exists():
        raise FileNotFoundError(f"Test script not found: {test_script}")
    
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root) + (
            ":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""
        )
        
        venv_python = project_root / "venv" / "bin" / "python3"
        python_cmd = str(venv_python) if venv_python.exists() else "python3"
        
        cmd = [
            python_cmd,
            "test.py",
            target,
            density,
            town,
            str(estimated_duration),
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        print(f"Working directory: {script_dir}")
        print(f"PYTHONPATH: {env['PYTHONPATH']}")
        print("Note: script/test.py will call init.sh to manage environment\n")
        
        result = subprocess.run(cmd, cwd=str(script_dir), env=env)
        
        if result.returncode == 0:
            print(f"\n✓ TM-Fuzzer completed successfully")
        else:
            print(f"\n[ERROR] TM-Fuzzer exited with code {result.returncode}")
            raise RuntimeError(f"TM-Fuzzer failed with exit code {result.returncode}")
    except Exception as e:
        print(f"[ERROR] Failed to run TM-Fuzzer: {e}")
        traceback.print_exc()
        raise
    finally:
        os.chdir(original_dir)


def run_tmfuzzer_timed(hours: float, output_dir: str = "./experiment_results",
                       target: str = "autoware", density: str = "0.4",
                       town: str = "3", timeout: int = 60):
    """
    TM-Fuzzer timed run.
    """
    duration_seconds = int(hours * 3600)
    
    print(f"\n{'='*60}")
    print(f"Running TM-Fuzzer Baseline")
    print(f"Duration: {hours} hours ({duration_seconds} seconds)")
    print(f"{'='*60}\n")
    
    project_root = PROJECT_ROOT
    script_dir = (project_root / "script").resolve()
    test_script = script_dir / "test.py"
    original_dir = os.getcwd()
    
    if not test_script.exists():
        raise FileNotFoundError(f"Test script not found: {test_script}")
    
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root) + (
            ":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""
        )
        
        venv_python = project_root / "venv" / "bin" / "python3"
        python_cmd = str(venv_python) if venv_python.exists() else "python3"
        
        cmd = [
            python_cmd,
            "test.py",
            target,
            density,
            town,
            str(duration_seconds),
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        print(f"Working directory: {script_dir}")
        print(f"PYTHONPATH: {env['PYTHONPATH']}")
        print("Note: script/test.py will call init.sh to manage environment\n")
        
        result = subprocess.run(cmd, cwd=str(script_dir), env=env)
        
        if result.returncode == 0:
            print(f"\n✓ TM-Fuzzer completed successfully")
        else:
            print(f"\n[ERROR] TM-Fuzzer exited with code {result.returncode}")
            raise RuntimeError(f"TM-Fuzzer failed with exit code {result.returncode}")
    except Exception as e:
        print(f"[ERROR] Failed to run TM-Fuzzer: {e}")
        traceback.print_exc()
        raise
    finally:
        os.chdir(original_dir)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run TM-Fuzzer baseline experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantitative: 100 scenarios
  python -m experiments.runners.tmfuzzer.baseline --num-scenarios 100
  
  # Timed: 2 hours
  python -m experiments.runners.tmfuzzer.baseline --hours 2
        """,
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--num-scenarios",
        type=int,
        help="Number of scenarios to generate",
    )
    group.add_argument(
        "--hours",
        type=float,
        help="Duration in hours",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiment_results",
        help="Output directory (default: ./experiment_results)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="autoware",
        choices=["autoware", "behavior"],
        help="Target ADS system (default: autoware)",
    )
    parser.add_argument(
        "--density",
        type=str,
        default="0.4",
        help="Vehicle density (default: 0.4)",
    )
    parser.add_argument(
        "--town",
        type=str,
        default="3",
        help="CARLA town number (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Scenario timeout in seconds (default: 60)",
    )
    
    args = parser.parse_args()
    
    if args.num_scenarios:
        run_tmfuzzer_quantitative(
            num_scenarios=args.num_scenarios,
            output_dir=args.output_dir,
            target=args.target,
            density=args.density,
            town=args.town,
            timeout=args.timeout,
        )
    elif args.hours:
        run_tmfuzzer_timed(
            hours=args.hours,
            output_dir=args.output_dir,
            target=args.target,
            density=args.density,
            town=args.town,
            timeout=args.timeout,
        )


if __name__ == "__main__":
    main()



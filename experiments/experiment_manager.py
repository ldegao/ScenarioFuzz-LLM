"""
Experiment Manager Module
Manages experiment execution with progress tracking and time estimation
"""

import sys
import os
import time
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazy imports to avoid api.json dependency at module load time
# fuzzer imports will be done when actually needed
from progress_tracker import ProgressTracker
from time_estimator import TimeEstimator


class ExperimentManager:
    """
    Manages experiment execution with progress tracking
    """
    
    def __init__(self, output_base_dir: str = "./experiment_results"):
        """
        Initialize experiment manager
        
        Args:
            output_base_dir: Base directory for experiment results
        """
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.progress_tracker = ProgressTracker(
            checkpoint_file=str(self.output_base_dir / "checkpoint.json")
        )
        self.time_estimator = TimeEstimator(
            history_file=str(self.output_base_dir / "time_history.json")
        )
    
    def run_quantitative_experiment(self, method_name: str, num_scenarios: int,
                                   experiment_id: Optional[str] = None, **kwargs):
        """
        Run experiment with quantitative control (N scenarios)
        
        Args:
            method_name: Name of the method
            num_scenarios: Number of scenarios to generate
            experiment_id: Optional experiment ID
            **kwargs: Additional configuration
        """
        if experiment_id is None:
            experiment_id = f"{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"Starting Quantitative Experiment: {method_name}")
        print(f"Target: {num_scenarios} scenarios")
        print(f"Experiment ID: {experiment_id}")
        print(f"{'='*60}\n")
        
        # Estimate time
        estimate = self.time_estimator.estimate_total_time(method_name, num_scenarios)
        print(f"Estimated time: {estimate['total_time_str']}")
        print(f"Estimated completion: {estimate['estimated_completion']}\n")
        
        # Start tracking
        self.progress_tracker.start_experiment(
            experiment_id=experiment_id,
            method_name=method_name,
            target_scenarios=num_scenarios
        )
        
        # Create output directory
        method_dir = self.output_base_dir / method_name / experiment_id
        method_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure
        args = self._create_args(method_name, method_dir, **kwargs)
        
        # Modify fuzzer to support quantitative control
        # This requires modifying fuzzer.py to accept scenario count limit
        start_time = time.time()
        scenario_count = 0
        
        try:
            # Method-specific configuration
            if method_name == "TM-Fuzzer":
                # TM-Fuzzer uses script/test.py, handled separately
                # script/test.py will manage environment via init.sh
                from run_tmfuzzer_baseline import run_tmfuzzer_quantitative
                run_tmfuzzer_quantitative(
                    num_scenarios=num_scenarios,
                    output_dir=str(self.output_base_dir),
                    target=kwargs.get('target', 'autoware'),
                    density=kwargs.get('density', '0.4'),
                    town=str(kwargs.get('town', 3)),
                    timeout=kwargs.get('timeout', 60)
                )
                return
            
            # For ScenarioFuzz-LLM and RAG-ScenarioFuzz:
            # Need to manage environment before running fuzzer
            from environment_manager import run_init_script, ensure_carla_running
            script_dir = Path(__file__).parent.parent / "script"
            
            # Ensure CARLA is running (calls init.sh if needed)
            project_root = Path(__file__).parent.parent
            ensure_carla_running(script_dir, project_root)
            
            # Run init.sh to clean environment before each run
            print("[INFO] Running init to clean environment...")
            run_init_script(script_dir, project_root)
            
            # Lazy import to avoid api.json dependency
            import config
            from fuzzer import main, init_env, set_args
            
            # Initialize environment
            conf, town, town_map, client, world, G = init_env(args)
            
            # Method-specific configuration
            if method_name == "RAG-ScenarioFuzz":
                conf.enable_rag = True
                conf.enable_rag_metrics = True
                conf.rag_k = kwargs.get('rag_k', 5)
            elif method_name == "ScenarioFuzz-LLM":
                conf.enable_rag = False
                conf.enable_rag_metrics = False
            # Note: DriveFuzz is temporarily disabled
            # elif method_name == "DriveFuzz":
            #     conf.enable_rag = False
            #     conf.enable_rag_metrics = False
            
            # Set scenario limit in config
            conf.max_scenarios = num_scenarios
            
            # Run fuzzing (modified to respect scenario limit)
            self._run_with_scenario_limit(conf, args, experiment_id, num_scenarios)
            
        except KeyboardInterrupt:
            print("\n[ERROR] Experiment interrupted by user")
            raise  # Re-raise to ensure interruption is visible
        except Exception as e:
            print(f"\n[ERROR] Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise to ensure error is visible and not masked
        finally:
            elapsed_time = time.time() - start_time
            try:
                summary = self.progress_tracker.get_summary(experiment_id)
                print(f"\nExperiment status:")
                print(f"  Scenarios generated: {summary.get('completed', 0)}/{summary.get('target', 0)}")
                print(f"  Elapsed time: {summary.get('elapsed_time_str', 'N/A')}")
            except Exception as summary_error:
                print(f"\n[WARNING] Could not get summary: {summary_error}")
    
    def run_timed_experiment(self, method_name: str, duration_hours: float,
                            experiment_id: Optional[str] = None, **kwargs):
        """
        Run experiment with time control (N hours)
        
        Args:
            method_name: Name of the method
            duration_hours: Duration in hours
            experiment_id: Optional experiment ID
            **kwargs: Additional configuration
        """
        duration_seconds = duration_hours * 3600
        
        if experiment_id is None:
            experiment_id = f"{method_name}_timed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"Starting Timed Experiment: {method_name}")
        print(f"Duration: {duration_hours} hours ({duration_seconds} seconds)")
        print(f"Experiment ID: {experiment_id}")
        print(f"{'='*60}\n")
        
        # Start tracking
        self.progress_tracker.start_experiment(
            experiment_id=experiment_id,
            method_name=method_name,
            target_scenarios=999999,  # Large number for time-based
            target_time=duration_seconds
        )
        
        # Create output directory
        method_dir = self.output_base_dir / method_name / experiment_id
        method_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure
        args = self._create_args(method_name, method_dir, **kwargs)
        
        start_time = time.time()
        
        try:
            # Method-specific configuration
            if method_name == "TM-Fuzzer":
                # TM-Fuzzer uses script/test.py, handled separately
                # script/test.py will manage environment via init.sh
                import sys
                import importlib.util
                baseline_path = Path(__file__).parent / "run_tmfuzzer_baseline.py"
                spec = importlib.util.spec_from_file_location("run_tmfuzzer_baseline", baseline_path)
                baseline_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(baseline_module)
                baseline_module.run_tmfuzzer_timed(
                    hours=duration_hours,
                    output_dir=str(self.output_base_dir),
                    target=kwargs.get('target', 'autoware'),
                    density=kwargs.get('density', '0.4'),
                    town=str(kwargs.get('town', 3)),
                    timeout=kwargs.get('timeout', 60)
                )
                return
            
            # For ScenarioFuzz-LLM and RAG-ScenarioFuzz:
            # Need to manage environment before running fuzzer
            from environment_manager import run_init_script, ensure_carla_running
            script_dir = Path(__file__).parent.parent / "script"
            
            # Ensure CARLA is running (calls init.sh if needed)
            project_root = Path(__file__).parent.parent
            ensure_carla_running(script_dir, project_root)
            
            # Run init.sh to clean environment before each run
            print("[INFO] Running init to clean environment...")
            run_init_script(script_dir, project_root)
            
            # Lazy import to avoid api.json dependency
            import config
            from fuzzer import main, init_env, set_args
            
            # Initialize environment
            conf, town, town_map, client, world, G = init_env(args)
            
            # Method-specific configuration
            if method_name == "RAG-ScenarioFuzz":
                conf.enable_rag = True
                conf.enable_rag_metrics = True
                conf.rag_k = kwargs.get('rag_k', 5)
            elif method_name == "ScenarioFuzz-LLM":
                conf.enable_rag = False
                conf.enable_rag_metrics = False
            # Note: DriveFuzz is temporarily disabled
            # elif method_name == "DriveFuzz":
            #     conf.enable_rag = False
            #     conf.enable_rag_metrics = False
            
            # Set timeout in config for fuzzer to check
            conf.experiment_timeout = duration_seconds
            conf.experiment_start_time = start_time
            
            # Run fuzzing - the main loop will check time limit
            # We use a wrapper that monitors time
            import threading
            
            def monitor_progress():
                """Monitor progress in background"""
                while (time.time() - start_time) < duration_seconds:
                    elapsed = time.time() - start_time
                    remaining = duration_seconds - elapsed
                    scenario_count = self._count_scenarios(method_dir)
                    self.progress_tracker.update_progress(experiment_id, scenario_count)
                    
                    if remaining > 0:
                        print(f"[Progress] Elapsed: {timedelta(seconds=int(elapsed))}, "
                              f"Remaining: {timedelta(seconds=int(remaining))}, "
                              f"Scenarios: {scenario_count}")
                    time.sleep(30)  # Update every 30 seconds
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
            monitor_thread.start()
            
            # Run main fuzzing loop
            # Note: fuzzer.py main loop needs to check conf.experiment_timeout
            main(args)
            
        except KeyboardInterrupt:
            print("\nExperiment interrupted by user")
        finally:
            elapsed_time = time.time() - start_time
            summary = self.progress_tracker.get_summary(experiment_id)
            print(f"\nExperiment completed:")
            print(f"  Scenarios generated: {summary['completed']}")
            print(f"  Elapsed time: {summary['elapsed_time_str']}")
    
    def _create_args(self, method_name: str, output_dir: Path, **kwargs):
        """Create argument parser for method"""
        # Lazy import to avoid api.json dependency
        import config
        from fuzzer import set_args
        parser = set_args()
        
        args_list = [
            '--out-dir', str(output_dir),
            '--target', kwargs.get('target', 'behavior'),
            '--max-mutations', str(kwargs.get('max_mutations', 5)),
            '--town', str(kwargs.get('town', 3)),
            '--timeout', str(kwargs.get('timeout', 60))
        ]
        
        if kwargs.get('debug'):
            args_list.append('--debug')
        
        args = parser.parse_args(args_list)
        return args
    
    def _run_with_scenario_limit(self, conf, args, experiment_id: str, max_scenarios: int):
        """
        Run fuzzing with scenario count limit
        Note: This requires modifying fuzzer.py to support scenario counting
        """
        # This is a placeholder - actual implementation requires modifying fuzzer.py
        # to track scenario count and stop when limit is reached
        print("Note: Scenario limit control requires modification to fuzzer.py main loop")
        main(args)
    
    def _count_scenarios(self, output_dir: Path) -> int:
        """Count generated scenarios in output directory"""
        queue_dir = output_dir / "queue"
        if queue_dir.exists():
            return len(list(queue_dir.glob("*.json")))
        return 0


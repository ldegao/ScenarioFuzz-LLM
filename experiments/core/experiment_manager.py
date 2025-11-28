"""
Experiment Manager Module
Manages experiment execution with progress tracking and time estimation
"""

import sys
import os
import time
import json
import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import importlib.util
import threading
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import fuzzer
from experiments.core.progress_tracker import ProgressTracker
from experiments.core.time_estimator import TimeEstimator
from experiments.runners.tmfuzzer.baseline import (
    run_tmfuzzer_quantitative,
    run_tmfuzzer_timed,
)
from experiments.core.environment_manager import run_init_script, ensure_carla_running
from experiments.aggregation.metrics_aggregator import (
    load_records_from_jsonl,
    aggregate_run_metrics,
    save_run_summary,
)


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
        # Pass max_scenarios via args so that fuzzer can enforce the limit
        args = self._create_args(method_name, method_dir, max_scenarios=num_scenarios, **kwargs)
        
        # Modify fuzzer to support quantitative control
        # This requires modifying fuzzer.py to accept scenario count limit
        start_time = time.time()
        scenario_count = 0
        
        try:
            # Method-specific configuration
            if method_name == "TM-Fuzzer":
                # TM-Fuzzer uses script/test.py, handled separately.
                # We still pass a per-experiment output directory to keep
                # the layout consistent with other methods:
                #   ./experiment_results/TM-Fuzzer/<experiment_id>/
                run_tmfuzzer_quantitative(
                    num_scenarios=num_scenarios,
                    output_dir=str(method_dir),
                    target=kwargs.get('target', 'autoware'),
                    density=kwargs.get('density', '0.4'),
                    town=str(kwargs.get('town', 3)),
                    timeout=kwargs.get('timeout', 60)
                )
                return
            
            # For ScenarioFuzz-LLM and RAG-ScenarioFuzz:
            # Need to manage environment before running fuzzer
            script_dir = PROJECT_ROOT / "script"
            
            # Ensure CARLA is running (calls init.sh if needed)
            project_root = PROJECT_ROOT
            ensure_carla_running(script_dir, project_root)
            
            # Run init.sh to clean environment before each run
            print("[INFO] Running init to clean environment...")
            run_init_script(script_dir, project_root)
            
            # Run fuzzing (modified to respect scenario limit)
            self._run_with_scenario_limit(args, experiment_id, num_scenarios)
            
        except KeyboardInterrupt:
            print("\n[ERROR] Experiment interrupted by user")
            raise  # Re-raise to ensure interruption is visible
        except Exception as e:
            print(f"\n[ERROR] Experiment failed: {e}")
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

            # Archive all artifacts for this run (results + metadata) for reproducibility
            try:
                self._archive_experiment_run(method_name, experiment_id, method_dir)
            except Exception as archive_error:
                print(f"[WARNING] Failed to archive experiment run {experiment_id}: {archive_error}")
    
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
                # TM-Fuzzer uses script/test.py, handled separately.
                # Use the per-experiment directory as output root to keep
                # the on-disk layout aligned with other methods.
                run_tmfuzzer_timed(
                    hours=duration_hours,
                    output_dir=str(method_dir),
                    target=kwargs.get('target', 'autoware'),
                    density=kwargs.get('density', '0.4'),
                    town=str(kwargs.get('town', 3)),
                    timeout=kwargs.get('timeout', 60)
                )
                return
            
            # For ScenarioFuzz-LLM and RAG-ScenarioFuzz:
            # Need to manage environment before running fuzzer
            script_dir = PROJECT_ROOT / "script"
            
            # Ensure CARLA is running (calls init.sh if needed)
            project_root = PROJECT_ROOT
            ensure_carla_running(script_dir, project_root)
            
            # Run init.sh to clean environment before each run
            print("[INFO] Running init to clean environment...")
            run_init_script(script_dir, project_root)
            
            # Lazy import to avoid api.json dependency
            # Initialize environment
            conf, town, town_map, client, world, G = fuzzer.init_env(args)
            
            # Method-specific configuration
            if method_name == "RAG-ScenarioFuzz":
                conf.enable_rag = True
                conf.enable_rag_metrics = True
                conf.rag_k = kwargs.get('rag_k', 5)
            elif method_name == "ScenarioFuzz-LLM":
                conf.enable_rag = False
                # Still enable metrics for non-RAG ScenarioFuzz-LLM
                conf.enable_rag_metrics = True
            # Note: DriveFuzz is temporarily disabled
            # elif method_name == "DriveFuzz":
            #     conf.enable_rag = False
            #     conf.enable_rag_metrics = False
            
            # Set timeout in config for fuzzer to check
            conf.experiment_timeout = duration_seconds
            conf.experiment_start_time = start_time
            
            # Run fuzzing - the main loop will check time limit
            # We use a wrapper that monitors time
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
            fuzzer.main(args)
            
        except KeyboardInterrupt:
            print("\nExperiment interrupted by user")
        finally:
            elapsed_time = time.time() - start_time
            try:
                summary = self.progress_tracker.get_summary(experiment_id)
                print(f"\nExperiment completed:")
                print(f"  Scenarios generated: {summary.get('completed', 0)}")
                print(f"  Elapsed time: {summary.get('elapsed_time_str', 'N/A')}")
            except Exception as summary_error:
                print(f"\n[WARNING] Could not get summary: {summary_error}")

            # Archive artifacts for timed run as well
            try:
                self._archive_experiment_run(method_name, experiment_id, method_dir)
            except Exception as archive_error:
                print(f"[WARNING] Failed to archive timed experiment run {experiment_id}: {archive_error}")
    
    def _create_args(self, method_name: str, output_dir: Path, **kwargs):
        """Create argument parser for method"""
        # Lazy import to avoid api.json dependency at module import time
        parser = fuzzer.set_args()
        
        args_list = [
            '--out-dir', str(output_dir),
            '--target', kwargs.get('target', 'behavior'),
            '--max-mutations', str(kwargs.get('max_mutations', 5)),
            '--town', str(kwargs.get('town', 3)),
            '--timeout', str(kwargs.get('timeout', 60)),
            '--sim-port', str(kwargs.get('sim_port', 4000)),
            '--max-scenarios', str(kwargs.get('max_scenarios', 0)),
            '--allow-out-dir-exists',
        ]
        # Always enable multi-dimensional metrics for experiments
        args_list.append('--enable-rag-metrics')
        
        # Enable RAG flags only for RAG-ScenarioFuzz method
        if method_name == "RAG-ScenarioFuzz":
            args_list.append('--enable-rag')
        
        if kwargs.get('debug'):
            args_list.append('--debug')
        
        args = parser.parse_args(args_list)
        return args
    
    def _run_with_scenario_limit(self, args, experiment_id: str, max_scenarios: int):
        """
        Run fuzzing with scenario count limit
        """
        print(f"[INFO] Running fuzzer with scenario limit = {max_scenarios}")

        # We may need to restart the CARLA simulator if it crashes or times out.
        # Wrap the main fuzzing loop in a small retry mechanism that:
        #  - detects the well-known CARLA timeout RuntimeError propagated by fuzzer.evaluation()
        #  - restarts / re-initializes the CARLA environment
        #  - retries the fuzzing run (discarding the current partial run)
        #
        # This prevents the whole experiment process from aborting when the simulator
        # becomes temporarily unavailable.
        max_restarts = 3
        attempt = 0

        # Resolve script and project paths here to avoid circular imports at module load time
        script_dir = PROJECT_ROOT / "script"
        project_root = PROJECT_ROOT

        while True:
            attempt += 1
            try:
                # Ensure CARLA is running and clean the environment before each attempt
                ensure_carla_running(script_dir, project_root)
                print("[INFO] Running init to clean environment before fuzzing attempt...")
                run_init_script(script_dir, project_root)

                print(f"[INFO] Starting fuzzer.main() (attempt {attempt}/{max_restarts})")
                # fuzzer.py tracks total_scenarios_generated via evaluation()
                fuzzer.main(args)
                # If we reach here, the fuzzing run completed successfully
                break
            except RuntimeError as e:
                msg = str(e)
                # Detect CARLA RPC timeout / simulator not responding
                if "time-out of 10000ms while waiting for the simulator" in msg:
                    print("\n[-] CARLA simulator timeout detected in ExperimentManager.")
                    if attempt >= max_restarts:
                        print(
                            f"[ERROR] Exceeded maximum CARLA restart attempts ({max_restarts}); "
                            "aborting experiment."
                        )
                        raise
                    print(
                        "[INFO] Discarding current fuzzing run, restarting CARLA, "
                        "and retrying from the beginning..."
                    )
                    # Loop will retry after re-running ensure_carla_running/init
                    continue
                # For other RuntimeErrors, let the caller handle them
                raise
        
        # After fuzzer exits, update progress tracker with the final count
        try:
            completed = getattr(fuzzer, "total_scenarios_generated", 0)
        except Exception:
            completed = 0
        
        if completed > 0:
            for sid in range(1, completed + 1):
                # We don't have per-scenario info here; record minimal data
                self.progress_tracker.update_progress(
                    experiment_id,
                    scenario_id=sid,
                    scenario_info={'note': 'Recorded via total_scenarios_generated'}
                )
    
    def _count_scenarios(self, output_dir: Path) -> int:
        """Count generated scenarios in output directory"""
        queue_dir = output_dir / "queue"
        if queue_dir.exists():
            return len(list(queue_dir.glob("*.json")))
        return 0

    def _archive_experiment_run(self, method_name: str, experiment_id: str, method_dir: Path):
        """
        Archive all artifacts for a single experiment run into a unified snapshot directory.
        
        This includes:
        - Per-experiment results (queue/metrics under method_dir)
        - Global scenario database (if any)
        - GPT conversation logs
        - Progress checkpoint and time history
        """
        project_root = PROJECT_ROOT
        archive_root = project_root / "data" / "experiment_snapshots" / method_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = archive_root / f"{experiment_id}_{timestamp}"
        
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # 1) Copy per-experiment results
        if method_dir.exists():
            target_results = archive_dir / "results"
            shutil.copytree(str(method_dir), str(target_results))
            print(f"[INFO] Archived results to {target_results}")

            # 1b) Aggregate metrics for this run if metrics_records.jsonl exists
            try:
                metrics_dir = method_dir / "metrics"
                records_path = metrics_dir / "metrics_records.jsonl"
                if records_path.exists():
                    records = load_records_from_jsonl(str(records_path))
                    summary = aggregate_run_metrics(records)
                    # Use the number of records as a fallback for num_scenarios
                    num_scenarios = summary.get("num_records", len(records))
                    summary_path = metrics_dir / "metrics_summary.json"
                    save_run_summary(
                        summary,
                        str(summary_path),
                        method_name=method_name,
                        experiment_id=experiment_id,
                        num_scenarios=num_scenarios,
                    )
            except Exception as metrics_err:
                print(f"[WARNING] Failed to aggregate metrics for run {experiment_id}: {metrics_err}")
        
        # 2) Copy scenario database (global) if present
        scenario_db = project_root / "data" / "scenario_db.json"
        if scenario_db.exists():
            shutil.copy2(str(scenario_db), str(archive_dir / "scenario_db.json"))
            print(f"[INFO] Archived scenario_db.json")
        
        # 3) Copy GPT logs (global) if present
        gpt_logs_dir = project_root / "data" / "gpt_logs"
        if gpt_logs_dir.exists() and any(gpt_logs_dir.iterdir()):
            target_logs = archive_dir / "gpt_logs"
            shutil.copytree(str(gpt_logs_dir), str(target_logs))
            print(f"[INFO] Archived GPT logs to {target_logs}")
        
        # 4) Copy checkpoint and time history for experiment manager
        checkpoint_path = Path(self.progress_tracker.checkpoint_file)
        if checkpoint_path.exists():
            shutil.copy2(str(checkpoint_path), str(archive_dir / "checkpoint.json"))
        
        history_path = Path(self.time_estimator.history_file)
        if history_path.exists():
            shutil.copy2(str(history_path), str(archive_dir / "time_history.json"))


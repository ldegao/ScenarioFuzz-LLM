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
from experiments.core.path_utils import (
    DEFAULT_RUN_ROOT,
    ensure_run_layout,
    snapshot_args,
    snapshot_config_file,
    reorganize_legacy_files,
)
from experiments.runners.tmfuzzer.baseline import (
    run_tmfuzzer_quantitative,
    run_tmfuzzer_timed,
)
from experiments.core.environment_manager import (
    run_init_script, 
    ensure_carla_running,
    restart_carla_container,
    wait_for_port
)
from experiments.core.token_tracker import get_tracker
from experiments.aggregation.metrics_aggregator import (
    load_records_from_jsonl,
    aggregate_run_metrics,
    save_run_summary,
)


class ExperimentManager:
    """
    Manages experiment execution with progress tracking
    """
    
    def __init__(self, output_base_dir: str = str(DEFAULT_RUN_ROOT)):
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
            if method_name == "SimilarityComparison":
                # Include similarity method in experiment ID
                similarity_method = kwargs.get('similarity_scoring_method', 'answer2')
                experiment_id = f"{method_name}_{similarity_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
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
        
        # Clear all old data if experiment directory already exists
        # This ensures each experiment starts completely fresh
        if method_dir.exists():
            queue_dir = method_dir / "queue"
            has_existing_data = False
            
            # Check if there are existing scenarios
            if queue_dir.exists() and any(queue_dir.glob("*.json")):
                has_existing_data = True
            
            # Check if checkpoint exists
            checkpoint_file = method_dir / "ga_checkpoint.pkl"
            if checkpoint_file.exists():
                has_existing_data = True
            
            if has_existing_data:
                print(f"[INFO] Found existing data for experiment {experiment_id}")
                print(f"[INFO] Clearing all old data to start fresh...")
                # Remove entire directory to ensure clean start
                shutil.rmtree(method_dir)
                print(f"[INFO] Removed old experiment directory: {method_dir}")
        
        # Create fresh directory
        method_dir.mkdir(parents=True, exist_ok=True)
        run_paths = ensure_run_layout(method_dir)
        print(f"[INFO] Created fresh experiment directory: {method_dir}")
        
        # Clean scenario database before run (unified for all methods)
        # This ensures a clean start for each experiment
        scenario_db_path = PROJECT_ROOT / "data" / "scenario_db.json"
        if scenario_db_path.exists():
            scenario_db_path.unlink()
            print(f"[INFO] 清空 RAG 场景库：{scenario_db_path}")
        else:
            print(f"[INFO] RAG 场景库为空：{scenario_db_path}")
        
        # Configure
        # Pass max_scenarios via args so that fuzzer can enforce the limit
        args = self._create_args(method_name, method_dir, max_scenarios=num_scenarios, **kwargs)
        # Snapshot args/config for复现
        snapshot_args(vars(args), run_paths["configs"])
        
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
            
            # For ScenarioFuzz-LLM, RAG-ScenarioFuzz, and SimilarityComparison:
            # Defer CARLA init into the retry loop to allow automatic restarts on failure.
            if method_name == "SimilarityComparison":
                similarity_method = kwargs.get('similarity_scoring_method', 'answer2')
                print(f"[SimilarityComparison] Using similarity scoring method: {similarity_method}")
            
            # Run fuzzing (modified to respect scenario limit) with built-in retry/restart.
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
                # Count actual scenarios from output directory
                if method_dir.exists():
                    actual_count = self._count_scenarios(method_dir)
                    # Update progress tracker with actual count
                    if actual_count > 0:
                        self.progress_tracker.update_progress(experiment_id, actual_count)
                    print(f"\nExperiment status:")
                    print(f"  Scenarios generated (from files): {actual_count}/{num_scenarios}")
                else:
                    # Fallback to summary
                    summary = self.progress_tracker.get_summary(experiment_id)
                    print(f"\nExperiment status:")
                    print(f"  Scenarios generated: {summary.get('completed', 0)}/{summary.get('target', 0)}")
                print(f"  Elapsed time: {timedelta(seconds=int(elapsed_time))}")
            except Exception as summary_error:
                print(f"\n[WARNING] Could not get summary: {summary_error}")
                traceback.print_exc()

            # Print and save token usage statistics
            try:
                token_tracker = get_tracker()
                token_tracker.print_summary()
                
                # Save token statistics to experiment directory
                token_stats_file = method_dir / "token_usage.json"
                token_tracker.save_to_file(token_stats_file)
                print(f"[INFO] Token usage statistics saved to {token_stats_file}")
            except Exception as token_error:
                print(f"[WARNING] Could not retrieve token statistics: {token_error}")

            # Copy scenario files from data/output/queue to experiment directory (for TM-Fuzzer compatibility)
            # This handles cases where script/test.py was called without --out-dir
            if method_name == "TM-Fuzzer":
                self._copy_scenario_files_to_experiment_dir(method_dir)
            
            # Copy recorder files from data/output/recorder to experiment directory before archiving
            # This is needed because Docker volume maps to data/output/recorder,
            # but experiment outputs are in experiment_results/.../
            self._copy_recorder_files_to_experiment_dir(method_dir)
            
            # Save scenario database after run (unified for all methods)
            # This ensures the database is saved even if _run_with_scenario_limit didn't save it
            scenario_db_path = PROJECT_ROOT / "data" / "scenario_db.json"
            if scenario_db_path.exists():
                try:
                    db_dest = method_dir / "scenario_db.json"
                    shutil.copy2(str(scenario_db_path), str(db_dest))
                    print(f"[INFO] 保存场景库到实验目录：{db_dest}")
                except Exception as copy_db_err:
                    print(f"[WARNING] Failed to save scenario DB snapshot: {copy_db_err}")

            try:
                reorganize_legacy_files(method_dir)
            except Exception as reorganize_error:
                print(f"[WARNING] Failed to reorganize run layout: {reorganize_error}")
            
            # Archive all artifacts for this run (results + metadata) for reproducibility
            # Only archive if experiment completed successfully (not interrupted)
            # Check if experiment was interrupted by checking if we reached the target
            try:
                if method_dir.exists():
                    actual_count = self._count_scenarios(method_dir)
                    # Only archive if we generated at least some scenarios (experiment ran)
                    # This prevents archiving on early failures
                    if actual_count > 0:
                        self._archive_experiment_run(method_name, experiment_id, method_dir)
                    else:
                        print(f"[INFO] Skipping archive: no scenarios generated for {experiment_id}")
                else:
                    print(f"[INFO] Skipping archive: experiment directory does not exist for {experiment_id}")
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
            if method_name == "SimilarityComparison":
                # Include similarity method in experiment ID
                similarity_method = kwargs.get('similarity_scoring_method', 'answer2')
                experiment_id = f"{method_name}_{similarity_method}_timed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
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
        run_paths = ensure_run_layout(method_dir)
        
        # Configure
        args = self._create_args(method_name, method_dir, **kwargs)
        snapshot_args(vars(args), run_paths["configs"])
        # Prepare scenario DB path (no auto-clear here to preserve state on unexpected interrupts)
        scenario_db_path = Path(getattr(args, "scenario_db", "./data/scenario_db.json"))
        
        start_time = time.time()
        experiment_start_time = start_time  # Record experiment start time for recorder file filtering
        
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
            
            # For ScenarioFuzz-LLM, RAG-ScenarioFuzz, and SimilarityComparison:
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
                # Note: Metrics calculation has been moved to experiments/analysis/calculate_metrics.py
                conf.rag_k = kwargs.get('rag_k', 5)
            elif method_name == "ScenarioFuzz-LLM":
                conf.enable_rag = False
                # Still enable metrics for non-RAG ScenarioFuzz-LLM
                # Note: Metrics calculation has been moved to experiments/analysis/calculate_metrics.py
            elif method_name == "SimilarityComparison":
                # SimilarityComparison: always enable RAG (needed for embedding/hybrid methods)
                conf.enable_rag = True
                # Note: Metrics calculation has been moved to experiments/analysis/calculate_metrics.py
                conf.rag_k = kwargs.get('rag_k', 5)
                # Configure similarity scoring method
                similarity_method = kwargs.get('similarity_scoring_method', 'answer2')
                conf.similarity_scoring_method = similarity_method
                # Configure hybrid method weights
                conf.hybrid_embedding_weight = kwargs.get('hybrid_embedding_weight', 0.6)
                # Configure feature method weights
                conf.feature_position_weight = kwargs.get('feature_position_weight', 0.3)
                conf.feature_speed_weight = kwargs.get('feature_speed_weight', 0.3)
                conf.feature_angular_accel_weight = kwargs.get('feature_angular_accel_weight', 0.2)
                conf.feature_relative_position_weight = kwargs.get('feature_relative_position_weight', 0.2)
                print(f"[SimilarityComparison] Using similarity scoring method: {similarity_method}")
            # Note: DriveFuzz is disabled
            # DriveFuzz support has been removed. If needed in the future,
            # uncomment and update the following:
            # elif method_name == "DriveFuzz":
            #     conf.enable_rag = False
            
            # Set timeout in config for fuzzer to check
            conf.experiment_timeout = duration_seconds
            conf.experiment_start_time = start_time
            
            # Get max_scenarios if specified (for dual constraint mode)
            max_scenarios = kwargs.get('max_scenarios', None)
            if max_scenarios is not None and max_scenarios > 0:
                # Ensure max_scenarios is set in args for fuzzer to check
                if hasattr(args, 'max_scenarios'):
                    args.max_scenarios = max_scenarios
                else:
                    args.max_scenarios = max_scenarios
                print(f"[INFO] Timed experiment with dual constraints: time limit = {duration_hours}h, scenario limit = {max_scenarios}")
            
            # Run fuzzing - the main loop will check time limit and scenario limit
            # We use a wrapper that monitors both constraints
            # Use threading.Event for proper thread management
            stop_monitoring = threading.Event()
            
            def monitor_progress():
                """Monitor progress in background, checking both time and scenario limits"""
                while not stop_monitoring.is_set():
                    elapsed = time.time() - start_time
                    remaining = duration_seconds - elapsed
                    
                    # Check if time limit reached
                    if elapsed >= duration_seconds:
                        print(f"[TIME LIMIT] Reached time limit: {elapsed:.0f}s / {duration_seconds:.0f}s")
                        break
                    
                    scenario_count = self._count_scenarios(method_dir)
                    self.progress_tracker.update_progress(experiment_id, scenario_count)
                    
                    # Check if scenario limit reached (if specified)
                    if max_scenarios is not None and max_scenarios > 0:
                        if scenario_count >= max_scenarios:
                            print(f"[SCENARIO LIMIT] Reached scenario limit: {scenario_count}/{max_scenarios}")
                            break
                        remaining_scenarios = max_scenarios - scenario_count
                        if remaining > 0:
                            print(f"[Progress] Elapsed: {timedelta(seconds=int(elapsed))}, "
                                  f"Remaining time: {timedelta(seconds=int(remaining))}, "
                                  f"Scenarios: {scenario_count}/{max_scenarios} (need {remaining_scenarios} more)")
                    else:
                        if remaining > 0:
                            print(f"[Progress] Elapsed: {timedelta(seconds=int(elapsed))}, "
                                  f"Remaining: {timedelta(seconds=int(remaining))}, "
                                  f"Scenarios: {scenario_count}")
                    
                    # Wait with timeout to allow checking stop_event
                    if stop_monitoring.wait(timeout=30):
                        break  # Event was set, exit loop
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
            monitor_thread.start()
            
            try:
                # Run main fuzzing loop
                # Note: fuzzer.py main loop needs to check conf.experiment_timeout
                fuzzer.main(args)
                # Snapshot scenario DB after run
                if scenario_db_path.exists():
                    try:
                        db_dest = method_dir / "scenario_db.json"
                        shutil.copy(scenario_db_path, db_dest)
                        print(f"[INFO] Saved scenario DB snapshot to {db_dest}")
                    except Exception as copy_db_err:
                        print(f"[WARNING] Failed to save scenario DB snapshot: {copy_db_err}")
            finally:
                # Signal monitoring thread to stop
                stop_monitoring.set()
                # Wait for thread to finish (with timeout)
                monitor_thread.join(timeout=5.0)
            
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

            # Print and save token usage statistics
            try:
                token_tracker = get_tracker()
                token_tracker.print_summary()
                
                # Save token statistics to experiment directory
                token_stats_file = method_dir / "token_usage.json"
                token_tracker.save_to_file(token_stats_file)
                print(f"[INFO] Token usage statistics saved to {token_stats_file}")
            except Exception as token_error:
                print(f"[WARNING] Could not retrieve token statistics: {token_error}")

            # Copy recorder files from data/output/recorder to experiment directory before archiving
            self._copy_recorder_files_to_experiment_dir(method_dir)

            try:
                reorganize_legacy_files(method_dir)
            except Exception as reorganize_error:
                print(f"[WARNING] Failed to reorganize run layout: {reorganize_error}")
            
            # Archive artifacts for timed run as well
            # Only archive if experiment completed successfully
            try:
                if method_dir.exists():
                    actual_count = self._count_scenarios(method_dir)
                    if actual_count > 0:
                        self._archive_experiment_run(method_name, experiment_id, method_dir)
                    else:
                        print(f"[INFO] Skipping archive: no scenarios generated for {experiment_id}")
                else:
                    print(f"[INFO] Skipping archive: experiment directory does not exist for {experiment_id}")
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
            '--rag-k', str(kwargs.get('rag_k', 5)),
            '--similarity-scoring-method', str(kwargs.get('similarity_scoring_method', 'answer2')),
            '--hybrid-embedding-weight', str(kwargs.get('hybrid_embedding_weight', 0.6)),
            '--feature-position-weight', str(kwargs.get('feature_position_weight', 0.3)),
            '--feature-speed-weight', str(kwargs.get('feature_speed_weight', 0.3)),
            '--feature-angular-accel-weight', str(kwargs.get('feature_angular_accel_weight', 0.2)),
            '--feature-relative-position-weight', str(kwargs.get('feature_relative_position_weight', 0.2)),
            '--allow-out-dir-exists',
        ]

        # Custom seed directory (e.g., reuse historical seed pool)
        if kwargs.get('seed_dir'):
            args_list.extend(['--seed-dir', str(kwargs.get('seed_dir'))])
        
        # Add determ-seed if provided
        if kwargs.get('determ_seed') is not None:
            args_list.extend(['--determ-seed', str(kwargs.get('determ_seed'))])

        # Ablation toggles
        if kwargs.get('disable_similarity'):
            args_list.append('--disable-similarity')
        if kwargs.get('disable_guided_mutation'):
            args_list.append('--disable-guided-mutation')
        
        # Note: Metrics calculation has been moved to experiments/analysis/calculate_metrics.py
        # Metrics are now calculated offline after experiment completion
        
        # Enable RAG flags for RAG-ScenarioFuzz and SimilarityComparison methods
        if method_name == "RAG-ScenarioFuzz" or method_name == "SimilarityComparison":
            args_list.append('--enable-rag')
        
        if kwargs.get('debug'):
            args_list.append('--debug')
        
        args = parser.parse_args(args_list)
        return args
    
    def _run_with_scenario_limit(self, args, experiment_id: str, max_scenarios: int):
        """
        Run fuzzing with scenario count limit.
        Ensures that exactly max_scenarios scenarios are generated by retrying if needed.
        """
        print(f"[INFO] Running fuzzer with scenario limit = {max_scenarios}")

        # We may need to restart the CARLA simulator if it crashes, times out, or connection fails.
        # Wrap the main fuzzing loop in a retry mechanism that:
        #  - detects connection failures (RuntimeError from utils.connect)
        #  - detects CARLA RPC timeout RuntimeError propagated by fuzzer.evaluation()
        #  - restarts / re-initializes the CARLA environment and waits for port to be available
        #  - retries the fuzzing run (discarding the current partial run)
        #  - infinite retries to ensure the test can run
        #  - checks scenario count after each run and continues until target is reached
        #
        # This prevents the whole experiment process from aborting when the simulator
        # becomes temporarily unavailable, and ensures we complete the target number of scenarios.
        attempt = 0
        # Retry limits: set to infinity to align with "no max time" expectation.
        # Change to finite values if operational safety limits are desired.
        MAX_RETRY_ATTEMPTS = float("inf")
        retry_start_time = time.time()
        MAX_RETRY_DURATION = float("inf")  # seconds

        # Resolve script and project paths here to avoid circular imports at module load time
        script_dir = PROJECT_ROOT / "script"
        project_root = PROJECT_ROOT
        
        # Get port from args
        sim_port = getattr(args, 'sim_port', 4000)
        
        # Get output directory to count scenarios
        output_dir = Path(args.out_dir) if hasattr(args, 'out_dir') else None
        print(f"[INFO] Output directory: {output_dir}")

        # Prepare scenario DB path (no auto-clear here; scripts handle clean between parts)
        scenario_db_path = Path(getattr(args, "scenario_db", "./data/scenario_db.json"))

        # Only run init script once at the beginning
        init_script_run = False

        while True:
            attempt += 1
            
            # Check retry limits to prevent infinite loops
            if attempt > MAX_RETRY_ATTEMPTS:
                print(f"[ERROR] Maximum retry attempts ({MAX_RETRY_ATTEMPTS}) reached. Aborting experiment.")
                raise RuntimeError(f"Maximum retry attempts ({MAX_RETRY_ATTEMPTS}) reached. Experiment aborted.")
            
            elapsed_retry_time = time.time() - retry_start_time
            if elapsed_retry_time > MAX_RETRY_DURATION:
                print(f"[ERROR] Maximum retry duration ({MAX_RETRY_DURATION/3600:.1f} hours) exceeded. Aborting experiment.")
                raise RuntimeError(f"Maximum retry duration exceeded. Experiment aborted.")
            
            # Count existing scenarios before this attempt
            if output_dir:
                existing_scenarios = self._count_scenarios(output_dir)
                print(f"[INFO] Existing scenarios in {output_dir}: {existing_scenarios}")
                
                # Copy recorder files from data/output/recorder to experiment directory
                # This ensures recorder files are copied even if experiment was interrupted
                self._copy_recorder_files_to_experiment_dir(output_dir)
            else:
                existing_scenarios = 0
                print(f"[WARNING] No output directory specified, cannot count scenarios")
            
            # Calculate how many more scenarios we need
            remaining_scenarios = max_scenarios - existing_scenarios
            
            if remaining_scenarios <= 0:
                print(f"[INFO] Target scenario count already reached: {existing_scenarios}/{max_scenarios}")
                break
            
            print(f"[INFO] Attempt {attempt}: {existing_scenarios}/{max_scenarios} scenarios completed, need {remaining_scenarios} more")
            
            # Update max_scenarios in args to reflect remaining scenarios needed
            # This ensures fuzzer knows how many more to generate
            # IMPORTANT: max_scenarios is controlled by the --num-scenarios parameter (passed as num_scenarios)
            # We set args.max_scenarios to the actual target so fuzzer continues until that target is reached
            # The fuzzer will check file count and stop when it reaches max_scenarios
            if hasattr(args, 'max_scenarios'):
                original_max = args.max_scenarios
                # Set to actual target (e.g., 100 from --num-scenarios 100)
                # This ensures fuzzer continues until the real target is reached
                args.max_scenarios = max_scenarios  # Use the actual target from --num-scenarios parameter
                print(f"[INFO] Set args.max_scenarios to {max_scenarios} (target from --num-scenarios parameter)")
            else:
                original_max = None
                args.max_scenarios = max_scenarios  # Use the actual target from --num-scenarios parameter
                print(f"[INFO] Set args.max_scenarios to {max_scenarios} (target from --num-scenarios parameter)")
            
            try:
                # Ensure CARLA is running
                # This will raise RuntimeError or FileNotFoundError if CARLA cannot be started
                try:
                    ensure_carla_running(script_dir, project_root)
                except (RuntimeError, FileNotFoundError) as carla_start_error:
                    print(f"[ERROR] Failed to ensure CARLA is running (attempt {attempt}): {carla_start_error}")
                    print("[INFO] Will retry in next attempt...")
                    # Restore original max_scenarios
                    if original_max is not None:
                        args.max_scenarios = original_max
                    else:
                        delattr(args, 'max_scenarios')
                    # Continue to retry
                    continue
                
                # Only run init script on first attempt to avoid cleaning experiment output
                if not init_script_run:
                    print("[INFO] Running init to clean environment (first time only)...")
                    try:
                        run_init_script(script_dir, project_root)
                    except (RuntimeError, FileNotFoundError) as init_error:
                        print(f"[ERROR] Failed to run init script (attempt {attempt}): {init_error}")
                        print("[INFO] Will retry in next attempt...")
                        # Restore original max_scenarios
                        if original_max is not None:
                            args.max_scenarios = original_max
                        else:
                            delattr(args, 'max_scenarios')
                        # Continue to retry
                        continue
                    init_script_run = True
                else:
                    print("[INFO] Skipping init script (already run, preserving experiment output)")
                
                # Wait for port to be available before starting fuzzer
                print(f"[INFO] Checking if port {sim_port} is available...")
                if not wait_for_port("localhost", sim_port, timeout=60):
                    print(f"[WARNING] Port {sim_port} not available, restarting CARLA container...")
                    if not restart_carla_container(script_dir, project_root, port=sim_port):
                        print(f"[WARNING] Failed to restart CARLA container, will retry...")
                        # Restore original max_scenarios
                        if original_max is not None:
                            args.max_scenarios = original_max
                        else:
                            delattr(args, 'max_scenarios')
                        continue

                print(f"[INFO] Starting fuzzer.main() (attempt {attempt}, target: {remaining_scenarios} more scenarios)")
                print(f"[INFO] Fuzzer args: out_dir={args.out_dir}, max_scenarios={args.max_scenarios}")
                # fuzzer.py tracks total_scenarios_generated via evaluation()
                try:
                    fuzzer.main(args)
                    print(f"[INFO] fuzzer.main() completed successfully")
                    
                    # Copy recorder files after successful fuzzer run
                    if output_dir:
                        self._copy_recorder_files_to_experiment_dir(output_dir)
                        # Snapshot scenario DB after run
                        if scenario_db_path.exists():
                            try:
                                db_dest = output_dir / "scenario_db.json"
                                shutil.copy(scenario_db_path, db_dest)
                                print(f"[INFO] Saved scenario DB snapshot to {db_dest}")
                            except Exception as copy_db_err:
                                print(f"[WARNING] Failed to save scenario DB snapshot: {copy_db_err}")
                except RuntimeError as runtime_err:
                    # Check if this is a CARLA connection/timeout error
                    msg = str(runtime_err)
                    if "time-out" in msg or "simulator" in msg.lower() or "connection" in msg.lower():
                        print(f"[WARNING] CARLA connection/timeout error in fuzzer.main(): {runtime_err}")
                        # Count scenarios to see if we made any progress
                        if output_dir:
                            completed_after_error = self._count_scenarios(output_dir)
                            print(f"[INFO] Scenarios after fuzzer error: {completed_after_error}/{max_scenarios}")
                            # Copy recorder files even if there was an error (they may have been generated)
                            self._copy_recorder_files_to_experiment_dir(output_dir)
                            if completed_after_error > existing_scenarios:
                                print(f"[INFO] Made progress: {completed_after_error - existing_scenarios} new scenarios")
                                # Made progress, but still need to handle the error
                                # Re-raise to trigger retry logic
                                raise
                            else:
                                print(f"[WARNING] No progress made, treating as connection/environment issue")
                                raise
                        else:
                            # No output dir, re-raise to trigger retry
                            raise
                    else:
                        # Other RuntimeError (e.g., AttributeError from mutation)
                        print(f"[WARNING] fuzzer.main() raised RuntimeError: {runtime_err}")
                        traceback.print_exc()
                        # Count scenarios to see if we made any progress
                        if output_dir:
                            completed_after_error = self._count_scenarios(output_dir)
                            print(f"[INFO] Scenarios after fuzzer error: {completed_after_error}/{max_scenarios}")
                            # Copy recorder files even if there was an error (they may have been generated)
                            self._copy_recorder_files_to_experiment_dir(output_dir)
                            if completed_after_error > existing_scenarios:
                                print(f"[INFO] Made progress: {completed_after_error - existing_scenarios} new scenarios")
                                # Made progress, but error occurred - re-raise to trigger retry
                                raise
                            else:
                                print(f"[WARNING] No progress made, treating as error")
                                raise
                        else:
                            raise
                except Exception as fuzzer_error:
                    # If fuzzer fails, check if we've made progress
                    print(f"[WARNING] fuzzer.main() raised exception: {fuzzer_error}")
                    traceback.print_exc()
                    # Count scenarios to see if we made any progress
                    if output_dir:
                        completed_after_error = self._count_scenarios(output_dir)
                        print(f"[INFO] Scenarios after fuzzer error: {completed_after_error}/{max_scenarios}")
                        # Copy recorder files even if there was an error (they may have been generated)
                        self._copy_recorder_files_to_experiment_dir(output_dir)
                        # If we made progress, continue; otherwise treat as connection error
                        if completed_after_error > existing_scenarios:
                            # Made some progress, continue to check completion
                            print(f"[INFO] Made progress: {completed_after_error - existing_scenarios} new scenarios")
                            # Still re-raise to trigger retry logic
                            raise
                        else:
                            # No progress, treat as connection/environment issue
                            print(f"[WARNING] No progress made, treating as connection/environment issue")
                            raise
                    else:
                        raise
                
                # Check if we've reached the target number of scenarios
                # Count scenarios from output directory (more reliable than global variable)
                if output_dir:
                    completed = self._count_scenarios(output_dir)
                    print(f"[INFO] Scenarios in output directory: {completed}")
                    # Copy recorder files after checking completion (in case of successful completion)
                    self._copy_recorder_files_to_experiment_dir(output_dir)
                else:
                    # Fallback to global variable
                    try:
                        completed = getattr(fuzzer, "total_scenarios_generated", 0)
                        print(f"[INFO] Scenarios from fuzzer global: {completed}")
                    except Exception:
                        completed = 0
                
                print(f"[INFO] Completed scenarios: {completed}/{max_scenarios}")
                
                # Update progress tracker
                if completed > 0:
                    self.progress_tracker.update_progress(experiment_id, completed)
                
                if completed >= max_scenarios:
                    print(f"[INFO] Target scenario count reached: {completed}/{max_scenarios}")
                    break
                else:
                    remaining = max_scenarios - completed
                    print(f"[INFO] Target not reached. Need {remaining} more scenarios. Continuing...")
                    # Restore original max_scenarios for next iteration
                    if original_max is not None:
                        args.max_scenarios = original_max
                    else:
                        delattr(args, 'max_scenarios')
                    # Continue to retry
                    continue
                    
            except (ConnectionError, RuntimeError) as e:
                msg = str(e)
                # Detect connection failure
                if "Failed to connect to CARLA" in msg or "Check client connection" in msg:
                    print(f"\n[-] CARLA connection failure detected (attempt {attempt}): {msg}")
                    print("[INFO] Restarting CARLA container and waiting for port to be available...")
                    if not restart_carla_container(script_dir, project_root, port=sim_port):
                        print(f"[ERROR] Failed to restart CARLA container after {attempt} attempts")
                        if attempt >= MAX_RETRY_ATTEMPTS:
                            raise RuntimeError(f"Failed to restart CARLA container after {MAX_RETRY_ATTEMPTS} attempts")
                    # Check if we've reached target before continuing
                    if output_dir:
                        completed = self._count_scenarios(output_dir)
                        if completed >= max_scenarios:
                            print(f"[INFO] Target scenario count reached: {completed}/{max_scenarios}")
                            break
                    # Continue to retry
                    continue
                # Detect CARLA RPC timeout / simulator not responding
                elif "time-out of 10000ms while waiting for the simulator" in msg:
                    print(f"\n[-] CARLA simulator timeout detected (attempt {attempt}).")
                    print("[INFO] Restarting CARLA container and waiting for port to be available...")
                    if not restart_carla_container(script_dir, project_root, port=sim_port):
                        print(f"[ERROR] Failed to restart CARLA container after {attempt} attempts")
                        if attempt >= MAX_RETRY_ATTEMPTS:
                            raise RuntimeError(f"Failed to restart CARLA container after {MAX_RETRY_ATTEMPTS} attempts")
                    # Check if we've reached target before continuing
                    if output_dir:
                        completed = self._count_scenarios(output_dir)
                        if completed >= max_scenarios:
                            print(f"[INFO] Target scenario count reached: {completed}/{max_scenarios}")
                            break
                    # Continue to retry
                    continue
                # For other RuntimeErrors, let the caller handle them
                raise
            except (OSError, TimeoutError) as e:
                # Catch OS-level errors (file system, network) and timeouts
                msg = str(e)
                if "connection" in msg.lower() or "timeout" in msg.lower() or "refused" in msg.lower():
                    print(f"\n[-] Connection-related error detected (attempt {attempt}): {msg}")
                    print("[INFO] Restarting CARLA container and waiting for port to be available...")
                    if not restart_carla_container(script_dir, project_root, port=sim_port):
                        print(f"[ERROR] Failed to restart CARLA container after {attempt} attempts")
                        if attempt >= MAX_RETRY_ATTEMPTS:
                            raise RuntimeError(f"Failed to restart CARLA container after {MAX_RETRY_ATTEMPTS} attempts")
                    # Check if we've reached target before continuing
                    if output_dir:
                        completed = self._count_scenarios(output_dir)
                        if completed >= max_scenarios:
                            print(f"[INFO] Target scenario count reached: {completed}/{max_scenarios}")
                            break
                    # Continue to retry
                    continue
                # For other OS errors, re-raise
                raise
            except RuntimeError as e:
                # Catch RuntimeError (including fatal errors from fuzzer)
                msg = str(e)
                # Check if this is a fatal error that should trigger retry
                if "Fatal error occurred during test" in msg or "ret == -1" in msg:
                    print(f"\n[ERROR] Fatal error detected (attempt {attempt}): {msg}")
                    print("[INFO] This is a recoverable error, will retry...")
                    # Count scenarios to see if we made any progress
                    if output_dir:
                        completed = self._count_scenarios(output_dir)
                        print(f"[INFO] Scenarios after fatal error: {completed}/{max_scenarios}")
                        if completed >= max_scenarios:
                            print(f"[INFO] Target scenario count reached despite error: {completed}/{max_scenarios}")
                            break
                    # Only retry if we haven't exceeded max attempts
                    if attempt >= MAX_RETRY_ATTEMPTS:
                        print(f"[ERROR] Maximum retry attempts reached. Re-raising exception.")
                        raise
                    # Restart CARLA and retry
                    print("[INFO] Restarting CARLA container and retrying...")
                    if not restart_carla_container(script_dir, project_root, port=sim_port):
                        print(f"[ERROR] Failed to restart CARLA container after {attempt} attempts")
                        if attempt >= MAX_RETRY_ATTEMPTS:
                            raise RuntimeError(f"Failed to restart CARLA container after {MAX_RETRY_ATTEMPTS} attempts")
                    time.sleep(10)
                    continue
                else:
                    # Other RuntimeErrors - re-raise
                    raise
            except Exception as e:
                # Catch any other unexpected exceptions
                msg = str(e)
                print(f"\n[ERROR] Unexpected error (attempt {attempt}): {type(e).__name__}: {msg}")
                # Only retry for known recoverable errors
                if attempt >= MAX_RETRY_ATTEMPTS:
                    print(f"[ERROR] Maximum retry attempts reached. Re-raising exception.")
                    raise
                # For unknown errors, wait a bit before retrying
                time.sleep(10)
                continue
        
        # After fuzzer exits, update progress tracker with the final count
        if output_dir:
            completed = self._count_scenarios(output_dir)
            print(f"[INFO] Final scenario count from output directory: {completed}")
        else:
            try:
                completed = getattr(fuzzer, "total_scenarios_generated", 0)
                print(f"[INFO] Final scenario count from fuzzer global: {completed}")
            except Exception:
                completed = 0
        
        # Update progress tracker with actual count
        if completed > 0:
            # Update progress tracker with the actual count
            self.progress_tracker.update_progress(experiment_id, completed)
            print(f"[INFO] Updated progress tracker: {completed} scenarios")
    
    def _count_scenarios(self, output_dir: Path) -> int:
        """Count generated scenarios in output directory"""
        queue_dir = output_dir / "queue"
        scenarios_dir = output_dir / "scenarios"
        if queue_dir.exists():
            return len(list(queue_dir.glob("*.json")))
        if scenarios_dir.exists():
            return len(list(scenarios_dir.glob("*.json")))
        return 0

    def continue_experiment(self, method_name: str, experiment_id: str, additional_scenarios: int, **kwargs):
        """
        Continue an existing experiment by generating additional scenarios.
        
        Args:
            method_name: Name of the method
            experiment_id: Existing experiment ID to continue
            additional_scenarios: Number of additional scenarios to generate
            **kwargs: Additional configuration (target, town, timeout, etc.)
        """
        print(f"\n{'='*60}")
        print(f"Continuing Experiment: {method_name}")
        print(f"Experiment ID: {experiment_id}")
        print(f"Additional scenarios: {additional_scenarios}")
        print(f"{'='*60}\n")
        
        # Find existing experiment directory
        method_dir = self.output_base_dir / method_name / experiment_id
        
        if not method_dir.exists():
            raise ValueError(f"Experiment directory not found: {method_dir}")
        
        run_paths = ensure_run_layout(method_dir)
        
        # Count existing scenarios
        existing_count = self._count_scenarios(method_dir)
        print(f"[INFO] Found {existing_count} existing scenarios in {method_dir}")
        
        # Calculate target scenario count
        target_scenarios = existing_count + additional_scenarios
        print(f"[INFO] Target: {target_scenarios} scenarios ({existing_count} existing + {additional_scenarios} new)")
        
        # Check if checkpoint exists
        checkpoint_file = method_dir / "ga_checkpoint.pkl"
        if not checkpoint_file.exists():
            print(f"[WARNING] Checkpoint file not found: {checkpoint_file}")
            print("[INFO] Will start from existing scenario files or create new run")
        
        # Update progress tracker
        self.progress_tracker.start_experiment(
            experiment_id=experiment_id,
            method_name=method_name,
            target_scenarios=target_scenarios
        )
        
        # Configure args for continuing experiment
        # Use existing output directory, don't clear it
        args = self._create_args(method_name, method_dir, max_scenarios=target_scenarios, **kwargs)
        snapshot_args(vars(args), run_paths["configs"])
        
        # Ensure we don't clear existing data
        args.allow_out_dir_exists = True
        
        start_time = time.time()
        
        try:
            # Method-specific configuration
            if method_name == "TM-Fuzzer":
                raise NotImplementedError("Continue experiment not supported for TM-Fuzzer")
            
            # For ScenarioFuzz-LLM and RAG-ScenarioFuzz:
            script_dir = PROJECT_ROOT / "script"
            project_root = PROJECT_ROOT
            
            # Ensure CARLA is running (calls init.sh if needed)
            ensure_carla_running(script_dir, project_root)
            
            # Don't run init script when continuing - preserve existing experiment output
            print("[INFO] Skipping init script (continuing experiment, preserving output)")
            
            # Run fuzzing with scenario limit
            self._run_with_scenario_limit(args, experiment_id, target_scenarios)
            
        except KeyboardInterrupt:
            print("\n[ERROR] Experiment interrupted by user")
            raise
        except Exception as e:
            print(f"\n[ERROR] Experiment failed: {e}")
            traceback.print_exc()
            raise
        finally:
            elapsed_time = time.time() - start_time
            try:
                if method_dir.exists():
                    actual_count = self._count_scenarios(method_dir)
                    if actual_count > 0:
                        self.progress_tracker.update_progress(experiment_id, actual_count)
                    print(f"\nExperiment status:")
                    print(f"  Scenarios generated (from files): {actual_count}/{target_scenarios}")
                    print(f"  Additional scenarios generated: {actual_count - existing_count}")
                else:
                    summary = self.progress_tracker.get_summary(experiment_id)
                    print(f"\nExperiment status:")
                    print(f"  Scenarios generated: {summary.get('completed', 0)}/{summary.get('target', 0)}")
                print(f"  Elapsed time: {timedelta(seconds=int(elapsed_time))}")
            except Exception as summary_error:
                print(f"\n[WARNING] Could not get summary: {summary_error}")
                traceback.print_exc()

            # Print and save token usage statistics
            try:
                token_tracker = get_tracker()
                token_tracker.print_summary()
                
                token_stats_file = method_dir / "token_usage.json"
                token_tracker.save_to_file(token_stats_file)
                print(f"[INFO] Token usage statistics saved to {token_stats_file}")
            except Exception as token_error:
                print(f"[WARNING] Could not retrieve token statistics: {token_error}")

            # Copy scenario files from data/output/queue to experiment directory (for TM-Fuzzer compatibility)
            if method_name == "TM-Fuzzer":
                try:
                    self._copy_scenario_files_to_experiment_dir(method_dir)
                except Exception as scenario_copy_error:
                    print(f"[WARNING] Could not copy scenario files: {scenario_copy_error}")
            
            # Copy recorder files and整理布局
            try:
                self._copy_recorder_files_to_experiment_dir(method_dir)
            except Exception as recorder_error:
                print(f"[WARNING] Could not copy recorder files: {recorder_error}")

            try:
                reorganize_legacy_files(method_dir)
            except Exception as reorganize_error:
                print(f"[WARNING] Failed to reorganize run layout: {reorganize_error}")

            # Archive all artifacts for this run
            # Only archive if experiment completed successfully
            try:
                if method_dir.exists():
                    actual_count = self._count_scenarios(method_dir)
                    if actual_count > 0:
                        self._archive_experiment_run(method_name, experiment_id, method_dir)
                    else:
                        print(f"[INFO] Skipping archive: no scenarios generated for {experiment_id}")
                else:
                    print(f"[INFO] Skipping archive: experiment directory does not exist for {experiment_id}")
            except Exception as archive_error:
                print(f"[WARNING] Failed to archive experiment run {experiment_id}: {archive_error}")
    
    def _copy_scenario_files_to_experiment_dir(self, method_dir: Path):
        """
        Copy scenario files from data/output/queue to experiment directory.
        
        This is needed for TM-Fuzzer when script/test.py was called without --out-dir,
        causing scenarios to be saved to data/output/queue instead of the experiment directory.
        
        Args:
            method_dir: Path to experiment method directory
        """
        try:
            if not method_dir.exists():
                return
            
            data_queue_dir = PROJECT_ROOT / "data" / "output" / "queue"
            experiment_queue_dir = method_dir / "queue"
            
            # Create experiment queue directory if it doesn't exist
            experiment_queue_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if experiment queue already has files
            existing_scenario_files = list(experiment_queue_dir.glob("*.pkl")) + list(experiment_queue_dir.glob("*.json"))
            if existing_scenario_files:
                print(f"[INFO] Experiment queue directory already has {len(existing_scenario_files)} file(s), skipping copy")
                return
            
            # Check if data/output/queue has scenario files
            if not data_queue_dir.exists() or not data_queue_dir.is_dir():
                return
            
            scenario_files = list(data_queue_dir.glob("*.pkl")) + list(data_queue_dir.glob("*.json"))
            if not scenario_files:
                return
            
            # Copy all scenario files from data/output/queue to experiment queue
            copied_count = 0
            for scenario_file in scenario_files:
                target_file = experiment_queue_dir / scenario_file.name
                if not target_file.exists():
                    shutil.copy2(str(scenario_file), str(target_file))
                    copied_count += 1
            
            if copied_count > 0:
                print(f"[INFO] Copied {copied_count} scenario file(s) from {data_queue_dir} to {experiment_queue_dir}")
        except Exception as scenario_copy_error:
            print(f"[WARNING] Failed to copy scenario files to experiment directory: {scenario_copy_error}")
            import traceback
            traceback.print_exc()
    
    def _copy_recorder_files_to_experiment_dir(self, method_dir: Path):
        """
        Copy recorder files from data/output/recorder to experiment directory.
        
        This is needed because Docker volume maps to data/output/recorder,
        but experiment outputs are in experiment_results/.../
        
        Strategy:
        1. If recorder directory already has files, skip (already copied)
        2. Otherwise, match recorder files to scenario files by gid/sid
        3. Copy matched recorder files to experiment directory
        
        Args:
            method_dir: Path to experiment method directory
        """
        try:
            if not method_dir.exists():
                return
            
            data_recorder_dir = PROJECT_ROOT / "data" / "output" / "recorder"
            experiment_recorder_dir = method_dir / "recorder"
            queue_dir = method_dir / "queue"
            
            # Create experiment recorder directory if it doesn't exist
            experiment_recorder_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if recorder directory already has files
            existing_recorder_files = list(experiment_recorder_dir.glob("*.log"))
            if existing_recorder_files:
                print(f"[INFO] Recorder directory already has {len(existing_recorder_files)} file(s), skipping copy")
                return
            
            # Get scenario files to match recorder files
            scenario_files = []
            if queue_dir.exists():
                scenario_files = list(queue_dir.glob("*.json"))
            
            if not scenario_files:
                print(f"[INFO] No scenario files found in {queue_dir}, cannot match recorder files")
                return
            
            # Extract gid/sid from scenario filenames
            # Format: gid_X_sid_Y.json or similar
            scenario_ids = set()
            for scenario_file in scenario_files:
                # Try to extract gid and sid from filename
                name = scenario_file.stem  # without .json extension
                # Common formats: gid_1_sid_106, gid:1_sid:106, etc.
                import re
                match = re.search(r'gid[:_](\d+)[^0-9]*sid[:_](\d+)', name)
                if match:
                    gid, sid = match.groups()
                    scenario_ids.add((int(gid), int(sid)))
            
            if not scenario_ids:
                print(f"[WARNING] Could not extract gid/sid from scenario filenames, trying to copy all recorder files")
                # Fallback: copy all recorder files if we can't match
                if data_recorder_dir.exists() and data_recorder_dir.is_dir():
                    recorder_files = list(data_recorder_dir.glob("*.log"))
                    if recorder_files:
                        copied_count = 0
                        for recorder_file in recorder_files:
                            target_file = experiment_recorder_dir / recorder_file.name
                            if not target_file.exists():
                                shutil.copy2(str(recorder_file), str(target_file))
                                copied_count += 1
                        if copied_count > 0:
                            print(f"[INFO] Copied {copied_count} recorder file(s) from {data_recorder_dir} to {experiment_recorder_dir}")
                return
            
            # Match and copy recorder files based on scenario gid/sid
            if data_recorder_dir.exists() and data_recorder_dir.is_dir():
                recorder_files = list(data_recorder_dir.glob("*.log"))
                if recorder_files:
                    copied_count = 0
                    for recorder_file in recorder_files:
                        # Extract gid/sid from recorder filename: gid:X_sid:Y.log
                        name = recorder_file.stem
                        match = re.search(r'gid[:_](\d+)[^0-9]*sid[:_](\d+)', name)
                        if match:
                            gid, sid = match.groups()
                            if (int(gid), int(sid)) in scenario_ids:
                                target_file = experiment_recorder_dir / recorder_file.name
                                if not target_file.exists():
                                    shutil.copy2(str(recorder_file), str(target_file))
                                    copied_count += 1
                    
                    if copied_count > 0:
                        print(f"[INFO] Copied {copied_count} matched recorder file(s) from {data_recorder_dir} to {experiment_recorder_dir}")
                    else:
                        print(f"[INFO] No matching recorder files found for {len(scenario_ids)} scenario(s)")
        except Exception as recorder_copy_error:
            print(f"[WARNING] Failed to copy recorder files to experiment directory: {recorder_copy_error}")
            import traceback
            traceback.print_exc()
    
    def _archive_experiment_run(self, method_name: str, experiment_id: str, method_dir: Path):
        """
        Archive all artifacts for a single experiment run into a unified snapshot directory.
        
        This includes:
        - Per-experiment results (queue/metrics under method_dir)
        - Global scenario database (if any)
        - GPT conversation logs
        - Progress checkpoint and time history
        
        Note: Uses experiment_id as directory name (without timestamp) to ensure
        each experiment has only one archive directory. If archive already exists,
        it will be updated/overwritten.
        """
        project_root = PROJECT_ROOT
        archive_root = project_root / "data" / "experiment_snapshots" / method_name
        # Use experiment_id only (no timestamp) to ensure one archive per experiment
        archive_dir = archive_root / experiment_id
        
        # Remove existing archive if it exists to ensure clean archive
        if archive_dir.exists():
            print(f"[INFO] Removing existing archive for {experiment_id} to create fresh archive...")
            shutil.rmtree(archive_dir)
        
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # 1) Copy per-experiment results
        # This includes all subdirectories under method_dir, including:
        # - queue/ (scenario files)
        # - metrics/ (metrics records)
        # - recorder/ (CARLA recorder log files)
        if method_dir.exists():
            # Ensure recorder files are copied before archiving
            # This handles the case where recorder files weren't copied during experiment run
            self._copy_recorder_files_to_experiment_dir(method_dir)
            
            target_results = archive_dir / "results"
            shutil.copytree(str(method_dir), str(target_results))
            print(f"[INFO] Archived results to {target_results}")
            
            # Verify recorder directory was archived
            recorder_source = method_dir / "recorder"
            recorder_target = target_results / "recorder"
            if recorder_source.exists() and recorder_target.exists():
                recorder_files = list(recorder_source.glob("*.log"))
                if recorder_files:
                    print(f"[INFO] Archived {len(recorder_files)} recorder log file(s) to {recorder_target}")
                else:
                    print(f"[WARNING] Recorder directory exists but contains no .log files")

            # 1b) Calculate metrics from scenario data (optional, can be done offline)
            # Metrics calculation has been moved to experiments/analysis/calculate_metrics.py
            # This allows offline recalculation and decouples data collection from metrics calculation
            # To calculate metrics, run:
            #   python -m experiments.analysis.calculate_metrics --experiment-dir <method_dir>
            # 
            # Optionally, we can call it here automatically:
            try:
                from experiments.analysis.calculate_metrics import calculate_metrics_from_scenarios
                metrics_dir = method_dir / "metrics"
                # Only calculate if metrics don't already exist (to avoid overwriting)
                records_path = metrics_dir / "metrics_records.jsonl"
                if not records_path.exists():
                    print(f"[ExperimentManager] Calculating metrics for {experiment_id}...")
                    calculate_metrics_from_scenarios(
                        experiment_dir=method_dir,
                        output_dir=metrics_dir,
                        incremental=False,
                        recalculate=False
                    )
                else:
                    # Aggregate existing metrics if they exist
                    records = load_records_from_jsonl(str(records_path))
                    if records:
                        summary = aggregate_run_metrics(records)
                        num_scenarios = summary.get("num_records", len(records))
                        summary_path = metrics_dir / "metrics_summary.json"
                        save_run_summary(
                            summary,
                            str(summary_path),
                            method_name=method_name,
                            experiment_id=experiment_id,
                            num_scenarios=num_scenarios,
                        )
            except ImportError:
                # calculate_metrics module not available, skip
                print(f"[ExperimentManager] Metrics calculation module not available, skipping for {experiment_id}")
            except Exception as metrics_err:
                print(f"[WARNING] Failed to calculate/aggregate metrics for run {experiment_id}: {metrics_err}")
        
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


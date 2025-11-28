"""
Tests for experiment progress tracking and time estimation logic.

These focus on JSON checkpoint/history persistence and basic calculations,
using temporary directories to avoid touching real experiment_results.
"""

from pathlib import Path
import time

import pytest

from experiments.core.progress_tracker import ProgressTracker
from experiments.core.time_estimator import TimeEstimator


def test_progress_tracker_checkpoint_lifecycle(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    tracker = ProgressTracker(checkpoint_file=str(checkpoint))

    exp_id = "exp_001"
    tracker.start_experiment(
        experiment_id=exp_id,
        method_name="RAG-ScenarioFuzz",
        target_scenarios=3,
        target_time=60.0,
    )

    assert checkpoint.exists()

    # Initial summary
    # get_summary uses datetime.fromisoformat when available; on Python 3.6
    # this falls back to manual parsing and should not raise.
    summary = tracker.get_summary(exp_id)
    assert summary["completed"] == 0
    assert summary["target"] == 3
    # Newly started experiment should not be complete
    assert summary["is_complete"] in (False, True)

    # Update progress for three scenarios
    for sid in range(3):
        tracker.update_progress(
            experiment_id=exp_id,
            scenario_id=sid,
            scenario_info={"metric": sid},
        )

    summary2 = tracker.get_summary(exp_id)
    assert summary2["completed"] == 3
    assert summary2["is_complete"] is True

    # Ensure scenarios info persisted
    prog = tracker.get_progress(exp_id)
    assert len(prog["scenarios"]) == 3

    # Clear checkpoint for this experiment
    tracker.clear_checkpoint(exp_id)
    assert tracker.get_progress(exp_id) is None


def test_time_estimator_history_and_estimates(tmp_path):
    history_file = tmp_path / "time_history.json"
    estimator = TimeEstimator(history_file=str(history_file))

    method = "RAG-ScenarioFuzz"

    # With no history, estimate_scenario_time should fall back to defaults
    default_time = estimator.estimate_scenario_time(method)
    assert default_time > 0

    # Record a few synthetic scenario times
    estimator.record_scenario_time(method, 10.0)
    estimator.record_scenario_time(method, 20.0)
    estimator.record_scenario_time(method, 30.0)

    assert history_file.exists()

    est_single = estimator.estimate_scenario_time(method)
    # Average of 10, 20, 30 = 20 (within a small tolerance)
    assert abs(est_single - 20.0) < 1e-6

    # Estimate total time for N scenarios
    total_info = estimator.estimate_total_time(method, num_scenarios=5)
    assert total_info["num_scenarios"] == 5
    assert total_info["total_seconds"] == pytest.approx(5 * est_single)

    # Update estimate based on elapsed time and completed scenarios
    elapsed = 40.0
    completed = 2
    updated = estimator.update_estimate(method, elapsed_time=elapsed, completed_scenarios=completed)
    assert updated["current_avg_time"] == pytest.approx(20.0)
    assert updated["completed_scenarios"] == completed



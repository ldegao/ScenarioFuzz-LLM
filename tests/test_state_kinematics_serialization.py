import json
from pathlib import Path

from experiments.analysis.scenario_loader import ScenarioDataLoader


def test_loader_preserves_kinematic_lists(tmp_path: Path):
    """
    Ensure yaw/lat_speed/lon_speed lists in queue-style JSON are preserved
    and allow metrics validation to pass.
    """
    queue_dir = tmp_path
    json_path = queue_dir / "gid:0_sid:1.json"

    sample_state = {
        "speed": [1.0, 2.0, 3.0],
        "yaw_list": [0.0, 1.0, 2.0],
        "yaw_rate_list": [0.1, 0.2, 0.3],
        "lat_speed_list": [0.5, 0.4, 0.3],
        "lon_speed_list": [5.0, 5.5, 6.0],
        "crashed": False,
        "stuck": False,
        "laneinvaded": False,
        "speed_lim": [30.0, 30.0, 30.0],
    }
    payload = {
        "events": {},
        "config": {},
        "state": sample_state,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loader = ScenarioDataLoader()
    scenario = loader.load_scenario_from_json(json_path)
    assert scenario is not None
    state = scenario.state

    # Fields are preserved
    assert state.yaw_list == sample_state["yaw_list"]
    assert state.lat_speed_list == sample_state["lat_speed_list"]
    assert state.lon_speed_list == sample_state["lon_speed_list"]
    assert state.speed == sample_state["speed"]

    # Validate passes when kinematic lists exist
    assert loader.validate_scenario_for_metrics(scenario) is True


"""
Basic tests for state and constants modules.

These tests check that core data structures can be instantiated and
that important constants are defined as expected.
"""

import constants
from states import ScenarioState


def test_constants_basic_attributes():
    # Check that key agent types are defined
    assert hasattr(constants, "AUTOWARE")
    assert hasattr(constants, "BEHAVIOR_AGENT") or hasattr(constants, "BEHAVIOR")


def test_scenario_state_initialization():
    state = ScenarioState()
    # Newly created state should have sensible defaults
    assert state.speed == [] or state.speed is None or isinstance(state.speed, list)
    assert hasattr(state, "yaw_list")
    assert hasattr(state, "lon_speed_list")
    assert hasattr(state, "lat_speed_list")



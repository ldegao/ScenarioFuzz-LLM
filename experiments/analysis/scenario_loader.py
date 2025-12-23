#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scenario Data Loader
--------------------

Loads scenario data from JSON or pickle files for metrics calculation.
This module is decoupled from the fuzzing loop and can be used offline.

The loader supports:
1. Loading from JSON files (with state data)
2. Loading from pickle files (complete Scenario objects)
3. Converting data to format required by metrics calculation
"""

import json
import pickle
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from scenario import Scenario
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Delay import of Scenario to avoid CARLA dependency at module import time
# Scenario will be imported only when needed (in load_scenario_from_pickle)
from states import ScenarioState


def _ensure_minimal_carla():
    """
    Provide minimal carla stubs when CARLA is absent or missing Location/Rotation/Transform.
    This lets offline metrics loading succeed on pickle files that contain these objects.
    Real CARLA (with full API) will not be modified.
    """
    try:
        import carla  # type: ignore
    except Exception:
        # Create minimal module
        class Location:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x, self.y, self.z = x, y, z
        class Rotation:
            def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
                self.pitch, self.yaw, self.roll = pitch, yaw, roll
        class Transform:
            def __init__(self, location=None, rotation=None):
                self.location = location or Location()
                self.rotation = rotation or Rotation()
        import types, sys as _sys
        _sys.modules['carla'] = types.SimpleNamespace(
            Location=Location, Rotation=Rotation, Transform=Transform
        )
        return
    # If carla exists but missing attributes, patch minimal ones
    missing = []
    for attr in ("Location", "Rotation", "Transform"):
        if not hasattr(carla, attr):
            missing.append(attr)
    if missing:
        class Location:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x, self.y, self.z = x, y, z
        class Rotation:
            def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
                self.pitch, self.yaw, self.roll = pitch, yaw, roll
        class Transform:
            def __init__(self, location=None, rotation=None):
                self.location = location or Location()
                self.rotation = rotation or Rotation()
        if "Location" in missing:
            carla.Location = Location  # type: ignore
        if "Rotation" in missing:
            carla.Rotation = Rotation  # type: ignore
        if "Transform" in missing:
            carla.Transform = Transform  # type: ignore


class ScenarioDataLoader:
    """
    Loads scenario data from files for metrics calculation.
    """
    
    def __init__(self):
        """Initialize the scenario loader"""
        pass
    
    def load_scenario_from_json(self, json_path: Path) -> Optional[Any]:
        """
        Load scenario data from JSON file.
        
        The JSON file should contain:
        - events: Event information
        - config: Configuration
        - state: Complete state data (speed, yaw_list, etc.)
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Scenario object with state data, or None if loading fails
        """
        if not json_path.exists():
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ScenarioLoader] WARNING: Failed to load JSON {json_path}: {e}")
            return None
        
        # Extract generation_id and scenario_id from filename
        # Format: gid:{generation_id}_sid:{scenario_id}.json
        filename = json_path.name
        match = re.match(r'gid:(\d+)_sid:(\d+)\.json', filename)
        if match:
            generation_id = int(match.group(1))
            scenario_id = int(match.group(2))
        else:
            # Try to extract from data if available
            generation_id = data.get('generation_id', -1)
            scenario_id = data.get('scenario_id', -1)
        
        # Create a minimal Scenario object
        # We don't need full Scenario initialization, just the state data
        class MinimalScenario:
            """Minimal Scenario object for metrics calculation"""
            def __init__(self):
                self.generation_id = generation_id
                self.scenario_id = scenario_id
                self.state = ScenarioState()
        
        scenario = MinimalScenario()
        
        # Load state data from JSON
        if 'state' in data:
            state_data = data['state']
            state = scenario.state
            
            # Load all state fields
            for key, value in state_data.items():
                if hasattr(state, key):
                    # Convert list back to set for drawn_points if needed
                    if key == 'drawn_points' and isinstance(value, list):
                        setattr(state, key, set(value))
                    else:
                        setattr(state, key, value)
        
        return scenario
    
    def load_scenario_from_pickle(self, pickle_path: Path) -> Optional[Any]:
        """
        Load complete Scenario object from pickle file.
        
        Args:
            pickle_path: Path to pickle file
            
        Returns:
            Scenario object, or None if loading fails
        """
        if not pickle_path.exists():
            return None
        
        try:
            # Ensure minimal CARLA types exist before unpickling
            _ensure_minimal_carla()

            # Delay import to avoid CARLA dependency
            try:
                from scenario import Scenario
            except (ImportError, AttributeError):
                # CARLA not available, but we can still try to load the pickle
                # The pickle might have been saved with CARLA objects
                Scenario = None
            
            with open(pickle_path, 'rb') as f:
                scenario = pickle.load(f)
            
            # Verify it's a Scenario object (if Scenario class is available)
            # If Scenario class is not available (CARLA not imported), we still accept the object
            # if it has the expected attributes
            if Scenario is not None:
                if not isinstance(scenario, Scenario):
                    print(f"[ScenarioLoader] WARNING: Pickle file {pickle_path} does not contain Scenario object")
                    return None
            else:
                # Scenario class not available, but we can still validate by attributes
                if not hasattr(scenario, 'generation_id') or not hasattr(scenario, 'scenario_id'):
                    print(f"[ScenarioLoader] WARNING: Loaded object from {pickle_path} does not have expected Scenario attributes")
                    return None
            
            # Basic validation: check if it has expected attributes
            if not hasattr(scenario, 'generation_id') or not hasattr(scenario, 'scenario_id'):
                print(f"[ScenarioLoader] WARNING: Loaded object from {pickle_path} does not have expected Scenario attributes")
                return None
            
            return scenario
        except Exception as e:
            print(f"[ScenarioLoader] WARNING: Failed to load pickle {pickle_path}: {e}")
            return None
    
    def load_all_scenarios(
        self, 
        experiment_dir: Path,
        prefer_pickle: bool = True
    ) -> List[Any]:
        """
        Load all scenarios from an experiment directory.
        
        Args:
            experiment_dir: Path to experiment directory (should contain queue/ subdirectory)
            prefer_pickle: If True, prefer pickle files over JSON files when both exist
            
        Returns:
            List of Scenario objects, sorted by generation_id and scenario_id
        """
        queue_dir = experiment_dir / "queue"
        if not queue_dir.exists():
            print(f"[ScenarioLoader] WARNING: Queue directory {queue_dir} does not exist")
            return []
        
        scenarios = []
        processed_ids = set()  # Track processed (generation_id, scenario_id) pairs
        
        # First, try to load from pickle files if prefer_pickle
        if prefer_pickle:
            pickle_files = sorted(queue_dir.glob("*.pkl"))
            for pickle_path in pickle_files:
                scenario = self.load_scenario_from_pickle(pickle_path)
                if scenario and hasattr(scenario, 'generation_id') and hasattr(scenario, 'scenario_id'):
                    key = (scenario.generation_id, scenario.scenario_id)
                    if key not in processed_ids:
                        scenarios.append(scenario)
                        processed_ids.add(key)
        
        # Then, load from JSON files (for scenarios without pickle or if prefer_pickle=False)
        json_files = sorted(queue_dir.glob("*.json"))
        for json_path in json_files:
            # Extract IDs from filename
            match = re.match(r'gid:(\d+)_sid:(\d+)\.json', json_path.name)
            if not match:
                continue
            
            generation_id = int(match.group(1))
            scenario_id = int(match.group(2))
            key = (generation_id, scenario_id)
            
            # Skip if already loaded from pickle
            if key in processed_ids:
                continue
            
            scenario = self.load_scenario_from_json(json_path)
            if scenario:
                scenarios.append(scenario)
                processed_ids.add(key)
        
        # Sort by generation_id, then scenario_id
        scenarios.sort(key=lambda s: (getattr(s, 'generation_id', -1), getattr(s, 'scenario_id', -1)))
        
        print(f"[ScenarioLoader] Loaded {len(scenarios)} scenarios from {experiment_dir}")
        return scenarios
    
    def validate_scenario_for_metrics(self, scenario: Any) -> bool:
        """
        Validate that a scenario has the data needed for metrics calculation.
        
        Args:
            scenario: Scenario object to validate
            
        Returns:
            True if scenario has valid state data, False otherwise
        """
        if not hasattr(scenario, 'state') or scenario.state is None:
            return False
        
        state = scenario.state
        
        # Check for critical fields needed by metrics
        # PC needs: yaw_list, lon_speed_list (for acceleration calculation)
        # PEC needs: speed, yaw_rate_list
        # TCD needs: lon_speed_list, lat_speed_list, yaw_list
        # BCM needs: crashed, stuck, laneinvaded, speed, speed_lim
        
        # At minimum, we need some state data
        has_speed_data = hasattr(state, 'speed') and state.speed is not None and len(state.speed) > 0
        has_movement_data = (
            (hasattr(state, 'yaw_list') and state.yaw_list and len(state.yaw_list) > 0) or
            (hasattr(state, 'lon_speed_list') and state.lon_speed_list and len(state.lon_speed_list) > 0) or
            (hasattr(state, 'lat_speed_list') and state.lat_speed_list and len(state.lat_speed_list) > 0)
        )
        
        return has_speed_data or has_movement_data


def load_scenario_from_json(json_path: Path) -> Optional[Any]:
    """
    Convenience function to load a scenario from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Scenario object or None
    """
    loader = ScenarioDataLoader()
    return loader.load_scenario_from_json(json_path)


def load_scenario_from_pickle(pickle_path: Path) -> Optional[Any]:
    """
    Convenience function to load a scenario from pickle file.
    
    Args:
        pickle_path: Path to pickle file
        
    Returns:
        Scenario object or None
    """
    loader = ScenarioDataLoader()
    return loader.load_scenario_from_pickle(pickle_path)


def load_all_scenarios(experiment_dir: Path, prefer_pickle: bool = True) -> List[Any]:
    """
    Convenience function to load all scenarios from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        prefer_pickle: If True, prefer pickle files over JSON files
        
    Returns:
        List of Scenario objects
    """
    loader = ScenarioDataLoader()
    return loader.load_all_scenarios(experiment_dir, prefer_pickle)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load scenario data from files")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Path to experiment directory"
    )
    parser.add_argument(
        "--prefer-pickle",
        action="store_true",
        help="Prefer pickle files over JSON files"
    )
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    scenarios = load_all_scenarios(experiment_dir, prefer_pickle=args.prefer_pickle)
    
    print(f"Loaded {len(scenarios)} scenarios")
    if scenarios:
        print(f"First scenario: gid={scenarios[0].generation_id}, sid={scenarios[0].scenario_id}")
        print(f"Last scenario: gid={scenarios[-1].generation_id}, sid={scenarios[-1].scenario_id}")


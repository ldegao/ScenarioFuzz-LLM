#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：测试场景数据加载功能

测试覆盖：
1. 从JSON文件加载场景数据
2. 从pickle文件加载场景数据
3. 从实验目录加载所有场景
4. 数据验证功能
5. 错误处理
"""

import unittest
import tempfile
import json
import pickle
import shutil
from pathlib import Path
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from experiments.analysis.scenario_loader import (
        ScenarioDataLoader,
        load_scenario_from_json,
        load_scenario_from_pickle,
        load_all_scenarios
    )
    from states import ScenarioState
    LOADER_AVAILABLE = True
    # Try to import Scenario, but don't fail if CARLA is not available
    try:
        from scenario import Scenario
        SCENARIO_AVAILABLE = True
    except (ImportError, AttributeError):
        # CARLA not available, we'll use mock Scenario
        SCENARIO_AVAILABLE = False
        Scenario = None
except ImportError as e:
    LOADER_AVAILABLE = False
    SCENARIO_AVAILABLE = False
    Scenario = None
    print(f"[WARNING] Scenario loader modules not available: {e}")


class TestScenarioLoader(unittest.TestCase):
    """测试场景数据加载功能"""
    
    def setUp(self):
        """设置测试环境"""
        if not LOADER_AVAILABLE:
            self.skipTest("Scenario loader modules not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_scenario_from_json(self):
        """测试从JSON文件加载场景数据"""
        # 创建测试JSON文件
        json_path = self.temp_path / "gid:1_sid:100.json"
        
        state_data = {
            "speed": [50.0, 55.0, 60.0],
            "speed_lim": [60.0] * 3,
            "yaw_list": [0.0, 1.0, 2.0],
            "yaw_rate_list": [0.0, 0.5, 1.0],
            "lon_speed_list": [50.0, 55.0, 60.0],
            "lat_speed_list": [0.0, 0.5, 1.0],
            "steer_angle_list": [0.0, 0.1, 0.2],
            "min_dist": 10.0,
            "crashed": False,
            "stuck": False,
            "laneinvaded": False,
            "num_frames": 100,
            "elapsed_time": 4.0,
            "distance": 100.0,
            "first_frame_id": 0,
            "first_sim_elapsed_time": 0.0,
            "sim_start_time": 0.0,
            "drawn_points": [1, 2, 3]
        }
        
        data = {
            "events": {
                "crash": False,
                "stuck": False,
                "lane_invasion": False,
                "red": False,
                "speeding": False,
                "other": False,
                "other_error_val": 0
            },
            "config": {
                "fps": 25,
                "max_dist_from_player": 40,
                "min_dist_from_player": 5,
                "abort_seconds": 60,
                "wait_autoware_num_topics": 64
            },
            "state": state_data
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 加载场景
        loader = ScenarioDataLoader()
        scenario = loader.load_scenario_from_json(json_path)
        
        # 验证加载结果
        self.assertIsNotNone(scenario)
        self.assertEqual(scenario.generation_id, 1)
        self.assertEqual(scenario.scenario_id, 100)
        self.assertIsNotNone(scenario.state)
        
        # 验证state数据
        state = scenario.state
        self.assertEqual(state.speed, [50.0, 55.0, 60.0])
        self.assertEqual(state.min_dist, 10.0)
        self.assertEqual(state.crashed, False)
        self.assertEqual(set(state.drawn_points), {1, 2, 3})
    
    def test_load_scenario_from_json_nonexistent(self):
        """测试加载不存在的JSON文件"""
        json_path = self.temp_path / "nonexistent.json"
        
        loader = ScenarioDataLoader()
        scenario = loader.load_scenario_from_json(json_path)
        
        self.assertIsNone(scenario)
    
    def test_load_scenario_from_pickle(self):
        """测试从pickle文件加载场景数据"""
        # 创建模拟Scenario对象
        class MockConfig:
            def __init__(self):
                self.debug = False
        
        # Use a simple mock Scenario class if real Scenario is not available
        if SCENARIO_AVAILABLE and Scenario is not None:
            seed_data = {
                "sp_x": 0.0, "sp_y": 0.0, "sp_z": 0.0,
                "roll": 0.0, "yaw": 0.0, "pitch": 0.0,
                "wp_x": 10.0, "wp_y": 10.0, "wp_z": 0.0, "wp_yaw": 0.0,
                "map": 3
            }
            scenario = Scenario(MockConfig(), seed_data)
        else:
            # Create a minimal mock Scenario
            class MockScenario:
                def __init__(self):
                    self.generation_id = 1
                    self.scenario_id = 100
                    self.state = ScenarioState()
            scenario = MockScenario()
        
        scenario.generation_id = 1
        scenario.scenario_id = 100
        scenario.state = ScenarioState()
        scenario.state.speed = [50.0, 55.0, 60.0]
        scenario.state.min_dist = 10.0
        
        # 保存为pickle
        pickle_path = self.temp_path / "gid:1_sid:100.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(scenario, f)
        
        # 加载场景
        loader = ScenarioDataLoader()
        loaded_scenario = loader.load_scenario_from_pickle(pickle_path)
        
        # 验证加载结果
        self.assertIsNotNone(loaded_scenario)
        self.assertEqual(loaded_scenario.generation_id, 1)
        self.assertEqual(loaded_scenario.scenario_id, 100)
        self.assertIsNotNone(loaded_scenario.state)
        self.assertEqual(loaded_scenario.state.speed, [50.0, 55.0, 60.0])
        self.assertEqual(loaded_scenario.state.min_dist, 10.0)
    
    def test_load_scenario_from_pickle_nonexistent(self):
        """测试加载不存在的pickle文件"""
        pickle_path = self.temp_path / "nonexistent.pkl"
        
        loader = ScenarioDataLoader()
        scenario = loader.load_scenario_from_pickle(pickle_path)
        
        self.assertIsNone(scenario)
    
    def test_load_all_scenarios(self):
        """测试从实验目录加载所有场景"""
        # 创建实验目录结构
        experiment_dir = self.temp_path / "test_experiment"
        queue_dir = experiment_dir / "queue"
        queue_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建多个场景JSON文件
        scenarios_data = [
            (1, 100, {"speed": [50.0, 55.0], "crashed": False}),
            (1, 101, {"speed": [60.0, 65.0], "crashed": True}),
            (2, 200, {"speed": [40.0, 45.0], "crashed": False})
        ]
        
        for gen_id, scen_id, state_data in scenarios_data:
            filename = f"gid:{gen_id}_sid:{scen_id}.json"
            json_path = queue_dir / filename
            
            data = {
                "events": {"crash": state_data["crashed"], "stuck": False, "lane_invasion": False},
                "config": {"fps": 25},
                "state": {
                    "speed": state_data["speed"],
                    "speed_lim": [60.0] * len(state_data["speed"]),
                    "crashed": state_data["crashed"],
                    "num_frames": 100,
                    "elapsed_time": 4.0,
                    "distance": 100.0
                }
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        
        # 加载所有场景
        scenarios = load_all_scenarios(experiment_dir, prefer_pickle=False)
        
        # 验证加载结果
        self.assertEqual(len(scenarios), 3)
        
        # 验证场景按ID排序
        self.assertEqual(scenarios[0].generation_id, 1)
        self.assertEqual(scenarios[0].scenario_id, 100)
        self.assertEqual(scenarios[1].generation_id, 1)
        self.assertEqual(scenarios[1].scenario_id, 101)
        self.assertEqual(scenarios[2].generation_id, 2)
        self.assertEqual(scenarios[2].scenario_id, 200)
    
    def test_load_all_scenarios_empty_dir(self):
        """测试加载空目录"""
        experiment_dir = self.temp_path / "empty_experiment"
        queue_dir = experiment_dir / "queue"
        queue_dir.mkdir(parents=True, exist_ok=True)
        
        scenarios = load_all_scenarios(experiment_dir)
        
        self.assertEqual(len(scenarios), 0)
    
    def test_load_all_scenarios_nonexistent_dir(self):
        """测试加载不存在的目录"""
        experiment_dir = self.temp_path / "nonexistent"
        
        scenarios = load_all_scenarios(experiment_dir)
        
        self.assertEqual(len(scenarios), 0)
    
    def test_validate_scenario_for_metrics(self):
        """测试场景数据验证功能"""
        loader = ScenarioDataLoader()
        
        # 创建有效场景
        class ValidScenario:
            def __init__(self):
                self.state = ScenarioState()
                self.state.speed = [50.0, 55.0, 60.0]
                self.state.yaw_list = [0.0, 1.0, 2.0]
        
        valid_scenario = ValidScenario()
        self.assertTrue(loader.validate_scenario_for_metrics(valid_scenario))
        
        # 创建无效场景（无state）
        class InvalidScenario1:
            def __init__(self):
                self.state = None
        
        invalid_scenario1 = InvalidScenario1()
        self.assertFalse(loader.validate_scenario_for_metrics(invalid_scenario1))
        
        # 创建无效场景（无state属性）
        class InvalidScenario2:
            pass
        
        invalid_scenario2 = InvalidScenario2()
        self.assertFalse(loader.validate_scenario_for_metrics(invalid_scenario2))
        
        # 创建无效场景（state为空）
        class InvalidScenario3:
            def __init__(self):
                self.state = ScenarioState()
                # 没有speed或movement数据
        
        invalid_scenario3 = InvalidScenario3()
        self.assertFalse(loader.validate_scenario_for_metrics(invalid_scenario3))
    
    def test_prefer_pickle_over_json(self):
        """测试优先使用pickle文件"""
        experiment_dir = self.temp_path / "test_experiment"
        queue_dir = experiment_dir / "queue"
        queue_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建JSON文件
        json_path = queue_dir / "gid:1_sid:100.json"
        json_data = {
            "events": {"crash": False},
            "config": {"fps": 25},
            "state": {"speed": [50.0], "crashed": False}
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f)
        
        # 创建pickle文件（相同ID）
        class MockConfig:
            def __init__(self):
                self.debug = False
        
        # Use a simple mock Scenario class if real Scenario is not available
        if SCENARIO_AVAILABLE and Scenario is not None:
            seed_data = {
                "sp_x": 0.0, "sp_y": 0.0, "sp_z": 0.0,
                "roll": 0.0, "yaw": 0.0, "pitch": 0.0,
                "wp_x": 10.0, "wp_y": 10.0, "wp_z": 0.0, "wp_yaw": 0.0,
                "map": 3
            }
            scenario = Scenario(MockConfig(), seed_data)
        else:
            # Create a minimal mock Scenario
            class MockScenario:
                def __init__(self):
                    self.generation_id = 1
                    self.scenario_id = 100
                    self.state = ScenarioState()
            scenario = MockScenario()
        
        scenario.generation_id = 1
        scenario.scenario_id = 100
        scenario.state = ScenarioState()
        scenario.state.speed = [60.0]  # 不同的值
        
        pickle_path = queue_dir / "gid:1_sid:100.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(scenario, f)
        
        # 加载（优先pickle）
        scenarios = load_all_scenarios(experiment_dir, prefer_pickle=True)
        
        self.assertEqual(len(scenarios), 1)
        # 应该使用pickle文件的值
        self.assertEqual(scenarios[0].state.speed, [60.0])


if __name__ == "__main__":
    unittest.main()


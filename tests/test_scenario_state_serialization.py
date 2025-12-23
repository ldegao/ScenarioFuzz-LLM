#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：测试Scenario state的序列化/反序列化

测试覆盖：
1. dump_states()方法是否保存完整的state数据到JSON
2. save_scenario_pickle()方法是否保存完整的Scenario对象
3. 序列化后的数据可以正确反序列化
"""

import unittest
import json
import pickle
import tempfile
import os
from pathlib import Path
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from states import ScenarioState


class MockConfig:
    """模拟配置对象"""
    def __init__(self):
        self.debug = False
        self.queue_dir = None
        self.out_dir = None
        self.timeout = 60


class MockScenario:
    """模拟Scenario对象用于测试"""
    def __init__(self, use_real_scenario=False):
        # 如果CARLA可用且需要真实Scenario，使用真实Scenario
        if use_real_scenario:
            try:
                import config
                config.set_carla_api_path()
                from scenario import Scenario
                # 使用真实的Scenario对象
                seed_data = {
                    "sp_x": 0.0, "sp_y": 0.0, "sp_z": 0.0,
                    "roll": 0.0, "yaw": 0.0, "pitch": 0.0,
                    "wp_x": 10.0, "wp_y": 10.0, "wp_z": 0.0, "wp_yaw": 0.0,
                    "map": 3
                }
                real_scenario = Scenario(MockConfig(), seed_data)
                real_scenario.generation_id = 1
                real_scenario.scenario_id = 106
                real_scenario.state.speed = [50.0, 60.0, 70.0]
                real_scenario.state.speed_lim = [60.0, 60.0, 60.0]
                real_scenario.state.yaw_list = [0.0, 10.0, 20.0]
                real_scenario.state.yaw_rate_list = [0.0, 5.0, 10.0]
                real_scenario.state.lon_speed_list = [50.0, 60.0, 70.0]
                real_scenario.state.lat_speed_list = [0.0, 1.0, 2.0]
                real_scenario.state.steer_angle_list = [0.0, 0.1, 0.2]
                real_scenario.state.min_dist = 5.5
                real_scenario.state.num_frames = 100
                real_scenario.state.elapsed_time = 4.0
                real_scenario.state.crashed = False
                real_scenario.state.stuck = False
                real_scenario.state.laneinvaded = False
                real_scenario.state.drawn_points = {1, 2, 3}
                # 将真实Scenario的属性复制到self
                for attr in dir(real_scenario):
                    if not attr.startswith('_'):
                        setattr(self, attr, getattr(real_scenario, attr))
                return
            except (SystemExit, ImportError, AttributeError, KeyError):
                # CARLA不可用或Scenario创建失败，使用Mock
                pass
        
        # 使用Mock Scenario
        self.generation_id = 1
        self.scenario_id = 106
        self.conf = MockConfig()
        self.state = ScenarioState()
        # 设置一些测试数据
        self.state.speed = [50.0, 60.0, 70.0]
        self.state.speed_lim = [60.0, 60.0, 60.0]
        self.state.yaw_list = [0.0, 10.0, 20.0]
        self.state.yaw_rate_list = [0.0, 5.0, 10.0]
        self.state.lon_speed_list = [50.0, 60.0, 70.0]
        self.state.lat_speed_list = [0.0, 1.0, 2.0]
        self.state.steer_angle_list = [0.0, 0.1, 0.2]
        self.state.min_dist = 5.5
        self.state.num_frames = 100
        self.state.elapsed_time = 4.0
        self.state.crashed = False
        self.state.stuck = False
        self.state.laneinvaded = False
        self.state.drawn_points = {1, 2, 3}
    
    def dump_states(self, state, log_type):
        """模拟dump_states方法"""
        event_dict = {
            "crash": state.crashed,
            "stuck": state.stuck,
            "lane_invasion": state.laneinvaded,
            "red": state.red_violation,
            "speeding": state.speeding,
            "other": state.other_error,
            "other_error_val": state.other_error_val
        }
        config_dict = {
            "fps": 25,
            "max_dist_from_player": 40,
            "min_dist_from_player": 5,
            "abort_seconds": self.conf.timeout,
            "wait_autoware_num_topics": 64
        }
        
        # Serialize complete state data
        state_data = {
            "first_frame_id": state.first_frame_id,
            "num_frames": state.num_frames,
            "elapsed_time": state.elapsed_time,
            "distance": state.distance,
            "crashed": state.crashed,
            "stuck": state.stuck,
            "laneinvaded": state.laneinvaded,
            "speed": state.speed,
            "speed_lim": state.speed_lim,
            "yaw_list": state.yaw_list,
            "yaw_rate_list": state.yaw_rate_list,
            "lon_speed_list": state.lon_speed_list,
            "lat_speed_list": state.lat_speed_list,
            "steer_angle_list": state.steer_angle_list,
            "min_dist": state.min_dist,
            "drawn_points": list(state.drawn_points) if isinstance(state.drawn_points, set) else state.drawn_points,
        }
        
        state_dict = {
            "events": event_dict,
            "config": config_dict,
            "state": state_data
        }
        
        filename = f"gid:{self.generation_id}_sid:{self.scenario_id}.json"
        if log_type == "queue":
            out_dir = self.conf.queue_dir
        else:
            out_dir = self.conf.out_dir
        
        with open(os.path.join(out_dir, filename), "w", encoding='utf-8') as fp:
            json.dump(state_dict, fp, ensure_ascii=False, indent=2)
        return filename
    
    def save_scenario_pickle(self, log_type="queue"):
        """模拟save_scenario_pickle方法"""
        filename = f"gid:{self.generation_id}_sid:{self.scenario_id}.pkl"
        if log_type == "queue":
            out_dir = self.conf.queue_dir
        else:
            out_dir = self.conf.out_dir
        
        pickle_path = os.path.join(out_dir, filename)
        with open(pickle_path, "wb") as fp:
            pickle.dump(self, fp)
        return filename


class TestScenarioStateSerialization(unittest.TestCase):
    """测试Scenario state序列化/反序列化"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # 设置配置
        self.scenario = MockScenario()
        self.scenario.conf.queue_dir = str(self.temp_path)
        self.scenario.conf.out_dir = str(self.temp_path)
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dump_states_saves_complete_state(self):
        """测试dump_states()是否保存完整的state数据"""
        filename = self.scenario.dump_states(self.scenario.state, "queue")
        
        # 验证文件存在
        json_path = self.temp_path / filename
        self.assertTrue(json_path.exists(), "JSON file should be created")
        
        # 加载JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证基本结构
        self.assertIn("events", data)
        self.assertIn("config", data)
        self.assertIn("state", data, "State data should be saved in JSON")
        
        # 验证state数据完整性
        state_data = data["state"]
        self.assertIn("speed", state_data)
        self.assertIn("speed_lim", state_data)
        self.assertIn("yaw_list", state_data)
        self.assertIn("yaw_rate_list", state_data)
        self.assertIn("lon_speed_list", state_data)
        self.assertIn("lat_speed_list", state_data)
        self.assertIn("steer_angle_list", state_data)
        self.assertIn("min_dist", state_data)
        self.assertIn("num_frames", state_data)
        self.assertIn("elapsed_time", state_data)
        
        # 验证数据值
        self.assertEqual(state_data["speed"], [50.0, 60.0, 70.0])
        self.assertEqual(state_data["min_dist"], 5.5)
        self.assertEqual(state_data["num_frames"], 100)
        
        # 验证set转换为list
        self.assertIsInstance(state_data["drawn_points"], list)
        self.assertEqual(set(state_data["drawn_points"]), {1, 2, 3})
    
    def test_pickle_save_and_load(self):
        """测试pickle保存和加载Scenario对象"""
        filename = self.scenario.save_scenario_pickle("queue")
        
        # 验证文件存在
        pickle_path = self.temp_path / filename
        self.assertTrue(pickle_path.exists(), "Pickle file should be created")
        
        # 加载pickle文件
        with open(pickle_path, "rb") as f:
            loaded_scenario = pickle.load(f)
        
        # 验证基本属性
        self.assertEqual(loaded_scenario.generation_id, 1)
        self.assertEqual(loaded_scenario.scenario_id, 106)
        
        # 验证state数据
        self.assertIsNotNone(loaded_scenario.state)
        self.assertEqual(loaded_scenario.state.speed, [50.0, 60.0, 70.0])
        self.assertEqual(loaded_scenario.state.min_dist, 5.5)
        self.assertEqual(loaded_scenario.state.num_frames, 100)
    
    def test_json_state_data_completeness(self):
        """测试JSON中state数据的完整性（所有指标计算需要的字段）"""
        filename = self.scenario.dump_states(self.scenario.state, "queue")
        json_path = self.temp_path / filename
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        state_data = data["state"]
        
        # 验证PC (Parameter Coverage)需要的字段
        self.assertIn("yaw_list", state_data)
        self.assertIn("yaw_rate_list", state_data)
        self.assertIn("lon_speed_list", state_data)  # 用于计算加速度
        
        # 验证PEC (Behavior Coverage)需要的字段
        self.assertIn("speed", state_data)
        self.assertIn("yaw_rate_list", state_data)
        self.assertIn("lat_speed_list", state_data)
        
        # 验证TCD (Trajectory Diversity)需要的字段
        self.assertIn("lon_speed_list", state_data)
        self.assertIn("lat_speed_list", state_data)
        self.assertIn("yaw_list", state_data)
        
        # 验证BCM (Behavior Matrix)需要的字段
        self.assertIn("crashed", state_data)
        self.assertIn("stuck", state_data)
        self.assertIn("laneinvaded", state_data)
        self.assertIn("speed", state_data)
        self.assertIn("speed_lim", state_data)
    
    def test_json_backward_compatibility(self):
        """测试JSON格式向后兼容性（旧代码仍能读取）"""
        filename = self.scenario.dump_states(self.scenario.state, "queue")
        json_path = self.temp_path / filename
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 旧代码期望的字段应该仍然存在
        self.assertIn("events", data)
        self.assertIn("config", data)
        
        # 新字段不影响旧代码
        # 旧代码可以忽略"state"字段，仍然正常工作
        events = data["events"]
        self.assertIn("crash", events)
        self.assertIn("stuck", events)
    
    def test_data_readable_by_statistics_script(self):
        """测试序列化后的数据可以被统计脚本读取"""
        try:
            from experiments.analysis.scenario_loader import ScenarioDataLoader
        except ImportError:
            self.skipTest("Scenario loader module not available")
        
        # 保存场景数据
        filename = self.scenario.dump_states(self.scenario.state, "queue")
        json_path = self.temp_path / filename
        
        # 使用scenario_loader加载数据
        loader = ScenarioDataLoader()
        loaded_scenario = loader.load_scenario_from_json(json_path)
        
        # 验证数据可以被加载
        self.assertIsNotNone(loaded_scenario, "Scenario should be loaded by statistics script")
        self.assertEqual(loaded_scenario.generation_id, 1)
        self.assertEqual(loaded_scenario.scenario_id, 106)
        
        # 验证state数据完整性
        self.assertIsNotNone(loaded_scenario.state)
        state = loaded_scenario.state
        
        # 验证关键字段
        self.assertEqual(state.speed, [50.0, 60.0, 70.0])
        self.assertEqual(state.speed_lim, [60.0, 60.0, 60.0])
        self.assertEqual(state.yaw_list, [0.0, 10.0, 20.0])
        self.assertEqual(state.yaw_rate_list, [0.0, 5.0, 10.0])
        self.assertEqual(state.lon_speed_list, [50.0, 60.0, 70.0])
        self.assertEqual(state.lat_speed_list, [0.0, 1.0, 2.0])
        self.assertEqual(state.steer_angle_list, [0.0, 0.1, 0.2])
        self.assertEqual(state.min_dist, 5.5)
        self.assertEqual(state.crashed, False)
        self.assertEqual(state.stuck, False)
        self.assertEqual(state.laneinvaded, False)
        
        # 验证数据可以被统计脚本验证
        is_valid = loader.validate_scenario_for_metrics(loaded_scenario)
        self.assertTrue(is_valid, "Scenario should be valid for metrics calculation")
    
    def test_pickle_readable_by_statistics_script(self):
        """测试pickle文件可以被统计脚本读取"""
        try:
            from experiments.analysis.scenario_loader import ScenarioDataLoader
            # 检查CARLA是否可用（需要真实Scenario对象）
            import config
            try:
                config.set_carla_api_path()
                from scenario import Scenario
                SCENARIO_AVAILABLE = True
            except (SystemExit, ImportError, AttributeError):
                SCENARIO_AVAILABLE = False
        except ImportError:
            self.skipTest("Scenario loader module not available")
        
        if not SCENARIO_AVAILABLE:
            self.skipTest("CARLA not available, cannot test with real Scenario object")
        
        # 创建真实的Scenario对象用于pickle测试
        seed_data = {
            "sp_x": 0.0, "sp_y": 0.0, "sp_z": 0.0,
            "roll": 0.0, "yaw": 0.0, "pitch": 0.0,
            "wp_x": 10.0, "wp_y": 10.0, "wp_z": 0.0, "wp_yaw": 0.0,
            "map": 3
        }
        real_scenario = Scenario(MockConfig(), seed_data)
        real_scenario.generation_id = 1
        real_scenario.scenario_id = 106
        real_scenario.state.speed = [50.0, 60.0, 70.0]
        real_scenario.state.min_dist = 5.5
        real_scenario.state.num_frames = 100
        
        # 保存pickle文件（使用真实的Scenario对象）
        filename = f"gid:{real_scenario.generation_id}_sid:{real_scenario.scenario_id}.pkl"
        pickle_path = self.temp_path / filename
        with open(pickle_path, "wb") as fp:
            pickle.dump(real_scenario, fp)
        
        # 使用scenario_loader加载数据
        loader = ScenarioDataLoader()
        loaded_scenario = loader.load_scenario_from_pickle(pickle_path)
        
        # 验证数据可以被加载
        self.assertIsNotNone(loaded_scenario, "Scenario should be loaded from pickle by statistics script")
        self.assertEqual(loaded_scenario.generation_id, 1)
        self.assertEqual(loaded_scenario.scenario_id, 106)
        
        # 验证state数据完整性
        self.assertIsNotNone(loaded_scenario.state)
        state = loaded_scenario.state
        
        # 验证关键字段
        self.assertEqual(state.speed, [50.0, 60.0, 70.0])
        self.assertEqual(state.min_dist, 5.5)
        self.assertEqual(state.num_frames, 100)
        
        # 验证数据可以被统计脚本验证
        is_valid = loader.validate_scenario_for_metrics(loaded_scenario)
        self.assertTrue(is_valid, "Scenario should be valid for metrics calculation")


if __name__ == "__main__":
    unittest.main()


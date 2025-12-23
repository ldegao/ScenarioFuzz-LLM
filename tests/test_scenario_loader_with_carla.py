#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：测试场景数据加载功能（需要CARLA环境）

测试覆盖：
1. 从JSON文件加载场景数据
2. 从pickle文件加载场景数据（需要真实Scenario对象）
3. 从实验目录加载所有场景
4. 数据验证功能
5. 错误处理

注意：这些测试需要CARLA环境运行，会检查CARLA容器是否运行。
"""

import unittest
import tempfile
import json
import pickle
import shutil
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 首先检查CARLA环境
def check_carla_environment():
    """检查CARLA环境是否可用"""
    import subprocess
    import socket
    
    try:
        # 检查Docker容器是否运行
        username = os.getlogin()
        docker_name = f"carla-{username}"
        
        # 检查容器状态
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", docker_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        container_running = (result.returncode == 0 and result.stdout.strip() == "running")
        
        if not container_running:
            print(f"[INFO] CARLA container {docker_name} is not running")
            # 尝试启动CARLA容器
            script_dir = PROJECT_ROOT / "script"
            try:
                from experiments.core.environment_manager import ensure_carla_running
                ensure_carla_running(script_dir, PROJECT_ROOT)
                container_running = True
            except Exception as e:
                print(f"[WARNING] Failed to start CARLA: {e}")
                return False
        
        # 检查端口是否可用（CARLA可能使用2000或4000端口，根据run_carla.sh使用4000）
        ports_to_check = [4000, 2000]  # 优先检查4000（run_carla.sh配置的端口）
        port_accessible = False
        for port in ports_to_check:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result_port = sock.connect_ex(('localhost', port))
                sock.close()
                if result_port == 0:
                    print(f"[INFO] CARLA port {port} is accessible")
                    port_accessible = True
                    break
            except Exception:
                continue
        
        if not port_accessible:
            print("[WARNING] CARLA ports 2000 and 4000 are not accessible")
            # 如果容器在运行，我们仍然允许测试继续
            # CARLA可能需要一些时间启动，或者端口检查可能不准确
            if container_running:
                print("[INFO] CARLA container is running, allowing test to proceed (port may not be ready yet)")
                return True
            return False
        
        return True
    except Exception as e:
        print(f"[WARNING] Failed to check CARLA environment: {e}")
        return False

# 检查CARLA导入
CARLA_AVAILABLE = False
try:
    import config
    try:
        config.set_carla_api_path()
        import carla
        CARLA_AVAILABLE = True
    except (SystemExit, ImportError, AttributeError):
        CARLA_AVAILABLE = False
except ImportError:
    CARLA_AVAILABLE = False

# 检查CARLA环境
CARLA_ENV_AVAILABLE = check_carla_environment()

try:
    from experiments.analysis.scenario_loader import (
        ScenarioDataLoader,
        load_scenario_from_json,
        load_scenario_from_pickle,
        load_all_scenarios
    )
    from states import ScenarioState
    LOADER_AVAILABLE = True
    
    # 尝试导入Scenario（需要CARLA）
    if CARLA_AVAILABLE:
        try:
            from scenario import Scenario
            SCENARIO_AVAILABLE = True
        except (ImportError, AttributeError):
            SCENARIO_AVAILABLE = False
    else:
        SCENARIO_AVAILABLE = False
except ImportError as e:
    LOADER_AVAILABLE = False
    SCENARIO_AVAILABLE = False
    print(f"[WARNING] Scenario loader modules not available: {e}")


class TestScenarioLoaderWithCarla(unittest.TestCase):
    """测试场景数据加载功能（需要CARLA环境）"""
    
    @classmethod
    def setUpClass(cls):
        """类级别的设置，检查CARLA环境"""
        if not CARLA_ENV_AVAILABLE:
            raise unittest.SkipTest("CARLA environment not available. Please ensure CARLA container is running.")
        if not CARLA_AVAILABLE:
            raise unittest.SkipTest("CARLA Python API not available. Please check CARLA installation.")
        if not SCENARIO_AVAILABLE:
            raise unittest.SkipTest("Scenario class not available. CARLA may not be properly configured.")
    
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
    
    def test_load_scenario_from_pickle(self):
        """测试从pickle文件加载场景数据（需要真实Scenario对象）"""
        if not SCENARIO_AVAILABLE:
            self.skipTest("Scenario class not available")
        
        # 创建真实的Scenario对象（需要有效的seed_data）
        class MockConfig:
            def __init__(self):
                self.debug = False
        
        # Scenario需要有效的seed_data
        seed_data = {
            "sp_x": 0.0, "sp_y": 0.0, "sp_z": 0.0,
            "roll": 0.0, "yaw": 0.0, "pitch": 0.0,
            "wp_x": 10.0, "wp_y": 10.0, "wp_z": 0.0, "wp_yaw": 0.0,
            "map": 3
        }
        
        scenario = Scenario(MockConfig(), seed_data)
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
        # 空的ScenarioState应该返回False（没有speed或movement数据）
        self.assertFalse(loader.validate_scenario_for_metrics(invalid_scenario3))


if __name__ == "__main__":
    unittest.main()


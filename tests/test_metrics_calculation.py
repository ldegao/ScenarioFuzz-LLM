#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：测试指标计算脚本的功能（需要CARLA环境）

测试覆盖：
1. 从场景数据计算累积指标
2. 增量计算功能
3. 重新计算功能
4. 指标值的正确性

注意：这些测试需要CARLA环境运行，会检查CARLA容器是否运行。
"""

import unittest
import tempfile
import json
import shutil
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 首先检查CARLA环境（与test_calculate_metrics.py相同）
def check_carla_environment():
    """检查CARLA环境是否可用"""
    import subprocess
    import socket
    
    try:
        username = os.getlogin()
        docker_name = f"carla-{username}"
        
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", docker_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        container_running = (result.returncode == 0 and result.stdout.strip() == "running")
        
        if not container_running:
            script_dir = PROJECT_ROOT / "script"
            try:
                from experiments.core.environment_manager import ensure_carla_running
                ensure_carla_running(script_dir, PROJECT_ROOT)
                container_running = True
            except Exception as e:
                print(f"[WARNING] Failed to start CARLA: {e}")
                return False
        
        ports_to_check = [4000, 2000]
        for port in ports_to_check:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result_port = sock.connect_ex(('localhost', port))
                sock.close()
                if result_port == 0:
                    return True
            except Exception:
                continue
        
        if container_running:
            return True
        return False
    except Exception:
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

CARLA_ENV_AVAILABLE = check_carla_environment()

try:
    from experiments.analysis.calculate_metrics import calculate_metrics_from_scenarios
    from experiments.analysis.scenario_loader import ScenarioDataLoader
    from states import ScenarioState
    METRICS_AVAILABLE = True
except ImportError as e:
    METRICS_AVAILABLE = False
    print(f"[WARNING] Metrics calculation modules not available: {e}")


class MockScenario:
    """模拟Scenario对象用于测试"""
    def __init__(self, scenario_id, generation_id=1):
        self.scenario_id = scenario_id
        self.generation_id = generation_id
        self.state = ScenarioState()
        
        # 设置不同的测试数据以产生不同的指标值
        if scenario_id == 1:
            # 场景1：正常驾驶
            self.state.speed = [50.0, 55.0, 60.0]
            self.state.speed_lim = [60.0] * 3
            self.state.yaw_list = [0.0, 1.0, 2.0]
            self.state.yaw_rate_list = [0.0, 0.5, 1.0]
            self.state.lon_speed_list = [50.0, 55.0, 60.0]
            self.state.lat_speed_list = [0.0, 0.5, 1.0]
            self.state.steer_angle_list = [0.0, 0.1, 0.2]
            self.state.min_dist = 10.0
            self.state.crashed = False
            self.state.stuck = False
            self.state.laneinvaded = False
        elif scenario_id == 2:
            # 场景2：急转弯
            self.state.speed = [40.0, 35.0, 30.0]
            self.state.speed_lim = [60.0] * 3
            self.state.yaw_list = [0.0, 15.0, 30.0]
            self.state.yaw_rate_list = [0.0, 10.0, 20.0]
            self.state.lon_speed_list = [40.0, 35.0, 30.0]
            self.state.lat_speed_list = [0.0, 2.0, 4.0]
            self.state.steer_angle_list = [0.0, 0.3, 0.6]
            self.state.min_dist = 8.0
            self.state.crashed = False
            self.state.stuck = False
            self.state.laneinvaded = True
        elif scenario_id == 3:
            # 场景3：碰撞
            self.state.speed = [60.0, 50.0, 0.0]
            self.state.speed_lim = [60.0] * 3
            self.state.yaw_list = [0.0, 5.0, 10.0]
            self.state.yaw_rate_list = [0.0, 2.0, 4.0]
            self.state.lon_speed_list = [60.0, 50.0, 0.0]
            self.state.lat_speed_list = [0.0, 1.0, 2.0]
            self.state.steer_angle_list = [0.0, 0.2, 0.4]
            self.state.min_dist = 2.0
            self.state.crashed = True
            self.state.stuck = False
            self.state.laneinvaded = False


class TestMetricsCalculation(unittest.TestCase):
    """测试指标计算脚本的功能（需要CARLA环境）"""
    
    @classmethod
    def setUpClass(cls):
        """类级别的设置，检查CARLA环境"""
        if not CARLA_ENV_AVAILABLE:
            raise unittest.SkipTest("CARLA environment not available. Please ensure CARLA container is running.")
        if not CARLA_AVAILABLE:
            raise unittest.SkipTest("CARLA Python API not available. Please check CARLA installation.")
    
    def setUp(self):
        """设置测试环境"""
        if not METRICS_AVAILABLE:
            self.skipTest("Metrics calculation modules not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # 创建实验目录结构
        self.experiment_dir = self.temp_path / "test_experiment"
        self.queue_dir = self.experiment_dir / "queue"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建测试场景的JSON文件
        self.scenarios = [
            MockScenario(1, generation_id=1),
            MockScenario(2, generation_id=1),
            MockScenario(3, generation_id=1)
        ]
        
        # 保存场景为JSON（简化版本，只包含state数据）
        for scenario in self.scenarios:
            filename = f"gid:{scenario.generation_id}_sid:{scenario.scenario_id}.json"
            json_path = self.queue_dir / filename
            
            # 创建包含state数据的JSON
            state_data = {
                "speed": scenario.state.speed,
                "speed_lim": scenario.state.speed_lim,
                "yaw_list": scenario.state.yaw_list,
                "yaw_rate_list": scenario.state.yaw_rate_list,
                "lon_speed_list": scenario.state.lon_speed_list,
                "lat_speed_list": scenario.state.lat_speed_list,
                "steer_angle_list": scenario.state.steer_angle_list,
                "min_dist": scenario.state.min_dist,
                "crashed": scenario.state.crashed,
                "stuck": scenario.state.stuck,
                "laneinvaded": scenario.state.laneinvaded,
                "num_frames": 100,
                "elapsed_time": 4.0,
                "distance": 100.0
            }
            
            data = {
                "events": {
                    "crash": scenario.state.crashed,
                    "stuck": scenario.state.stuck,
                    "lane_invasion": scenario.state.laneinvaded
                },
                "config": {
                    "fps": 25,
                    "max_dist_from_player": 40,
                    "min_dist_from_player": 5,
                    "abort_seconds": 60
                },
                "state": state_data
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_calculate_metrics_from_scenarios(self):
        """测试从场景数据计算指标"""
        result = calculate_metrics_from_scenarios(
            experiment_dir=self.experiment_dir,
            output_dir=None,
            incremental=False,
            recalculate=False
        )
        
        # 验证结果
        self.assertIn('num_scenarios', result)
        self.assertIn('num_calculated', result)
        self.assertIn('metrics', result)
        
        self.assertEqual(result['num_scenarios'], 3)
        self.assertEqual(result['num_calculated'], 3)
        
        # 验证指标值
        metrics = result['metrics']
        self.assertIn('pc', metrics)
        self.assertIn('pec', metrics)
        self.assertIn('tcd', metrics)
        self.assertIn('bcm', metrics)
        
        # 指标值应该在合理范围内
        for metric_name, value in metrics.items():
            self.assertGreaterEqual(value, 0.0, f"{metric_name} should be >= 0")
            self.assertLessEqual(value, 1.0, f"{metric_name} should be <= 1")
        
        # 验证文件已创建
        metrics_dir = self.experiment_dir / "metrics"
        records_path = metrics_dir / "metrics_records.jsonl"
        summary_path = metrics_dir / "metrics_summary.json"
        
        self.assertTrue(records_path.exists(), "metrics_records.jsonl should be created")
        self.assertTrue(summary_path.exists(), "metrics_summary.json should be created")
        
        # 验证records文件内容
        with open(records_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            self.assertEqual(len(lines), 3, "Should have 3 records")
            
            # 验证第一条记录
            first_record = json.loads(lines[0])
            self.assertIn('pc', first_record)
            self.assertIn('pec', first_record)
            self.assertIn('tcd', first_record)
            self.assertIn('bcm', first_record)
    
    def test_incremental_calculation(self):
        """测试增量计算功能"""
        # 第一次计算
        result1 = calculate_metrics_from_scenarios(
            experiment_dir=self.experiment_dir,
            incremental=False,
            recalculate=False
        )
        
        self.assertEqual(result1['num_calculated'], 3)
        
        # 第二次计算（增量模式，应该跳过已计算的）
        result2 = calculate_metrics_from_scenarios(
            experiment_dir=self.experiment_dir,
            incremental=True,
            recalculate=False
        )
        
        # 应该跳过所有场景（因为都已计算）
        self.assertEqual(result2['num_calculated'], 0)
        self.assertEqual(result2['num_skipped'], 3)
    
    def test_recalculate(self):
        """测试重新计算功能"""
        # 第一次计算
        calculate_metrics_from_scenarios(
            experiment_dir=self.experiment_dir,
            incremental=False,
            recalculate=False
        )
        
        # 重新计算
        result = calculate_metrics_from_scenarios(
            experiment_dir=self.experiment_dir,
            incremental=False,
            recalculate=True
        )
        
        # 应该重新计算所有场景
        self.assertEqual(result['num_calculated'], 3)
        
        # 验证文件被覆盖（记录数应该还是3，不是6）
        metrics_dir = self.experiment_dir / "metrics"
        records_path = metrics_dir / "metrics_records.jsonl"
        
        with open(records_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            self.assertEqual(len(lines), 3, "Should have 3 records after recalculate")


if __name__ == "__main__":
    unittest.main()

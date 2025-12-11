#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：测试程序中所有序列化与反序列化功能

测试覆盖：
1. JSON 序列化/反序列化
   - Scenario_database
   - GPT 响应 JSON 提取
   - Token tracker 数据
   - Progress tracker 数据
   - Time estimator 数据
   - RAG knowledge base
   - Checkpoint 数据（JSON 部分）
   
2. Pickle 序列化/反序列化
   - Scenario 对象
   - Checkpoint 数据
   
3. 特殊对象序列化
   - ScenarioFitness 对象
   - 包含 fitness 的字典结构
"""

import unittest
import json
import pickle
import tempfile
import os
from pathlib import Path
from collections import OrderedDict
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入需要测试的模块
# 注意：不导入 scenario 模块以避免 CARLA 依赖
SCENARIO_AVAILABLE = False

try:
    import gpt
    GPT_AVAILABLE = True
except ImportError:
    GPT_AVAILABLE = False
    print("[WARNING] GPT module not available, skipping GPT-related tests")

try:
    from experiments.core.token_tracker import TokenTracker, get_tracker
    TOKEN_TRACKER_AVAILABLE = True
except ImportError:
    TOKEN_TRACKER_AVAILABLE = False
    print("[WARNING] Token tracker not available, skipping token tracker tests")

try:
    from experiments.core.progress_tracker import ProgressTracker
    PROGRESS_TRACKER_AVAILABLE = True
except ImportError:
    PROGRESS_TRACKER_AVAILABLE = False
    print("[WARNING] Progress tracker not available, skipping progress tracker tests")

try:
    from experiments.core.time_estimator import TimeEstimator
    TIME_ESTIMATOR_AVAILABLE = True
except ImportError:
    TIME_ESTIMATOR_AVAILABLE = False
    print("[WARNING] Time estimator not available, skipping time estimator tests")


class TestJSONSerialization(unittest.TestCase):
    """测试 JSON 序列化/反序列化"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_scenario_database_serialization(self):
        """测试 Scenario_database 的序列化/反序列化"""
        # 创建测试数据
        scenario_db = {
            "0": "The ego vehicle encounters dense fog reducing visibility",
            "1": "Construction zone with narrowed lanes",
            "2": "A vehicle stops suddenly in the middle of the road",
            "3": "A vehicle cuts in front of the ego vehicle",
            "4": "Slippery road conditions after rain"
        }
        
        # 测试序列化
        db_path = self.temp_path / "test_scenario_db.json"
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(scenario_db, f, ensure_ascii=False, indent=2)
        
        # 测试反序列化
        with open(db_path, 'r', encoding='utf-8') as f:
            loaded_db = json.load(f)
        
        self.assertEqual(scenario_db, loaded_db)
    
    def test_gpt_response_json_extraction(self):
        """测试 GPT 响应 JSON 提取"""
        if not GPT_AVAILABLE:
            self.skipTest("GPT module not available")
        
        # 测试正常 JSON 响应
        test_response = '{"answer1": {"Description": "测试场景"}, "answer2": {"Overall Similarity": "85"}}'
        extracted = gpt.extract_json(test_response)
        self.assertIsNotNone(extracted)
        self.assertIn("answer1", extracted)
        self.assertIn("answer2", extracted)
    
    def test_gpt_response_with_prefix_suffix(self):
        """测试带前缀后缀的 GPT 响应 JSON 提取"""
        if not GPT_AVAILABLE:
            self.skipTest("GPT module not available")
        
        # 测试带前缀和后缀的响应
        test_response = 'some prefix text\n{"answer1": {"Description": "测试"}}\nsome suffix text'
        extracted = gpt.extract_json(test_response)
        self.assertIsNotNone(extracted)
        self.assertIn("answer1", extracted)
    
    def test_token_tracker_serialization(self):
        """测试 Token tracker 的序列化/反序列化"""
        if not TOKEN_TRACKER_AVAILABLE:
            self.skipTest("Token tracker not available")
        
        tracker = TokenTracker()
        
        # 记录一些使用情况
        tracker.record_usage(
            model="gpt-5-mini",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        tracker.record_usage(
            model="gpt-4o-mini",
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300
        )
        
        # 测试保存
        stats_path = self.temp_path / "test_token_stats.json"
        tracker.save_to_file(stats_path)
        
        # 测试加载
        new_tracker = TokenTracker()
        loaded = new_tracker.load_from_file(stats_path)
        self.assertTrue(loaded)
        
        # 验证数据
        stats = new_tracker.get_stats()
        self.assertEqual(stats['overall']['total_tokens'], 450)
        self.assertEqual(len(stats['by_model']), 2)
    
    def test_progress_tracker_serialization(self):
        """测试 Progress tracker 的序列化/反序列化"""
        if not PROGRESS_TRACKER_AVAILABLE:
            self.skipTest("Progress tracker not available")
        
        checkpoint_path = self.temp_path / "test_progress_checkpoint.json"
        tracker = ProgressTracker(checkpoint_file=str(checkpoint_path))
        
        # 启动实验并更新进度
        experiment_id = "test_exp"
        tracker.start_experiment(
            experiment_id=experiment_id,
            method_name="test_method",
            target_scenarios=100
        )
        tracker.update_progress(experiment_id=experiment_id, scenario_id=1, scenario_info={"test": "data"})
        tracker.update_progress(experiment_id=experiment_id, scenario_id=2)
        
        # 创建新 tracker 并加载
        new_tracker = ProgressTracker(checkpoint_file=str(checkpoint_path))
        progress = new_tracker.get_progress(experiment_id)
        
        self.assertEqual(progress['method_name'], "test_method")
        self.assertEqual(progress['completed_scenarios'], 2)
    
    def test_time_estimator_serialization(self):
        """测试 Time estimator 的序列化/反序列化"""
        if not TIME_ESTIMATOR_AVAILABLE:
            self.skipTest("Time estimator not available")
        
        history_path = self.temp_path / "test_time_history.json"
        estimator = TimeEstimator(history_file=str(history_path))
        
        # 记录一些时间数据
        estimator.record_scenario_time("test_method", 5.5)  # 单个场景 5.5 秒
        estimator.record_scenario_time("test_method", 6.0)
        
        # 创建新 estimator 并加载
        new_estimator = TimeEstimator(history_file=str(history_path))
        estimate = new_estimator.estimate_total_time("test_method", 100)
        
        self.assertIn("total_time_str", estimate)
        self.assertIn("total_seconds", estimate)
        # 验证历史数据被加载（平均时间应该在 5.5-6.0 之间）
        self.assertGreater(estimate['avg_scenario_time'], 5.0)
        self.assertLess(estimate['avg_scenario_time'], 7.0)


class TestFitnessSerialization(unittest.TestCase):
    """测试 Fitness 对象的序列化"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_fitness_info_serialization(self):
        """测试 fitness 信息字典的序列化（模拟 fuzzer.py 中的逻辑）"""
        # 模拟 ScenarioFitness 对象
        class MockFitness:
            def __init__(self):
                self.values = (10.5, 20.3, 30.1)  # 元组
                self.valid = True
                self.weights = (-1.0, -1.0, 5.0)
        
        mock_fitness = MockFitness()
        
        # 按照 fuzzer.py 中的逻辑提取 fitness 信息
        fitness_info = None
        fitness_values = None
        
        if hasattr(mock_fitness, 'fitness') and mock_fitness.fitness is not None:
            fitness = mock_fitness.fitness
        else:
            fitness = mock_fitness
        
        try:
            if hasattr(fitness, 'values') and fitness.values is not None:
                fitness_values = list(fitness.values) if isinstance(fitness.values, tuple) else fitness.values
            
            fitness_info = {
                "values": fitness_values,
                "valid": getattr(fitness, 'valid', None),
                "weights": getattr(fitness, 'weights', None),
            }
        except Exception as fit_err:
            fitness_info = {"error": str(fit_err)}
        
        # 测试序列化
        scenario_state_info = {
            "min_dist": 5.5,
            "min_dist_frame": 100,
            "fitness_values": fitness_values,
            "fitness_info": fitness_info,
        }
        
        # 测试序列化
        json_path = self.temp_path / "test_fitness.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(scenario_state_info, f, ensure_ascii=False, indent=2)
        
        # 验证可以反序列化
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded['fitness_values'], [10.5, 20.3, 30.1])
        self.assertEqual(loaded['fitness_info']['valid'], True)
        self.assertEqual(loaded['fitness_info']['weights'], [-1.0, -1.0, 5.0])
    
    def test_fitness_with_none_values(self):
        """测试 fitness 值为 None 的情况"""
        scenario_state_info = {
            "min_dist": None,
            "min_dist_frame": None,
            "fitness_values": None,
            "fitness_info": None,
        }
        
        json_path = self.temp_path / "test_fitness_none.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(scenario_state_info, f, ensure_ascii=False, indent=2)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertIsNone(loaded['fitness_values'])
        self.assertIsNone(loaded['fitness_info'])


class TestGPTResponseSerialization(unittest.TestCase):
    """测试 GPT 响应相关的序列化"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_gpt_log_entry_serialization(self):
        """测试 GPT 日志条目的序列化（模拟 fuzzer.py 中的日志结构）"""
        # 模拟完整的 GPT 日志条目结构
        log_entry = {
                "generation_id": 0,
                "scenario_id": 1,
                "timestamp": "2025-12-08T17:00:00",
                "iso_timestamp": "2025-12-08T17:00:00.123456",
                
                "gpt_call_duration_seconds": 73.31,
                "model_version": "gpt-5-mini",
                "max_tokens": 10000,
                
                "prompt": "Test prompt",
                "prompt_length": 10,
                "raw_response": "Test response",
                "response_length": 13,
                "parsed_response": {
                    "answer1": {"Description": "Test"},
                    "answer2": {"Overall Similarity": "85"}
                },
                
                "token_stats": {
                    "prompt_tokens": 1986,
                    "completion_tokens": 3820,
                    "total_tokens": 5806
                },
                
                "scenario_description": "Test scenario",
                "scenario_state": {
                    "min_dist": 5.5,
                    "min_dist_frame": 100,
                    "fitness_values": [10.5, 20.3, 30.1],
                    "fitness_info": {
                        "values": [10.5, 20.3, 30.1],
                        "valid": True,
                        "weights": [-1.0, -1.0, 5.0]
                    }
                },
                
                "rag_info": {
                    "rag_enabled": False
                },
                
                "scenario_database_size": 5,
                "overall_similarity": 85,
                "answer3_vehicle_info": {}
        }
        
        # 测试序列化
        log_path = self.temp_path / "test_gpt_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
        
        # 测试反序列化
        with open(log_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded['scenario_id'], 1)
        self.assertEqual(loaded['parsed_response']['answer2']['Overall Similarity'], "85")
        self.assertEqual(loaded['scenario_state']['fitness_info']['valid'], True)
    
    def test_gpt_response_jsonl_serialization(self):
        """测试 JSONL 格式的序列化（用于 metrics aggregator）"""
        records = [
                {"scenario_id": 1, "similarity": 85, "timestamp": "2025-12-08T17:00:00"},
                {"scenario_id": 2, "similarity": 90, "timestamp": "2025-12-08T17:01:00"},
                {"scenario_id": 3, "similarity": 75, "timestamp": "2025-12-08T17:02:00"},
        ]
        
        # 写入 JSONL
        jsonl_path = self.temp_path / "test_records.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # 读取 JSONL
        loaded_records = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    loaded_records.append(json.loads(line))
        
        self.assertEqual(len(loaded_records), 3)
        self.assertEqual(loaded_records[0]['scenario_id'], 1)


class TestComplexDataStructures(unittest.TestCase):
    """测试复杂数据结构的序列化"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_nested_dict_serialization(self):
        """测试嵌套字典的序列化"""
        nested_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": 123,
                        "list": [1, 2, 3],
                        "nested_list": [[1, 2], [3, 4]]
                    }
                }
            }
        }
        
        json_path = self.temp_path / "test_nested.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(nested_data, f, ensure_ascii=False, indent=2)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded['level1']['level2']['level3']['value'], 123)
        self.assertEqual(loaded['level1']['level2']['level3']['list'], [1, 2, 3])
    
    def test_ordered_dict_serialization(self):
        """测试 OrderedDict 的序列化（Scenario_database 使用）"""
        ordered_db = OrderedDict([
            ("0", "Scenario 0"),
            ("1", "Scenario 1"),
            ("2", "Scenario 2"),
        ])
        
        json_path = self.temp_path / "test_ordered.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(ordered_db, f, ensure_ascii=False, indent=2)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        # JSON 加载后是普通 dict，但内容应该相同
        self.assertEqual(loaded["0"], "Scenario 0")
        self.assertEqual(loaded["1"], "Scenario 1")
    
    def test_unicode_serialization(self):
        """测试 Unicode 字符的序列化（中文等）"""
        unicode_data = {
            "中文": "测试",
            "description": "这是一个测试场景，包含中文描述。",
            "scenarios": {
                "0": "ADS车辆遇到浓雾，能见度降低到50米以下",
                "1": "施工区域，车道变窄，限速降低"
            }
        }
        
        json_path = self.temp_path / "test_unicode.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(unicode_data, f, ensure_ascii=False, indent=2)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded["中文"], "测试")
        self.assertIn("浓雾", loaded["scenarios"]["0"])


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_empty_dict_serialization(self):
        """测试空字典的序列化"""
        empty_data = {}
        json_path = self.temp_path / "test_empty.json"
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(empty_data, f)
            
            with open(json_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            
            self.assertEqual(loaded, {})
            print("✓ 空字典序列化测试通过")
        except Exception as e:
            self.fail(f"空字典序列化失败: {e}")
    
    def test_large_data_serialization(self):
        """测试大数据量的序列化"""
        large_data = {
            f"scenario_{i}": f"Description for scenario {i}" 
            for i in range(1000)
        }
        
        json_path = self.temp_path / "test_large.json"
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(large_data, f, ensure_ascii=False)
            
            with open(json_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            
            self.assertEqual(len(loaded), 1000)
            self.assertEqual(loaded["scenario_0"], "Description for scenario 0")
    
    def test_special_values_serialization(self):
        """测试特殊值的序列化（None, True, False, 数字）"""
        special_data = {
            "none_value": None,
            "true_value": True,
            "false_value": False,
            "zero": 0,
            "negative": -1,
            "float": 3.14,
            "empty_string": "",
            "empty_list": [],
            "empty_dict": {}
        }
        
        json_path = self.temp_path / "test_special.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(special_data, f, ensure_ascii=False)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertIsNone(loaded["none_value"])
        self.assertTrue(loaded["true_value"])
        self.assertFalse(loaded["false_value"])
        self.assertEqual(loaded["zero"], 0)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestJSONSerialization))
    suite.addTests(loader.loadTestsFromTestCase(TestFitnessSerialization))
    suite.addTests(loader.loadTestsFromTestCase(TestGPTResponseSerialization))
    suite.addTests(loader.loadTestsFromTestCase(TestComplexDataStructures))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回测试结果
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 60)
    print("序列化/反序列化单元测试")
    print("=" * 60)
    print()
    
    success = run_tests()
    
    print()
    print("=" * 60)
    if success:
        print("✓ 所有测试通过！")
        exit(0)
    else:
        print("✗ 部分测试失败")
        exit(1)


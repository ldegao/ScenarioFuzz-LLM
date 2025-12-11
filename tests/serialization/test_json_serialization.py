#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON 序列化/反序列化测试

测试所有使用 JSON 的序列化功能，不依赖 CARLA。
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
from collections import OrderedDict
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入需要测试的模块
try:
    import gpt
    GPT_AVAILABLE = True
except ImportError:
    GPT_AVAILABLE = False

try:
    from experiments.core.token_tracker import TokenTracker
    TOKEN_TRACKER_AVAILABLE = True
except ImportError:
    TOKEN_TRACKER_AVAILABLE = False

try:
    from experiments.core.progress_tracker import ProgressTracker
    PROGRESS_TRACKER_AVAILABLE = True
except ImportError:
    PROGRESS_TRACKER_AVAILABLE = False

try:
    from experiments.core.time_estimator import TimeEstimator
    TIME_ESTIMATOR_AVAILABLE = True
except ImportError:
    TIME_ESTIMATOR_AVAILABLE = False


class TestScenarioDatabaseSerialization(unittest.TestCase):
    """测试 Scenario_database 的序列化/反序列化"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_serialization(self):
        """测试基本序列化"""
        scenario_db = {
            "0": "The ego vehicle encounters dense fog",
            "1": "Construction zone with narrowed lanes",
            "2": "A vehicle stops suddenly"
        }
        
        db_path = self.temp_path / "test_scenario_db.json"
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(scenario_db, f, ensure_ascii=False, indent=2)
        
        with open(db_path, 'r', encoding='utf-8') as f:
            loaded_db = json.load(f)
        
        self.assertEqual(scenario_db, loaded_db)
    
    def test_ordered_dict_serialization(self):
        """测试 OrderedDict 序列化"""
        ordered_db = OrderedDict([
            ("0", "Scenario 0"),
            ("1", "Scenario 1"),
            ("2", "Scenario 2"),
        ])
        
        db_path = self.temp_path / "test_ordered.json"
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(ordered_db, f, ensure_ascii=False, indent=2)
        
        with open(db_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded["0"], "Scenario 0")
        self.assertEqual(loaded["1"], "Scenario 1")
    
    def test_unicode_serialization(self):
        """测试 Unicode 字符序列化"""
        unicode_data = {
            "中文": "测试",
            "description": "这是一个测试场景",
            "scenarios": {
                "0": "ADS车辆遇到浓雾，能见度降低"
            }
        }
        
        db_path = self.temp_path / "test_unicode.json"
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(unicode_data, f, ensure_ascii=False, indent=2)
        
        with open(db_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded["中文"], "测试")
        self.assertIn("浓雾", loaded["scenarios"]["0"])


class TestGPTResponseSerialization(unittest.TestCase):
    """测试 GPT 响应 JSON 提取"""
    
    def setUp(self):
        if not GPT_AVAILABLE:
            self.skipTest("GPT module not available")
    
    def test_simple_json_extraction(self):
        """测试简单 JSON 提取"""
        test_response = '{"answer1": {"Description": "测试场景"}, "answer2": {"Overall Similarity": "85"}}'
        extracted = gpt.extract_json(test_response)
        
        self.assertIsNotNone(extracted)
        self.assertIn("answer1", extracted)
        self.assertIn("answer2", extracted)
        self.assertEqual(extracted["answer2"]["Overall Similarity"], "85")
    
    def test_json_with_prefix_suffix(self):
        """测试带前缀后缀的 JSON 提取"""
        test_response = 'some prefix text\n{"answer1": {"Description": "测试"}}\nsome suffix text'
        extracted = gpt.extract_json(test_response)
        
        self.assertIsNotNone(extracted)
        self.assertIn("answer1", extracted)
    
    def test_invalid_json_handling(self):
        """测试无效 JSON 的处理"""
        invalid_responses = [
            "not json at all",
            "{invalid json}",
            '{"incomplete":',
            "",
        ]
        
        for invalid_response in invalid_responses:
            with self.subTest(response=invalid_response[:20]):
                extracted = gpt.extract_json(invalid_response)
                self.assertIsNone(extracted, "无效 JSON 应该返回 None")
    
    def test_nested_json_extraction(self):
        """测试嵌套 JSON 提取"""
        nested_response = '''{
            "answer1": {
                "Description": "测试场景",
                "ADS Vehicle": {
                    "Location": "(100, 200)",
                    "Speed": "50 km/h"
                }
            },
            "answer2": {
                "Overall Similarity": "85"
            }
        }'''
        extracted = gpt.extract_json(nested_response)
        
        self.assertIsNotNone(extracted)
        self.assertIn("ADS Vehicle", extracted["answer1"])
        self.assertEqual(extracted["answer1"]["ADS Vehicle"]["Speed"], "50 km/h")


class TestTokenTrackerSerialization(unittest.TestCase):
    """测试 Token tracker 序列化"""
    
    def setUp(self):
        if not TOKEN_TRACKER_AVAILABLE:
            self.skipTest("Token tracker not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load(self):
        """测试保存和加载"""
        tracker = TokenTracker()
        
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
        
        stats_path = self.temp_path / "test_token_stats.json"
        tracker.save_to_file(stats_path)
        
        new_tracker = TokenTracker()
        loaded = new_tracker.load_from_file(stats_path)
        
        self.assertTrue(loaded)
        stats = new_tracker.get_stats()
        self.assertEqual(stats['overall']['total_tokens'], 450)
        self.assertEqual(len(stats['by_model']), 2)
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        tracker = TokenTracker()
        nonexistent_path = self.temp_path / "nonexistent.json"
        
        loaded = tracker.load_from_file(nonexistent_path)
        self.assertFalse(loaded)
    
    def test_empty_tracker_serialization(self):
        """测试空 tracker 序列化"""
        tracker = TokenTracker()
        stats_path = self.temp_path / "test_empty_stats.json"
        
        tracker.save_to_file(stats_path)
        
        new_tracker = TokenTracker()
        loaded = new_tracker.load_from_file(stats_path)
        
        self.assertTrue(loaded)
        stats = new_tracker.get_stats()
        self.assertEqual(stats['overall']['total_tokens'], 0)


class TestProgressTrackerSerialization(unittest.TestCase):
    """测试 Progress tracker 序列化"""
    
    def setUp(self):
        if not PROGRESS_TRACKER_AVAILABLE:
            self.skipTest("Progress tracker not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_progress_save_and_load(self):
        """测试进度保存和加载"""
        checkpoint_path = self.temp_path / "test_progress_checkpoint.json"
        tracker = ProgressTracker(checkpoint_file=str(checkpoint_path))
        
        experiment_id = "test_exp"
        tracker.start_experiment(
            experiment_id=experiment_id,
            method_name="test_method",
            target_scenarios=100
        )
        tracker.update_progress(experiment_id=experiment_id, scenario_id=1)
        tracker.update_progress(experiment_id=experiment_id, scenario_id=2)
        
        new_tracker = ProgressTracker(checkpoint_file=str(checkpoint_path))
        progress = new_tracker.get_progress(experiment_id)
        
        self.assertIsNotNone(progress)
        self.assertEqual(progress['method_name'], "test_method")
        self.assertEqual(progress['completed_scenarios'], 2)


class TestTimeEstimatorSerialization(unittest.TestCase):
    """测试 Time estimator 序列化"""
    
    def setUp(self):
        if not TIME_ESTIMATOR_AVAILABLE:
            self.skipTest("Time estimator not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_time_history_save_and_load(self):
        """测试时间历史保存和加载"""
        history_path = self.temp_path / "test_time_history.json"
        estimator = TimeEstimator(history_file=str(history_path))
        
        estimator.record_scenario_time("test_method", 5.5)
        estimator.record_scenario_time("test_method", 6.0)
        
        new_estimator = TimeEstimator(history_file=str(history_path))
        estimate = new_estimator.estimate_total_time("test_method", 100)
        
        self.assertIn("total_time_str", estimate)
        self.assertGreater(estimate['avg_scenario_time'], 5.0)
        self.assertLess(estimate['avg_scenario_time'], 7.0)


class TestJSONLSerialization(unittest.TestCase):
    """测试 JSONL 格式序列化"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_jsonl_write_and_read(self):
        """测试 JSONL 写入和读取"""
        records = [
            {"scenario_id": 1, "similarity": 85, "timestamp": "2025-12-08T17:00:00"},
            {"scenario_id": 2, "similarity": 90, "timestamp": "2025-12-08T17:01:00"},
            {"scenario_id": 3, "similarity": 75, "timestamp": "2025-12-08T17:02:00"},
        ]
        
        jsonl_path = self.temp_path / "test_records.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        loaded_records = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    loaded_records.append(json.loads(line))
        
        self.assertEqual(len(loaded_records), 3)
        self.assertEqual(loaded_records[0]['scenario_id'], 1)
        self.assertEqual(loaded_records[1]['similarity'], 90)
    
    def test_jsonl_with_empty_lines(self):
        """测试包含空行的 JSONL"""
        records = [
            {"id": 1, "data": "test1"},
            {"id": 2, "data": "test2"},
        ]
        
        jsonl_path = self.temp_path / "test_with_empty.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
            f.write("\n")  # 空行
            f.write(json.dumps({"id": 3, "data": "test3"}) + "\n")
        
        loaded_records = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    loaded_records.append(json.loads(line))
        
        self.assertEqual(len(loaded_records), 3)


class TestComplexJSONStructures(unittest.TestCase):
    """测试复杂 JSON 结构"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_deeply_nested_structure(self):
        """测试深层嵌套结构"""
        nested_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": 123,
                            "list": [1, 2, [3, 4, [5, 6]]]
                        }
                    }
                }
            }
        }
        
        json_path = self.temp_path / "test_nested.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(nested_data, f, ensure_ascii=False, indent=2)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded['level1']['level2']['level3']['level4']['value'], 123)
        self.assertEqual(loaded['level1']['level2']['level3']['level4']['list'][2][2][0], 5)
    
    def test_large_dataset(self):
        """测试大数据集序列化"""
        large_data = {
            f"scenario_{i}": {
                "id": i,
                "description": f"Description for scenario {i}",
                "data": list(range(i % 100))
            }
            for i in range(1000)
        }
        
        json_path = self.temp_path / "test_large.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(large_data, f, ensure_ascii=False)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertEqual(len(loaded), 1000)
        self.assertEqual(loaded["scenario_0"]["id"], 0)
        self.assertEqual(loaded["scenario_999"]["id"], 999)
    
    def test_special_values(self):
        """测试特殊值序列化"""
        special_data = {
            "none_value": None,
            "true_value": True,
            "false_value": False,
            "zero": 0,
            "negative": -1,
            "float": 3.14159,
            "scientific": 1e10,
            "empty_string": "",
            "empty_list": [],
            "empty_dict": {},
            "unicode": "测试中文 🚗",
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
        self.assertEqual(loaded["negative"], -1)
        self.assertAlmostEqual(loaded["float"], 3.14159, places=5)
        self.assertEqual(loaded["unicode"], "测试中文 🚗")


if __name__ == "__main__":
    unittest.main()


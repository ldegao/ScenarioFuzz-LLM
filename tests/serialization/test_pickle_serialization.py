#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pickle 序列化测试

测试使用 pickle 的序列化功能，包括：
- Checkpoint 数据序列化
- 复杂对象序列化
- 循环引用处理
"""

import unittest
import pickle
import tempfile
from pathlib import Path
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestPickleBasicSerialization(unittest.TestCase):
    """测试基本 Pickle 序列化"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_dict_pickle(self):
        """测试基本字典的 pickle 序列化"""
        data = {
            "key1": "value1",
            "key2": 123,
            "key3": [1, 2, 3],
            "key4": {"nested": "dict"}
        }
        
        pickle_path = self.temp_path / "test_basic.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        
        with open(pickle_path, 'rb') as f:
            loaded = pickle.load(f)
        
        self.assertEqual(data, loaded)
    
    def test_nested_structure_pickle(self):
        """测试嵌套结构的 pickle 序列化"""
        nested_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "list": [1, 2, [3, 4]],
                        "tuple": (1, 2, 3),
                        "set": {1, 2, 3}
                    }
                }
            }
        }
        
        pickle_path = self.temp_path / "test_nested.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(nested_data, f)
        
        with open(pickle_path, 'rb') as f:
            loaded = pickle.load(f)
        
        self.assertEqual(loaded['level1']['level2']['level3']['list'], [1, 2, [3, 4]])
        # 注意：set 在 pickle 后仍然是 set
        self.assertEqual(loaded['level1']['level2']['level3']['set'], {1, 2, 3})
    
    def test_list_of_dicts_pickle(self):
        """测试字典列表的 pickle 序列化"""
        data_list = [
            {"id": 1, "name": "item1"},
            {"id": 2, "name": "item2"},
            {"id": 3, "name": "item3"}
        ]
        
        pickle_path = self.temp_path / "test_list.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(data_list, f)
        
        with open(pickle_path, 'rb') as f:
            loaded = pickle.load(f)
        
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[0]['id'], 1)
        self.assertEqual(loaded[2]['name'], "item3")


class TestCheckpointDataSerialization(unittest.TestCase):
    """测试 Checkpoint 数据序列化（模拟 fuzzer.py 中的结构）"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checkpoint_data_structure(self):
        """测试 checkpoint 数据结构序列化"""
        checkpoint_data = {
            'curr_gen': 5,
            'total_scenarios_generated': 100,
            'next_scenario_id': 101,
            'population': [],  # 实际是 Scenario 对象列表，这里用空列表测试
            'archive': [],
            'hof_items': [],
            'hof': None,  # 不保存 ParetoFront 对象
            'stats': None,  # 不保存 stats（包含 lambda 函数）
            'logbook': None,  # 不保存 logbook
            'determ_seed': 1234567890.0,
            'cur_time': 1234567890.0,
        }
        
        pickle_path = self.temp_path / "test_checkpoint.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        with open(pickle_path, 'rb') as f:
            loaded = pickle.load(f)
        
        self.assertEqual(loaded['curr_gen'], 5)
        self.assertEqual(loaded['total_scenarios_generated'], 100)
        self.assertEqual(loaded['next_scenario_id'], 101)
        self.assertIsNone(loaded['hof'])
        self.assertIsNone(loaded['stats'])
        self.assertIsNone(loaded['logbook'])
    
    def test_checkpoint_with_none_values(self):
        """测试包含 None 值的 checkpoint"""
        checkpoint_data = {
            'curr_gen': 0,
            'total_scenarios_generated': 0,
            'next_scenario_id': 1,
            'population': None,
            'archive': None,
            'hof_items': None,
        }
        
        pickle_path = self.temp_path / "test_checkpoint_none.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        with open(pickle_path, 'rb') as f:
            loaded = pickle.load(f)
        
        self.assertIsNone(loaded['population'])
        self.assertIsNone(loaded['archive'])


class TestNonSerializableObjects(unittest.TestCase):
    """测试不可序列化对象的处理"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_lambda_function_not_serializable(self):
        """测试 lambda 函数不可序列化"""
        data = {
            "normal_key": "value",
            "lambda_func": lambda x: x * 2
        }
        
        pickle_path = self.temp_path / "test_lambda.pkl"
        with self.assertRaises((pickle.PicklingError, AttributeError)):
            with open(pickle_path, 'wb') as f:
                pickle.dump(data, f)
    
    def test_removing_non_serializable(self):
        """测试移除不可序列化对象后的序列化"""
        # 模拟 fuzzer.py 中的处理方式
        data = {
            "serializable": "value",
            "lambda_func": lambda x: x * 2
        }
        
        # 移除不可序列化的对象
        data_clean = {k: v for k, v in data.items() if not callable(v)}
        data_clean['lambda_func'] = None
        
        pickle_path = self.temp_path / "test_cleaned.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(data_clean, f)
        
        with open(pickle_path, 'rb') as f:
            loaded = pickle.load(f)
        
        self.assertEqual(loaded['serializable'], "value")
        self.assertIsNone(loaded['lambda_func'])


if __name__ == "__main__":
    unittest.main()


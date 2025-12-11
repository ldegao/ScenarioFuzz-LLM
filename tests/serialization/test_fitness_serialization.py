#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fitness 对象序列化测试

测试 ScenarioFitness 对象和包含 fitness 的字典结构的序列化。
"""

import unittest
import json
import tempfile
from pathlib import Path
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestFitnessInfoSerialization(unittest.TestCase):
    """测试 fitness 信息字典的序列化（模拟 fuzzer.py 中的逻辑）"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_fitness_values_serialization(self):
        """测试 fitness values 序列化"""
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
        
        # 提取 fitness values（元组转换为列表）
        if hasattr(fitness, 'values') and fitness.values is not None:
            fitness_values = list(fitness.values) if isinstance(fitness.values, tuple) else fitness.values
        
        fitness_info = {
            "values": fitness_values,
            "valid": getattr(fitness, 'valid', None),
            "weights": getattr(fitness, 'weights', None),
        }
        
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
    
    def test_fitness_with_invalid_values(self):
        """测试 fitness 值无效的情况"""
        # 模拟无效的 fitness 对象
        class MockInvalidFitness:
            def __init__(self):
                self.values = None
                self.valid = False
                self.weights = None
        
        mock_fitness = MockInvalidFitness()
        
        fitness_info = {
            "values": None,
            "valid": False,
            "weights": None,
        }
        
        scenario_state_info = {
            "fitness_values": None,
            "fitness_info": fitness_info,
        }
        
        json_path = self.temp_path / "test_fitness_invalid.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(scenario_state_info, f, ensure_ascii=False, indent=2)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertIsNone(loaded['fitness_values'])
        self.assertFalse(loaded['fitness_info']['valid'])
    
    def test_fitness_error_handling(self):
        """测试 fitness 提取错误处理"""
        # 模拟提取失败的情况
        fitness_info = {"error": "Failed to extract fitness info"}
        
        scenario_state_info = {
            "fitness_values": None,
            "fitness_info": fitness_info,
        }
        
        json_path = self.temp_path / "test_fitness_error.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(scenario_state_info, f, ensure_ascii=False, indent=2)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertIn("error", loaded['fitness_info'])
        self.assertEqual(loaded['fitness_info']['error'], "Failed to extract fitness info")


class TestFitnessTupleToListConversion(unittest.TestCase):
    """测试 fitness values 从元组到列表的转换"""
    
    def test_tuple_to_list_conversion(self):
        """测试元组到列表转换"""
        # 模拟 fitness values 是元组的情况
        fitness_values_tuple = (10.5, 20.3, 30.1)
        fitness_values_list = list(fitness_values_tuple) if isinstance(fitness_values_tuple, tuple) else fitness_values_tuple
        
        self.assertIsInstance(fitness_values_list, list)
        self.assertEqual(fitness_values_list, [10.5, 20.3, 30.1])
    
    def test_already_list_no_conversion(self):
        """测试已经是列表的情况"""
        fitness_values_list = [10.5, 20.3, 30.1]
        converted = list(fitness_values_list) if isinstance(fitness_values_list, tuple) else fitness_values_list
        
        self.assertIsInstance(converted, list)
        self.assertEqual(converted, [10.5, 20.3, 30.1])


if __name__ == "__main__":
    unittest.main()


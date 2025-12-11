#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
序列化边界情况测试

测试各种边界情况和异常情况。
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

try:
    import gpt
    GPT_AVAILABLE = True
except ImportError:
    GPT_AVAILABLE = False


class TestEmptyAndNoneValues(unittest.TestCase):
    """测试空值和 None 值"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_empty_dict(self):
        """测试空字典"""
        empty_data = {}
        json_path = self.temp_path / "test_empty.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(empty_data, f)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded, {})
    
    def test_empty_list(self):
        """测试空列表"""
        empty_data = []
        json_path = self.temp_path / "test_empty_list.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(empty_data, f)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded, [])
    
    def test_all_none_values(self):
        """测试所有值都是 None"""
        none_data = {
            "key1": None,
            "key2": None,
            "nested": {
                "key3": None
            }
        }
        
        json_path = self.temp_path / "test_all_none.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(none_data, f)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertIsNone(loaded['key1'])
        self.assertIsNone(loaded['nested']['key3'])


class TestLargeDataSerialization(unittest.TestCase):
    """测试大数据量序列化"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_large_scenario_database(self):
        """测试大型 Scenario_database"""
        large_db = {
            f"scenario_{i}": f"Description for scenario {i} with some additional text to make it longer"
            for i in range(5000)
        }
        
        json_path = self.temp_path / "test_large_db.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(large_db, f, ensure_ascii=False)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertEqual(len(loaded), 5000)
        self.assertEqual(loaded["scenario_0"], "Description for scenario 0 with some additional text to make it longer")
        self.assertEqual(loaded["scenario_4999"], "Description for scenario 4999 with some additional text to make it longer")
    
    def test_deeply_nested_structure(self):
        """测试深层嵌套结构"""
        depth = 20
        nested = "value"
        for i in range(depth):
            nested = {f"level_{i}": nested}
        
        json_path = self.temp_path / "test_deep_nested.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(nested, f, ensure_ascii=False)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        # 验证可以访问到最深层（从最外层开始）
        current = loaded
        for i in range(depth - 1, -1, -1):  # 从 level_19 到 level_0
            self.assertIn(f"level_{i}", current)
            current = current[f"level_{i}"]
        self.assertEqual(current, "value")


class TestSpecialCharacters(unittest.TestCase):
    """测试特殊字符序列化"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_unicode_emojis(self):
        """测试 Unicode 表情符号"""
        emoji_data = {
            "vehicles": "🚗🚕🚙🚌🚎",
            "description": "测试场景包含表情符号 🎯"
        }
        
        json_path = self.temp_path / "test_emoji.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(emoji_data, f, ensure_ascii=False)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertIn("🚗", loaded["vehicles"])
        self.assertIn("🎯", loaded["description"])
    
    def test_special_json_characters(self):
        """测试 JSON 特殊字符"""
        special_data = {
            "quotes": 'Text with "quotes"',
            "newline": "Line 1\nLine 2",
            "tab": "Column1\tColumn2",
            "backslash": "Path\\to\\file",
            "unicode": "测试中文"
        }
        
        json_path = self.temp_path / "test_special_chars.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(special_data, f, ensure_ascii=False)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded["quotes"], 'Text with "quotes"')
        self.assertIn("\n", loaded["newline"])
        self.assertIn("\t", loaded["tab"])


class TestGPTResponseEdgeCases(unittest.TestCase):
    """测试 GPT 响应边界情况"""
    
    def setUp(self):
        if not GPT_AVAILABLE:
            self.skipTest("GPT module not available")
    
    def test_multiple_json_objects(self):
        """测试包含多个 JSON 对象的情况"""
        response = '{"first": "object"} some text {"second": "object"}'
        extracted = gpt.extract_json(response)
        
        # extract_json 可能只提取第一个，或者返回 None
        # 根据实际实现调整期望
        if extracted is not None:
            # 如果提取成功，应该包含第一个对象
            self.assertIn("first", extracted)
        # 如果返回 None 也是可以接受的（取决于实现）
    
    def test_json_with_comments(self):
        """测试包含注释的 JSON（虽然 JSON 不支持注释）"""
        # JSON 标准不支持注释，但测试是否能处理
        response = '/* comment */ {"answer": "test"} // comment'
        extracted = gpt.extract_json(response)
        
        # 应该能提取 JSON 部分
        self.assertIsNotNone(extracted)
    
    def test_malformed_json(self):
        """测试格式错误的 JSON"""
        malformed_responses = [
            '{"incomplete":',
            '{"key": "value"',
            '{key: "value"}',  # 缺少引号
            '{"key": value}',  # 值缺少引号
        ]
        
        for response in malformed_responses:
            with self.subTest(response=response[:20]):
                extracted = gpt.extract_json(response)
                # 应该返回 None 或处理错误
                # 具体行为取决于 extract_json 的实现
                self.assertIsNone(extracted)


class TestNumericPrecision(unittest.TestCase):
    """测试数值精度"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_float_precision(self):
        """测试浮点数精度"""
        precision_data = {
            "small_float": 1e-10,
            "large_float": 1e10,
            "pi": 3.14159265358979323846,
            "negative": -123.456789
        }
        
        json_path = self.temp_path / "test_precision.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(precision_data, f, ensure_ascii=False)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertAlmostEqual(loaded["pi"], 3.14159265358979323846, places=15)
        self.assertAlmostEqual(loaded["small_float"], 1e-10, places=20)
    
    def test_integer_limits(self):
        """测试整数边界"""
        limit_data = {
            "zero": 0,
            "negative": -1,
            "large": 999999999999999,
            "negative_large": -999999999999999
        }
        
        json_path = self.temp_path / "test_limits.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(limit_data, f, ensure_ascii=False)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded["zero"], 0)
        self.assertEqual(loaded["large"], 999999999999999)


if __name__ == "__main__":
    unittest.main()


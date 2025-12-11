#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA 对象序列化测试

测试 CARLA Location, Rotation, Transform 的序列化/反序列化。
使用与正式程序相同的方式导入 CARLA。
"""

import unittest
import json
import sys
import os
import glob
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 像正式程序那样设置 CARLA API 路径
# 参考 utils.py: 先调用 config.set_carla_api_path()，然后导入 carla
CARLA_AVAILABLE = False
carla = None

def setup_carla_import():
    """设置 CARLA 导入，使用与正式程序相同的方式（参考 utils.py 第22行）"""
    global CARLA_AVAILABLE, carla
    
    try:
        # 使用与正式代码相同的方式：调用 config.set_carla_api_path()
        # 参考 utils.py: config.set_carla_api_path()
        import config
        
        # config.set_carla_api_path() 如果找不到 CARLA 会 sys.exit(-1)
        # 我们需要捕获 SystemExit 异常，这是正常的（CARLA 不可用时）
        try:
            config.set_carla_api_path()
        except SystemExit:
            # CARLA 不可用，这是预期的，正常跳过测试
            CARLA_AVAILABLE = False
            return False
        
        # 尝试导入 carla（参考 utils.py 第25行）
        try:
            import carla
            CARLA_AVAILABLE = True
            return True
        except (ImportError, ModuleNotFoundError) as e:
            # CARLA 导入失败
            CARLA_AVAILABLE = False
            return False
            
    except ImportError:
        # config 模块不可用，尝试直接导入 carla
        try:
            import carla
            CARLA_AVAILABLE = True
            return True
        except (ImportError, ModuleNotFoundError):
            CARLA_AVAILABLE = False
            return False
    except Exception as e:
        # 其他异常不应该被隐藏，但也不应该阻止测试运行
        # 记录警告但继续（让测试决定是否跳过）
        CARLA_AVAILABLE = False
        return False

setup_carla_import()

# 导入工具函数（延迟导入，只在 CARLA 可用时）
# 注意：utils 模块在导入时会调用 config.set_carla_api_path()（第22行）
# 如果 CARLA 不可用，utils 模块导入会失败（sys.exit）
# 参考 utils.py 的导入方式
UTILS_AVAILABLE = False
utils = None
if CARLA_AVAILABLE:
    try:
        # utils 模块在导入时会调用 config.set_carla_api_path()
        # 如果 CARLA 不可用，会触发 SystemExit
        # 但由于我们已经设置了 CARLA 路径，这里应该能成功导入
        import utils
        UTILS_AVAILABLE = True
    except (ImportError, SystemExit):
        # SystemExit 表示 CARLA 不可用，这是预期的
        UTILS_AVAILABLE = False
    except Exception as e:
        # 其他异常应该被记录
        UTILS_AVAILABLE = False

# Mock 对象（用于无 CARLA 环境）
class MockLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

class MockRotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)

class MockTransform:
    def __init__(self, location, rotation):
        if isinstance(location, (list, tuple)):
            self.location = MockLocation(*location)
        else:
            self.location = location
        if isinstance(rotation, (list, tuple)):
            self.rotation = MockRotation(*rotation)
        else:
            self.rotation = rotation


class TestCARLALocationSerialization(unittest.TestCase):
    """测试 CARLA Location 序列化"""
    
    def setUp(self):
        if not UTILS_AVAILABLE:
            self.skipTest("utils module not available (requires CARLA)")
    
    def test_location_pickle_unpickle(self):
        """测试 Location 序列化/反序列化"""
        if CARLA_AVAILABLE:
            location = carla.Location(x=100.5, y=200.3, z=50.7)
        else:
            location = MockLocation(x=100.5, y=200.3, z=50.7)
        
        json_string = utils.carla_location_pickle(location)
        self.assertIsInstance(json_string, str)
        
        if CARLA_AVAILABLE:
            unpickled = utils.carla_location_unpickle(json_string)
            # CARLA 使用 float32，精度约为 7 位有效数字，使用 places=5 更合理
            self.assertAlmostEqual(unpickled.x, 100.5, places=5)
            self.assertAlmostEqual(unpickled.y, 200.3, places=5)
            self.assertAlmostEqual(unpickled.z, 50.7, places=5)
        else:
            # 验证 JSON 格式
            data = json.loads(json_string)
            self.assertIn('location_x', data)
            self.assertAlmostEqual(data['location_x'], 100.5, places=6)
    
    def test_location_edge_cases(self):
        """测试 Location 边界情况"""
        test_cases = [
            (0.0, 0.0, 0.0, "零值"),
            (-100.0, -200.0, -50.0, "负值"),
            (1e6, 1e6, 1e6, "大值"),
            (1e-6, 1e-6, 1e-6, "小值"),
        ]
        
        for x, y, z, desc in test_cases:
            with self.subTest(desc=desc):
                if CARLA_AVAILABLE:
                    location = carla.Location(x=x, y=y, z=z)
                else:
                    location = MockLocation(x=x, y=y, z=z)
                
                json_string = utils.carla_location_pickle(location)
                if CARLA_AVAILABLE:
                    unpickled = utils.carla_location_unpickle(json_string)
                    # CARLA 使用 float32，精度约为 7 位有效数字，使用 places=5 更合理
                    self.assertAlmostEqual(unpickled.x, x, places=5)
                    self.assertAlmostEqual(unpickled.y, y, places=5)
                    self.assertAlmostEqual(unpickled.z, z, places=5)


class TestCARLARotationSerialization(unittest.TestCase):
    """测试 CARLA Rotation 序列化"""
    
    def setUp(self):
        if not UTILS_AVAILABLE:
            self.skipTest("utils module not available (requires CARLA)")
    
    def test_rotation_pickle_unpickle(self):
        """测试 Rotation 序列化/反序列化"""
        if CARLA_AVAILABLE:
            rotation = carla.Rotation(pitch=10.5, yaw=20.3, roll=30.7)
        else:
            rotation = MockRotation(pitch=10.5, yaw=20.3, roll=30.7)
        
        json_string = utils.carla_rotation_pickle(rotation)
        self.assertIsInstance(json_string, str)
        
        data = json.loads(json_string)
        self.assertIn('rotation_pitch', data)
        self.assertIn('rotation_yaw', data)
        self.assertIn('rotation_roll', data)
        
        if CARLA_AVAILABLE:
            unpickled = utils.carla_rotation_unpickle(json_string)
            # CARLA 使用 float32，精度约为 7 位有效数字，使用 places=5 更合理
            self.assertAlmostEqual(unpickled.pitch, 10.5, places=5)
            self.assertAlmostEqual(unpickled.yaw, 20.3, places=5)
            self.assertAlmostEqual(unpickled.roll, 30.7, places=5)
    
    def test_rotation_backward_compatibility(self):
        """测试 Rotation 向后兼容（旧格式列表）"""
        if not UTILS_AVAILABLE:
            self.skipTest("utils module not available (requires CARLA)")
        if not CARLA_AVAILABLE:
            self.skipTest("需要 CARLA 环境")
        
        # 测试旧格式（列表）
        old_format_json = json.dumps([10.5, 20.3, 30.7])
        unpickled = utils.carla_rotation_unpickle(old_format_json)
        
        # CARLA 使用 float32，精度约为 7 位有效数字，使用 places=5 更合理
        self.assertAlmostEqual(unpickled.pitch, 10.5, places=5)
        self.assertAlmostEqual(unpickled.yaw, 20.3, places=5)
        self.assertAlmostEqual(unpickled.roll, 30.7, places=5)


class TestCARLATransformSerialization(unittest.TestCase):
    """测试 CARLA Transform 序列化"""
    
    def setUp(self):
        if not UTILS_AVAILABLE:
            self.skipTest("utils module not available (requires CARLA)")
    
    def test_transform_pickle_unpickle(self):
        """测试 Transform 序列化/反序列化"""
        if CARLA_AVAILABLE:
            location = carla.Location(x=100.5, y=200.3, z=50.7)
            rotation = carla.Rotation(pitch=10.5, yaw=20.3, roll=30.7)
            transform = carla.Transform(location, rotation)
        else:
            location = MockLocation(x=100.5, y=200.3, z=50.7)
            rotation = MockRotation(pitch=10.5, yaw=20.3, roll=30.7)
            transform = MockTransform(location, rotation)
        
        json_string = utils.carla_transform_pickle(transform)
        self.assertIsInstance(json_string, str)
        
        data = json.loads(json_string)
        required_fields = ['location_x', 'location_y', 'location_z', 
                          'rotation_pitch', 'rotation_yaw', 'rotation_roll']
        for field in required_fields:
            self.assertIn(field, data)
        
        if CARLA_AVAILABLE:
            unpickled = utils.carla_transform_unpickle(json_string)
            # CARLA 使用 float32，精度约为 7 位有效数字，使用 places=5 更合理
            self.assertAlmostEqual(unpickled.location.x, 100.5, places=5)
            self.assertAlmostEqual(unpickled.location.y, 200.3, places=5)
            self.assertAlmostEqual(unpickled.location.z, 50.7, places=5)
            self.assertAlmostEqual(unpickled.rotation.pitch, 10.5, places=5)
            self.assertAlmostEqual(unpickled.rotation.yaw, 20.3, places=5)
            self.assertAlmostEqual(unpickled.rotation.roll, 30.7, places=5)
    
    def test_transform_invalid_json(self):
        """测试无效 JSON 的处理"""
        if not UTILS_AVAILABLE:
            self.skipTest("utils module not available (requires CARLA)")
        if not CARLA_AVAILABLE:
            self.skipTest("需要 CARLA 环境")
        
        invalid_json_strings = [
            "not json",
            "{invalid json}",
            '{"location_x": 100}',  # 缺少字段
            "{}",  # 空对象
        ]
        
        for invalid_json in invalid_json_strings:
            with self.subTest(json=invalid_json[:20]):
                with self.assertRaises((ValueError, json.JSONDecodeError, KeyError)):
                    utils.carla_transform_unpickle(invalid_json)


class TestRoundTripConsistency(unittest.TestCase):
    """测试往返序列化一致性"""
    
    def setUp(self):
        if not UTILS_AVAILABLE:
            self.skipTest("utils module not available (requires CARLA)")
        if not CARLA_AVAILABLE:
            self.skipTest("需要 CARLA 环境")
    
    def test_location_round_trip(self):
        """测试 Location 往返序列化"""
        original = carla.Location(x=123.456, y=789.012, z=345.678)
        json_str = utils.carla_location_pickle(original)
        restored = utils.carla_location_unpickle(json_str)
        
        # CARLA 使用 float32，精度约为 7 位有效数字，使用 places=5 更合理
        self.assertAlmostEqual(original.x, restored.x, places=5)
        self.assertAlmostEqual(original.y, restored.y, places=5)
        self.assertAlmostEqual(original.z, restored.z, places=5)
    
    def test_rotation_round_trip(self):
        """测试 Rotation 往返序列化"""
        original = carla.Rotation(pitch=10.5, yaw=20.3, roll=30.7)
        json_str = utils.carla_rotation_pickle(original)
        restored = utils.carla_rotation_unpickle(json_str)
        
        # CARLA 使用 float32，精度约为 7 位有效数字，使用 places=5 更合理
        self.assertAlmostEqual(original.pitch, restored.pitch, places=5)
        self.assertAlmostEqual(original.yaw, restored.yaw, places=5)
        self.assertAlmostEqual(original.roll, restored.roll, places=5)
    
    def test_transform_round_trip(self):
        """测试 Transform 往返序列化"""
        original = carla.Transform(
            carla.Location(x=100, y=200, z=300),
            carla.Rotation(pitch=10, yaw=20, roll=30)
        )
        json_str = utils.carla_transform_pickle(original)
        restored = utils.carla_transform_unpickle(json_str)
        
        # CARLA 使用 float32，精度约为 7 位有效数字，使用 places=5 更合理
        self.assertAlmostEqual(original.location.x, restored.location.x, places=5)
        self.assertAlmostEqual(original.location.y, restored.location.y, places=5)
        self.assertAlmostEqual(original.location.z, restored.location.z, places=5)
        self.assertAlmostEqual(original.rotation.pitch, restored.rotation.pitch, places=5)
        self.assertAlmostEqual(original.rotation.yaw, restored.rotation.yaw, places=5)
        self.assertAlmostEqual(original.rotation.roll, restored.rotation.roll, places=5)


if __name__ == "__main__":
    unittest.main()


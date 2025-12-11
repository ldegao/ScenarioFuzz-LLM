#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 CARLA 导入设置的总结脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 60)
print("CARLA 导入测试总结")
print("=" * 60)
print()

# 测试 config 模块导入
print("1. 测试 config 模块导入...")
try:
    import config
    print("   ✓ config 模块导入成功")
    proj_root = config.get_proj_root()
    print("   项目根目录: {}".format(proj_root))
except Exception as e:
    print("   ✗ config 模块导入失败: {}".format(e))
    sys.exit(1)

# 测试 CARLA egg 查找
print("\n2. 测试 CARLA egg 文件查找...")
target_version = "0.9.13"
py_ver = "py{}.{}".format(sys.version_info.major, sys.version_info.minor)
platform_tag = "win-amd64" if os.name == "nt" else "linux-x86_64"

import glob
dist_path = os.path.join(proj_root, "carla", "PythonAPI", "carla", "dist")
exact_pattern = os.path.join(dist_path, "carla-{}-{}-{}.egg".format(target_version, py_ver, platform_tag))

candidate_paths = []
if os.path.exists(exact_pattern):
    candidate_paths.append(exact_pattern)
    print("   ✓ 找到精确匹配: {}".format(exact_pattern))

if not candidate_paths and os.path.isdir(dist_path):
    for path in glob.glob(os.path.join(dist_path, "carla-*.egg")):
        if target_version in os.path.basename(path):
            candidate_paths.append(path)
            print("   ✓ 找到匹配文件: {}".format(path))

if not candidate_paths:
    print("   ⚠️  未找到 CARLA egg 文件")
    print("   预期路径: {}".format(exact_pattern))
else:
    print("   找到 {} 个候选文件".format(len(candidate_paths)))

# 测试 CARLA 导入
print("\n3. 测试 CARLA 模块导入...")
if candidate_paths:
    api_path = sorted(candidate_paths)[0]
    if api_path not in sys.path:
        sys.path.append(api_path)
        print("   已添加路径: {}".format(api_path))
    
    try:
        import carla
        print("   ✓ CARLA 模块导入成功")
        print("   CARLA 版本: {}".format(getattr(carla, '__version__', 'unknown')))
    except ImportError as e:
        print("   ✗ CARLA 模块导入失败: {}".format(e))
else:
    print("   ⚠️  跳过 CARLA 导入（egg 文件不存在）")

# 测试 utils 模块导入
print("\n4. 测试 utils 模块导入...")
try:
    import utils
    print("   ✓ utils 模块导入成功")
    
    # 检查关键函数
    functions_to_check = [
        'carla_location_pickle',
        'carla_location_unpickle',
        'carla_rotation_pickle',
        'carla_rotation_unpickle',
        'carla_transform_pickle',
        'carla_transform_unpickle'
    ]
    
    for func_name in functions_to_check:
        if hasattr(utils, func_name):
            print("     ✓ {} 可用".format(func_name))
        else:
            print("     ✗ {} 不可用".format(func_name))
            
except ImportError as e:
    print("   ✗ utils 模块导入失败: {}".format(e))
except Exception as e:
    print("   ⚠️  utils 模块导入时出错: {}".format(e))

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)


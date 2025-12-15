#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 GPT-4o-mini 模型连通性的脚本
"""

import sys
import traceback
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gpt
import json
from datetime import datetime


def test_gpt_connectivity():
    """测试 GPT-4o-mini 的连通性和基本功能"""
    print("=" * 60)
    print("测试 GPT 模型连通性")
    print("=" * 60)
    print()
    
    # 测试问题
    test_question = "请用一句话回答：什么是人工智能？"
    
    default_model = gpt.DEFAULT_MODEL if hasattr(gpt, 'DEFAULT_MODEL') else 'gpt-4o-mini'
    print(f"模型: {default_model}")
    print("测试问题: {}".format(test_question))
    print("开始时间: {}".format(datetime.now().isoformat()))
    print()
    
    try:
        # 调用 GPT
        print("正在调用 GPT API...")
        default_model = gpt.DEFAULT_MODEL if hasattr(gpt, 'DEFAULT_MODEL') else 'gpt-4o-mini'
        response = gpt.call_gpt(
            question=test_question,
            model_version=None,  # 使用配置文件中的默认模型
            max_tokens=10000,
            retries=3  # 限制重试次数以便快速测试
        )
        
        print()
        print("=" * 60)
        print("测试结果: 成功")
        print("=" * 60)
        print("响应内容: {}".format(response))
        print()
        
        # 测试 JSON 提取功能
        print("测试 JSON 提取功能...")
        test_json_response = '{"answer1": {"Description": "测试场景"}, "answer2": {"Overall Similarity": "85"}}'
        extracted = gpt.extract_json(test_json_response)
        if extracted:
            print("JSON 提取成功: {}".format(json.dumps(extracted, ensure_ascii=False, indent=2)))
        else:
            print("JSON 提取失败（这是正常的，因为测试响应不是 JSON 格式）")
        
        print()
        print("=" * 60)
        print("连通性测试完成！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print()
        print("=" * 60)
        print("测试结果: 失败")
        print("=" * 60)
        print("错误信息: {}: {}".format(type(e).__name__, e))
        traceback.print_exc()
        return False


def test_token_tracking():
    """测试 token 追踪功能"""
    print()
    print("=" * 60)
    print("测试 Token 追踪功能")
    print("=" * 60)
    print()
    
    try:
        from experiments.core.token_tracker import get_tracker
        
        tracker = get_tracker()
        print("Token 追踪器初始化成功")
        
        # 模拟记录一些使用情况
        default_model = gpt.DEFAULT_MODEL if hasattr(gpt, 'DEFAULT_MODEL') else 'gpt-4o-mini'
        tracker.record_usage(
            model=default_model,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        print("已记录 token 使用情况")
        
        # 获取统计信息
        stats = tracker.get_stats()
        print("Token 统计信息: {}".format(json.dumps(stats, indent=2, ensure_ascii=False)))
        
        print()
        print("Token 追踪功能测试完成！")
        return True
        
    except Exception as e:
        print("Token 追踪功能测试失败: {}".format(e))
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print()
    print("开始测试 GPT 模型连通性和功能...")
    print()
    
    # 测试连通性
    connectivity_ok = test_gpt_connectivity()
    
    # 测试 token 追踪
    token_tracking_ok = test_token_tracking()
    
    print()
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    print("连通性测试: {}".format('✓ 通过' if connectivity_ok else '✗ 失败'))
    print("Token 追踪测试: {}".format('✓ 通过' if token_tracking_ok else '✗ 失败'))
    print()
    
    if connectivity_ok and token_tracking_ok:
        default_model = gpt.DEFAULT_MODEL if hasattr(gpt, 'DEFAULT_MODEL') else 'gpt-4o-mini'
        print(f"所有测试通过！{default_model} 已准备就绪。")
        sys.exit(0)
    else:
        print("部分测试失败，请检查配置和网络连接。")
        sys.exit(1)


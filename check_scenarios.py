#!/usr/bin/env python3
"""
辅助脚本：检查实验实际生成的场景数量
用法: python check_scenarios.py <experiment_directory>
"""

import sys
import os
from pathlib import Path
import json

def count_scenarios_in_directory(exp_dir):
    """统计实验目录中的场景文件数量"""
    exp_path = Path(exp_dir)
    
    if not exp_path.exists():
        print(f"错误: 目录不存在: {exp_dir}")
        return
    
    # 查找queue目录
    queue_dir = exp_path / "queue"
    if not queue_dir.exists():
        print(f"警告: queue目录不存在: {queue_dir}")
        # 尝试查找其他可能的场景文件位置
        json_files = list(exp_path.rglob("*.json"))
        print(f"找到 {len(json_files)} 个JSON文件:")
        for f in sorted(json_files)[:10]:  # 只显示前10个
            print(f"  - {f}")
        if len(json_files) > 10:
            print(f"  ... 还有 {len(json_files) - 10} 个文件")
        return
    
    # 统计queue目录中的场景文件
    scenario_files = [f for f in queue_dir.glob("*.json") if f.is_file()]
    scenario_count = len(scenario_files)
    
    print(f"\n实验目录: {exp_dir}")
    print(f"Queue目录: {queue_dir}")
    print(f"场景文件数量: {scenario_count}")
    
    if scenario_count > 0:
        # 获取场景ID范围
        scenario_ids = []
        for f in scenario_files:
            try:
                with open(f, 'r') as jf:
                    data = json.load(jf)
                    if 'scenario_id' in data:
                        scenario_ids.append(data['scenario_id'])
                    elif 'state' in data and 'scenario_id' in data['state']:
                        scenario_ids.append(data['state']['scenario_id'])
            except Exception as e:
                pass
        
        if scenario_ids:
            print(f"场景ID范围: {min(scenario_ids)} - {max(scenario_ids)}")
            print(f"场景ID列表: {sorted(scenario_ids)}")
        
        # 显示前几个文件
        print(f"\n前5个场景文件:")
        for f in sorted(scenario_files)[:5]:
            print(f"  - {f.name}")
    
    # 检查检查点文件
    checkpoint_file = exp_path / "ga_checkpoint.pkl"
    if checkpoint_file.exists():
        print(f"\n检查点文件存在: {checkpoint_file}")
        import pickle
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                print(f"  检查点中的场景数: {checkpoint.get('total_scenarios_generated', 'N/A')}")
                print(f"  检查点中的generation: {checkpoint.get('curr_gen', 'N/A')}")
                print(f"  检查点中的next_scenario_id: {checkpoint.get('next_scenario_id', 'N/A')}")
        except Exception as e:
            print(f"  无法读取检查点: {e}")
    else:
        print(f"\n检查点文件不存在")
    
    return scenario_count

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_scenarios.py <experiment_directory>")
        print("示例: python check_scenarios.py experiment_results/ScenarioFuzz-LLM/ScenarioFuzz-LLM_20251203_202534")
        sys.exit(1)
    
    exp_dir = sys.argv[1]
    count_scenarios_in_directory(exp_dir)


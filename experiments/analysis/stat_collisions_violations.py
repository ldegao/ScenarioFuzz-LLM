#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计碰撞和违规脚本
------------------

统计实验中的碰撞和违规情况：
- 碰撞次数和比例
- 违规次数和比例（红灯违规等）
- 最小距离统计
- 按实验分组统计

Usage:
    python -m experiments.analysis.stat_collisions_violations \\
        --experiment-dir ./experiments/runs/ScenarioFuzz-LLM/... \\
        [--output-dir ./reports]
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.scenario_loader import ScenarioDataLoader


def load_scenario_file(file_path: Path):
    """Load scenario from JSON or pickle file."""
    if file_path.suffix == '.pkl':
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to load pickle {file_path}: {e}")
            return None
    elif file_path.suffix == '.json':
        loader = ScenarioDataLoader()
        return loader.load_scenario_from_json(file_path)
    return None


def analyze_collisions_violations(experiment_dir: Path) -> Dict:
    """
    分析实验中的碰撞和违规情况。
    
    Args:
        experiment_dir: 实验目录路径
        
    Returns:
        统计结果字典
    """
    queue_dir = experiment_dir / "queue"
    if not queue_dir.exists():
        # Try scenarios directory
        queue_dir = experiment_dir / "scenarios"
    
    if not queue_dir.exists():
        print(f"[ERROR] Queue/scenarios directory not found in {experiment_dir}")
        return {}
    
    # Find all scenario files
    scenario_files = list(queue_dir.glob("*.pkl")) + list(queue_dir.glob("*.json"))
    
    if not scenario_files:
        print(f"[WARNING] No scenario files found in {queue_dir}")
        return {}
    
    print(f"[StatCollisions] Found {len(scenario_files)} scenario file(s)")
    
    # Statistics
    total_scenarios = 0
    collisions = 0
    violations = 0
    errors = 0
    min_distances = []
    collision_details = []
    violation_details = []
    
    for scenario_file in scenario_files:
        scenario = load_scenario_file(scenario_file)
        if scenario is None:
            continue
        
        total_scenarios += 1
        
        # Check found_error
        found_error = getattr(scenario, 'found_error', False)
        if found_error:
            errors += 1
        
        # Check collision
        state = getattr(scenario, 'state', None)
        if state is not None:
            collision_to = getattr(state, 'collision_to', None)
            if collision_to is not None:
                collisions += 1
                collision_details.append({
                    'file': scenario_file.name,
                    'generation_id': getattr(scenario, 'generation_id', -1),
                    'scenario_id': getattr(scenario, 'scenario_id', -1),
                    'collision_to': collision_to if isinstance(collision_to, (int, str)) else getattr(collision_to, 'id', None),
                })
            
            # Check red violation
            red_violation = getattr(state, 'red_violation', False)
            if red_violation:
                violations += 1
                violation_details.append({
                    'file': scenario_file.name,
                    'generation_id': getattr(scenario, 'generation_id', -1),
                    'scenario_id': getattr(scenario, 'scenario_id', -1),
                })
            
            # Collect min_dist
            min_dist = getattr(state, 'min_dist', None)
            if min_dist is not None and isinstance(min_dist, (int, float)):
                min_distances.append(float(min_dist))
    
    # Calculate statistics
    collision_rate = (collisions / total_scenarios * 100) if total_scenarios > 0 else 0.0
    violation_rate = (violations / total_scenarios * 100) if total_scenarios > 0 else 0.0
    error_rate = (errors / total_scenarios * 100) if total_scenarios > 0 else 0.0
    
    stats = {
        'total_scenarios': total_scenarios,
        'collisions': {
            'count': collisions,
            'rate': round(collision_rate, 2),
            'details': collision_details[:20]  # Limit to first 20 for summary
        },
        'violations': {
            'count': violations,
            'rate': round(violation_rate, 2),
            'details': violation_details[:20]  # Limit to first 20 for summary
        },
        'errors': {
            'count': errors,
            'rate': round(error_rate, 2)
        },
        'min_distance': {
            'count': len(min_distances),
            'min': round(min(min_distances), 3) if min_distances else None,
            'max': round(max(min_distances), 3) if min_distances else None,
            'mean': round(sum(min_distances) / len(min_distances), 3) if min_distances else None,
            'median': round(sorted(min_distances)[len(min_distances)//2], 3) if min_distances else None,
        }
    }
    
    return stats


def generate_report(stats: Dict, experiment_dir: Path, output_dir: Optional[Path] = None):
    """生成统计报告"""
    if output_dir is None:
        output_dir = experiment_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    json_path = output_dir / "collisions_violations_stats.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[StatCollisions] Saved JSON report: {json_path}")
    
    # Generate Markdown report
    md_path = output_dir / "collisions_violations_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 碰撞和违规统计报告\n\n")
        f.write(f"**实验目录**: {experiment_dir}\n\n")
        
        f.write("## 总体统计\n\n")
        f.write(f"- **总场景数**: {stats['total_scenarios']}\n")
        f.write(f"- **碰撞场景数**: {stats['collisions']['count']} ({stats['collisions']['rate']}%)\n")
        f.write(f"- **违规场景数**: {stats['violations']['count']} ({stats['violations']['rate']}%)\n")
        f.write(f"- **错误场景数**: {stats['errors']['count']} ({stats['errors']['rate']}%)\n\n")
        
        f.write("## 最小距离统计\n\n")
        min_dist = stats['min_distance']
        if min_dist['count'] > 0:
            f.write(f"- **统计场景数**: {min_dist['count']}\n")
            f.write(f"- **最小值**: {min_dist['min']} m\n")
            f.write(f"- **最大值**: {min_dist['max']} m\n")
            f.write(f"- **平均值**: {min_dist['mean']} m\n")
            f.write(f"- **中位数**: {min_dist['median']} m\n\n")
        else:
            f.write("无最小距离数据\n\n")
        
        f.write("## 碰撞详情（前20个）\n\n")
        if stats['collisions']['details']:
            f.write("| 文件 | Generation ID | Scenario ID | Collision To |\n")
            f.write("|------|---------------|-------------|--------------|\n")
            for detail in stats['collisions']['details']:
                f.write(f"| {detail['file']} | {detail['generation_id']} | {detail['scenario_id']} | {detail['collision_to']} |\n")
        else:
            f.write("无碰撞记录\n")
        f.write("\n")
        
        f.write("## 违规详情（前20个）\n\n")
        if stats['violations']['details']:
            f.write("| 文件 | Generation ID | Scenario ID |\n")
            f.write("|------|---------------|-------------|\n")
            for detail in stats['violations']['details']:
                f.write(f"| {detail['file']} | {detail['generation_id']} | {detail['scenario_id']} |\n")
        else:
            f.write("无违规记录\n")
        f.write("\n")
    
    print(f"[StatCollisions] Saved Markdown report: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="统计实验中的碰撞和违规情况"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="实验目录路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认：实验目录/reports）"
    )
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"[ERROR] Experiment directory {experiment_dir} does not exist")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Analyze
    stats = analyze_collisions_violations(experiment_dir)
    
    if not stats:
        print("[ERROR] No statistics generated")
        return
    
    # Generate report
    generate_report(stats, experiment_dir, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("统计摘要")
    print("="*60)
    print(f"总场景数: {stats['total_scenarios']}")
    print(f"碰撞: {stats['collisions']['count']} ({stats['collisions']['rate']}%)")
    print(f"违规: {stats['violations']['count']} ({stats['violations']['rate']}%)")
    print(f"错误: {stats['errors']['count']} ({stats['errors']['rate']}%)")
    if stats['min_distance']['count'] > 0:
        print(f"最小距离 - 平均: {stats['min_distance']['mean']} m, 最小: {stats['min_distance']['min']} m")
    print("="*60)


if __name__ == "__main__":
    main()


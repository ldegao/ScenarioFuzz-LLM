#!/usr/bin/env python3
"""
Check for outdated data that might affect comparison results.

This script identifies:
1. Experiment directories without metrics_summary.json
2. Old experiment snapshots in data/ folder
3. Multiple runs of the same method to identify which should be kept
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


def check_experiment_dirs(results_dir: Path) -> Dict[str, List[Tuple[Path, bool, str]]]:
    """
    Check experiment directories for each method.
    
    Returns:
        Dictionary mapping method names to list of (dir_path, has_metrics, timestamp)
    """
    method_dirs = {
        "answer2": [],
        "embedding": [],
        "feature": [],
        "hybrid": []
    }
    
    similarity_dir = results_dir / "SimilarityComparison"
    if not similarity_dir.exists():
        return method_dirs
    
    for exp_dir in similarity_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        if exp_name.startswith("SimilarityComparison_"):
            parts = exp_name.split("_")
            if len(parts) >= 3:
                method = parts[1]
                if method in method_dirs:
                    # Check if metrics_summary.json exists (try both possible locations)
                    metrics_path1 = exp_dir / "metrics" / "metrics_summary.json"
                    metrics_path2 = exp_dir / "metrics_summary.json"
                    has_metrics = metrics_path1.exists() or metrics_path2.exists()
                    
                    # Extract timestamp
                    timestamp = "_".join(parts[2:]) if len(parts) > 2 else "unknown"
                    
                    method_dirs[method].append((exp_dir, has_metrics, timestamp))
    
    # Sort by timestamp (newest first)
    for method in method_dirs:
        method_dirs[method].sort(key=lambda x: x[2], reverse=True)
    
    return method_dirs


def check_data_snapshots(data_dir: Path) -> List[Path]:
    """Check for old experiment snapshots in data/ folder."""
    snapshots_dir = data_dir / "experiment_snapshots"
    if not snapshots_dir.exists():
        return []
    
    snapshots = []
    for snapshot_dir in snapshots_dir.iterdir():
        if snapshot_dir.is_dir():
            snapshots.append(snapshot_dir)
    
    return sorted(snapshots, key=lambda x: x.name, reverse=True)


def generate_report(method_dirs: Dict, data_snapshots: List[Path], output_path: Path):
    """Generate a report on outdated data."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 过时数据检查报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 实验结果目录检查\n\n")
        
        for method in ["answer2", "embedding", "feature", "hybrid"]:
            if method not in method_dirs or not method_dirs[method]:
                continue
            
            f.write(f"### {method.upper()} 方法\n\n")
            
            valid_dirs = [d for d in method_dirs[method] if d[1]]
            invalid_dirs = [d for d in method_dirs[method] if not d[1]]
            
            if valid_dirs:
                f.write("**有效实验（有 metrics_summary.json）**:\n")
                for exp_dir, has_metrics, timestamp in valid_dirs:
                    f.write(f"- `{exp_dir.name}` (时间戳: {timestamp})\n")
                f.write("\n")
            
            if invalid_dirs:
                f.write("**⚠️ 无效/过时实验（无 metrics_summary.json）**:\n")
                for exp_dir, has_metrics, timestamp in invalid_dirs:
                    f.write(f"- `{exp_dir.name}` (时间戳: {timestamp}) - **建议删除**\n")
                f.write("\n")
        
        f.write("## data/ 文件夹检查\n\n")
        
        if data_snapshots:
            f.write("**实验快照目录**:\n")
            for snapshot in data_snapshots:
                f.write(f"- `{snapshot}`\n")
            f.write("\n")
            f.write("> 注意: 这些快照是旧的实验数据，分析脚本不会使用它们，但可能占用存储空间。\n\n")
        else:
            f.write("未发现实验快照。\n\n")
        
        f.write("## 影响分析\n\n")
        f.write("### 对分析结果的影响\n\n")
        f.write("1. **无效实验目录**: 分析脚本会尝试读取这些目录，但找不到 `metrics_summary.json`，")
        f.write("   会跳过这些目录。不会影响分析结果，但会产生警告信息。\n\n")
        f.write("2. **data/ 文件夹中的快照**: 分析脚本**不会**读取这些数据，因此不会影响分析结果。\n\n")
        f.write("3. **多个有效实验**: 如果同一方法有多个有效实验，分析脚本会聚合所有实验的结果。")
        f.write("   这是正常行为，用于计算平均值和标准差。\n\n")
        
        f.write("## 建议操作\n\n")
        
        has_invalid = any(any(not d[1] for d in dirs) for dirs in method_dirs.values() if dirs)
        
        if has_invalid:
            f.write("### 清理无效实验目录\n\n")
            f.write("以下命令可以删除无效的实验目录（请先确认）：\n\n")
            f.write("```bash\n")
            for method in ["answer2", "embedding", "feature", "hybrid"]:
                if method not in method_dirs:
                    continue
                for exp_dir, has_metrics, timestamp in method_dirs[method]:
                    if not has_metrics:
                        f.write(f"# rm -rf experiment_results/SimilarityComparison/{exp_dir.name}\n")
            f.write("```\n\n")
        
        if data_snapshots:
            f.write("### 清理旧快照（可选）\n\n")
            f.write("如果不需要保留旧快照，可以删除：\n\n")
            f.write("```bash\n")
            for snapshot in data_snapshots:
                f.write(f"# rm -rf data/experiment_snapshots/{snapshot.name}\n")
            f.write("```\n\n")


def main():
    results_dir = Path("./experiment_results")
    data_dir = Path("./data")
    output_path = Path("./reports/similarity_comparison/outdated_data_check.md")
    
    print("[CheckOutdatedData] Checking experiment directories...")
    method_dirs = check_experiment_dirs(results_dir)
    
    print("[CheckOutdatedData] Checking data snapshots...")
    data_snapshots = check_data_snapshots(data_dir)
    
    print("[CheckOutdatedData] Generating report...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_report(method_dirs, data_snapshots, output_path)
    
    print(f"\n[CheckOutdatedData] Report saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    for method in ["answer2", "embedding", "feature", "hybrid"]:
        if method not in method_dirs or not method_dirs[method]:
            continue
        
        valid = [d for d in method_dirs[method] if d[1]]
        invalid = [d for d in method_dirs[method] if not d[1]]
        
        print(f"\n{method.upper()}:")
        print(f"  Valid experiments: {len(valid)}")
        if invalid:
            print(f"  ⚠️  Invalid/outdated: {len(invalid)}")
            for exp_dir, _, timestamp in invalid:
                print(f"     - {exp_dir.name}")
    
    if data_snapshots:
        print(f"\ndata/experiment_snapshots: {len(data_snapshots)} snapshots found")
        print("  (These do not affect analysis results)")


if __name__ == "__main__":
    main()


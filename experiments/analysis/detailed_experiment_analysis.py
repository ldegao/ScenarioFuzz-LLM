#!/usr/bin/env python3
"""
Detailed analysis of experiment execution status.

This script analyzes:
1. File structure and contents
2. GA checkpoint files (indicates fuzzing ran)
3. Queue files (indicates scenarios were generated)
4. Metrics records (indicates metrics were calculated)
5. Code logic to understand when experiments are considered successful
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def analyze_experiment_directory(exp_dir: Path) -> Dict[str, Any]:
    """Comprehensive analysis of an experiment directory."""
    analysis = {
        "experiment_id": exp_dir.name,
        "exists": exp_dir.exists(),
        "file_count": 0,
        "directories": [],
        "key_files": {},
        "ga_checkpoint": None,
        "queue_files": [],
        "metrics_records": None,
        "metrics_summary": None,
        "token_usage": None,
        "diagnosis": []
    }
    
    if not exp_dir.exists():
        analysis["diagnosis"].append("❌ 实验目录不存在")
        return analysis
    
    # Count files
    all_files = list(exp_dir.rglob("*"))
    analysis["file_count"] = len([f for f in all_files if f.is_file()])
    
    # List directories
    analysis["directories"] = [d.name for d in exp_dir.iterdir() if d.is_dir()]
    
    # Check key files
    key_files = {
        "ga_checkpoint": exp_dir / "ga_checkpoint.pkl",
        "metrics_records": exp_dir / "metrics" / "metrics_records.jsonl",
        "metrics_summary": exp_dir / "metrics" / "metrics_summary.json",
        "token_usage": exp_dir / "token_usage.json"
    }
    
    for key, path in key_files.items():
        if path.exists():
            analysis["key_files"][key] = {
                "exists": True,
                "path": str(path),
                "size": path.stat().st_size,
                "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            analysis["key_files"][key] = {"exists": False}
    
    # Analyze GA checkpoint
    ga_path = exp_dir / "ga_checkpoint.pkl"
    if ga_path.exists():
        try:
            with open(ga_path, "rb") as f:
                ga_data = pickle.load(f)
            analysis["ga_checkpoint"] = {
                "size": ga_path.stat().st_size,
                "has_data": ga_data is not None,
                "type": type(ga_data).__name__ if ga_data is not None else None
            }
            analysis["diagnosis"].append(f"✓ GA checkpoint 存在 ({ga_path.stat().st_size:,} bytes)")
            analysis["diagnosis"].append("   这表明遗传算法（fuzzing）至少运行了一段时间")
        except Exception as e:
            analysis["ga_checkpoint"] = {"error": str(e)}
            analysis["diagnosis"].append(f"⚠️  GA checkpoint 存在但无法读取: {e}")
    else:
        analysis["diagnosis"].append("❌ 没有 GA checkpoint 文件")
        analysis["diagnosis"].append("   这表明遗传算法（fuzzing）可能没有运行")
    
    # Analyze queue files
    queue_dir = exp_dir / "queue"
    if queue_dir.exists():
        queue_files = list(queue_dir.glob("*.json"))
        analysis["queue_files"] = [f.name for f in queue_files]
        if queue_files:
            analysis["diagnosis"].append(f"✓ 队列目录包含 {len(queue_files)} 个场景文件")
            analysis["diagnosis"].append("   这表明实验生成了场景并保存到队列")
        else:
            analysis["diagnosis"].append("⚠️  队列目录存在但为空")
    else:
        analysis["diagnosis"].append("❌ 没有队列目录")
        analysis["diagnosis"].append("   这表明实验可能没有生成任何场景")
    
    # Analyze metrics_records.jsonl
    records_path = exp_dir / "metrics" / "metrics_records.jsonl"
    if records_path.exists():
        try:
            with open(records_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            analysis["metrics_records"] = {
                "count": len(lines),
                "size": records_path.stat().st_size
            }
            if lines:
                # Try to parse first and last records
                try:
                    first_record = json.loads(lines[0])
                    last_record = json.loads(lines[-1])
                    analysis["metrics_records"]["first_scenario_id"] = first_record.get("scenario_id")
                    analysis["metrics_records"]["last_scenario_id"] = last_record.get("scenario_id")
                except:
                    pass
            analysis["diagnosis"].append(f"✓ metrics_records.jsonl 存在，包含 {len(lines)} 条记录")
        except Exception as e:
            analysis["metrics_records"] = {"error": str(e)}
    else:
        analysis["diagnosis"].append("❌ 没有 metrics_records.jsonl 文件")
        analysis["diagnosis"].append("   这表明实验没有计算和保存指标")
    
    # Analyze metrics_summary.json
    summary_path = exp_dir / "metrics" / "metrics_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            analysis["metrics_summary"] = summary
            analysis["diagnosis"].append(f"✓ metrics_summary.json 存在")
            analysis["diagnosis"].append(f"   场景数: {summary.get('num_scenarios', 'N/A')}")
        except Exception as e:
            analysis["metrics_summary"] = {"error": str(e)}
    else:
        analysis["diagnosis"].append("❌ 没有 metrics_summary.json 文件")
        if analysis["metrics_records"] and analysis["metrics_records"].get("count", 0) > 0:
            analysis["diagnosis"].append("   虽然有 metrics_records.jsonl，但聚合过程可能失败")
        else:
            analysis["diagnosis"].append("   因为没有 metrics_records.jsonl，所以无法生成")
    
    # Analyze token usage
    token_path = exp_dir / "token_usage.json"
    if token_path.exists():
        try:
            with open(token_path, "r", encoding="utf-8") as f:
                token_data = json.load(f)
            analysis["token_usage"] = token_data.get("overall", {})
            total_tokens = analysis["token_usage"].get("total_tokens", 0)
            if total_tokens > 0:
                analysis["diagnosis"].append(f"✓ Token 使用: {total_tokens:,} tokens")
            else:
                analysis["diagnosis"].append("ℹ️  Token 使用量为 0（对于 embedding/feature 方法是正常的）")
        except Exception as e:
            analysis["token_usage"] = {"error": str(e)}
    
    # Overall assessment
    # Check if GA checkpoint file exists (even if we can't read it due to carla import issues)
    ga_file = exp_dir / "ga_checkpoint.pkl"
    has_ga = ga_file.exists() and ga_file.stat().st_size > 0
    has_queue = len(analysis["queue_files"]) > 0
    has_metrics_records = analysis["metrics_records"] and analysis["metrics_records"].get("count", 0) > 0
    has_metrics_summary = analysis["metrics_summary"] is not None
    
    analysis["overall_status"] = "unknown"
    if has_ga and has_queue and has_metrics_records and has_metrics_summary:
        analysis["overall_status"] = "complete"
        analysis["diagnosis"].append("\n✅ 实验状态: 完整运行")
    elif has_ga and has_queue and has_metrics_records:
        analysis["overall_status"] = "complete_no_summary"
        analysis["diagnosis"].append("\n✅ 实验状态: 完整运行（但缺少聚合的 summary，可以重新生成）")
    elif has_ga and has_queue:
        analysis["overall_status"] = "partial"
        analysis["diagnosis"].append("\n⚠️  实验状态: 部分运行（生成了场景但可能缺少指标）")
    elif has_ga:
        analysis["overall_status"] = "started"
        analysis["diagnosis"].append("\n⚠️  实验状态: 已启动但可能未完成")
    else:
        analysis["overall_status"] = "failed"
        analysis["diagnosis"].append("\n❌ 实验状态: 未运行或失败")
    
    return analysis


def main():
    results_dir = Path("./experiment_results/SimilarityComparison")
    
    experiments = [
        "SimilarityComparison_answer2_20251208_211248",
        "SimilarityComparison_embedding_20251209_095659",
        "SimilarityComparison_feature_20251209_095726",
        "SimilarityComparison_hybrid_20251209_095745"
    ]
    
    print("="*80)
    print("详细实验运行状态分析")
    print("="*80)
    print("\n根据代码逻辑，实验成功运行的标志：")
    print("1. GA checkpoint (ga_checkpoint.pkl) - 表明遗传算法运行了")
    print("2. Queue 文件 - 表明场景被生成并保存")
    print("3. metrics_records.jsonl - 表明指标被计算和记录")
    print("4. metrics_summary.json - 从 metrics_records.jsonl 聚合生成")
    print("\n注意：embedding 和 feature 方法不需要调用 LLM API，所以 token 为 0 是正常的")
    print("="*80)
    
    all_analyses = {}
    
    for exp_id in experiments:
        exp_dir = results_dir / exp_id
        print(f"\n{'='*80}")
        print(f"实验: {exp_id}")
        print(f"{'='*80}")
        
        analysis = analyze_experiment_directory(exp_dir)
        all_analyses[exp_id] = analysis
        
        print(f"\n文件统计:")
        print(f"  总文件数: {analysis['file_count']}")
        print(f"  目录: {', '.join(analysis['directories'])}")
        
        print(f"\n关键文件:")
        for key, info in analysis["key_files"].items():
            if info.get("exists"):
                print(f"  ✓ {key}: {info['size']:,} bytes, 修改时间: {info['modified']}")
            else:
                print(f"  ✗ {key}: 不存在")
        
        if analysis["ga_checkpoint"] and 'size' in analysis["ga_checkpoint"]:
            print(f"\nGA Checkpoint:")
            print(f"  大小: {analysis['ga_checkpoint']['size']:,} bytes")
            print(f"  类型: {analysis['ga_checkpoint'].get('type', 'N/A')}")
        
        if analysis["queue_files"]:
            print(f"\n队列文件: {len(analysis['queue_files'])} 个")
            print(f"  示例: {analysis['queue_files'][:3]}")
        
        if analysis["metrics_records"]:
            print(f"\nMetrics Records:")
            print(f"  记录数: {analysis['metrics_records']['count']}")
            if 'first_scenario_id' in analysis['metrics_records']:
                print(f"  场景ID范围: {analysis['metrics_records']['first_scenario_id']} - {analysis['metrics_records']['last_scenario_id']}")
        
        if analysis["metrics_summary"]:
            print(f"\nMetrics Summary:")
            summary = analysis["metrics_summary"]
            print(f"  场景数: {summary.get('num_scenarios', 'N/A')}")
            print(f"  PC: {summary.get('pc', 'N/A')}")
            print(f"  PEC: {summary.get('pec', 'N/A')}")
            print(f"  TCD: {summary.get('tcd', 'N/A')}")
            print(f"  BCM: {summary.get('bcm', 'N/A')}")
        
        print(f"\n诊断:")
        for item in analysis["diagnosis"]:
            print(f"  {item}")
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("总结对比")
    print(f"{'='*80}")
    print(f"{'实验':<50} {'状态':<15} {'GA':<5} {'队列':<5} {'指标':<5} {'汇总':<5}")
    print("-"*80)
    
    for exp_id in experiments:
        analysis = all_analyses[exp_id]
        status = analysis["overall_status"]
        has_ga = "✓" if analysis["ga_checkpoint"] else "✗"
        has_queue = "✓" if len(analysis["queue_files"]) > 0 else "✗"
        has_records = "✓" if analysis["metrics_records"] and analysis["metrics_records"].get("count", 0) > 0 else "✗"
        has_summary = "✓" if analysis["metrics_summary"] else "✗"
        
        print(f"{exp_id:<50} {status:<15} {has_ga:<5} {has_queue:<5} {has_records:<5} {has_summary:<5}")
    
    print(f"\n{'='*80}")
    print("结论")
    print(f"{'='*80}")
    print("""
根据代码分析（experiment_manager.py）：
1. 实验运行时会创建 GA checkpoint (ga_checkpoint.pkl)
2. 生成的场景会保存到 queue/ 目录
3. 指标会记录到 metrics/metrics_records.jsonl
4. 实验结束后，从 metrics_records.jsonl 聚合生成 metrics_summary.json

有效实验的标准：
- 有 GA checkpoint（表明 fuzzing 运行了）
- 有队列文件（表明场景生成了）
- 有 metrics_records.jsonl（表明指标计算了）
- 有 metrics_summary.json（表明聚合完成了）

如果缺少这些文件，说明实验在相应阶段失败了。
    """)


if __name__ == "__main__":
    main()


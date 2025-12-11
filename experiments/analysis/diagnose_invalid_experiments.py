#!/usr/bin/env python3
"""
Diagnose why embedding and feature experiments are invalid.

This script checks:
1. Whether metrics_records.jsonl exists
2. Whether any scenarios were generated
3. Token usage (to see if experiment actually ran)
4. Error logs
"""

import json
from pathlib import Path
from typing import Dict, List


def diagnose_experiment(exp_dir: Path) -> Dict:
    """Diagnose an experiment directory."""
    diagnosis = {
        "experiment_id": exp_dir.name,
        "has_metrics_dir": (exp_dir / "metrics").exists(),
        "has_metrics_records": (exp_dir / "metrics" / "metrics_records.jsonl").exists(),
        "has_metrics_summary": (exp_dir / "metrics" / "metrics_summary.json").exists(),
        "has_token_usage": (exp_dir / "token_usage.json").exists(),
        "token_usage": None,
        "metrics_records_count": 0,
        "has_errors": (exp_dir / "errors").exists() and any((exp_dir / "errors").iterdir()),
        "error_files": [],
        "has_queue": (exp_dir / "queue").exists() and any((exp_dir / "queue").iterdir()),
        "queue_files_count": 0,
        "diagnosis": []
    }
    
    # Check token usage
    token_path = exp_dir / "token_usage.json"
    if token_path.exists():
        try:
            with open(token_path, "r", encoding="utf-8") as f:
                diagnosis["token_usage"] = json.load(f)
        except:
            pass
    
    # Check metrics_records.jsonl
    records_path = exp_dir / "metrics" / "metrics_records.jsonl"
    if records_path.exists():
        try:
            with open(records_path, "r", encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
            diagnosis["metrics_records_count"] = count
        except:
            pass
    
    # Check error files
    errors_dir = exp_dir / "errors"
    if errors_dir.exists():
        diagnosis["error_files"] = [f.name for f in errors_dir.iterdir() if f.is_file()]
    
    # Check queue files
    queue_dir = exp_dir / "queue"
    if queue_dir.exists():
        diagnosis["queue_files_count"] = sum(1 for f in queue_dir.iterdir() if f.is_file())
    
    # Generate diagnosis
    if not diagnosis["has_metrics_records"]:
        diagnosis["diagnosis"].append("❌ 没有 metrics_records.jsonl 文件")
        diagnosis["diagnosis"].append("   这意味着实验没有生成任何场景数据")
    
    if not diagnosis["has_metrics_summary"]:
        diagnosis["diagnosis"].append("❌ 没有 metrics_summary.json 文件")
        if diagnosis["has_metrics_records"]:
            diagnosis["diagnosis"].append("   虽然存在 metrics_records.jsonl，但聚合过程可能失败")
        else:
            diagnosis["diagnosis"].append("   因为没有 metrics_records.jsonl，所以无法生成 summary")
    
    if diagnosis["token_usage"]:
        total_tokens = diagnosis["token_usage"].get("overall", {}).get("total_tokens", 0)
        if total_tokens == 0:
            diagnosis["diagnosis"].append("⚠️  Token 使用量为 0")
            diagnosis["diagnosis"].append("   这表明实验可能根本没有运行，或者没有调用 API")
        else:
            diagnosis["diagnosis"].append(f"✓ Token 使用量: {total_tokens:,}")
    
    if diagnosis["metrics_records_count"] == 0:
        diagnosis["diagnosis"].append("❌ 没有场景记录")
        diagnosis["diagnosis"].append("   实验可能启动后立即失败，或者没有生成任何场景")
    
    if diagnosis["has_errors"]:
        diagnosis["diagnosis"].append(f"⚠️  发现错误目录，包含 {len(diagnosis['error_files'])} 个错误文件")
    
    if diagnosis["queue_files_count"] > 0:
        diagnosis["diagnosis"].append(f"ℹ️  队列目录包含 {diagnosis['queue_files_count']} 个文件")
    
    return diagnosis


def main():
    results_dir = Path("./experiment_results/SimilarityComparison")
    
    embedding_dir = results_dir / "SimilarityComparison_embedding_20251209_095659"
    feature_dir = results_dir / "SimilarityComparison_feature_20251209_095726"
    
    print("="*60)
    print("实验诊断报告")
    print("="*60)
    
    if embedding_dir.exists():
        print("\n## EMBEDDING 实验诊断")
        print("-"*60)
        embedding_diag = diagnose_experiment(embedding_dir)
        print(f"实验ID: {embedding_diag['experiment_id']}")
        print(f"有 metrics 目录: {embedding_diag['has_metrics_dir']}")
        print(f"有 metrics_records.jsonl: {embedding_diag['has_metrics_records']}")
        print(f"有 metrics_summary.json: {embedding_diag['has_metrics_summary']}")
        print(f"场景记录数: {embedding_diag['metrics_records_count']}")
        print(f"Token 使用: {embedding_diag['token_usage']}")
        print("\n诊断结果:")
        for item in embedding_diag['diagnosis']:
            print(f"  {item}")
    
    if feature_dir.exists():
        print("\n## FEATURE 实验诊断")
        print("-"*60)
        feature_diag = diagnose_experiment(feature_dir)
        print(f"实验ID: {feature_diag['experiment_id']}")
        print(f"有 metrics 目录: {feature_diag['has_metrics_dir']}")
        print(f"有 metrics_records.jsonl: {feature_diag['has_metrics_records']}")
        print(f"有 metrics_summary.json: {feature_diag['has_metrics_summary']}")
        print(f"场景记录数: {feature_diag['metrics_records_count']}")
        print(f"Token 使用: {feature_diag['token_usage']}")
        print("\n诊断结果:")
        for item in feature_diag['diagnosis']:
            print(f"  {item}")
    
    print("\n" + "="*60)
    print("结论")
    print("="*60)
    print("""
根据诊断结果，embedding 和 feature 实验无效的原因是：

1. **没有生成场景数据**: 实验没有生成 metrics_records.jsonl 文件
   - metrics_summary.json 是从 metrics_records.jsonl 聚合生成的
   - 如果没有 metrics_records.jsonl，就无法生成 metrics_summary.json

2. **实验可能未运行**: Token 使用量为 0，说明：
   - 实验可能启动后立即失败
   - 或者实验配置有问题，导致没有调用 API
   - 或者实验被提前终止

3. **可能的原因**:
   - 实验脚本运行时出错
   - 配置参数不正确
   - 依赖服务（如 CARLA）未启动
   - 实验被手动终止或系统资源不足

**建议**:
- 检查实验运行日志，查看是否有错误信息
- 重新运行这两个实验，确保它们能正常完成
- 如果实验确实失败，可以删除这些无效的实验目录
    """)


if __name__ == "__main__":
    main()


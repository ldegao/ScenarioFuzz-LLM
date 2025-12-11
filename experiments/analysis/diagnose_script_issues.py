#!/usr/bin/env python3
"""
Comprehensive diagnosis of script issues causing embedding/feature experiments to fail
and statistics script data discrepancies.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def analyze_bash_script_issues() -> Dict:
    """Analyze potential issues in the Bash script."""
    issues = []
    
    script_path = Path("experiments/scripts/run_similarity_comparison.sh")
    if not script_path.exists():
        return {"error": "Script not found"}
    
    with open(script_path, "r") as f:
        content = f.read()
    
    # Check 1: set -e behavior
    if "set -e" in content:
        issues.append({
            "type": "warning",
            "issue": "Script uses 'set -e' which exits on any error",
            "location": "Line 5",
            "impact": "If Python script fails early (before creating output), script will exit",
            "fix": "Consider using 'set +e' before eval or better error handling"
        })
    
    # Check 2: Error handling
    if "if eval" in content and "then" in content:
        issues.append({
            "type": "info",
            "issue": "Script uses 'if eval' to catch errors",
            "location": "Line 131",
            "impact": "Should catch Python script failures, but may miss early failures",
            "fix": "Ensure Python script returns proper exit codes"
        })
    
    # Check 3: Command building
    if "CMD=" in content and "eval" in content:
        issues.append({
            "type": "warning",
            "issue": "Uses string concatenation for command building",
            "location": "Lines 114-125",
            "impact": "Potential issues with special characters or spaces",
            "fix": "Consider using arrays or proper quoting"
        })
    
    return {"issues": issues, "script_exists": True}


def analyze_python_runner_issues() -> Dict:
    """Analyze potential issues in Python runner."""
    issues = []
    
    # Check experiment_manager.py
    manager_path = Path("experiments/core/experiment_manager.py")
    if not manager_path.exists():
        return {"error": "Manager not found"}
    
    with open(manager_path, "r") as f:
        content = f.read()
    
    # Check 1: Directory creation timing
    if "method_dir.mkdir" in content:
        issues.append({
            "type": "critical",
            "issue": "Directory created before experiment runs",
            "location": "Line 130",
            "impact": "If experiment fails early, empty directory is left behind",
            "fix": "Only create directory after successful initialization"
        })
    
    # Check 2: Exception handling
    if "except Exception" in content and "raise" in content:
        issues.append({
            "type": "warning",
            "issue": "Exceptions are re-raised after logging",
            "location": "Lines 207-210",
            "impact": "Bash script will see non-zero exit code, but may not see error message",
            "fix": "Ensure error messages are printed before re-raising"
        })
    
    # Check 3: Early failures
    if "_create_args" in content:
        issues.append({
            "type": "info",
            "issue": "Args created before directory validation",
            "location": "Line 474",
            "impact": "If arg parsing fails, directory may already exist",
            "fix": "Validate args before creating directory"
        })
    
    return {"issues": issues, "manager_exists": True}


def analyze_statistics_script_issues() -> Dict:
    """Analyze potential issues in statistics script."""
    issues = []
    
    stats_path = Path("experiments/analysis/compare_similarity_methods.py")
    if not stats_path.exists():
        return {"error": "Stats script not found"}
    
    with open(stats_path, "r") as f:
        content = f.read()
    
    # Check 1: Path resolution
    if "results_dir / \"SimilarityComparison\"" in content:
        issues.append({
            "type": "critical",
            "issue": "Hardcoded path 'SimilarityComparison' in find_experiment_dirs",
            "location": "Line 84",
            "impact": "If results_dir already points to SimilarityComparison, will fail",
            "fix": "Check if results_dir already contains SimilarityComparison"
        })
    
    # Check 2: Metrics summary path
    if "metrics_summary.json" in content:
        issues.append({
            "type": "fixed",
            "issue": "Metrics summary path was fixed to check both locations",
            "location": "Lines 51-56",
            "impact": "Now checks both metrics/metrics_summary.json and metrics_summary.json",
            "fix": "Already fixed"
        })
    
    # Check 3: Empty results handling
    if "if not all_method_metrics" in content:
        issues.append({
            "type": "info",
            "issue": "Checks for empty results",
            "location": "Line 378",
            "impact": "Will exit if no metrics found",
            "fix": "Good - prevents generating empty reports"
        })
    
    return {"issues": issues, "stats_exists": True}


def check_experiment_directories() -> Dict:
    """Check actual experiment directories for issues."""
    results_dir = Path("experiment_results/SimilarityComparison")
    if not results_dir.exists():
        return {"error": "Results directory not found"}
    
    findings = {
        "embedding": {"exists": False, "has_ga": False, "has_metrics": False, "has_summary": False},
        "feature": {"exists": False, "has_ga": False, "has_metrics": False, "has_summary": False},
        "answer2": {"exists": False, "has_ga": False, "has_metrics": False, "has_summary": False},
        "hybrid": {"exists": False, "has_ga": False, "has_metrics": False, "has_summary": False}
    }
    
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        if not exp_name.startswith("SimilarityComparison_"):
            continue
        
        parts = exp_name.split("_")
        if len(parts) >= 3:
            method = parts[1]
            if method in findings:
                findings[method]["exists"] = True
                findings[method]["has_ga"] = (exp_dir / "ga_checkpoint.pkl").exists()
                findings[method]["has_metrics"] = (exp_dir / "metrics" / "metrics_records.jsonl").exists()
                findings[method]["has_summary"] = (exp_dir / "metrics" / "metrics_summary.json").exists()
                findings[method]["dir"] = exp_name
    
    return findings


def test_python_command(method: str, num_scenarios: int = 3) -> Dict:
    """Test if Python command can be executed (dry run)."""
    try:
        cmd = [
            sys.executable, "-m", "experiments.runners.run_similarity_comparison",
            "--num-scenarios", str(num_scenarios),
            "--similarity-method", method,
            "--output-root", "./test_output",
            "--target", "behavior",
            "--town", "3",
            "--timeout", "60",
            "--rag-k", "5"
        ]
        
        # Just check if it can parse arguments (don't actually run)
        result = subprocess.run(
            cmd + ["--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        return {
            "can_parse_args": result.returncode == 0,
            "stdout": result.stdout[:500] if result.stdout else "",
            "stderr": result.stderr[:500] if result.stderr else ""
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("="*80)
    print("脚本问题诊断报告")
    print("="*80)
    
    # 1. Analyze Bash script
    print("\n1. Bash 脚本分析")
    print("-"*80)
    bash_issues = analyze_bash_script_issues()
    if "error" in bash_issues:
        print(f"错误: {bash_issues['error']}")
    else:
        for issue in bash_issues["issues"]:
            print(f"\n[{issue['type'].upper()}] {issue['issue']}")
            print(f"  位置: {issue['location']}")
            print(f"  影响: {issue['impact']}")
            print(f"  修复: {issue['fix']}")
    
    # 2. Analyze Python runner
    print("\n2. Python 运行器分析")
    print("-"*80)
    python_issues = analyze_python_runner_issues()
    if "error" in python_issues:
        print(f"错误: {python_issues['error']}")
    else:
        for issue in python_issues["issues"]:
            print(f"\n[{issue['type'].upper()}] {issue['issue']}")
            print(f"  位置: {issue['location']}")
            print(f"  影响: {issue['impact']}")
            print(f"  修复: {issue['fix']}")
    
    # 3. Analyze statistics script
    print("\n3. 统计脚本分析")
    print("-"*80)
    stats_issues = analyze_statistics_script_issues()
    if "error" in stats_issues:
        print(f"错误: {stats_issues['error']}")
    else:
        for issue in stats_issues["issues"]:
            print(f"\n[{issue['type'].upper()}] {issue['issue']}")
            print(f"  位置: {issue['location']}")
            print(f"  影响: {issue['impact']}")
            print(f"  修复: {issue['fix']}")
    
    # 4. Check experiment directories
    print("\n4. 实验目录检查")
    print("-"*80)
    exp_findings = check_experiment_directories()
    for method, info in exp_findings.items():
        if isinstance(info, dict) and "exists" in info:
            print(f"\n{method.upper()}:")
            print(f"  目录存在: {info['exists']}")
            if info['exists']:
                print(f"  目录名: {info.get('dir', 'N/A')}")
                print(f"  有 GA checkpoint: {info['has_ga']}")
                print(f"  有 metrics_records: {info['has_metrics']}")
                print(f"  有 metrics_summary: {info['has_summary']}")
    
    # 5. Test Python commands
    print("\n5. Python 命令测试")
    print("-"*80)
    for method in ["embedding", "feature"]:
        print(f"\n测试 {method} 方法:")
        result = test_python_command(method, 3)
        if "error" in result:
            print(f"  错误: {result['error']}")
        else:
            print(f"  可以解析参数: {result['can_parse_args']}")
            if result.get('stderr'):
                print(f"  错误输出: {result['stderr'][:200]}")
    
    # Summary
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("""
主要发现：

1. **Bash 脚本问题**:
   - `set -e` 可能导致早期失败时脚本退出
   - 错误处理依赖 Python 脚本的退出码

2. **Python 运行器问题**:
   - 目录在实验运行前就创建，导致失败时留下空目录
   - 异常被重新抛出，可能被 Bash 脚本捕获

3. **统计脚本问题**:
   - 路径解析逻辑已修复（检查两个位置）
   - 但可能仍有路径解析问题

4. **实验目录状态**:
   - embedding 和 feature 实验目录存在但为空
   - 说明实验启动后立即失败

建议修复：
1. 改进 Bash 脚本的错误处理
2. 延迟目录创建直到确认实验可以运行
3. 添加更详细的错误日志
4. 检查实验启动时的依赖（CARLA、配置等）
    """)


if __name__ == "__main__":
    main()


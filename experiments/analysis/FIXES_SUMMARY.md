# 脚本问题修复总结

## 问题诊断完成

已全面分析 embedding 和 feature 实验未运行的原因，以及统计脚本数据偏差问题。

## 主要发现

### 1. embedding 和 feature 实验未运行

**根本原因**:
- 实验在启动阶段就失败了（可能在参数解析、配置设置、CARLA 连接等阶段）
- 目录在实验运行前就创建，导致失败时留下空目录
- 错误信息可能没有正确传递到 Bash 脚本

**证据**:
- 实验目录存在但完全为空（只有 token_usage.json，token 为 0）
- 没有 GA checkpoint（遗传算法未运行）
- 没有队列文件（未生成场景）
- 没有 metrics 文件（未计算指标）

### 2. 统计脚本数据偏差

**根本原因**:
- 部分实验有 `metrics_records.jsonl` 但缺少 `metrics_summary.json`
- 聚合过程在归档时执行，如果归档失败或实验被中断，聚合不会执行
- 聚合失败时只打印警告，不阻止归档

**已修复**:
- ✅ `load_metrics_summary()` 现在检查两个路径位置
- ✅ 提供了修复工具 `fix_script_issues.py` 可以重新生成缺少的 summary

## 修复方案

### 立即可以执行的修复

1. **修复缺少的 metrics_summary.json**:
   ```bash
   python3 -m experiments.analysis.fix_script_issues --results-dir ./experiment_results
   ```
   这会为所有有 `metrics_records.jsonl` 但缺少 `metrics_summary.json` 的实验重新生成 summary。

### 代码修复建议

#### 1. Bash 脚本改进 (`run_similarity_comparison.sh`)

**问题**: 错误处理可能不够完善

**修复** (Line 130-138):
```bash
# 改进错误处理
set +e  # 临时禁用 exit on error
eval "$CMD"
EXIT_CODE=$?
set -e  # 重新启用

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Completed: $METHOD"
else
    echo ""
    echo "✗ Failed: $METHOD (exit code: $EXIT_CODE)"
    echo "查看上方错误信息。继续下一个方法..."
fi
```

#### 2. Python 运行器改进 (`experiment_manager.py`)

**问题**: 目录创建过早，错误信息可能丢失

**修复 A** (Line 130 之前):
```python
# 添加验证步骤
try:
    # 验证实验可以启动
    self._validate_experiment_config(method_name, **kwargs)
except Exception as validation_error:
    print(f"[ERROR] Experiment validation failed: {validation_error}", file=sys.stderr)
    raise

# 现在可以安全创建目录
method_dir.mkdir(parents=True, exist_ok=True)
```

**修复 B** (Line 207-210):
```python
except Exception as e:
    print(f"\n[ERROR] Experiment failed: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)  # 明确退出码
```

#### 3. 统计脚本改进 (`compare_similarity_methods.py`)

**问题**: 路径解析可能有歧义

**修复** (Line 84):
```python
# 处理两种情况：results_dir 可能已经包含 SimilarityComparison
if results_dir.name == "SimilarityComparison":
    similarity_dir = results_dir
else:
    similarity_dir = results_dir / "SimilarityComparison"
```

## 诊断工具

已创建以下诊断工具：

1. **`diagnose_script_issues.py`**: 全面诊断脚本问题
2. **`detailed_experiment_analysis.py`**: 详细分析实验运行状态
3. **`fix_script_issues.py`**: 修复缺少的 metrics_summary.json
4. **`check_outdated_data.py`**: 检查过时数据
5. **`diagnose_invalid_experiments.py`**: 诊断无效实验

## 下一步行动

### 短期（立即执行）

1. ✅ 运行修复工具生成缺少的 summary:
   ```bash
   python3 -m experiments.analysis.fix_script_issues --results-dir ./experiment_results
   ```

2. ✅ 重新运行统计脚本:
   ```bash
   python3 -m experiments.analysis.compare_similarity_methods \
     --results-dir ./experiment_results \
     --output-dir ./reports/similarity_comparison
   ```

### 中期（建议实施）

1. 实施 Bash 脚本的错误处理改进
2. 实施 Python 运行器的验证步骤
3. 改进统计脚本的路径解析

### 长期（可选）

1. 添加实验启动前的完整验证
2. 实现聚合过程的自动重试
3. 添加更详细的错误日志和报告

## 关于 embedding 和 feature 实验

**当前状态**: 这两个实验确实没有运行

**可能的原因**:
1. 实验在启动阶段遇到错误（配置、依赖、环境等）
2. 错误被静默处理或没有正确报告
3. Bash 脚本继续执行下一个方法，没有详细记录错误

**建议**:
1. 手动运行这两个实验，捕获详细错误:
   ```bash
   python3 -m experiments.runners.run_similarity_comparison \
     --num-scenarios 10 \
     --similarity-method embedding \
     --output-root ./test_output \
     --target behavior
   ```

2. 检查错误输出，确定失败原因
3. 根据错误信息修复配置或代码
4. 重新运行完整实验

## 文件清单

所有分析报告和工具已保存在 `experiments/analysis/` 目录：

- `root_cause_analysis.md` - 详细根本原因分析
- `FIXES_SUMMARY.md` - 本文件，修复总结
- `diagnose_script_issues.py` - 脚本问题诊断工具
- `fix_script_issues.py` - 修复工具
- `detailed_experiment_analysis.py` - 实验状态分析工具
- `check_outdated_data.py` - 过时数据检查工具


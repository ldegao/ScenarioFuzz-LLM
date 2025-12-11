# 根本原因分析报告

## 问题总结

1. **embedding 和 feature 实验未运行**: 实验目录存在但完全为空（无 GA checkpoint、无队列文件、无 metrics）
2. **统计脚本数据偏差**: 部分实验有 metrics_records.jsonl 但缺少 metrics_summary.json

## 根本原因分析

### 1. embedding 和 feature 实验未运行的原因

#### 问题现象
- 实验目录已创建（`SimilarityComparison_embedding_20251209_095659` 和 `SimilarityComparison_feature_20251209_095726`）
- 目录中只有 `token_usage.json`（token 为 0，这是正常的，因为这两个方法不需要 LLM API）
- 没有 `ga_checkpoint.pkl`（说明遗传算法没有运行）
- 没有队列文件（说明没有生成场景）
- 没有 `metrics_records.jsonl`（说明没有计算指标）

#### 根本原因

**A. 实验启动流程问题**

根据代码分析（`experiment_manager.py`）：

1. **目录创建时机过早** (Line 130):
   ```python
   method_dir.mkdir(parents=True, exist_ok=True)
   ```
   - 目录在实验实际运行前就创建了
   - 如果实验在初始化阶段失败，会留下空目录

2. **异常处理机制** (Lines 207-210):
   ```python
   except Exception as e:
       print(f"\n[ERROR] Experiment failed: {e}")
       traceback.print_exc()
       raise  # Re-raise to ensure error is visible
   ```
   - 异常被重新抛出，Bash 脚本会看到非零退出码
   - 但错误信息可能没有正确传递

3. **Bash 脚本的错误处理** (Line 131):
   ```bash
   if eval "$CMD"; then
       echo "✓ Completed: $METHOD"
   else
       echo "✗ Failed: $METHOD"
       echo "Continuing with next method..."
   fi
   ```
   - 使用 `if eval` 捕获错误，但可能无法捕获所有类型的失败
   - `set -e` 在脚本开头，但被 `if` 语句覆盖

**B. 可能的失败点**

实验可能在以下阶段失败：

1. **参数解析阶段**: `_create_args()` 可能失败
2. **配置设置阶段**: 相似度方法配置可能有问题
3. **CARLA 连接阶段**: `ensure_carla_running()` 可能失败
4. **初始化脚本阶段**: `run_init_script()` 可能失败
5. **Fuzzer 启动阶段**: `fuzzer.main()` 可能在导入或初始化时失败

**C. 为什么 answer2 和 hybrid 成功了？**

- 这两个方法可能需要 LLM API，如果 API 可用，实验可以继续
- embedding 和 feature 方法可能有特殊的代码路径，在某些条件下会失败
- 可能是配置问题，某些配置只对 answer2 和 hybrid 有效

### 2. 统计脚本数据偏差的原因

#### 问题现象
- `SimilarityComparison_answer2_20251208_154241` 有 `metrics_records.jsonl` 但缺少 `metrics_summary.json`
- 其他一些实验也可能有类似问题

#### 根本原因

**A. 聚合过程失败** (Line 1025-1026):
```python
except Exception as metrics_err:
    print(f"[WARNING] Failed to aggregate metrics for run {experiment_id}: {metrics_err}")
```
- 聚合过程在 `_archive_experiment_run()` 中执行
- 如果聚合失败，只打印警告，不阻止归档
- 导致 `metrics_records.jsonl` 存在但 `metrics_summary.json` 不存在

**B. 聚合时机问题**
- 聚合只在归档时执行（实验结束后）
- 如果实验被中断或归档失败，聚合不会执行
- 需要手动运行聚合脚本

**C. 路径查找问题（已修复）**
- 之前 `load_metrics_summary()` 只查找 `experiment_dir / "metrics_summary.json"`
- 实际文件在 `experiment_dir / "metrics" / "metrics_summary.json"`
- 已修复为检查两个位置

## 修复方案

### 1. 改进 Bash 脚本错误处理

**问题**: `set -e` 和错误捕获可能不够完善

**修复**:
```bash
# 在循环中临时禁用 set -e
set +e
if eval "$CMD"; then
    echo "✓ Completed: $METHOD"
    EXIT_CODE=0
else
    EXIT_CODE=$?
    echo "✗ Failed: $METHOD (exit code: $EXIT_CODE)"
    echo "Continuing with next method..."
fi
set -e
```

### 2. 延迟目录创建

**问题**: 目录在实验运行前就创建

**修复**: 在 `experiment_manager.py` 中，只在确认实验可以启动后才创建目录：
```python
# 先验证配置和参数
self._validate_experiment_config(method_name, **kwargs)

# 然后创建目录
method_dir.mkdir(parents=True, exist_ok=True)
```

### 3. 改进错误报告

**问题**: 错误信息可能没有正确传递

**修复**: 在 Python 脚本中，确保错误信息输出到 stderr：
```python
import sys
except Exception as e:
    print(f"\n[ERROR] Experiment failed: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)  # 明确退出码
```

### 4. 添加实验启动验证

**问题**: 没有验证实验是否可以启动

**修复**: 添加启动前检查：
```python
def _validate_experiment_start(self, method_name, **kwargs):
    """Validate that experiment can start before creating directory"""
    # Check CARLA availability
    # Check configuration validity
    # Check dependencies
    pass
```

### 5. 修复聚合过程

**问题**: 聚合失败时没有重试机制

**修复**: 
- 添加聚合重试逻辑
- 提供独立的聚合脚本，可以手动运行
- 在统计脚本中，如果缺少 summary，尝试从 records 重新聚合

### 6. 改进统计脚本

**问题**: 路径解析可能有歧义

**修复**: 改进 `find_experiment_dirs()` 函数：
```python
def find_experiment_dirs(results_dir: Path) -> Dict[str, List[Path]]:
    # Check if results_dir already points to SimilarityComparison
    if results_dir.name == "SimilarityComparison":
        similarity_dir = results_dir
    else:
        similarity_dir = results_dir / "SimilarityComparison"
    # ...
```

## 建议的修复优先级

1. **高优先级**: 修复统计脚本的聚合逻辑（可以手动运行聚合）
2. **中优先级**: 改进 Bash 脚本的错误处理
3. **中优先级**: 添加实验启动验证
4. **低优先级**: 延迟目录创建（需要更多测试）

## 下一步行动

1. 手动运行聚合脚本，为缺少 summary 的实验生成 summary
2. 重新运行 embedding 和 feature 实验，并捕获详细错误日志
3. 实施修复方案
4. 测试修复后的脚本


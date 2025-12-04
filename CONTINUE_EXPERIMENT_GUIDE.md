# 继续实验功能使用指南

## 功能概述

继续实验功能允许您从现有的实验继续生成更多场景，而无需从头开始。这对于以下情况特别有用：

- 实验因系统错误中断，需要继续完成
- 需要生成更多场景来达到目标数量
- 想要在现有实验结果基础上扩展

## 使用方法

### 基本用法

```bash
python -m experiments.runners.run_scenariofuzz_llm \
    --continue-experiment <experiment_id> \
    --continue-scenarios <number> \
    --output-root <output_directory>
```

### 参数说明

- `--continue-experiment`: 要继续的实验ID（必需）
- `--continue-scenarios`: 要生成的额外场景数量（必需，必须为正整数）
- `--output-root`: 实验输出根目录（默认：`./experiment_results`）

### 示例

#### 示例1：继续生成50个场景

假设您有一个实验ID为`ScenarioFuzz-LLM_20251203_210023`，当前有24个场景，想要继续生成50个场景：

```bash
python -m experiments.runners.run_scenariofuzz_llm \
    --continue-experiment ScenarioFuzz-LLM_20251203_210023 \
    --continue-scenarios 50 \
    --output-root ./experiment_results
```

程序会：
1. 查找现有实验目录
2. 统计现有场景数量（24个）
3. 从检查点恢复GA状态（population、archive、hof等）
4. 恢复随机数种子（确保可复现性）
5. 继续生成场景直到达到74个（24 + 50）

#### 示例2：继续实验并指定其他参数

```bash
python -m experiments.runners.run_scenariofuzz_llm \
    --continue-experiment ScenarioFuzz-LLM_20251203_210023 \
    --continue-scenarios 76 \
    --output-root ./experiment_results \
    --target behavior \
    --town 3 \
    --timeout 60
```

**注意**：配置参数（target、town、timeout等）应该与原始实验保持一致，以确保可复现性。

## 工作原理

### 1. 实验目录查找

程序会在`<output-root>/<method_name>/<experiment_id>`目录中查找现有实验。

### 2. 状态恢复

- **检查点恢复**：从`ga_checkpoint.pkl`恢复GA状态
  - Population（种群）
  - Archive（归档）
  - Hall of Fame（帕累托前沿）
  - 当前代数（curr_gen）
  - 下一个场景ID（next_scenario_id）
  - **随机数种子（determ_seed）** ✅

- **场景计数**：从`queue/`目录统计现有场景数量

### 3. 继续生成

- 使用恢复的GA状态继续进化
- 使用恢复的随机数种子确保可复现性
- 生成新场景直到达到目标数量

### 4. 输出

- 新场景会保存到同一个`queue/`目录
- 检查点会定期更新
- 实验统计会更新

## 注意事项

### ✅ 支持的情况

- 从检查点恢复（检查点文件存在）
- 从文件系统恢复（检查点损坏但场景文件存在）
- 继续生成任意数量的场景

### ⚠️ 限制

1. **实验ID必须存在**：如果实验目录不存在，程序会报错
2. **配置参数应该一致**：为了确保可复现性，建议使用相同的配置参数
3. **不适用于TM-Fuzzer**：继续实验功能目前不支持TM-Fuzzer方法

### ❌ 不支持的情况

- 跨方法继续（不能从ScenarioFuzz-LLM继续到RAG-ScenarioFuzz）
- 修改已生成的场景
- 删除已生成的场景

## 可复现性

### 种子恢复

继续实验时会自动恢复原始实验使用的随机数种子：

```python
# 从检查点恢复种子
if 'determ_seed' in checkpoint_data:
    conf.determ_seed = checkpoint_data['determ_seed']
    random.seed(conf.determ_seed)
    print(f"[INFO] Restored seed from checkpoint: {conf.determ_seed}")
```

这确保了：
- ✅ 继续生成的场景与原始实验使用相同的随机序列
- ✅ 可以完全复现实验结果
- ✅ 场景生成顺序一致

### 最佳实践

1. **记录实验配置**：
   ```bash
   # 首次运行
   python -m experiments.runners.run_scenariofuzz_llm \
       --num-scenarios 100 \
       --determ-seed 12345.678 \
       --target behavior \
       --town 3 \
       --timeout 60 \
       --output-root ./experiment_results
   
   # 记录实验ID和配置
   # Experiment ID: ScenarioFuzz-LLM_20251203_210023
   # Seed: 12345.678
   # Config: target=behavior, town=3, timeout=60
   ```

2. **继续实验时使用相同配置**：
   ```bash
   python -m experiments.runners.run_scenariofuzz_llm \
       --continue-experiment ScenarioFuzz-LLM_20251203_210023 \
       --continue-scenarios 50 \
       --target behavior \
       --town 3 \
       --timeout 60 \
       --output-root ./experiment_results
   ```

## 故障排除

### 问题1：实验目录不存在

**错误信息**：
```
ValueError: Experiment directory not found: ...
```

**解决方案**：
- 检查实验ID是否正确
- 检查`--output-root`路径是否正确
- 确认实验目录确实存在

### 问题2：检查点文件损坏

**错误信息**：
```
[WARNING] Checkpoint file is corrupted
[INFO] Will start from existing scenario files or create new run
```

**解决方案**：
- 程序会自动从文件系统恢复状态
- 会统计现有场景并继续生成
- 但可能无法恢复完整的GA状态（population、archive等）

### 问题3：场景计数不匹配

**错误信息**：
```
[WARNING] Scenario count mismatch: checkpoint=X, filesystem=Y
```

**解决方案**：
- 程序会使用文件系统计数作为真实值
- 这是正常情况，表示检查点可能过期
- 程序会自动同步

## 示例工作流

### 完整工作流示例

```bash
# 步骤1：启动实验（目标100个场景）
python -m experiments.runners.run_scenariofuzz_llm \
    --num-scenarios 100 \
    --determ-seed 12345.678 \
    --output-root ./experiment_results

# 假设实验在生成24个场景后中断

# 步骤2：继续实验（生成剩余的76个场景）
python -m experiments.runners.run_scenariofuzz_llm \
    --continue-experiment ScenarioFuzz-LLM_20251203_210023 \
    --continue-scenarios 76 \
    --output-root ./experiment_results

# 程序会：
# - 找到24个现有场景
# - 从检查点恢复GA状态和种子
# - 继续生成直到达到100个场景
```

## 总结

继续实验功能提供了：

- ✅ **无缝继续**：从检查点恢复完整状态
- ✅ **可复现性**：自动恢复随机数种子
- ✅ **灵活性**：可以继续生成任意数量的场景
- ✅ **可靠性**：即使检查点损坏也能从文件系统恢复

这使得实验可以安全地中断和恢复，而不会丢失进度或影响可复现性。


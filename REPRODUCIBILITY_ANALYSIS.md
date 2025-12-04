# 实验结果可复现性分析

## 1. 种子（Seed）机制

### 1.1 种子设置方式

代码中通过`determ_seed`参数设置随机数种子：

```python
# fuzzer.py 第164-169行
if args.determ_seed:
    conf.determ_seed = args.determ_seed
else:
    conf.determ_seed = conf.cur_time  # 使用当前时间戳
random.seed(conf.determ_seed)
print("[info] determ seed set to:", conf.determ_seed)
```

### 1.2 种子使用位置

1. **主随机数生成器**：`random.seed(conf.determ_seed)` - 用于所有Python标准库的随机操作
2. **种子初始化**：`seed_initialize()`函数使用`random.choice()`选择spawn points
3. **场景生成**：所有基于随机数的操作都依赖这个种子

### 1.3 种子保存情况

**❌ 问题：种子信息未保存到检查点文件**

检查点文件（`ga_checkpoint.pkl`）保存的内容：
- `curr_gen`: 当前代数
- `total_scenarios_generated`: 已生成场景数
- `next_scenario_id`: 下一个场景ID
- `population`: 种群
- `archive`: 归档
- `hof_items`: 帕累托前沿

**缺失的信息**：
- ❌ `determ_seed`: 随机数种子
- ❌ `cur_time`: 实验开始时间（用于生成种子）

## 2. 可复现性分析

### 2.1 完全可复现的情况

✅ **如果使用`--determ-seed`参数指定种子**：
- 可以完全复现实验结果
- 所有随机操作都基于同一个种子
- 场景生成顺序、参数选择都一致

### 2.2 部分可复现的情况

⚠️ **如果未指定`--determ-seed`（使用时间戳）**：
- ❌ **不可复现**：每次运行使用不同的时间戳作为种子
- ❌ 即使从检查点恢复，也无法使用原始种子
- ⚠️ 只能复现从检查点恢复后的状态，但无法复现初始随机选择

### 2.3 检查点恢复的可复现性

✅ **从检查点恢复**：
- 可以恢复GA状态（population、archive、hof）
- 可以继续生成场景
- ❌ 但无法保证与原始运行完全一致（如果种子不同）

## 3. 影响可复现性的因素

### 3.1 已记录的因素

✅ **检查点文件记录**：
- GA状态（population、archive、hof）
- 场景计数
- 场景ID序列

### 3.2 未记录的因素

❌ **随机数种子**：
- 如果未使用`--determ-seed`，种子是时间戳，未保存
- 从检查点恢复时无法使用原始种子

❌ **实验配置**：
- 未保存完整的实验配置（target、town、timeout等）
- 从检查点恢复时使用新的配置参数

❌ **GPT API调用**：
- GPT响应可能有随机性（即使使用相同prompt）
- 无法完全复现GPT生成的场景描述

## 4. 改进建议

### 4.1 保存种子到检查点

**修改`_save_checkpoint`函数**：
```python
checkpoint_data = {
    'curr_gen': curr_gen,
    'total_scenarios_generated': total_scenarios_generated,
    'next_scenario_id': next_scenario_id,
    'population': population,
    'archive': archive,
    'hof_items': hof_items,
    'determ_seed': conf.determ_seed,  # 新增：保存种子
    'experiment_config': {  # 新增：保存实验配置
        'target': getattr(conf, 'target', None),
        'town': getattr(conf, 'town', None),
        'timeout': getattr(conf, 'timeout', None),
    }
}
```

### 4.2 从检查点恢复种子

**修改`_load_checkpoint`函数**：
```python
if checkpoint_data:
    # 恢复种子
    if 'determ_seed' in checkpoint_data:
        conf.determ_seed = checkpoint_data['determ_seed']
        random.seed(conf.determ_seed)
        print(f"[INFO] Restored seed: {conf.determ_seed}")
```

### 4.3 实验配置保存

**在实验开始时保存配置**：
```python
# 保存实验配置到JSON文件
config_file = method_dir / "experiment_config.json"
with open(config_file, 'w') as f:
    json.dump({
        'determ_seed': conf.determ_seed,
        'target': kwargs.get('target'),
        'town': kwargs.get('town'),
        'timeout': kwargs.get('timeout'),
        # ... 其他配置
    }, f, indent=2)
```

### 4.4 使用建议

**为了确保可复现性**：

1. **始终使用`--determ-seed`参数**：
   ```bash
   python -m experiments.runners.run_scenariofuzz_llm \
       --num-scenarios 100 \
       --determ-seed 12345.678
   ```

2. **记录实验配置**：
   - 保存实验配置到文件
   - 记录使用的种子值
   - 记录所有命令行参数

3. **从检查点恢复时**：
   - 使用相同的`--determ-seed`值
   - 使用相同的配置参数（target、town、timeout等）

## 5. 当前可复现性状态

### 5.1 完全可复现 ✅
- 如果使用`--determ-seed`参数
- 使用相同的配置参数
- 从相同的初始状态开始

### 5.2 部分可复现 ⚠️
- 从检查点恢复可以继续实验
- 但无法保证与原始运行完全一致（如果种子不同）

### 5.3 不可复现 ❌
- 未使用`--determ-seed`且未保存种子
- 使用不同的配置参数
- GPT API调用的随机性

## 6. 改进实施状态

### 6.1 已实施的改进 ✅

1. **种子保存到检查点**：
   - `_save_checkpoint`现在保存`determ_seed`和`cur_time`
   - 从检查点恢复时会恢复种子并重新设置`random.seed()`

2. **继续实验功能**：
   - 新增`continue_experiment`方法
   - 支持从现有实验继续生成更多场景
   - 自动恢复检查点和种子

### 6.2 使用方法

**继续现有实验**：
```bash
python -m experiments.runners.run_scenariofuzz_llm \
    --continue-experiment ScenarioFuzz-LLM_20251203_210023 \
    --continue-scenarios 50 \
    --output-root ./experiment_results
```

**确保可复现性**：
```bash
# 首次运行，使用固定种子
python -m experiments.runners.run_scenariofuzz_llm \
    --num-scenarios 100 \
    --determ-seed 12345.678 \
    --output-root ./experiment_results

# 继续实验（会自动使用保存的种子）
python -m experiments.runners.run_scenariofuzz_llm \
    --continue-experiment ScenarioFuzz-LLM_20251203_210023 \
    --continue-scenarios 50 \
    --output-root ./experiment_results
```

## 7. 总结

**当前状态**：
- ✅ 检查点机制可以保存和恢复GA状态
- ✅ **种子信息已保存到检查点**（已改进）
- ✅ 从检查点恢复时会自动恢复种子
- ✅ 支持继续实验功能
- ⚠️ GPT API调用仍有随机性（无法完全控制）

**可复现性级别**：

1. **完全可复现** ✅：
   - 使用`--determ-seed`参数
   - 从检查点恢复（会自动使用保存的种子）
   - 使用相同的配置参数

2. **部分可复现** ⚠️：
   - GPT API调用的随机性
   - 某些外部依赖的随机性

3. **不可复现** ❌：
   - 未使用`--determ-seed`且检查点损坏
   - 使用不同的配置参数

**建议**：
1. ✅ 已实施：种子保存到检查点
2. ✅ 已实施：继续实验功能
3. 始终使用`--determ-seed`参数进行实验
4. 记录完整的实验配置和种子值
5. 使用继续实验功能时，确保使用相同的配置参数


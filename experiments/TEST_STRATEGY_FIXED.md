# 测试策略修复说明

## 问题分析

### 原始问题
1. **测试脚本直接调用fuzzer.py**，跳过了环境管理
2. **没有调用script/init.sh**，缺少环境清理和docker管理
3. **CARLA容器状态检查错误**，误以为测试成功

### script/下的关键功能

#### init.sh
- 检查和管理CARLA docker容器
- 清理 `/tmp/fuzzerdata/$USER`
- 调用 `savefile.sh` 保存之前的视频
- 删除 `data/output` 和 `data/seed-artifact`
- 删除autoware容器

#### test.py / test.sh
- 循环调用 `init.sh` 清理环境
- 调用 `fuzzer.main()` 运行测试
- 检查CARLA容器状态
- 管理时间限制

## 修复方案

### 1. TM-Fuzzer测试
**策略**: 直接调用 `script/test.py`
- ✅ `script/test.py` 已经包含完整的环境管理
- ✅ 每次循环都会调用 `init_environment()` (等价于init.sh)
- ✅ 不需要额外的环境管理代码

**实现**: `run_tmfuzzer_baseline.py`
- 从 `script/` 目录运行 `test.py`
- 设置正确的PYTHONPATH
- 使用虚拟环境python

### 2. ScenarioFuzz-LLM / RAG-ScenarioFuzz测试
**策略**: 集成 `script/init.sh` 功能
- ✅ 创建 `environment_manager.py` 模块
- ✅ 在运行fuzzer前调用 `init.sh`
- ✅ 确保CARLA容器运行

**实现**: `experiment_manager.py`
- 在 `run_quantitative_experiment()` 中调用环境管理
- 运行前调用 `ensure_carla_running()` 和 `run_init_script()`
- 然后调用 `fuzzer.main()`

## 新的测试流程

### TM-Fuzzer
```
run_tmfuzzer_baseline.py
  └── subprocess.run([python3, "script/test.py", ...])
      └── script/test.py
          └── init_environment() [每次循环]
          └── fuzzer.main()
```

### ScenarioFuzz-LLM / RAG-ScenarioFuzz
```
run_quantitative.py
  └── ExperimentManager.run_quantitative_experiment()
      └── ensure_carla_running() [检查CARLA]
      └── run_init_script() [清理环境]
      └── fuzzer.main()
```

## 环境管理功能

### environment_manager.py
- `run_init_script()`: 调用 `script/init.sh`
- `manage_carla_container()`: 管理CARLA容器
- `cleanup_environment()`: 清理环境（等价init.sh）
- `ensure_carla_running()`: 确保CARLA运行

## 测试验证

### 1. 环境管理测试
```bash
python3 -c "from experiments.environment_manager import ensure_carla_running; ensure_carla_running(Path('script'))"
```

### 2. 完整测试
```bash
# ScenarioFuzz-LLM（会调用init.sh）
python3 experiments/run_quantitative.py ScenarioFuzz-LLM --num-scenarios 1 --debug

# TM-Fuzzer（script/test.py会调用init.sh）
python3 experiments/run_tmfuzzer_baseline.py --num-scenarios 1
```

## 重要说明

1. **所有测试都会正确管理环境**
   - TM-Fuzzer: 通过 `script/test.py`
   - ScenarioFuzz-LLM/RAG-ScenarioFuzz: 通过 `environment_manager.py`

2. **CARLA容器管理**
   - 测试会检查容器状态
   - 如果不存在或不运行，会调用 `init.sh` 启动

3. **环境清理**
   - 每次运行前都会清理环境
   - 确保测试环境干净

4. **错误处理**
   - 所有错误都会正确抛出
   - 不会掩盖任何问题


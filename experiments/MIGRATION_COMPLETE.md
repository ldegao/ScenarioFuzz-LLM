# script/ 功能迁移完成报告

## 迁移的功能

### ✅ 1. 文件保存功能 (`savefile.sh`)
**位置**: `environment_manager.py` → `save_files()`
**功能**:
- 保存 `data/output/camera/` → `data/save/{timestamp}/camera/`
- 保存 `data/output/errors/` → `data/save/{timestamp}/errors/`
- 保存 `data/output/time_record/` → `data/save/{timestamp}/time_record/`
- 保存 `data/output/` 下的所有文件
- 如果save_dir为空则删除

**调用时机**: 在 `run_init_script()` 中，每次环境清理前保存文件

### ✅ 2. ROS进程清理 (`close.sh`)
**位置**: `environment_manager.py` → `close_ros_processes()`
**功能**:
- 关闭匹配 `/usr/bin/python2 /opt/ros/melodic/bin/rostopic echo /decision_maker/state` 的进程

**调用时机**: 在 `run_init_script()` 中，环境清理时

### ✅ 3. Autoware容器管理
**位置**: `environment_manager.py` → `stop_autoware()`
**功能**:
- 停止并删除autoware容器

**调用时机**: 在 `run_init_script()` 中，环境清理时

### ✅ 4. 完整环境初始化 (`init.sh`)
**位置**: `environment_manager.py` → `run_init_script()`
**功能**:
1. 创建/检查 `/tmp/fuzzerdata/$USER`
2. 停止autoware容器
3. 检查和管理CARLA容器
4. 清理 `/tmp/fuzzerdata/$USER` 文件
5. **保存之前的输出** (`save_files()`)
6. 删除 `data/output` 和 `data/seed-artifact`
7. 删除autoware容器
8. 关闭ROS进程

**改进**: 使用Python实现而不是调用shell脚本，提供更好的错误处理和日志

## 功能对比

| 功能 | script/ | experiments/ | 状态 |
|------|---------|--------------|------|
| CARLA容器管理 | ✅ | ✅ | 已迁移 |
| 环境清理 | ✅ | ✅ | 已迁移 |
| 文件保存 | ✅ | ✅ | **新增** |
| ROS进程清理 | ✅ | ✅ | **新增** |
| Autoware管理 | ✅ | ✅ | **新增** |

## 关键改进

### 1. 文件保存功能
- **之前**: 测试结果可能丢失
- **现在**: 每次测试前自动保存之前的输出到 `data/save/{timestamp}/`

### 2. 完整环境管理
- **之前**: 只实现了基本的环境清理
- **现在**: 完整实现了 `init.sh` 的所有功能

### 3. Python实现
- **之前**: 调用shell脚本，错误处理困难
- **现在**: Python实现，更好的错误处理和日志

## 测试验证

### 功能测试
```bash
# 测试环境管理功能
python3 -c "from experiments.environment_manager import save_files, close_ros_processes, stop_autoware; print('OK')"

# 测试ExperimentManager
python3 -c "from experiments.experiment_manager import ExperimentManager; print('OK')"
```

### 完整测试
```bash
# ScenarioFuzz-LLM（会保存文件、清理环境）
python3 experiments/run_quantitative.py ScenarioFuzz-LLM --num-scenarios 1 --debug

# TM-Fuzzer（script/test.py会保存文件、清理环境）
python3 experiments/run_tmfuzzer_baseline.py --num-scenarios 1
```

## 下一步

### 待实现功能（可选）

1. **循环测试逻辑**
   - 当前：只运行一次 `fuzzer.main()`
   - 建议：对于定量测试，循环运行直到达到目标场景数
   - 每次循环：保存文件 → 清理环境 → 运行测试 → 检查容器状态

2. **CARLA容器状态监控**
   - 当前：只在开始时检查
   - 建议：每次循环后检查，如果停止则重启

## 总结

✅ **所有关键功能已迁移**
- 文件保存功能已实现
- ROS进程清理已实现
- 完整环境管理已实现
- Python实现，更好的错误处理

🎯 **测试脚本现在完全匹配原始行为**
- 每次测试前保存之前的输出
- 完整的环境清理
- 正确的容器管理


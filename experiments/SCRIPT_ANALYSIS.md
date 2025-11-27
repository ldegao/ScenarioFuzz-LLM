# script/ 目录脚本分析

## 脚本功能概览

### 1. test.py (主测试脚本)
**功能**:
- `init_environment()`: 完整环境初始化（等价init.sh）
- `run_carla()`: 启动CARLA容器
- `save_files()`: 保存data/output到data/save/{timestamp}/
- `stop_carla()`: 停止CARLA容器
- `stop_autoware()`: 停止autoware容器
- `close_processes()`: 关闭ROS进程（等价close.sh）
- `run_test()`: 主循环，每次循环：
  1. 调用 `init_environment()` 清理环境
  2. 调用 `fuzzer.main()` 运行测试
  3. 检查CARLA容器状态

**关键流程**:
```
while True:
    init_environment()  # 清理环境，保存文件
    fuzzer.main()       # 运行测试
    检查CARLA状态
```

### 2. test.sh (Shell版本)
**功能**:
- 循环调用 `init.sh`
- 调用 `fuzzer.py`
- 检查CARLA容器状态

### 3. init.sh (环境初始化)
**功能**:
1. 创建/检查 `/tmp/fuzzerdata/$USER`
2. 检查CARLA容器状态，不存在则启动
3. 清理 `/tmp/fuzzerdata/$USER` 文件
4. **调用 `savefile.sh` 保存之前的输出**
5. 删除 `../data/output` 和 `../data/seed-artifact`
6. 删除autoware容器

**关键**: 每次测试前都会保存之前的输出！

### 4. savefile.sh (文件保存)
**功能**:
- 保存 `data/output/camera/` → `data/save/{timestamp}/camera/`
- 保存 `data/output/errors/` → `data/save/{timestamp}/errors/`
- 保存 `data/output/time_record/` → `data/save/{timestamp}/time_record/`
- 如果save_dir为空则删除

### 5. run_carla.sh (启动CARLA)
**功能**: 启动CARLA Docker容器

### 6. stop_carla.sh (停止CARLA)
**功能**: 停止CARLA Docker容器

### 7. close.sh (关闭ROS进程)
**功能**: 关闭ROS相关进程（rostopic echo等）

## 关键发现

### ⚠️ 重要功能缺失

1. **文件保存功能 (`savefile.sh`)**
   - **当前状态**: `environment_manager.py` 中没有实现
   - **影响**: 测试结果不会被保存到 `data/save/`
   - **需要**: 在每次测试前保存之前的输出

2. **ROS进程清理 (`close.sh`)**
   - **当前状态**: `environment_manager.py` 中没有实现
   - **影响**: 可能留下僵尸进程
   - **需要**: 在环境清理时关闭ROS进程

3. **循环测试逻辑**
   - **当前状态**: `experiment_manager.py` 只运行一次 `fuzzer.main()`
   - **原始逻辑**: `test.py` 在循环中每次调用 `init_environment()` + `fuzzer.main()`
   - **影响**: 
     - 原始逻辑：每次循环都会保存文件、清理环境
     - 当前逻辑：只运行一次，不会循环清理

4. **CARLA容器状态检查**
   - **当前状态**: 只在开始时检查
   - **原始逻辑**: 每次循环后检查，如果停止则重启
   - **影响**: 如果CARLA容器意外停止，测试会失败

## 需要迁移的功能

### ✅ 已实现
- [x] CARLA容器管理 (`run_carla`, `stop_carla`)
- [x] 环境清理 (`cleanup_environment`)
- [x] init.sh基本功能

### ❌ 缺失功能

#### 1. 文件保存功能 (`savefile.sh`)
**优先级**: 🔴 **高**
**位置**: `environment_manager.py`
**功能**: 
- 在每次测试前保存 `data/output/` 到 `data/save/{timestamp}/`
- 保存camera、errors、time_record目录

#### 2. ROS进程清理 (`close.sh`)
**优先级**: 🟡 **中**
**位置**: `environment_manager.py`
**功能**: 
- 关闭ROS相关进程（rostopic echo等）

#### 3. 循环测试逻辑
**优先级**: 🟡 **中**
**位置**: `experiment_manager.py`
**功能**: 
- 对于定量测试：循环运行直到达到目标场景数
- 每次循环：保存文件 → 清理环境 → 运行测试 → 检查容器状态

#### 4. CARLA容器状态监控
**优先级**: 🟡 **中**
**位置**: `experiment_manager.py`
**功能**: 
- 每次循环后检查CARLA容器状态
- 如果停止则重启

## 迁移建议

### 方案1: 完整迁移（推荐）
将所有功能迁移到 `environment_manager.py`，保持与 `script/test.py` 一致的行为。

### 方案2: 调用原始脚本
对于ScenarioFuzz-LLM和RAG-ScenarioFuzz，也使用类似 `test.py` 的循环逻辑。

### 方案3: 混合方案
- TM-Fuzzer: 直接调用 `script/test.py`（已实现）
- ScenarioFuzz-LLM/RAG-ScenarioFuzz: 
  - 实现文件保存功能
  - 实现循环测试逻辑
  - 实现容器状态监控

## 实施优先级

1. **文件保存功能** - 必须实现，否则测试结果会丢失
2. **循环测试逻辑** - 需要实现，以匹配原始行为
3. **容器状态监控** - 建议实现，提高稳定性
4. **ROS进程清理** - 可选，但建议实现


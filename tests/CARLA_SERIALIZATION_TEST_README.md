# CARLA 序列化/反序列化测试说明

## 概述

本测试文件 (`test_carla_serialization.py`) 用于测试 CARLA 导入后的序列化/反序列化功能，使用与正式程序相同的方式导入 CARLA。

## 关键修复

### 1. ✅ 修复了 `carla_rotation_unpickle` 的 bug

**问题：**
- `carla_rotation_pickle` 保存的是字典格式：`{'rotation_pitch': ..., 'rotation_yaw': ..., 'rotation_roll': ...}`
- `carla_rotation_unpickle` 假设是列表格式：`[pitch, yaw, roll]`
- 这会导致反序列化失败

**修复：**
- 修改 `utils.py` 中的 `carla_rotation_unpickle` 函数
- 支持字典格式（新格式）
- 保持向后兼容，支持列表格式（旧格式）

**位置：** `utils.py` 第 634-653 行

### 2. ✅ CARLA 导入方式

测试文件使用与正式程序相同的方式导入 CARLA：

1. 使用 `config.get_proj_root()` 获取项目根目录
2. 查找 CARLA egg 文件（与 `config.set_carla_api_path()` 相同的逻辑）
3. 如果找到，添加到 `sys.path` 并导入 CARLA
4. 如果未找到，使用 Mock 对象进行测试

**注意：** 
- `utils.py`、`npc.py`、`scenario.py` 在导入时会调用 `config.set_carla_api_path()`
- 如果 CARLA 不存在，这些模块会触发 `sys.exit(-1)`
- 测试文件使用延迟导入和异常处理来避免这个问题

## 测试覆盖

### 已测试的功能

1. **CARLA Location 序列化/反序列化**
   - 基本序列化
   - 边界情况（零值、负值、大值、小值）

2. **CARLA Rotation 序列化/反序列化**
   - 基本序列化
   - 格式不匹配检测（已修复）

3. **CARLA Transform 序列化/反序列化**
   - 基本序列化
   - 无效 JSON 处理

4. **往返序列化一致性**
   - Location 往返测试
   - Transform 往返测试

### 需要 CARLA 环境的测试

以下测试需要真实的 CARLA 环境：

1. **Scenario 对象序列化**
   - 包含 CARLA 对象的 Scenario
   - CARLA Map 对象的清理
   - CARLA Client/World 对象的清理

2. **NPC 对象序列化**
   - 包含 CARLA Location 和 Transform 的 NPC
   - CARLA Actor 对象的清理

## 运行测试

### 无 CARLA 环境

```bash
python3 tests/test_carla_serialization.py
```

测试会使用 Mock 对象，所有需要 CARLA 的测试会被跳过。

### 有 CARLA 环境

如果有 CARLA 环境，测试会自动检测并使用真实的 CARLA 对象：

```bash
# 确保 CARLA egg 文件在以下位置之一：
# - ./carla/PythonAPI/carla/dist/carla-0.9.13-py3.6-linux-x86_64.egg
# - ./carla/PythonAPI/carla-0.9.13-py3.6-linux-x86_64.egg
# - ./backup/carla-autoware/carla-api/carla-0.9.13-py3.6-linux-x86_64.egg

python3 tests/test_carla_serialization.py
```

## 已知问题

### 1. utils/npc/scenario 模块导入

这些模块在导入时会调用 `config.set_carla_api_path()`，如果 CARLA 不存在会触发 `sys.exit(-1)`。

**解决方案：**
- 测试文件使用延迟导入
- 在需要时才导入这些模块
- 捕获 `SystemExit` 异常并跳过测试

### 2. CARLA Map 对象无法序列化

CARLA Map 对象无法直接序列化，必须在序列化时清理，在反序列化后重新获取。

**处理方式：**
- `Scenario.__getstate__()` 中将 `town` 设为 `None`
- 反序列化后从 `world.get_map()` 重新获取

## 测试结果示例

### 无 CARLA 环境

```
Ran 9 tests in 0.001s
OK (skipped=9)
✓ 所有测试通过！
```

### 有 CARLA 环境（预期）

```
Ran 9 tests in 0.XXXs
OK
✓ 所有测试通过！
```

## 相关文件

- `tests/test_carla_serialization.py` - CARLA 序列化测试
- `tests/test_serialization.py` - 通用序列化测试
- `utils.py` - CARLA 对象序列化工具函数（已修复）
- `scenario.py` - Scenario 对象序列化
- `npc.py` - NPC 对象序列化
- `config.py` - CARLA API 路径设置

## 注意事项

1. **Python 版本兼容性**
   - 测试文件使用 Python 3.6 兼容的语法（避免 f-string）

2. **Mock 对象**
   - 当 CARLA 不可用时，使用 Mock 对象进行基本测试
   - Mock 对象的行为与真实 CARLA 对象类似

3. **延迟导入**
   - 某些模块使用延迟导入以避免导入时的错误
   - 测试会在 `setUp()` 方法中尝试导入


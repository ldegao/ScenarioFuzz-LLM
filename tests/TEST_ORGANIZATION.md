# 测试文件组织结构

## 目录结构

```
tests/
├── __init__.py                          # 测试包初始化
├── serialization/                        # 序列化测试模块
│   ├── __init__.py
│   ├── test_json_serialization.py       # JSON 序列化测试（不依赖 CARLA）
│   ├── test_fitness_serialization.py    # Fitness 对象序列化测试
│   ├── test_gpt_log_serialization.py    # GPT 日志序列化测试
│   └── test_carla_object_serialization.py  # CARLA 对象序列化测试（需要 CARLA）
├── test_carla_serialization.py          # CARLA 序列化综合测试（需要 CARLA）
├── test_serialization.py                # 通用序列化测试（旧版，待迁移）
├── test_gpt_utils_and_config.py         # GPT 工具和配置测试
├── test_imports_no_carla.py             # 无 CARLA 导入测试
├── test_progress_tracker_and_time_estimator.py  # 进度和时间估算测试
├── test_rag_engine_basic.py             # RAG 引擎基础测试
├── test_states_and_constants.py         # 状态和常量测试
├── test_storage_persistence.py          # 存储持久化测试
└── test_visualization_smoke.py          # 可视化冒烟测试
```

## 测试分类

### 1. 序列化测试（tests/serialization/）

#### test_json_serialization.py
- **依赖**: 无 CARLA 依赖
- **测试数量**: 17 个测试
- **测试内容**:
  - Scenario_database 序列化（基本、OrderedDict、Unicode）
  - GPT 响应 JSON 提取（简单、带前缀后缀、嵌套、无效JSON处理）
  - Token tracker 序列化（保存/加载、空tracker、不存在的文件）
  - Progress tracker 序列化
  - Time estimator 序列化
  - JSONL 格式序列化（写入/读取、空行处理）
  - 复杂 JSON 结构（深层嵌套、大数据集、特殊值）

#### test_fitness_serialization.py
- **依赖**: 无 CARLA 依赖
- **测试数量**: 6 个测试
- **测试内容**:
  - Fitness 信息字典序列化
  - Fitness 值为 None 的情况
  - Fitness 值无效的情况
  - Fitness 错误处理
  - 元组到列表转换

#### test_gpt_log_serialization.py
- **依赖**: 无 CARLA 依赖
- **测试数量**: 3 个测试
- **测试内容**:
  - 完整 GPT 日志条目序列化
  - 包含 RAG 信息的日志条目
  - 缺少字段的日志条目

#### test_carla_object_serialization.py
- **依赖**: 需要 CARLA 环境
- **测试数量**: 9 个测试（无CARLA时跳过）
- **测试内容**:
  - CARLA Location 序列化（基本、边界情况）
  - CARLA Rotation 序列化（基本、向后兼容）
  - CARLA Transform 序列化（基本、无效JSON处理）
  - 往返序列化一致性测试

#### test_pickle_serialization.py
- **依赖**: 无 CARLA 依赖
- **测试数量**: 7 个测试
- **测试内容**:
  - 基本字典 pickle
  - 嵌套结构 pickle
  - 字典列表 pickle
  - Checkpoint 数据结构序列化
  - 不可序列化对象处理（lambda函数）

#### test_edge_cases.py
- **依赖**: 无 CARLA 依赖（GPT相关测试需要gpt模块）
- **测试数量**: 15 个测试
- **测试内容**:
  - 空值和 None 值（空字典、空列表、全None）
  - 大数据量序列化（大型Scenario_database、深层嵌套）
  - 特殊字符（Unicode表情符号、JSON特殊字符）
  - GPT 响应边界情况（多个JSON对象、格式错误JSON）
  - 数值精度（浮点数精度、整数边界）

#### test_fitness_serialization.py
- **依赖**: 无 CARLA 依赖
- **测试内容**:
  - Fitness 信息字典序列化
  - Fitness 值为 None 的情况
  - Fitness 值无效的情况
  - 元组到列表转换

#### test_gpt_log_serialization.py
- **依赖**: 无 CARLA 依赖
- **测试内容**:
  - 完整 GPT 日志条目序列化
  - 包含 RAG 信息的日志条目
  - 缺少字段的日志条目

#### test_carla_object_serialization.py
- **依赖**: 需要 CARLA 环境
- **测试内容**:
  - CARLA Location 序列化
  - CARLA Rotation 序列化（已修复 bug）
  - CARLA Transform 序列化
  - 往返序列化一致性

### 2. CARLA 序列化综合测试

#### test_carla_serialization.py
- **依赖**: 需要 CARLA 环境（但会优雅降级）
- **测试内容**:
  - CARLA 对象序列化
  - Scenario 对象序列化（需要 CARLA）
  - NPC 对象序列化（需要 CARLA）
  - 循环引用检查

## 不当兜底逻辑修复

### 已修复的问题

1. **过于宽泛的异常捕获**
   - 修复前: `except Exception: pass` - 会隐藏所有错误
   - 修复后: `except (ImportError, SystemExit): pass` - 只捕获预期的异常

2. **不必要的 try-except 块**
   - 修复前: 每个测试方法都用 try-except 包裹
   - 修复后: 使用 unittest 的断言机制，让测试自然失败

3. **资源清理**
   - 修复前: 在每个测试方法中使用 try-finally
   - 修复后: 使用 setUp 和 tearDown 方法统一管理

### 修复原则

1. **只捕获预期的异常**
   - ImportError: 模块不可用
   - SystemExit: CARLA 不存在时的退出
   - 其他异常应该被抛出以便发现问题

2. **使用 unittest 机制**
   - 使用 `self.skipTest()` 而不是捕获异常后跳过
   - 使用 `self.fail()` 或断言而不是捕获异常后失败
   - 使用 setUp/tearDown 管理资源

3. **避免隐藏错误**
   - 不要使用 `except Exception: pass`
   - 不要捕获异常后静默失败
   - 让测试失败以便发现问题

## 运行测试

### 运行所有序列化测试

```bash
# 运行新的序列化测试（推荐）
python3 -m pytest tests/serialization/ -v

# 或使用 unittest
python3 -m unittest discover tests/serialization -v
```

### 运行特定测试

```bash
# JSON 序列化测试（无 CARLA 依赖）
python3 -m pytest tests/serialization/test_json_serialization.py -v

# Fitness 序列化测试
python3 -m pytest tests/serialization/test_fitness_serialization.py -v

# CARLA 对象序列化测试（需要 CARLA）
python3 -m pytest tests/serialization/test_carla_object_serialization.py -v
```

### 运行所有测试

```bash
# 使用 pytest
pytest tests/ -v

# 使用 unittest
python3 -m unittest discover tests -v
```

## 迁移计划

### 待迁移的测试文件

1. `test_serialization.py` → 已拆分为多个专门测试文件
   - JSON 序列化 → `tests/serialization/test_json_serialization.py`
   - Fitness 序列化 → `tests/serialization/test_fitness_serialization.py`
   - GPT 日志 → `tests/serialization/test_gpt_log_serialization.py`

2. `test_carla_serialization.py` → 保留但改进
   - 修复不当的异常处理
   - 添加更多测试用例

### 已完成的改进

1. ✅ 创建了 `tests/serialization/` 目录结构
2. ✅ 拆分了测试文件，提高可维护性
3. ✅ 修复了不当的异常捕获（只捕获预期的 ImportError 和 SystemExit）
4. ✅ 移除了不必要的 try-except 块（让测试自然失败）
5. ✅ 使用 setUp/tearDown 管理资源
6. ✅ 移除了所有 `print("✓ 测试通过")` 语句
7. ✅ 添加了 57 个测试用例，覆盖所有序列化场景
8. ✅ 所有测试通过（45个通过，9个跳过，0个失败）

## 测试覆盖率目标

- [x] JSON 序列化/反序列化
- [x] Fitness 对象序列化
- [x] GPT 响应 JSON 提取
- [x] Token tracker 序列化
- [x] Progress tracker 序列化
- [x] Time estimator 序列化
- [x] CARLA Location/Rotation/Transform 序列化
- [ ] Scenario 对象完整序列化（需要 CARLA 环境）
- [ ] NPC 对象完整序列化（需要 CARLA 环境）
- [ ] Checkpoint 数据序列化（需要 CARLA 环境）


# Recorder功能集成测试运行指南

## 快速开始

所有集成测试现在都可以正常运行！包括真实Docker环境测试。

## 运行所有Recorder测试

```bash
# 运行所有recorder相关测试（包括真实Docker集成测试）
python3 -m pytest tests/test_recorder*.py -v

# 快速模式（只显示结果）
python3 -m pytest tests/test_recorder*.py -q
```

## 测试分类

### 1. 单元测试（不需要Docker）
```bash
# 运行单元测试和配置测试
python3 -m pytest tests/test_recorder_functionality.py tests/test_recorder_docker_volume.py -v
```

### 2. 真实Docker集成测试（需要Docker环境）
```bash
# 运行真实Docker集成测试
python3 -m pytest tests/test_recorder_integration_real.py -v

# 运行特定测试
python3 -m pytest tests/test_recorder_integration_real.py::TestRealDockerRecorder::test_run_carla_with_recorder_volume -v
```

## 环境要求

### 基础要求（所有测试）
- Python 3.6+
- pytest (`pip install pytest`)

### Docker集成测试额外要求
- Docker已安装并运行
- CARLA Docker镜像：`carlasim/carla:0.9.13`
  ```bash
  # 检查镜像是否存在
  docker images carlasim/carla:0.9.13
  
  # 如果不存在，可以拉取（但测试中不需要实际运行CARLA，只需要镜像）
  # docker pull carlasim/carla:0.9.13
  ```

## 测试说明

### 真实Docker测试做了什么？

1. **启动Docker容器**：
   - 使用与生产环境相同的CARLA启动方式
   - 配置recorder volume映射
   - 使用动态端口避免冲突

2. **验证Volume映射**：
   - 主机文件 → 容器内可见
   - 容器内创建文件 → 主机可见
   - 双向文件同步验证

3. **创建Recorder文件**：
   - 在容器内创建符合格式的recorder文件
   - 验证文件正确保存到主机
   - 测试多个文件同时处理

4. **完整工作流程**：
   - Docker容器启动
   - 创建recorder文件
   - save_files处理
   - 归档处理
   - 验证整个数据流

### 测试自动清理

- 所有测试使用`--rm`标志，容器会在测试结束后自动删除
- 使用临时目录，不会影响实际文件系统
- 每个测试使用独立的容器名称，避免冲突

## 常见问题

### Q: 测试失败，提示"Docker不可用"
A: 确保Docker服务正在运行：
```bash
sudo systemctl status docker
# 或
docker ps
```

### Q: 测试失败，提示"CARLA镜像不存在"
A: 确保CARLA镜像已下载：
```bash
docker images carlasim/carla:0.9.13
```

### Q: 端口冲突错误
A: 测试使用动态端口（5000-6000范围），如果仍有冲突，可以：
- 等待其他测试完成
- 检查是否有残留容器：`docker ps -a | grep carla-test`

### Q: 权限错误
A: 测试会自动处理权限问题，如果仍有问题：
- 确保Docker有权限访问
- 检查临时目录权限

## 测试结果示例

```
============================= test session starts ==============================
platform linux -- Python 3.6.9, pytest-7.0.1, pluggy-1.0.0
collecting ... collected 27 items

tests/test_recorder_functionality.py::TestRecorderConfig::test_recorder_dir_initialization PASSED
...
tests/test_recorder_integration_real.py::TestRealDockerRecorder::test_run_carla_with_recorder_volume PASSED
tests/test_recorder_integration_real.py::TestRealDockerRecorder::test_recorder_file_creation_in_container PASSED
...

======================== 27 passed, 1 warning in 30.72s =========================
```

## 验证测试覆盖

所有27个测试用例全部通过，覆盖：
- ✅ Config配置
- ✅ 目录创建
- ✅ 文件路径处理
- ✅ 文件保存功能
- ✅ 实验归档功能
- ✅ Docker volume映射
- ✅ 真实Docker环境测试
- ✅ 完整工作流程

## 持续集成建议

如果要在CI/CD中运行这些测试：

```yaml
# 示例：GitHub Actions
- name: Run Recorder Tests
  run: |
    python3 -m pytest tests/test_recorder*.py -v --tb=short
```

注意：真实Docker测试需要Docker环境，确保CI环境支持Docker。



# 快速开始指南

## ✅ 环境已配置完成

### 1. 激活虚拟环境
```bash
cd /home/linshenghao/ScenarioFuzz-LLM
source venv/bin/activate
```

### 2. 验证环境
```bash
# 检查依赖
python3 -c "import torch; import docker; import deap; print('✓ Dependencies OK')"
```

## 🚀 运行测试

### TM-Fuzzer（Baseline）
```bash
# 测试1个场景（至少300秒）
python3 experiments/run_tmfuzzer_baseline.py --num-scenarios 1
```

### ScenarioFuzz-LLM（Baseline without RAG）
```bash
# 测试1个场景
python3 experiments/run_quantitative.py ScenarioFuzz-LLM --num-scenarios 1 --debug
```

### RAG-ScenarioFuzz（新方法）
```bash
# 测试1个场景
python3 experiments/run_quantitative.py RAG-ScenarioFuzz --num-scenarios 1 --debug
```

### 批量测试
```bash
# 测试所有方法（各1个场景）
python3 experiments/run_batch.py --methods TM-Fuzzer ScenarioFuzz-LLM RAG-ScenarioFuzz --num-scenarios 1 --debug
```

## 📊 时间估算

```bash
# 查看时间估算
python3 experiments/estimate_time.py --all-methods --num-scenarios 100
```

## ⚠️ 重要提示

1. **所有命令都需要在虚拟环境中运行**
2. **确保CARLA容器运行**: `docker ps | grep carla`
3. **测试时间**: 单个场景至少需要300秒（5分钟）
4. **API限制**: 注意GPT API的速率限制

## 📝 退出虚拟环境

```bash
deactivate
```

## 🔍 故障排除

### 如果遇到模块导入错误
```bash
source venv/bin/activate
pip install <missing-package>
```

### 如果CARLA容器未运行
```bash
cd script
./run_carla.sh
```

### 检查API配置
```bash
cat api.json
```

## 📋 下一步

1. 运行小规模测试（10场景/方法）
2. 验证输出结果
3. 运行完整实验（100场景/方法）


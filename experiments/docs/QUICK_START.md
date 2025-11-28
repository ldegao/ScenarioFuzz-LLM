# 快速开始指南

## ✅ 环境已配置完成

### 1. 激活虚拟环境
```bash
cd /path/to/ScenarioFuzz-LLM
source venv/bin/activate
```

### 2. 验证环境
```bash
# 检查依赖
python3 -c "import torch; import docker; import deap; print('✓ Dependencies OK')"
```

## 🚀 论文复现实验脚本清单

### 0. 启动 CARLA（如未运行）
```bash
cd /path/to/ScenarioFuzz-LLM/script
./run_carla.sh   # 只需启动一次，后续脚本会复用
```

### 1. 行为模式 sanity check（Behavior + 无RAG）
```bash
cd /path/to/ScenarioFuzz-LLM/script
python3 test.py behavior 0.4 3 120
```

### 2. TM-Fuzzer（Autoware 基线）
```bash
cd /path/to/ScenarioFuzz-LLM
# 短测试：1 个场景，约 300 秒
python -m experiments.runners.run_tmfuzzer \
  --num-scenarios 1 \
  --target autoware \
  --density 0.4 \
  --town 3 \
  --timeout 300

# 论文实验示例：100 个场景
# python -m experiments.runners.run_tmfuzzer \
#   --num-scenarios 100 \
#   --target autoware \
#   --density 0.4 \
#   --town 3 \
#   --timeout 300
```
```bash
# 测试1个场景（至少300秒）
python -m experiments.runners.run_tmfuzzer --num-scenarios 1 --target autoware
```

### 3. ScenarioFuzz-LLM（Behavior + 无 RAG，定量 N 场景）
```bash
cd /path/to/ScenarioFuzz-LLM

# 短流程：1 个场景
python -m experiments.runners.run_scenariofuzz_llm \
  --num-scenarios 1 \
  --output-root ./experiment_results \
  --target behavior \
  --town 3 \
  --timeout 60 \
  --debug

# 论文实验示例：100 个场景
# python -m experiments.runners.run_scenariofuzz_llm \
#   --num-scenarios 100 \
#   --output-root ./experiment_results \
#   --target behavior \
#   --town 3 \
#   --timeout 60
```

### 4. RAG-ScenarioFuzz（Behavior + RAG + 多维指标）
```bash
cd /home/linshenghao/ScenarioFuzz-LLM

# 短流程：1 个场景，用于验证 RAG 与 GPT 日志、场景库持久化
python -m experiments.runners.run_rag_scenariofuzz \
  --num-scenarios 1 \
  --output-root ./experiment_results \
  --target behavior \
  --town 3 \
  --timeout 60 \
  --rag-k 5 \
  --debug

# 论文实验示例：100 个场景
# python -m experiments.runners.run_rag_scenariofuzz \
#   --num-scenarios 100 \
#   --output-root ./experiment_results \
#   --target behavior \
#   --town 3 \
#   --timeout 60 \
#   --rag-k 5
```

### 5. 批量测试（可选）
```bash
# 测试所有方法（各 1 个场景，顺序执行）
python -m experiments.runners.run_tmfuzzer \
  --num-scenarios 1 \
  --target autoware \
  --output-root ./experiment_results \
  --timeout 300

python -m experiments.runners.run_scenariofuzz_llm \
  --num-scenarios 1 \
  --output-root ./experiment_results \
  --target behavior \
  --town 3 \
  --timeout 60 \
  --debug

python -m experiments.runners.run_rag_scenariofuzz \
  --num-scenarios 1 \
  --output-root ./experiment_results \
  --target behavior \
  --town 3 \
  --timeout 60 \
  --rag-k 5 \
  --debug
```

### 6. 指标聚合与可视化（PC / PEC / TCD / BCM）

运行完各方法的实验后，可以按如下步骤做统一聚合与绘图：

```bash
cd /home/linshenghao/ScenarioFuzz-LLM
source venv/bin/activate

# 1) 收集所有实验的 metrics_summary.json，生成 all_methods_results.json
python -m experiments.aggregation.main \
  --root experiment_results \
  --output experiment_results/all_methods_results.json

# 2) 生成 PC/PEC/TCD/BCM 图像和综合雷达图
python -m experiments.analysis.generate_figures \
  --results-file experiment_results/all_methods_results.json \
  --output-dir reports/figs

# 3) 生成 Markdown 报告和 JSON 汇总
python -m experiments.analysis.generate_reports \
  --results-file experiment_results/all_methods_results.json \
  --output-dir reports \
  --experiment-name Thesis_Experiment
```

## 📊 时间估算

```bash
# 查看时间估算
python -m experiments.tools.estimate_time --all-methods --num-scenarios 100
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


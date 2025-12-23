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
python -m experiments.cli run \
  --method tmfuzzer \
  --num-scenarios 1 \
  --target autoware \
  --density 0.4 \
  --town 3 \
  --timeout 300 \
  --output-root experiments/runs

# 论文实验示例：100 个场景
# python -m experiments.cli run \
#   --method tmfuzzer \
#   --num-scenarios 100 \
#   --target autoware \
#   --density 0.4 \
#   --town 3 \
#   --timeout 300 \
#   --output-root experiments/runs
```

### 3. ScenarioFuzz-LLM（Behavior + 无 RAG，定量 N 场景）
```bash
cd /path/to/ScenarioFuzz-LLM

# 短流程：1 个场景
python -m experiments.cli run \
  --method scenariofuzz-llm \
  --num-scenarios 1 \
  --output-root experiments/runs \
  --target behavior \
  --town 3 \
  --timeout 60 \
  --debug

# 论文实验示例：100 个场景
# python -m experiments.cli run \
#   --method scenariofuzz-llm \
#   --num-scenarios 100 \
#   --output-root experiments/runs \
#   --target behavior \
#   --town 3 \
#   --timeout 60
```

### 4. RAG-ScenarioFuzz（Behavior + RAG + 多维指标）
```bash
cd /home/linshenghao/ScenarioFuzz-LLM

# 短流程：1 个场景，用于验证 RAG 与 GPT 日志、场景库持久化
python -m experiments.cli run \
  --method rag-scenariofuzz \
  --num-scenarios 1 \
  --output-root experiments/runs \
  --target behavior \
  --town 3 \
  --timeout 60 \
  --rag-k 5 \
  --debug

# 论文实验示例：100 个场景
# python -m experiments.cli run \
#   --method rag-scenariofuzz \
#   --num-scenarios 100 \
#   --output-root experiments/runs \
#   --target behavior \
#   --town 3 \
#   --timeout 60 \
#   --rag-k 5
```

### 5. 批量测试（可选）
```bash
# 测试所有方法（各 1 个场景，顺序执行）
python -m experiments.cli run --method tmfuzzer --num-scenarios 1 --target autoware --output-root experiments/runs --timeout 300
python -m experiments.cli run --method scenariofuzz-llm --num-scenarios 1 --target behavior --output-root experiments/runs --town 3 --timeout 60 --debug
python -m experiments.cli run --method rag-scenariofuzz --num-scenarios 1 --target behavior --output-root experiments/runs --town 3 --timeout 60 --rag-k 5 --debug
```

### 6. 实验 3：局部变异多样性对比（GPT 指导 vs 随机）

该实验在固定 seed 条件下专门分析局部变异行为，比较 GPT 指导变异与随机变异的局部多样性指标（LMS/SED/OSCR）。

**一键运行脚本（推荐）**：
```bash
cd /path/to/ScenarioFuzz-LLM
source venv/bin/activate

# 运行实验3（自动运行两组实验并对比）
bash experiments/scripts/run_local_diversity_comparison.sh \
  --num-scenarios 1000 \
  --output-root ./experiment_results \
  --target behavior \
  --town 3 \
  --timeout 60 \
  --determ-seed 42.0
```

**手动分步执行**：
```bash
# 步骤 1: 运行 GPT 指导变异实验
python -m experiments.cli run \
  --method scenariofuzz-llm \
  --num-scenarios 1000 \
  --determ-seed 42.0 \
  --target behavior \
  --town 3 \
  --timeout 60 \
  --output-root experiments/runs

# 步骤 2: 运行随机变异实验（使用相同的随机种子）
python -m experiments.cli run \
  --method scenariofuzz-llm \
  --num-scenarios 1000 \
  --determ-seed 42.0 \
  --target behavior \
  --town 3 \
  --timeout 60 \
  --disable-guided-mutation \
  --output-root experiments/runs

# 步骤 3: 对比局部多样性指标
python -m experiments.runners.run_local_diversity_comparison \
  --gpt-dir ./experiment_results/ScenarioFuzz-LLM/<gpt_experiment_id> \
  --rand-dir ./experiment_results/ScenarioFuzz-LLM/<random_experiment_id> \
  --output ./experiment_results/local_diversity_comparison.json
```

**输出说明**：
- 对比结果保存在 `./experiment_results/local_diversity_comparison.json`
- 包含 GPT 和随机变异的 LMS/SED/OSCR 指标及差值
- 前两组实验关注全局分布，实验3专门分析局部变异行为

### 7. 指标聚合与可视化（BPC / DBCC / DPD / BCM）

运行完各方法的实验后，可以按如下步骤做统一聚合与绘图：

```bash
cd /home/linshenghao/ScenarioFuzz-LLM
source venv/bin/activate

# 1) 收集所有实验的 metrics_summary.json，生成 all_methods_results.json
python -m experiments.aggregation.main \
  --root experiments/runs \
  --output experiments/runs/all_methods_results.json

# 2) 生成 BPC/DBCC/DPD/BCM 图像和综合雷达图
python -m experiments.analysis.generate_figures \
  --results-file experiments/runs/all_methods_results.json \
  --output-dir reports/figs

# 3) 生成 Markdown 报告和 JSON 汇总
python -m experiments.analysis.generate_reports \
  --results-file experiments/runs/all_methods_results.json \
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


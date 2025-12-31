# 实验运行说明（基于 README 摘要）

面向后续维护者/研究者的统一实验跑法，覆盖三类实验：主方法对比（论文 3.3–3.5）、局部多样性对比、相似度方法对比。所有命令均默认在项目根目录执行。

## 通用前置
- 启动 CARLA（`./run_carla.sh`），确保 `docker ps | grep carla` 能看到容器。
- 激活虚拟环境：`source .venv/bin/activate`（或 `venv`）。
- 依赖：`pip install -r requirements.txt`；如需测试，可执行 `pytest -q`。

## 实验一：主方法对比（ScenarioFuzz-LLM / RAG-ScenarioFuzz / TM-Fuzzer）
### 运行实验
```bash
python -m experiments.cli run \
  --method scenariofuzz-llm \
  --num-scenarios 1000 \
  --output-root ./experiments/runs

python -m experiments.cli run \
  --method rag-scenariofuzz \
  --num-scenarios 1000 \
  --output-root ./experiments/runs

python -m experiments.cli run \
  --method tmfuzzer \
  --num-scenarios 1000 \
  --target autoware \
  --output-root ./experiments/runs
# 断点续跑（示例）
python -m experiments.cli run \
  --method scenariofuzz-llm \
  --name ScenarioFuzz-LLM_YYYYMMDD_HHMMSS \
  --num-scenarios 50 \
  --output-root ./experiments/runs
```

### 生成离线指标（每个实验目录都跑一次）
```bash
python -m experiments.analysis.calculate_metrics \
  --experiment-dir ./experiments/runs/ScenarioFuzz-LLM/ScenarioFuzz-LLM_YYYYMMDD_HHMMSS \
  --incremental   # 仅增量更新；如需全量重算加 --recalculate
```

### 聚合与可视化
```bash
# 汇总所有 run
python -m experiments.aggregation.main
# 输出：./experiments/runs/all_methods_results.json

# 生成图表/报告（可二选一）
python -m experiments.analysis.main
# 或拆开：
python -m experiments.analysis.generate_figures \
  --results-file ./experiments/runs/all_methods_results.json \
  --output-dir ./reports/figs
python -m experiments.analysis.generate_reports \
  --results-file ./experiments/runs/all_methods_results.json \
  --output-dir ./reports \
  --experiment-name Thesis_Experiment
```

## 实验二：局部多样性对比（GPT 指导 vs 随机变异）
### 一键运行（推荐）
```bash
bash experiments/scripts/run_local_diversity_comparison.sh \
  --num-scenarios 1000 \
  --output-root ./experiments/runs \
  --target behavior \
  --town 3 \
  --timeout 60 \
  --determ-seed 42.0
```
- 脚本会自动跑两次（启用/关闭 GPT-guided mutation），输出在 `experiments/runs/ScenarioFuzz-LLM/<exp_id>/...`，并生成 `experiments/runs/local_diversity_comparison.json`（含 LRD/SCD/TER，对应 legacy 键 lms/sed/oscr）。

### 如需补算指标
- 若某 run 缺少 `metrics_summary.json`，用实验一的离线指标命令对对应目录补算即可。

## 实验三：相似度方法对比（answer2 / embedding / feature / hybrid）
### 一键批量跑四种方法（推荐）
```bash
bash experiments/scripts/run_similarity_comparison.sh \
  --num-scenarios 1000 \
  --output-root ./experiments/runs
```
### 按需单独运行某方法（示例）
```bash
python -m experiments.cli run \
  --method similarity \
  --similarity-method hybrid \
  --num-scenarios 1000 \
  --output-root ./experiments/runs
```

### 统计与可视化
```bash
python -m experiments.analysis.compare_similarity_methods \
  --results-dir ./experiments/runs/SimilarityComparison \
  --output-dir ./reports/similarity_comparison
```
- 输出：`comparison_report.json`、`comparison_report.md`、`comparison_figures/`（PCE/BCE/DPE/CCE 对比柱状图 + 综合雷达图）。

## 已有结果对比（按三类实验，指标均为前四个：PCE/BCE/DPE/CCE）
- 说明：以下对比基于已跑完的目录；若缺少 `metrics_summary.json`，请先对对应目录运行离线指标计算（见“生成离线指标”）。

### 1) 相似度对比
- 数据位置：`experiments/runs/SimilarityComparison/SimilarityComparison_answer2_20251218_140032` 及同级的 embedding/feature/hybrid 目录。
- 操作：
  - 如需补算指标：对上述四个目录分别执行 `python -m experiments.analysis.calculate_metrics --experiment-dir <dir> --incremental`
  - 生成对比：`python -m experiments.analysis.compare_similarity_methods --results-dir ./experiments/runs/SimilarityComparison --output-dir ./reports/similarity_comparison`
  - 输出包含四个指标的柱状图与雷达图。

### 2) 消融实验
- 数据位置：`experiments/runs/ScenarioFuzz-LLM/` 下的 `S1`、`S2`、`S3`、`S4` 目录。
- 操作：
  - 补算指标：对 `S1`–`S4` 分别执行 `python -m experiments.analysis.calculate_metrics --experiment-dir <dir> --incremental`
  - 汇总：`python -m experiments.aggregation.main`（会扫描 `experiments/runs`，将 S1–S4 写入 `all_methods_results.json`）
  - 可视化：`python -m experiments.analysis.generate_figures --results-file ./experiments/runs/all_methods_results.json --output-dir ./reports/figs` （聚焦前四个指标）
  - 如需 Markdown/JSON 报告：`python -m experiments.analysis.generate_reports --results-file ./experiments/runs/all_methods_results.json --output-dir ./reports --experiment-name Ablation`

### 3) 变异实验（GPT 引导 vs 随机）
- 数据位置：`ScenarioFuzz-LLM/experiment_results/ScenarioFuzz-LLM/` 下的两个实验目录（GPT 引导与随机变异）。
- 操作：
  - 补算指标：对两个目录分别执行 `python -m experiments.analysis.calculate_metrics --experiment-dir <dir> --incremental`
  - 若需并列图表，可先将结果拷贝到 `experiments/runs/ScenarioFuzz-LLM/`（或为生成的 `metrics_summary.json` 添加唯一 `method` 字段），再运行：
    - `python -m experiments.aggregation.main`
    - `python -m experiments.analysis.generate_figures --results-file ./experiments/runs/all_methods_results.json --output-dir ./reports/figs`
  - 对比重点：PCE/BCE/DPE/CCE（前四个指标）。

---
- 所有命令默认在项目根目录执行；需要自定义 GPU/端口可在启动 CARLA 时调整 `run_carla.sh`。
- 若出现 `Unknown` 方法汇总，检查相关 run 是否缺少 `method` 字段或离线指标未补全。


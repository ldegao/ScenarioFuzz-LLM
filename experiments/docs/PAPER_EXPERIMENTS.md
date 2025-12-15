### 论文 3.3–3.5 复现实验说明（ScenarioFuzz-LLM / RAG-ScenarioFuzz / TM-Fuzzer）

本说明文档面向论文第 3.3–3.5 节的复现实验，提供一个**统一且精简**的使用入口。

- **三类参评方法（3.3.2）**：
  - ScenarioFuzz-LLM
  - RAG-ScenarioFuzz
  - TM-Fuzzer（Autoware 基线）
- **统一数据流（3.4 + 3.5）**：
  - 三种方法的运行结果全部写入 `./experiment_results`
  - 指标聚合后得到 `all_methods_results.json`
  - 可视化与报告生成统一从该文件读取

---

### 1. 环境准备（简要）

详细环境准备请参考仓库根目录下的 `README.md`，这里只列出与实验脚本相关的要点：

- 已安装并可运行：
  - CARLA 模拟器（通过仓库中的 `script/run_carla.sh` 管理）
  - Autoware（TM-Fuzzer 基线所需）
  - Docker 与 ROS（TM-Fuzzer 基线依赖）
- 已安装 Python 依赖（建议使用虚拟环境）：
  - 在仓库根目录执行 `pip install -r requirements.txt`（或按照主 README 的说明）
- 默认假设当前工作目录为仓库根目录：
  - `/home/linshenghao/ScenarioFuzz-LLM`

环境的具体初始化与清理逻辑由：

- `experiments/environment_manager.py`
- `experiments/experiment_manager.py`

自动调用，不需要手动执行。

---

### 2. 三类方法的统一 Runner 接口

三种方法各有一个**方法专属入口**，均支持两种模式：

- **定量模式（quantitative）**：指定 `--num-scenarios`
- **定时模式（timed）**：指定 `--hours`

#### 2.1 ScenarioFuzz-LLM

入口：

```bash
python -m experiments.runners.run_scenariofuzz_llm ...
```

常用参数（核心）：

- `--num-scenarios N`：运行 N 个场景（与论文 3.3 中的“定量实验”对应）
- `--hours H`：运行 H 小时（与论文 3.3 中的“定时实验”对应）
- `--output-root PATH`：结果根目录（默认 `./experiment_results`）
- `--target {behavior,autoware}`：目标 ADS（默认 `behavior`）
- `--town INT`：CARLA Town ID（默认 `3`）
- `--timeout INT`：单场景超时（秒，默认 `60`）

示例：

```bash
# 以行为模型为目标，生成 1000 个场景
python -m experiments.runners.run_scenariofuzz_llm \
  --num-scenarios 1000 \
  --output-root ./experiment_results

# 以 Autoware 为目标，运行 2 小时
python -m experiments.runners.run_scenariofuzz_llm \
  --hours 2 \
  --target autoware \
  --output-root ./experiment_results
```

#### 2.2 RAG-ScenarioFuzz

入口：

```bash
python -m experiments.runners.run_rag_scenariofuzz ...
```

在 ScenarioFuzz-LLM 的参数基础上，增加：

- `--rag-k INT`：RAG 检索的 top-k，默认 `5`

示例：

```bash
# RAG-ScenarioFuzz，生成 1000 个场景
python -m experiments.runners.run_rag_scenariofuzz \
  --num-scenarios 1000 \
  --rag-k 5 \
  --output-root ./experiment_results

# RAG-ScenarioFuzz，运行 2 小时
python -m experiments.runners.run_rag_scenariofuzz \
  --hours 2 \
  --rag-k 5 \
  --output-root ./experiment_results
```

#### 2.3 TM-Fuzzer（Autoware 基线）

入口：

```bash
python -m experiments.runners.run_tmfuzzer ...
```

核心参数：

- `--num-scenarios N` 或 `--hours H`
- `--output-root PATH`（默认 `./experiment_results`）
- `--target {autoware,behavior}`（默认 `autoware`，推荐保持不变）
- `--density FLOAT`：车辆密度，默认 `0.4`
- `--town INT`：CARLA Town ID（默认 `3`）
- `--timeout INT`：单场景超时（默认 `60`）

示例：

```bash
# TM-Fuzzer，Autoware 目标，估算时间对应约 1000 个场景
python -m experiments.runners.run_tmfuzzer \
  --num-scenarios 1000 \
  --target autoware \
  --output-root ./experiment_results

# TM-Fuzzer，Autoware 目标，运行 2 小时
python -m experiments.runners.run_tmfuzzer \
  --hours 2 \
  --target autoware \
  --output-root ./experiment_results
```

---

### 3. 统一的结果目录结构

所有三种方法都通过 `ExperimentManager` 写入**统一布局**：

```text
./experiment_results/
  ScenarioFuzz-LLM/
    <experiment_id_1>/
      ... # 该次运行的所有输出（队列、错误、图片、metrics 等）
    <experiment_id_2>/
      ...
  RAG-ScenarioFuzz/
    <experiment_id_3>/
      ...
  TM-Fuzzer/
    <experiment_id_4>/
      ...
  all_methods_results.json        # 指标聚合后的多方法结果（第 2 阶段生成）
```

其中：

- `experiment_id` 若不显式指定，会由 `ExperimentManager` 自动生成（包含方法名与时间戳），便于多次实验对比；
- 每个 `<experiment_id>` 目录内部：
  - `metrics/metrics_records.jsonl`：按场景记录的指标（由 `fuzzer.py` 写入，适用于 ScenarioFuzz-LLM / RAG-ScenarioFuzz）
  - `metrics/metrics_summary.json`：单次运行的聚合指标（由 `MetricsAggregator` 自动生成）

在 TM-Fuzzer 路径下：

- `ExperimentManager` 会为每次 TM-Fuzzer 运行创建独立的 `./experiment_results/TM-Fuzzer/<experiment_id>/` 目录；
- 如后续补充了 TM-Fuzzer 的 per-scenario metrics（JSONL），即可与现有 `MetricsAggregator` 对齐进行聚合。

---

### 4. 第二阶段：指标聚合（experiments.aggregation）

在运行完三条方法之后（可以按任意顺序、多次运行），使用聚合入口：

```bash
python -m experiments.aggregation.main
```

默认行为：

- 从 `./experiment_results` 开始递归搜索所有 `metrics_summary.json`；
- 读取其中的 `method` 字段，将结果聚合为以下结构：

```json
{
  "ScenarioFuzz-LLM": [ { "bpc": ..., "dbcc": ..., "dpd": ..., "bcm": ..., ... }, ... ],
  "RAG-ScenarioFuzz": [ ... ],
  "TM-Fuzzer":        [ ... ]
}
```

- 写入文件：

```text
./experiment_results/all_methods_results.json
```

这一步对应论文 3.4 + 3.5 中“多方法汇总的数据线”。

如果你想指定其他根目录或输出路径，可使用：

```bash
python -m experiments.aggregation.collect_results \
  --root ./experiment_results \
  --output ./experiment_results/all_methods_results.json
```

---

### 5. 第三阶段：可视化与报告生成（experiments.analysis）

分析部分依赖聚合后的 `all_methods_results.json`，提供三种调用方式：

#### 5.1 仅生成图像

```bash
python -m experiments.analysis.generate_figures \
  --results-file ./experiment_results/all_methods_results.json \
  --output-dir ./reports/figs
```

- 输出：
  - `./reports/figs/bpc_coverage.png`：BPC 对比柱状图
  - `./reports/figs/metrics_radar.png`：BPC/DBCC/DPD/BCM 雷达图

#### 5.2 仅生成文本/JSON 报告

```bash
python -m experiments.analysis.generate_reports \
  --results-file ./experiment_results/all_methods_results.json \
  --output-dir ./reports \
  --experiment-name "Thesis_Experiment"
```

- 输出：
  - `./reports/*.md`：Markdown 报告
  - `./reports/*.json`：总结 JSON

#### 5.3 一键生成图像 + 报告

```bash
python -m experiments.analysis.main
```

默认使用：

- `./experiment_results/all_methods_results.json` 作为输入
- 将图像写入 `./reports/figs`
- 将报告写入 `./reports`

---

### 6. 数据流总览（文字版）

1. **运行三类方法（可多次重复）**：

   ```text
   python -m experiments.runners.run_scenariofuzz_llm   ...
   python -m experiments.runners.run_rag_scenariofuzz   ...
   python -m experiments.runners.run_tmfuzzer           ...
   ```

   结果写入：

   ```text
   ./experiment_results/<Method>/<experiment_id>/metrics/metrics_summary.json
   ```

2. **统一聚合（多方法统一指标）**：

   ```bash
   python -m experiments.aggregation.main
   ```

   结果写入：

   ```text
   ./experiment_results/all_methods_results.json
   ```

3. **可视化 + 报告**：

   ```bash
   python -m experiments.analysis.main
   ```

   结果写入：

   ```text
   ./reports/figs/*.png
   ./reports/*.md
   ./reports/*.json
   ```

---

### 7. 目录整理说明

最新的 `experiments/` 目录经过归档与精简，核心入口如下：

```text
experiments/core/             # ExperimentManager + Progress/Time tracking + 环境管理
experiments/runners/          # 统一 CLI 入口（run_*.py + 各方法 runner 包）
experiments/evaluation/       # 离线评估（如 rag_embedding）
experiments/aggregation/      # 指标聚合
experiments/analysis/         # 图表与报告
experiments/tools/estimate_time.py   # 实验耗时预估 CLI
experiments/scripts/          # shell 辅助脚本（quick_start.sh, run_example.sh）
experiments/docs/             # 当前文档（Quick Start、Paper Experiments）
```

- 早期的 `experiments/dev_legacy/` 脚本已删除，功能已经融入上述正式入口；
- 论文 3.3–3.5 的复现实验**请只使用** `experiments.runners.*` 入口，必要时可配合 `experiments/tools/estimate_time.py` 做耗时预估。

---

### 8. 一条典型的 3.3–3.5 复现流水线（示例）

从“空的 `experiment_results` 目录”开始，你可以按以下顺序完成一次完整的实验：

1. 运行三种方法（例如各 1000 场景）：

   ```bash
   python -m experiments.runners.run_scenariofuzz_llm   --num-scenarios 1000
   python -m experiments.runners.run_rag_scenariofuzz   --num-scenarios 1000
   python -m experiments.runners.run_tmfuzzer           --num-scenarios 1000 --target autoware
   ```

2. 聚合所有运行的指标：

   ```bash
   python -m experiments.aggregation.main
   ```

3. 生成图像与报告：

   ```bash
   python -m experiments.analysis.main
   ```

4. 在论文撰写中引用：

   - 表格/图 3.x：来自 `./reports/figs/*.png`
   - 文字总结：来自 `./reports/*.md` 与 `./reports/*.json`



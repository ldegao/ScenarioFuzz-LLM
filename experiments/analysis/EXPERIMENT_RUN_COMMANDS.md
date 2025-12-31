# 实验运行命令说明

## 失败的实验目录

以下两个实验目录是失败的（已确认为空，没有运行）：

1. `SimilarityComparison_embedding_20251209_095659`
2. `SimilarityComparison_feature_20251209_095726`

## 删除失败的实验目录

### 方法 1: 使用删除脚本（推荐）

```bash
cd ~/ScenarioFuzz-LLM
./experiments/analysis/delete_failed_experiments.sh
```

脚本会：
- 显示要删除的目录信息
- 要求确认
- 安全删除失败的实验目录

### 方法 2: 手动删除

```bash
cd ~/ScenarioFuzz-LLM
rm -rf experiment_results/SimilarityComparison/SimilarityComparison_embedding_20251209_095659
rm -rf experiment_results/SimilarityComparison/SimilarityComparison_feature_20251209_095726
```

## 运行脚本说明

### 1. 批量运行脚本（运行所有方法）

**Bash 脚本**: `experiments/scripts/run_similarity_comparison.sh`

**使用方法**:
```bash
cd ~/ScenarioFuzz-LLM
./experiments/scripts/run_similarity_comparison.sh \
  --num-scenarios 100 \
  --output-root ./experiment_results
```

**参数说明**:
- `--num-scenarios N`: 要生成的场景数量（默认: 1000）
- `--output-root PATH`: 输出根目录（默认: ./experiment_results）
- `--target {behavior|autoware}`: 目标 ADS 系统（默认: behavior）
- `--town N`: CARLA 城镇编号（默认: 3）
- `--timeout N`: 场景超时时间（秒，默认: 60）
- `--rag-k N`: RAG 检索的 top-k（默认: 5）
- `--hybrid-embedding-weight F`: Hybrid 方法中 embedding 的权重（默认: 0.6）

**运行的方法**:
- answer2 (LLM-based)
- embedding (Embedding-based)
- feature (Feature-based)
- hybrid (Hybrid)

### 2. 单独运行某个方法

**Python 模块**: `experiments.runners.run_similarity_comparison`

#### 运行 embedding 方法

```bash
cd ~/ScenarioFuzz-LLM
python3 -m experiments.runners.run_similarity_comparison \
  --num-scenarios 100 \
  --similarity-method embedding \
  --output-root ./experiment_results \
  --target behavior \
  --town 3 \
  --timeout 60 \
  --rag-k 5
```

#### 运行 feature 方法

```bash
cd ~/ScenarioFuzz-LLM
python3 -m experiments.runners.run_similarity_comparison \
  --num-scenarios 100 \
  --similarity-method feature \
  --output-root ./experiment_results \
  --target behavior \
  --town 3 \
  --timeout 60 \
  --rag-k 5
```

#### 运行 answer2 方法

```bash
cd ~/ScenarioFuzz-LLM
python3 -m experiments.runners.run_similarity_comparison \
  --num-scenarios 100 \
  --similarity-method answer2 \
  --output-root ./experiment_results \
  --target behavior \
  --town 3 \
  --timeout 60 \
  --rag-k 5
```

#### 运行 hybrid 方法

```bash
cd ~/ScenarioFuzz-LLM
python3 -m experiments.runners.run_similarity_comparison \
  --num-scenarios 100 \
  --similarity-method hybrid \
  --output-root ./experiment_results \
  --target behavior \
  --town 3 \
  --timeout 60 \
  --rag-k 5 \
  --hybrid-embedding-weight 0.6
```

### 3. 查看帮助信息

```bash
# Bash 脚本帮助
./experiments/scripts/run_similarity_comparison.sh --help

# Python 模块帮助
python3 -m experiments.runners.run_similarity_comparison --help
```

## 实验输出位置

所有实验结果保存在：
```
./experiment_results/SimilarityComparison/SimilarityComparison_{method}_{timestamp}/
```

每个实验目录包含：
- `ga_checkpoint.pkl` - 遗传算法检查点
- `queue/` - 生成的场景文件
- `metrics/metrics_records.jsonl` - 指标记录
- `metrics/metrics_summary.json` - 指标汇总
- `token_usage.json` - Token 使用统计
- 其他文件（rosbags, pictures, etc.）

## 注意事项

1. **embedding 和 feature 方法不需要 LLM API**，所以 token 使用量为 0 是正常的
2. 确保 CARLA 模拟器正在运行或可以启动
3. 实验可能需要较长时间，建议使用 `screen` 或 `tmux` 运行
4. 如果实验失败，检查错误日志和输出信息

## 调试失败的实验

如果 embedding 或 feature 实验再次失败，可以：

1. **添加调试模式**:
```bash
python3 -m experiments.runners.run_similarity_comparison \
  --num-scenarios 10 \
  --similarity-method embedding \
  --output-root ./test_output \
  --target behavior \
  --debug
```

2. **检查详细输出**: 查看标准输出和标准错误
3. **检查依赖**: 确保所有依赖（CARLA、Python 包等）已正确安装
4. **检查配置**: 验证配置文件是否正确




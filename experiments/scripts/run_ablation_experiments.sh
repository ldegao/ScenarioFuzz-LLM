#!/usr/bin/env bash
set -u

# 一键运行主线 + 两个消融（无相似度 / 无指导变异）
# 用法：
#   bash experiments/scripts/run_ablation_experiments.sh [num_scenarios] [output_root]
# 默认：num_scenarios=100，output_root=./experiments/runs

NUM_SCENARIOS="${1:-100}"
OUTPUT_ROOT="${2:-./experiments/runs}"
DB_PATH="./data/scenario_db.json"

clean_scenario_db() {
  if [ -f "$DB_PATH" ]; then
    rm -f "$DB_PATH"
    echo "[INFO] 清空 RAG 场景库：$DB_PATH"
  else
    echo "[INFO] RAG 场景库为空：$DB_PATH"
  fi
}

run_with_retry() {
  local desc="$1"; shift
  local cmd=("$@")
  local retries=3
  local attempt=0
  echo ">>> 开始：${desc}"
  until "${cmd[@]}"; do
    exit_code=$?
    attempt=$((attempt + 1))
    if [ $attempt -ge $retries ]; then
      echo "!!! 失败：${desc} （已重试 ${attempt} 次，仍未成功，退出码 ${exit_code}）"
      return $exit_code
    fi
    echo "--- 重试 ${attempt}/${retries}：${desc} （退出码 ${exit_code}），等待 5 秒再试..."
    sleep 5
  done
  echo "✓ 完成：${desc}"
}

run_with_retry "[1/4] ScenarioFuzz-LLM（无 RAG）" \
  clean_scenario_db && \
  python -m experiments.cli run \
    --method scenariofuzz-llm \
    --num-scenarios "${NUM_SCENARIOS}" \
    --output-root "${OUTPUT_ROOT}" \
    --target behavior \
    --town 3 \
    --timeout 60

run_with_retry "[2/4] RAG-ScenarioFuzz（带 RAG）" \
  clean_scenario_db && \
  python -m experiments.cli run \
    --method rag-scenariofuzz \
    --num-scenarios "${NUM_SCENARIOS}" \
    --output-root "${OUTPUT_ROOT}" \
    --target behavior \
    --town 3 \
    --timeout 60 \
    --rag-k 5

run_with_retry "[3/4] ScenarioFuzz-LLM 消融1（禁用相似度模块，无多样性分数）" \
  clean_scenario_db && \
  python -m experiments.cli run \
    --method scenariofuzz-llm \
    --num-scenarios "${NUM_SCENARIOS}" \
    --output-root "${OUTPUT_ROOT}" \
    --target behavior \
    --town 3 \
    --timeout 60 \
    --disable-similarity

run_with_retry "[4/4] ScenarioFuzz-LLM 消融2（禁用相似度 + 禁用指导变异，纯随机）" \
  clean_scenario_db && \
  python -m experiments.cli run \
    --method scenariofuzz-llm \
    --num-scenarios "${NUM_SCENARIOS}" \
    --output-root "${OUTPUT_ROOT}" \
    --target behavior \
    --town 3 \
    --timeout 60 \
    --disable-similarity \
    --disable-guided-mutation

echo "[DONE] 全部实验已提交运行。结果输出目录：${OUTPUT_ROOT}"


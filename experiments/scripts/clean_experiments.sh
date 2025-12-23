#!/usr/bin/env bash
# 清理实验产物/缓存目录，默认仅预览将删除的内容，使用 --force 才会实际删除。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

FORCE=0
KEEP_REPORTS=0

usage() {
  cat <<'EOF'
用法:
  experiments/scripts/clean_experiments.sh [--force] [--keep-reports]

默认只预览将删除的目录/文件；加 --force 才会执行删除。

目标（若存在）:
  - experiments/runs/                  # 新实验结果根
  - experiment_results/                # 旧实验结果根
  - experiment_result_analysis/        # 旧分析输出
  - reports/figs/ 与 reports/*.md      # 报告与图（可用 --keep-reports 保留）
  - rag_short_output/                  # RAG 短输出缓存
  - data/output/recorder/              # 录制缓存
  - metrics/metrics_records.jsonl      # 历史指标记录
  - token_usage.json                   # 运行期 token 统计
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    --keep-reports) KEEP_REPORTS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1"; usage; exit 1 ;;
  esac
  shift
done

TARGETS=(
  "$PROJECT_ROOT/experiments/runs"
  "$PROJECT_ROOT/experiment_results"
  "$PROJECT_ROOT/experiment_result_analysis"
  "$PROJECT_ROOT/rag_short_output"
  "$PROJECT_ROOT/data/output/recorder"
  "$PROJECT_ROOT/metrics/metrics_records.jsonl"
  "$PROJECT_ROOT/token_usage.json"
)

if [[ $KEEP_REPORTS -eq 0 ]]; then
  TARGETS+=(
    "$PROJECT_ROOT/reports/figs"
    "$PROJECT_ROOT/reports"/*.md
  )
fi

echo "[INFO] 项目根: $PROJECT_ROOT"
echo "[INFO] 目标列表:"
for t in "${TARGETS[@]}"; do
  echo "  - $t"
done

if [[ $FORCE -eq 0 ]]; then
  echo "[DRY-RUN] 未加 --force，不会执行删除。"
  exit 0
fi

echo "[INFO] 开始删除..."
for t in "${TARGETS[@]}"; do
  if compgen -G "$t" > /dev/null; then
    rm -rf $t
    echo "  removed: $t"
  fi
done
echo "[INFO] 清理完成。"


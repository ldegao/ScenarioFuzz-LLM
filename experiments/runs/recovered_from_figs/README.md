# 从图像/报告恢复的聚合数据快照

本目录用于弥补 `experiments/runs/` 被清理后的缺失数据，内容均来源于 `experiment_results/` 下的已生成图像/报告/聚合 JSON，**不包含原始逐场景运行日志或指标明细**。如需完全复现，请重新运行实验。

## 文件说明
- `all_methods_results.json`：主方法对比（ScenarioFuzz-LLM 系列 + RAG-ScenarioFuzz + SimilarityComparison 衍生）汇总指标，含 `experiment_id` 和样本数量。
- `local_diversity/local_diversity_incremental.json`：局部多样性增量指标（GPT 引导 vs 随机）随检查点的 LMS/SED/OSCR 变化。
- `similarity_comparison/comparison_report.{json,md}`：相似度方法对比的聚合统计（PCE/BCE/DPE/CCE）。
- `similarity_comparison/comparison_figures/`：对应的对比图像备份。

## 已知缺失
- 未包含原始生成的场景文件、逐样本指标或日志；若需重现，请按 `experiments/docs/EXPERIMENT_RUN_GUIDE.md` 重新运行对应实验。

## 用途
- 作为图像/报告的数值依据，便于后续撰写文档或复刻图表，无需改动现有图像。


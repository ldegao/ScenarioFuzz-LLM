#!/usr/bin/env python3
"""
Local Diversity Comparison Runner
---------------------------------

对同一初始种子集合，比较 “GPT 指导变异” 与 “随机变异” 的局部多样性指标（LMS/SED/OSCR）。
输入为两份实验目录（各自包含 queue/ 与 pickle/json 场景数据），输出一个汇总 JSON。
"""

import argparse
import json
from pathlib import Path
from typing import Dict

from experiments.analysis.local_diversity_metrics import summarize_local_diversity
from experiments.analysis.scenario_loader import ScenarioDataLoader


def _compute_for_dir(exp_dir: Path, prefer_pickle: bool = True) -> Dict:
    loader = ScenarioDataLoader()
    scenarios = loader.load_all_scenarios(exp_dir, prefer_pickle=prefer_pickle)
    metrics = summarize_local_diversity(scenarios)
    metrics.update(
        {
            "num_scenarios": len(scenarios),
            "experiment_id": exp_dir.name,
            "experiment_dir": str(exp_dir),
        }
    )
    return metrics


def _diff(gpt: Dict, rand: Dict) -> Dict[str, float]:
    keys = ["lms", "sed", "oscr"]
    return {f"delta_{k}": float(gpt.get(k, 0.0) - rand.get(k, 0.0)) for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local diversity (LMS/SED/OSCR) between GPT-guided and random mutations."
    )
    parser.add_argument(
        "--gpt-dir",
        required=True,
        help="实验目录（GPT 指导变异），应包含 queue/ 及场景数据。",
    )
    parser.add_argument(
        "--rand-dir",
        required=True,
        help="实验目录（随机变异），应包含 queue/ 及场景数据。",
    )
    parser.add_argument(
        "--output",
        default="./experiment_results/local_diversity_comparison.json",
        help="输出 JSON 路径（默认: ./experiment_results/local_diversity_comparison.json）。",
    )
    parser.add_argument(
        "--no-prefer-pickle",
        action="store_true",
        help="若设置，则优先使用 JSON 而非 pickle 进行加载（默认优先 pickle）。",
    )

    args = parser.parse_args()
    prefer_pickle = not args.no_prefer_pickle

    gpt_dir = Path(args.gpt_dir).resolve()
    rand_dir = Path(args.rand_dir).resolve()

    if not gpt_dir.exists():
        parser.error(f"GPT 目录不存在: {gpt_dir}")
    if not rand_dir.exists():
        parser.error(f"随机变异目录不存在: {rand_dir}")

    gpt_metrics = _compute_for_dir(gpt_dir, prefer_pickle=prefer_pickle)
    rand_metrics = _compute_for_dir(rand_dir, prefer_pickle=prefer_pickle)

    result = {
        "gpt": gpt_metrics,
        "random": rand_metrics,
    }
    result.update(_diff(gpt_metrics, rand_metrics))

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[LocalDiversity] 已保存对比结果 -> {output_path}")
    print(
        f"[LocalDiversity] GPT: LMS={gpt_metrics.get('lms',0):.4f}, "
        f"SED={gpt_metrics.get('sed',0):.4f}, OSCR={gpt_metrics.get('oscr',0):.4f}"
    )
    print(
        f"[LocalDiversity] Rand: LMS={rand_metrics.get('lms',0):.4f}, "
        f"SED={rand_metrics.get('sed',0):.4f}, OSCR={rand_metrics.get('oscr',0):.4f}"
    )


if __name__ == "__main__":
    main()


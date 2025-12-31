#!/usr/bin/env python3
"""
统一实验 CLI：
  python -m experiments.cli run ...
  python -m experiments.cli metrics ...
  python -m experiments.cli aggregate ...
  python -m experiments.cli report ...

保持核心逻辑不变，主要整合路径与指令入口。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover - 兜底
    yaml = None

from experiments.core.experiment_manager import ExperimentManager
from experiments.core.path_utils import DEFAULT_RUN_ROOT, generate_run_id


METHOD_MAP = {
    "scenariofuzz-llm": "ScenarioFuzz-LLM",
    "rag-scenariofuzz": "RAG-ScenarioFuzz",
    "tmfuzzer": "TM-Fuzzer",
    "similarity": "SimilarityComparison",
}


def load_config(path: Path) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    if yaml is None:
        raise RuntimeError("缺少 PyYAML，无法读取配置文件，请安装 pyyaml 或改用 CLI 参数。")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    for k, v in (override or {}).items():
        if v is not None:
            merged[k] = v
    return merged


def run_experiment(args: argparse.Namespace) -> None:
    method_key = args.method.lower()
    if method_key not in METHOD_MAP:
        raise ValueError(f"不支持的 method: {args.method}")
    method_name = METHOD_MAP[method_key]

    cfg = load_config(Path(args.config)) if args.config else {}

    merged = merge_config(
        cfg,
        {
            "num_scenarios": args.num_scenarios,
            "hours": args.hours,
            "target": args.target,
            "town": args.town,
            "timeout": args.timeout,
            "density": args.density,
            "rag_k": args.rag_k,
            "similarity_scoring_method": args.similarity_method,
            "hybrid_embedding_weight": args.hybrid_embedding_weight,
            "feature_position_weight": args.feature_position_weight,
            "feature_speed_weight": args.feature_speed_weight,
            "feature_angular_accel_weight": args.feature_angular_accel_weight,
            "feature_relative_position_weight": args.feature_relative_position_weight,
            "disable_similarity": args.disable_similarity,
            "disable_guided_mutation": args.disable_guided_mutation,
            "determ_seed": args.determ_seed,
            "seed_dir": args.seed_dir,
            "debug": args.debug,
        },
    )

    run_id = args.name or merged.get("experiment_id") or generate_run_id(method_key)
    manager = ExperimentManager(output_base_dir=args.output_root)

    kwargs = dict(
        experiment_id=run_id,
        target=merged.get("target", "behavior"),
        town=merged.get("town", 3),
        timeout=merged.get("timeout", 60),
        density=merged.get("density", 0.4),
        rag_k=merged.get("rag_k", 5),
        similarity_scoring_method=merged.get("similarity_scoring_method", "answer2"),
        hybrid_embedding_weight=merged.get("hybrid_embedding_weight", 0.6),
        feature_position_weight=merged.get("feature_position_weight", 0.3),
        feature_speed_weight=merged.get("feature_speed_weight", 0.3),
        feature_angular_accel_weight=merged.get("feature_angular_accel_weight", 0.2),
        feature_relative_position_weight=merged.get("feature_relative_position_weight", 0.2),
        disable_similarity=bool(merged.get("disable_similarity", False)),
        disable_guided_mutation=bool(merged.get("disable_guided_mutation", False)),
        determ_seed=merged.get("determ_seed"),
        seed_dir=merged.get("seed_dir"),
        debug=bool(merged.get("debug", False)),
    )

    # 运行模式选择
    num_scenarios = merged.get("num_scenarios")
    hours = merged.get("hours")

    if method_name == "TM-Fuzzer" and num_scenarios and hours:
        raise ValueError("TM-Fuzzer 不支持同时设置 --num-scenarios 与 --hours")

    if num_scenarios and not hours:
        manager.run_quantitative_experiment(method_name, int(num_scenarios), **kwargs)
    elif hours and not num_scenarios:
        manager.run_timed_experiment(method_name, float(hours), **kwargs)
    elif num_scenarios and hours:
        # 双约束：时间 + 场景数
        kwargs["max_scenarios"] = int(num_scenarios)
        manager.run_timed_experiment(method_name, float(hours), **kwargs)
    else:
        raise ValueError("需要指定 --num-scenarios 或 --hours 中至少一个。")


def run_metrics(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "experiments.analysis.calculate_metrics",
        "--experiment-dir",
        args.run_dir,
    ]
    if args.incremental:
        cmd.append("--incremental")
    if args.recalculate:
        cmd.append("--recalculate")
    subprocess.run(cmd, check=True)


def run_aggregate(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "experiments.aggregation.main",
        "--root",
        args.root,
        "--output",
        args.output,
    ]
    subprocess.run(cmd, check=True)


def run_report(args: argparse.Namespace) -> None:
    # 先生成图，再生成报告
    figs_cmd = [
        sys.executable,
        "-m",
        "experiments.analysis.generate_figures",
        "--results-file",
        args.results,
        "--output-dir",
        args.output_dir,
    ]
    rep_cmd = [
        sys.executable,
        "-m",
        "experiments.analysis.generate_reports",
        "--results-file",
        args.results,
        "--output-dir",
        args.output_dir,
        "--experiment-name",
        args.experiment_name,
    ]
    subprocess.run(figs_cmd, check=True)
    subprocess.run(rep_cmd, check=True)


def run_migrate(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "experiments.tools.migrate_results",
        "--legacy-root",
        args.legacy_root,
        "--new-root",
        args.new_root,
    ]
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ScenarioFuzz-LLM 统一实验 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="运行实验")
    run_p.add_argument("--method", required=True, choices=METHOD_MAP.keys())
    run_p.add_argument("--config", type=str, help="YAML 配置文件路径")
    run_p.add_argument("--name", type=str, help="自定义 run_id（缺省自动生成）")
    run_p.add_argument("--num-scenarios", type=int, help="场景数量")
    run_p.add_argument("--hours", type=float, help="运行时长（小时）")
    run_p.add_argument("--target", type=str, choices=["behavior", "autoware"], help="目标系统")
    run_p.add_argument("--town", type=int, help="CARLA town")
    run_p.add_argument("--timeout", type=int, help="单场景超时时间")
    run_p.add_argument("--density", type=float, help="交通密度")
    run_p.add_argument("--rag-k", type=int, help="RAG top-k")
    run_p.add_argument("--similarity-method", type=str, choices=["answer2", "embedding", "feature", "hybrid"], help="相似度方法（similarity 模式必填）")
    run_p.add_argument("--hybrid-embedding-weight", type=float, help="hybrid 方法 embedding 权重")
    run_p.add_argument("--feature-position-weight", type=float, help="feature 方法位置权重")
    run_p.add_argument("--feature-speed-weight", type=float, help="feature 方法速度权重")
    run_p.add_argument("--feature-angular-accel-weight", type=float, help="feature 方法角加速度权重")
    run_p.add_argument("--feature-relative-position-weight", type=float, help="feature 方法相对位置权重")
    run_p.add_argument("--disable-similarity", action="store_true", help="禁用相似度模块（相似度记0，跳过 GPT/embedding/feature 计算）")
    run_p.add_argument("--disable-guided-mutation", action="store_true", help="禁用指导变异（不调用 GPT answer3，使用纯随机变异参数）")
    run_p.add_argument("--determ-seed", type=float, help="固定随机种子（用于可复现性，实验3需要相同种子）")
    run_p.add_argument("--seed-dir", type=str, help="指定初始种子库目录（默认 ./data/seed，可指向历史实验的 queue 作为起始种子）")
    run_p.add_argument("--output-root", type=str, default=str(DEFAULT_RUN_ROOT), help="输出根目录（默认 experiments/runs）")
    run_p.add_argument("--debug", action="store_true")
    run_p.set_defaults(func=run_experiment)

    metrics_p = sub.add_parser("metrics", help="离线指标计算")
    metrics_p.add_argument("--run-dir", required=True, help="单次实验目录")
    metrics_p.add_argument("--incremental", action="store_true")
    metrics_p.add_argument("--recalculate", action="store_true")
    metrics_p.set_defaults(func=run_metrics)

    agg_p = sub.add_parser("aggregate", help="聚合多次实验指标")
    agg_p.add_argument("--root", default=str(DEFAULT_RUN_ROOT), help="运行根目录（默认 experiments/runs）")
    agg_p.add_argument("--output", default="experiments/runs/all_methods_results.json", help="输出文件")
    agg_p.set_defaults(func=run_aggregate)

    rep_p = sub.add_parser("report", help="生成图表与报告")
    rep_p.add_argument("--results", required=True, help="all_methods_results.json 路径")
    rep_p.add_argument("--output-dir", default="reports", help="输出目录")
    rep_p.add_argument("--experiment-name", default="Thesis_Experiment", help="报告名称")
    rep_p.set_defaults(func=run_report)

    mig_p = sub.add_parser("migrate", help="迁移旧 experiment_results 目录")
    mig_p.add_argument("--legacy-root", default="experiment_results", help="旧根目录")
    mig_p.add_argument("--new-root", default=str(DEFAULT_RUN_ROOT), help="新根目录")
    mig_p.set_defaults(func=run_migrate)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()


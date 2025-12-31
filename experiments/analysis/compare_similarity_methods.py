#!/usr/bin/env python3
"""
Compare similarity scoring methods based on four diversity metrics (PCE, BCE, DPE, CCE).

This script reads metrics_summary.json files from all similarity method experiments,
compares the four metrics, and generates comparison reports and visualizations.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import font_manager, rcParams
    from matplotlib.backends.backend_pdf import PdfPages
    matplotlib.use('Agg')  # Use non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not available, skipping figure generation")

METHOD_NAMES = {
    "answer2": "LLM similarity",
    "embedding": "Embedding-based",
    "feature": "Feature-based",
    "hybrid": "Hybrid",
    "random": "Random baseline",
    "ragllm": "RAG-LLM similarity",
    "S1": "RAG-ScenarioFuzz-LLM (baseline)",
    "S2": "RAG-ScenarioFuzz-LLM (no similarity)",
    "S3": "RAG-ScenarioFuzz-LLM (no LLM guidance)",
    "S4": "Random mutation baseline",
}

METHOD_NAMES_CN = {
    "answer2": "LLM 相似度",
    "embedding": "Embedding 相似度",
    "feature": "特征距离",
    "hybrid": "混合策略",
    "random": "随机变异基线",
    "ragllm": "RAG-LLM 相似度",
    "S1": "S1：RAG-ScenarioFuzz-LLM",
    "S2": "S2：RAG-ScenarioFuzz-LLM（无相似度模块）",
    "S3": "S3：RAG-ScenarioFuzz-LLM（无 LLM 变异引导）",
    "S4": "S4：随机变异基线",
}

METRIC_NAMES = {
    "pc": "Parameter Configuration Entropy (PCE)",
    "pec": "Behavior Category Entropy (BCE)",
    "tcd": "Driving Pattern Entropy (DPE)",
    "bcm": "Combination Coverage Entropy (CCE)"
}

METRIC_NAMES_CN = {
    "pc": "PCE",
    "pec": "BCE",
    "tcd": "DPE",
    "bcm": "CCE",
}

METRIC_ABBR = {
    "pc": "PCE",
    "pec": "BCE",
    "tcd": "DPE",
    "bcm": "CCE",
}

METRIC_FILE_SLUG = {
    "pc": "pce",
    "pec": "bce",
    "tcd": "dpe",
    "bcm": "cce",
}

TITLE_SIZE = 10.5  # 五号
LABEL_SIZE = 9     # 小五


HOME = Path.home()

CHINESE_FONT_CANDIDATES: List[Tuple[str, str]] = [
    ("Noto Serif CJK SC", str(HOME / ".local/share/fonts/custom/NotoSerifSC-Regular.otf")),
    ("Noto Serif CJK SC", str(HOME / ".local/share/fonts/custom/NotoSerifSC-Medium.otf")),
    ("Noto Serif CJK SC", str(HOME / ".local/share/fonts/custom/NotoSerifSC-Bold.otf")),
    ("Noto Serif CJK SC", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
    ("Noto Sans CJK SC", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ("DejaVu Serif", ""),
]

LATIN_FONT_CANDIDATES: List[Tuple[str, str]] = [
    ("DejaVu Serif", ""),
]


def _select_font(candidates: List[Tuple[str, str]]) -> str:
    for name, path in candidates:
        if path:
            p = Path(path)
            if p.exists():
                try:
                    font_manager.fontManager.addfont(str(p))
                    real_name = font_manager.FontProperties(fname=str(p)).get_name()
                    if real_name:
                        return real_name
                except Exception:
                    pass
        try:
            found_path = font_manager.findfont(name, fallback_to_default=False)
            if found_path:
                return name
        except Exception:
            continue
    print(f"[Font] 未找到候选字体，使用 {candidates[-1][0]} 作为兜底")
    return candidates[-1][0]


def configure_fonts():
    chinese = _select_font(CHINESE_FONT_CANDIDATES)
    latin = _select_font(LATIN_FONT_CANDIDATES)
    rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": [chinese, latin, "serif"],
            "font.serif": [latin, chinese, "DejaVu Serif"],
            "font.sans-serif": [chinese, latin, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": LABEL_SIZE,
            "ytick.labelsize": LABEL_SIZE,
            "legend.fontsize": LABEL_SIZE,
        }
    )


def load_metrics_summary(experiment_dir: Path) -> Optional[Dict]:
    """
    Load metrics_summary.json from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary with metrics or None if not found
    """
    # Try metrics/metrics_summary.json first (standard location)
    summary_path = experiment_dir / "metrics" / "metrics_summary.json"
    if not summary_path.exists():
        # Fallback to metrics_summary.json in root
        summary_path = experiment_dir / "metrics_summary.json"
        if not summary_path.exists():
            return None
    
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to load {summary_path}: {e}")
        return None


def find_experiment_dirs(results_dir: Path) -> Dict[str, List[Path]]:
    """
    Find all experiment directories for each similarity method.
    
    Args:
        results_dir: Root directory containing SimilarityComparison experiments
        
    Returns:
        Dictionary mapping method names to lists of experiment directories
    """
    method_dirs = {
        "answer2": [],
        "embedding": [],
        "hybrid": [],
        "random": [],
        "ragllm": [],
        "S1": [],
        "S2": [],
        "S3": [],
        "S4": [],
    }
    
    similarity_dir = results_dir / "SimilarityComparison"
    if not similarity_dir.exists():
        print(f"[ERROR] Directory {similarity_dir} does not exist")
        return method_dirs
    
    # Find all experiment directories
    for exp_dir in similarity_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        # Experiment ID format: SimilarityComparison_{method}_{timestamp}
        if exp_name.startswith("SimilarityComparison_"):
            parts = exp_name.split("_")
            if len(parts) >= 3:
                method = parts[1]  # Extract method name
                if method in method_dirs:
                    method_dirs[method].append(exp_dir)
    
    return method_dirs


def aggregate_method_metrics(method_dirs: List[Path]) -> Dict:
    """
    Aggregate metrics from multiple runs of the same method.
    
    Args:
        method_dirs: List of experiment directories for a method
        
    Returns:
        Dictionary with aggregated metrics (mean, std, min, max, count)
    """
    all_metrics = {
        "pc": [],
        "pec": [],
        "tcd": [],
        "bcm": []
    }
    
    for exp_dir in method_dirs:
        summary = load_metrics_summary(exp_dir)
        if summary is None:
            continue
        
        for metric in all_metrics.keys():
            value = summary.get(metric, 0.0)
            if isinstance(value, (int, float)):
                all_metrics[metric].append(float(value))
    
    # Compute statistics
    result = {}
    for metric, values in all_metrics.items():
        if values:
            result[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
                "values": values
            }
        else:
            result[metric] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
                "values": []
            }
    
    return result


def generate_comparison_report(all_method_metrics: Dict[str, Dict], output_dir: Path):
    """
    Generate a comparison report in JSON and Markdown formats.
    
    Args:
        all_method_metrics: Dictionary mapping method names to aggregated metrics
        output_dir: Directory to save reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON report
    json_path = output_dir / "comparison_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_method_metrics, f, indent=2, ensure_ascii=False)
    print(f"[CompareSimilarity] Saved JSON report: {json_path}")
    
    # Markdown report（中文叙述）
    md_path = output_dir / "comparison_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 相似度评分方法对比报告\n\n")
        f.write("本报告基于四个多样性指标（PCE/BCE/DPE/CCE）比较不同相似度评分策略。\n\n")
        
        f.write("## 方法列表\n\n")
        for method, display_name in METHOD_NAMES_CN.items():
            if method in all_method_metrics:
                f.write(f"- **{display_name}**（{method}）\n")
        f.write("\n")
        
        f.write("## 指标说明\n\n")
        for metric, display_name in METRIC_NAMES_CN.items():
            abbr = METRIC_ABBR.get(metric, metric.upper())
            f.write(f"- **{display_name}**（{abbr}）\n")
        f.write("\n")
        
        f.write("## 结果汇总（均值±标准差）\n\n")
        f.write("| 方法 | PCE | BCE | DPE | CCE | 运行次数 |\n")
        f.write("|------|-----|-----|-----|-----|---------|\n")
        
        for method in ["answer2", "embedding", "feature", "hybrid"]:
            if method not in all_method_metrics:
                continue
            
            metrics = all_method_metrics[method]
            pc = metrics.get("pc", {})
            pec = metrics.get("pec", {})
            tcd = metrics.get("tcd", {})
            bcm = metrics.get("bcm", {})
            
            display_name = METHOD_NAMES_CN.get(method, method)
            f.write(
                f"| {display_name} | "
                f"{pc.get('mean', 0):.3f}±{pc.get('std', 0):.3f} | "
                f"{pec.get('mean', 0):.3f}±{pec.get('std', 0):.3f} | "
                f"{tcd.get('mean', 0):.3f}±{tcd.get('std', 0):.3f} | "
                f"{bcm.get('mean', 0):.3f}±{bcm.get('std', 0):.3f} | "
                f"{pc.get('count', 0)} |\n"
            )
        
        f.write("\n## 细节统计\n\n")
        for method in ["answer2", "embedding", "feature", "hybrid"]:
            if method not in all_method_metrics:
                continue
            
            display_name = METHOD_NAMES_CN.get(method, method)
            f.write(f"### {display_name}\n\n")
            metrics = all_method_metrics[method]
            
            for metric_name, display_name_metric in METRIC_NAMES_CN.items():
                metric_data = metrics.get(metric_name, {})
                f.write(f"**{display_name_metric}**:\n")
                f.write(f"- 均值: {metric_data.get('mean', 0):.4f}\n")
                f.write(f"- 标准差: {metric_data.get('std', 0):.4f}\n")
                f.write(f"- 最小值: {metric_data.get('min', 0):.4f}\n")
                f.write(f"- 最大值: {metric_data.get('max', 0):.4f}\n")
                f.write(f"- 运行次数: {metric_data.get('count', 0)}\n")
                f.write("\n")
    
    print(f"[CompareSimilarity] Saved Markdown report: {md_path}")


def generate_comparison_figures(all_method_metrics: Dict[str, Dict], output_dir: Path):
    """
    Generate comparison figures for the four metrics.
    
    Args:
        all_method_metrics: Dictionary mapping method names to aggregated metrics
        output_dir: Directory to save figures
    """
    if not HAS_MATPLOTLIB:
        print("[WARNING] matplotlib not available, skipping figure generation")
        return
    
    configure_fonts()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "comparison_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figures_dir / "all_figures.pdf"
    pdf_pages = PdfPages(pdf_path)
    
    method_order = ["answer2", "ragllm", "embedding", "hybrid", "random", "S1", "S2", "S3", "S4"]
    methods = [m for m in method_order if m in all_method_metrics]
    if not methods:
        print("[WARNING] No methods found for figure generation")
        return
    
    # Prepare data for plotting
    method_labels = [METHOD_NAMES_CN.get(m, m) for m in methods]
    metrics_data = {
        "pc": {"means": [], "stds": [], "name": METRIC_ABBR["pc"], "slug": METRIC_FILE_SLUG["pc"]},
        "pec": {"means": [], "stds": [], "name": METRIC_ABBR["pec"], "slug": METRIC_FILE_SLUG["pec"]},
        "tcd": {"means": [], "stds": [], "name": METRIC_ABBR["tcd"], "slug": METRIC_FILE_SLUG["tcd"]},
        "bcm": {"means": [], "stds": [], "name": METRIC_ABBR["bcm"], "slug": METRIC_FILE_SLUG["bcm"]},
    }
    
    for method in methods:
        metrics = all_method_metrics[method]
        for metric_name in metrics_data.keys():
            metric_data = metrics.get(metric_name, {})
            metrics_data[metric_name]["means"].append(metric_data.get("mean", 0.0))
            metrics_data[metric_name]["stds"].append(metric_data.get("std", 0.0))
    
    # Create bar chart for each metric (矢量 PDF，无图内标题)
    for metric_name, metric_info in metrics_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        means = metric_info["means"]
        stds = metric_info["stds"]
        
        x_pos = np.arange(len(methods))
        has_std = any(s > 1e-9 for s in stds)
        ax.bar(
            x_pos,
            means,
            yerr=stds if has_std else None,
            capsize=5 if has_std else 0,
            alpha=0.8,
            color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(methods)],
        )
        
        ax.set_xlabel('相似度评分方法')
        if metric_name == "pc":
            ax.set_ylabel(f'{metric_info["name"]}')
        else:
            ax.set_ylabel(f'{metric_info["name"]}（更大更好）')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_labels, rotation=0, ha='center')
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis='y', alpha=0.2)
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + (std if has_std else 0) + 0.015, f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        slug = metric_info.get("slug", METRIC_FILE_SLUG.get(metric_name, metric_name))
        fig_path = figures_dir / f"{slug}_comparison.pdf"
        plt.savefig(fig_path, bbox_inches='tight')
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close()
        print(f"[CompareSimilarity] Saved figure: {fig_path}")
    
    # Create combined radar chart
    try:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Normalize metrics to [0, 1] for radar chart
        all_values = []
        for metric_name in metrics_data.keys():
            all_values.extend(metrics_data[metric_name]["means"])
        max_val = max(all_values) if all_values else 1.0
        min_val = min(all_values) if all_values else 0.0
        range_val = max_val - min_val if max_val > min_val else 1.0
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_data), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, method in enumerate(methods):
            values = []
            for metric_name in metrics_data.keys():
                mean = metrics_data[metric_name]["means"][i]
                normalized = (mean - min_val) / range_val if range_val > 0 else 0.0
                values.append(normalized)
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=METHOD_NAMES_CN.get(method, method))
            ax.fill(angles, values, alpha=0.2)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([METRIC_ABBR[m] for m in metrics_data.keys()])
        ax.set_ylim(0, 1)
        # no in-figure title; rely on external caption
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05))
        ax.grid(True)
        
        plt.tight_layout()
        fig_path = figures_dir / "radar_comparison.pdf"
        plt.savefig(fig_path, bbox_inches='tight')
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close()
        print(f"[CompareSimilarity] Saved figure: {fig_path}")
    except Exception as e:
        print(f"[WARNING] Failed to generate radar chart: {e}")
    
    pdf_pages.close()
    print(f"[CompareSimilarity] Combined PDF -> {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare similarity scoring methods based on diversity metrics."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing SimilarityComparison experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reports/similarity_comparison",
        help="Directory to save comparison reports and figures",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Results directory {results_dir} does not exist")
        return
    
    # Find all experiment directories
    method_dirs = find_experiment_dirs(results_dir)
    
    # Aggregate metrics for each method
    all_method_metrics = {}
    for method, exp_dirs in method_dirs.items():
        if not exp_dirs:
            print(f"[WARNING] No experiments found for method: {method}")
            continue
        
        print(f"[CompareSimilarity] Found {len(exp_dirs)} experiment(s) for {method}")
        aggregated = aggregate_method_metrics(exp_dirs)
        all_method_metrics[method] = aggregated
    
    if not all_method_metrics:
        print("[ERROR] No metrics found for any method")
        return
    
    # Generate reports
    output_dir = Path(args.output_dir)
    generate_comparison_report(all_method_metrics, output_dir)
    generate_comparison_figures(all_method_metrics, output_dir)
    
    print(f"\n[CompareSimilarity] Comparison complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


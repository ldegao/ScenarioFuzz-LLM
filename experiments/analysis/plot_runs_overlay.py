#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对多条实验记录，将同一指标的折线叠加绘制（累计/增量），输出中文 PNG 与 PDF。

用法示例：
  python -m experiments.analysis.plot_runs_overlay \
    --run "基线=S1_base_no_rag:experiments/runs/ScenarioFuzz-LLM/S1_base_no_rag" \
    --run "禁用相似度=S3_disable_similarity:experiments/runs/ScenarioFuzz-LLM/S3_disable_similarity" \
    --run "禁用相似度+引导=S4_disable_similarity_guided:experiments/runs/ScenarioFuzz-LLM/S4_disable_similarity_guided" \
    --run "随机变异=S4_disable_similarity_guided_2:experiments/runs/ScenarioFuzz-LLM/S4_disable_similarity_guided_2" \
    --output-dir experiments/runs/ScenarioFuzz-LLM/overlay_compare
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not available; skipping figure generation.")

from matplotlib.backends.backend_pdf import PdfPages

HOME = Path.home()
METRICS = ["pc", "pec", "tcd", "bcm"]
METRIC_LABELS = {
    "pc": "PCE",
    "pec": "BCE",
    "tcd": "DPE",
    "bcm": "CCE",
}
TITLE_SIZE = 10.5  # 五号
LABEL_SIZE = 9     # 小五

# line/marker map to ensure BW-friendly distinction
_LINESTYLES = [
    ("solid", "o"),
    ("solid", "^"),
    ("solid", "s"),
    ("solid", "x"),
]

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


def load_records(records_path: Path) -> List[Dict]:
    records = []
    if not records_path.exists():
        return records
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _normalize_label(label: str) -> str:
    norm = label.lower()
    if norm in {"基线", "s1_base_no_rag"}:
        return "RAG-ScenarioFuzz-LLM"
    if norm in {"禁用相似度", "s3_disable_similarity"}:
        return "RAG-ScenarioFuzz-LLM（无相似度模块）"
    if norm in {"禁用相似度+引导", "s4_disable_similarity_guided"}:
        return "RAG-ScenarioFuzz-LLM（无 LLM 变异引导）"
    if norm in {"随机变异", "s4_disable_similarity_guided_2"}:
        return "随机变异基线（非 RAG）"
    return label


def _linestyle(label: str) -> str:
    #统一使用实线，保证连续性
    return "solid"


def _marker(label: str) -> str:
    labels = sorted(list(_LINESTYLES))
    idx = abs(hash(label)) % len(_LINESTYLES)
    return _LINESTYLES[idx][1]


def parse_run_arg(arg: str) -> Tuple[str, Path]:
    """
    接收形如 '标签=别名:路径' 或 '标签=路径' 或 '路径' 的输入。
    优先使用左侧标签，其次目录名。
    """
    label = None
    path_str = arg
    if "=" in arg:
        label, path_str = arg.split("=", 1)
    if ":" in path_str and not Path(path_str).exists():
        # 允许使用 Label:Path 的形式
        lbl, maybe_path = path_str.split(":", 1)
        label = label or lbl
        path_str = maybe_path
    path = Path(path_str).resolve()
    if label is None:
        label = path.name
    label = _normalize_label(label)
    return label, path


def plot_overlay(run_records: Dict[str, List[Dict]], output_dir: Path):
    if not HAS_MPL:
        print("[PlotRunsOverlay] matplotlib not available; abort.")
        return
    configure_fonts()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_pdf = output_dir / "overlay_all_metrics.pdf"
    pdf_pages = PdfPages(combined_pdf)

    # 单指标图（直接矢量 PDF）
    for metric in METRICS:
        metric_label = METRIC_LABELS.get(metric, metric.upper())

        # cumulative
        fig_cum, ax_cum = plt.subplots(figsize=(8, 5))
        for label, records in run_records.items():
            if not records:
                continue
            x = [rec.get("num_accumulated_scenarios", idx + 1) for idx, rec in enumerate(records)]
            y = [float(rec.get(metric, 0.0)) for rec in records]
            ax_cum.plot(
                x,
                y,
                label=label,
                linewidth=1.2,
                linestyle=_linestyle(label),
            )
        ax_cum.set_xlabel("场景数")
        if metric == "pc":
            ax_cum.set_ylabel(f"{metric_label}")
        else:
            ax_cum.set_ylabel(f"{metric_label}（更大更好）")
        ax_cum.set_ylim(0, 1)
        ax_cum.set_yticks(np.linspace(0, 1, 6))
        ax_cum.legend()
        ax_cum.grid(axis='y', linestyle="--", alpha=0.2)
        fig_cum.tight_layout()
        pdf_cum = output_dir / f"overlay_cumulative_{metric}.pdf"
        fig_cum.savefig(pdf_cum, bbox_inches="tight")
        pdf_pages.savefig(fig_cum, bbox_inches="tight")
        plt.close(fig_cum)

        # incremental
        inc_key = f"{metric}_incremental"
        fig_inc, ax_inc = plt.subplots(figsize=(8, 5))
        for label, records in run_records.items():
            if not records:
                continue
            x = [rec.get("num_accumulated_scenarios", idx + 1) for idx, rec in enumerate(records)]
            y = [float(rec.get(inc_key, 0.0)) for rec in records]
            ax_inc.plot(
                x,
                y,
                label=label,
                linewidth=1.2,
                linestyle=_linestyle(label),
            )
        ax_inc.set_xlabel("场景数")
        if metric == "pc":
            ax_inc.set_ylabel(f"{metric_label}")
        else:
            ax_inc.set_ylabel(f"{metric_label}（更大更好）")
        ax_inc.set_ylim(0, 1)
        ax_inc.set_yticks(np.linspace(0, 1, 6))
        ax_inc.legend()
        ax_inc.grid(axis='y', linestyle="--", alpha=0.2)
        fig_inc.tight_layout()
        pdf_inc = output_dir / f"overlay_incremental_{metric}.pdf"
        fig_inc.savefig(pdf_inc, bbox_inches="tight")
        pdf_pages.savefig(fig_inc, bbox_inches="tight")
        plt.close(fig_inc)

    print(f"[PlotRunsOverlay] Saved figures to {output_dir}")

    # 汇总排版：累计 1x4 横排；增量 2x2
    def _plot_grid(kind: str, layout: Tuple[int, int], filename: str):
        metrics_for_grid = [m for m in METRICS if m != "pc"]
        rows, cols = layout
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols + 2, 3.2 * rows))
        axes = axes.flatten() if isinstance(axes, (list, tuple)) else axes.ravel()
        legend_handles = []
        legend_labels = []
        for idx, metric in enumerate(metrics_for_grid):
            if idx >= len(axes):
                break
            ax = axes[idx]
            metric_label = METRIC_LABELS.get(metric, metric.upper())
            key = metric if kind == "cumulative" else f"{metric}_incremental"
            y_label = metric_label
            for label, records in run_records.items():
                if not records:
                    continue
                x = [rec.get("num_accumulated_scenarios", i + 1) for i, rec in enumerate(records)]
                y = [float(rec.get(key, 0.0)) for rec in records]
                ax.plot(
                    x,
                    y,
                    label=label,
                    linewidth=1.2,
                    linestyle=_linestyle(label),
                )
            ax.set_ylabel(f"{y_label}（更大更好）")
            ax.set_ylim(0, 1)
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.grid(axis='y', linestyle="--", alpha=0.2)
            # 仅底行显示 x 轴标签
            if idx >= cols * (rows - 1):
                ax.set_xlabel("场景数")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            if not legend_handles and ax.get_legend_handles_labels()[0]:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
        # 清理多余子图
        for j in range(len(metrics_for_grid), len(axes)):
            axes[j].axis("off")
        if legend_handles and legend_labels:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.04),
                ncol=min(len(legend_labels), 4),
                frameon=False,
            )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        pdf_path = output_dir / f"{filename}.pdf"
        fig.savefig(pdf_path, bbox_inches='tight')
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # 专门绘制 PCE 单图（累计/增量）
    def _plot_single_pc(kind: str, filename: str):
        fig, ax = plt.subplots(figsize=(6, 4))
        key = "pc" if kind == "cumulative" else "pc_incremental"
        for label, records in run_records.items():
            if not records:
                continue
            x = [rec.get("num_accumulated_scenarios", i + 1) for i, rec in enumerate(records)]
            y = [float(rec.get(key, 0.0)) for rec in records]
            ax.plot(x, y, label=label, marker=_marker(label), markersize=3, linewidth=1.4, linestyle=_linestyle(label))
        ax.set_xlabel("场景数")
        ax.set_ylabel("PCE")
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis='y', linestyle="--", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        pdf_path = output_dir / f"{filename}.pdf"
        fig.savefig(pdf_path, bbox_inches='tight')
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    _plot_grid("cumulative", (1, 3), "overlay_cumulative_grid_nonpc")
    _plot_grid("cumulative", (1, 3), "overlay_incremental_grid_nonpc")  # call later for incremental? adjust below
    _plot_single_pc("cumulative", "overlay_cumulative_pc")
    _plot_single_pc("incremental", "overlay_incremental_pc")
    _plot_grid("incremental", (1, 3), "overlay_incremental_grid_nonpc")
    pdf_pages.close()
    print(f"[PlotRunsOverlay] Combined PDF -> {combined_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description="叠加绘制多实验的增量/累计折线（每个指标一张图，中文输出）"
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="形如 label=path 或 path；可多次提供。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reports/overlay_runs",
        help="输出目录",
    )
    args = parser.parse_args()

    runs: Dict[str, List[Dict]] = {}
    for run_arg in args.run:
        label, run_path = parse_run_arg(run_arg)
        records_path = run_path / "metrics" / "metrics_records.jsonl"
        records = load_records(records_path)
        if not records:
            print(f"[PlotRunsOverlay] 无记录：{records_path}")
        runs[label] = records

    output_dir = Path(args.output_dir).resolve()
    plot_overlay(runs, output_dir)


if __name__ == "__main__":
    main()


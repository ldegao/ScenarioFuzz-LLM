#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot per-metric curves with all similarity methods on the same figure.

For each metric (PCE/BCE/DPE/CCE), generates:
  - cumulative_<metric>_compare.png
  - incremental_<metric>_compare.png

Usage example:
  python3 -m experiments.analysis.plot_incremental_compare \
    --results-dir ./experiment_results/SimilarityComparison \
    --output-dir ./reports/similarity_comparison/incremental_compare

Assumptions:
  - Each method directory follows naming: SimilarityComparison_<method>_<timestamp>
  - metrics/metrics_records.jsonl exists per experiment
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not available; skipping figure generation.")
METHODS = ["answer2", "embedding", "feature", "hybrid"]
METRICS = ["pc", "pec", "tcd", "bcm"]
METRIC_LABELS = {
    "pc": "PCE 参数配置熵",
    "pec": "BCE 行为类别熵",
    "tcd": "DPE 驾驶模式熵",
    "bcm": "CCE 组合覆盖熵",
}
METRIC_SLUGS = {
    "pc": "bpc",
    "pec": "dbcc",
    "tcd": "dpd",
    "bcm": "bcm",
}
METHOD_LABELS = {
    "answer2": "Answer2（LLM相似度）",
    "embedding": "Embedding 相似度",
    "feature": "特征距离",
    "hybrid": "混合策略",
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
    """Return preferred font if available; otherwise the first available fallback."""
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


def find_latest_run(results_dir: Path, method: str) -> Path:
    base = results_dir / "SimilarityComparison"
    candidates = sorted(
        [d for d in base.glob(f"SimilarityComparison_{method}_*") if d.is_dir()],
        reverse=True,
    )
    return candidates[0] if candidates else None


def _save_pdf_from_png(png_path: Path, pdf_path: Path):
    if not HAS_PIL:
        print(f"[PlotIncrementalCompare] Pillow 未安装，跳过 PDF 生成: {pdf_path}")
        return
    with Image.open(png_path) as img:
        rgb = img.convert("RGB")
        rgb.save(pdf_path, "PDF")


def plot_compare(method_records: Dict[str, List[Dict]], output_dir: Path):
    if not HAS_MPL:
        return
    configure_fonts()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_pdf = output_dir / "incremental_compare_all.pdf"
    pdf_pages = PdfPages(combined_pdf)

    for metric in METRICS:
        metric_label = METRIC_LABELS.get(metric, metric.upper())

        # cumulative
        fig_cum, ax_cum = plt.subplots(figsize=(8, 5))
        for method, records in method_records.items():
            if not records:
                continue
            x = [rec.get("num_accumulated_scenarios", idx + 1) for idx, rec in enumerate(records)]
            y = [float(rec.get(metric, 0.0)) for rec in records]
            ax_cum.plot(x, y, label=METHOD_LABELS.get(method, method), marker="o", linewidth=1.6)
        ax_cum.set_xlabel("累积场景数")
        ax_cum.set_ylabel(f"{metric_label}（累计值）")
        ax_cum.set_title(f"{metric_label}累计（全部方法）", fontweight="bold")
        ax_cum.legend()
        ax_cum.grid(True, linestyle="--", alpha=0.5)
        fig_cum.tight_layout()
        slug = METRIC_SLUGS.get(metric, metric)
        pdf_cum = output_dir / f"cumulative_{slug}_compare.pdf"
        fig_cum.savefig(pdf_cum, bbox_inches="tight")
        pdf_pages.savefig(fig_cum, bbox_inches="tight")
        plt.close(fig_cum)

        # incremental
        inc_key = f"{metric}_incremental"
        fig_inc, ax_inc = plt.subplots(figsize=(8, 5))
        for method, records in method_records.items():
            if not records:
                continue
            x = [rec.get("num_accumulated_scenarios", idx + 1) for idx, rec in enumerate(records)]
            y = [float(rec.get(inc_key, 0.0)) for rec in records]
            ax_inc.plot(x, y, label=METHOD_LABELS.get(method, method), marker="o", linewidth=1.6)
        ax_inc.set_xlabel("累积场景数")
        ax_inc.set_ylabel(f"{metric_label}（增量贡献）")
        ax_inc.set_title(f"{metric_label}增量（全部方法）", fontweight="bold")
        ax_inc.legend()
        ax_inc.grid(True, linestyle="--", alpha=0.5)
        fig_inc.tight_layout()
        pdf_inc = output_dir / f"incremental_{slug}_compare.pdf"
        fig_inc.savefig(pdf_inc, bbox_inches="tight")
        pdf_pages.savefig(fig_inc, bbox_inches="tight")
        plt.close(fig_inc)

    print(f"[PlotIncrementalCompare] Saved figures to {output_dir}")
    pdf_pages.close()
    print(f"[PlotIncrementalCompare] Combined PDF -> {combined_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Compare incremental/cumulative metrics across methods on per-metric plots.")
    parser.add_argument("--results-dir", type=str, required=True, help="Root results dir (contains SimilarityComparison/...)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for comparison plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    method_records: Dict[str, List[Dict]] = {}
    for method in METHODS:
        exp_dir = find_latest_run(results_dir, method)
        if not exp_dir:
            print(f"[PlotIncrementalCompare] No experiment found for method {method}")
            method_records[method] = []
            continue
        records_path = exp_dir / "metrics" / "metrics_records.jsonl"
        records = load_records(records_path)
        if not records:
            print(f"[PlotIncrementalCompare] No records for {method} at {records_path}")
        method_records[method] = records

    plot_compare(method_records, output_dir)


if __name__ == "__main__":
    main()


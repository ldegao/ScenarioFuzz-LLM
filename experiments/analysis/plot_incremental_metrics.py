#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot incremental and cumulative metrics from metrics_records.jsonl.

Usage examples:
  python -m experiments.analysis.plot_incremental_metrics \
    --experiment-dir ./experiment_results/SimilarityComparison/SimilarityComparison_hybrid_20251217_084906 \
    --output-dir ./reports/similarity_comparison/incremental/hybrid

  python -m experiments.analysis.plot_incremental_metrics \
    --records-file ./experiment_results/SimilarityComparison/SimilarityComparison_hybrid_20251217_084906/metrics/metrics_records.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not available; skipping figure generation.")

METRIC_KEYS = ["pc", "pec", "tcd", "bcm"]
METRIC_LABELS = {
    "pc": "PCE 参数配置熵",
    "pec": "BCE 行为类别熵",
    "tcd": "DPE 驾驶模式熵",
    "bcm": "CCE 组合覆盖熵",
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


def load_records(records_path: Path) -> List[Dict]:
    records = []
    if not records_path.exists():
        raise FileNotFoundError(f"{records_path} not found")
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


def _save_pdf_from_png(png_path: Path, pdf_path: Path):
    if not HAS_PIL:
        print(f"[PlotIncremental] Pillow 未安装，跳过 PDF 生成: {pdf_path}")
        return
    with Image.open(png_path) as img:
        rgb = img.convert("RGB")
        rgb.save(pdf_path, "PDF")


def plot_metrics(records: List[Dict], output_dir: Path):
    if not HAS_MATPLOTLIB:
        return
    configure_fonts()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_pdf = output_dir / "incremental_metrics.pdf"
    pdf_pages = PdfPages(combined_pdf)

    x = [rec.get("num_accumulated_scenarios", idx + 1) for idx, rec in enumerate(records)]

    for key in METRIC_KEYS:
        metric_label = METRIC_LABELS.get(key, key.upper())
        y_cum = [float(rec.get(key, 0.0)) for rec in records]
        fig_cum, ax_cum = plt.subplots(figsize=(8, 5))
        ax_cum.plot(x, y_cum, label=f"{metric_label}（累计）", color="C0", linewidth=1.6)
        ax_cum.set_xlabel("累积场景数")
        ax_cum.set_ylabel(f"{metric_label}（累计值）")
        ax_cum.set_title(f"{metric_label}累计随场景变化", fontweight="bold")
        ax_cum.grid(True, linestyle="--", alpha=0.5)
        ax_cum.legend()
        fig_cum.tight_layout()
        pdf_cum = output_dir / f"cumulative_{key}.pdf"
        fig_cum.savefig(pdf_cum, bbox_inches="tight")
        pdf_pages.savefig(fig_cum, bbox_inches="tight")
        plt.close(fig_cum)

        inc_key = f"{key}_incremental"
        y_inc = [float(rec.get(inc_key, 0.0)) for rec in records]
        fig_inc, ax_inc = plt.subplots(figsize=(8, 5))
        ax_inc.plot(x, y_inc, label=f"{metric_label}（增量）", color="C1", linewidth=1.6)
        ax_inc.set_xlabel("累积场景数")
        ax_inc.set_ylabel(f"{metric_label}（增量贡献）")
        ax_inc.set_title(f"{metric_label}单场景增量", fontweight="bold")
        ax_inc.grid(True, linestyle="--", alpha=0.5)
        ax_inc.legend()
        fig_inc.tight_layout()
        pdf_inc = output_dir / f"incremental_{key}.pdf"
        fig_inc.savefig(pdf_inc, bbox_inches="tight")
        pdf_pages.savefig(fig_inc, bbox_inches="tight")
        plt.close(fig_inc)

    pdf_pages.close()
    print(f"[PlotIncremental] Saved figures to {output_dir}")
    print(f"[PlotIncremental] Combined PDF -> {combined_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Plot incremental/cumulative metrics from metrics_records.jsonl")
    parser.add_argument("--experiment-dir", type=str, default=None, help="Experiment directory containing metrics/")
    parser.add_argument("--records-file", type=str, default=None, help="Path to metrics_records.jsonl")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    records_path = None
    if args.records_file:
        records_path = Path(args.records_file)
    elif args.experiment_dir:
        records_path = Path(args.experiment_dir) / "metrics" / "metrics_records.jsonl"
    else:
        raise ValueError("Either --experiment-dir or --records-file must be provided")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        base_dir = Path(args.experiment_dir) if args.experiment_dir else records_path.parent
        output_dir = base_dir / "metrics" / "incremental_plots"

    records = load_records(records_path)
    if not records:
        print(f"[PlotIncremental] No records found in {records_path}")
        return

    plot_metrics(records, output_dir)


if __name__ == "__main__":
    main()


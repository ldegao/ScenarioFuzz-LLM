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
from typing import List, Dict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not available; skipping figure generation.")


METRIC_KEYS = ["pc", "pec", "tcd", "bcm"]


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


def plot_metrics(records: List[Dict], output_dir: Path):
    if not HAS_MATPLOTLIB:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    x = [rec.get("num_accumulated_scenarios", idx + 1) for idx, rec in enumerate(records)]

    # Per-metric figures (cumulative & incremental)
    for key in METRIC_KEYS:
        y_cum = [float(rec.get(key, 0.0)) for rec in records]
        plt.figure(figsize=(8, 5))
        plt.plot(x, y_cum, label=f"{key.upper()} cumulative", color="C0")
        plt.xlabel("Accumulated scenarios")
        plt.ylabel("Cumulative value")
        plt.title(f"Cumulative {key.upper()} over scenarios")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / f"cumulative_{key}.png")
        plt.close()

        inc_key = f"{key}_incremental"
        y_inc = [float(rec.get(inc_key, 0.0)) for rec in records]
        plt.figure(figsize=(8, 5))
        plt.plot(x, y_inc, label=f"{inc_key.upper()}", color="C1")
        plt.xlabel("Accumulated scenarios")
        plt.ylabel("Incremental contribution")
        plt.title(f"Incremental {key.upper()} per scenario")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / f"incremental_{key}.png")
        plt.close()

    print(f"[PlotIncremental] Saved figures to {output_dir}")


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


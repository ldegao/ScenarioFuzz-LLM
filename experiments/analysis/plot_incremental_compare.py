#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot per-metric curves with all similarity methods on the same figure.

For each metric (PC/PEC/TCD/BCM), generates:
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
from typing import Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not available; skipping figure generation.")

METHODS = ["answer2", "embedding", "feature", "hybrid"]
METRICS = ["pc", "pec", "tcd", "bcm"]


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


def plot_compare(method_records: Dict[str, List[Dict]], output_dir: Path):
    if not HAS_MPL:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        # cumulative
        plt.figure(figsize=(8, 5))
        for method, records in method_records.items():
            if not records:
                continue
            x = [rec.get("num_accumulated_scenarios", idx + 1) for idx, rec in enumerate(records)]
            y = [float(rec.get(metric, 0.0)) for rec in records]
            plt.plot(x, y, label=method)
        plt.xlabel("Accumulated scenarios")
        plt.ylabel("Cumulative value")
        plt.title(f"{metric.upper()} cumulative (all methods)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / f"cumulative_{metric}_compare.png")
        plt.close()

        # incremental
        plt.figure(figsize=(8, 5))
        inc_key = f"{metric}_incremental"
        for method, records in method_records.items():
            if not records:
                continue
            x = [rec.get("num_accumulated_scenarios", idx + 1) for idx, rec in enumerate(records)]
            y = [float(rec.get(inc_key, 0.0)) for rec in records]
            plt.plot(x, y, label=method)
        plt.xlabel("Accumulated scenarios")
        plt.ylabel("Incremental contribution")
        plt.title(f"{metric.upper()} incremental (all methods)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / f"incremental_{metric}_compare.png")
        plt.close()

    print(f"[PlotIncrementalCompare] Saved figures to {output_dir}")


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


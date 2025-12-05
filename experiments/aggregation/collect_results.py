#!/usr/bin/env python3
"""
Collect metrics summaries from multiple experiments and methods.

This script scans an experiment results root directory for
`metrics_summary.json` files and aggregates them into a single
`all_methods_results.json` structure:

{
  "ScenarioFuzz-LLM": [ {..summary..}, ... ],
  "RAG-ScenarioFuzz": [ {..summary..}, ... ],
  "TM-Fuzzer":        [ {..summary..}, ... ]
}
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def find_summary_files(root: Path) -> List[Path]:
    """
    Recursively find all metrics_summary.json files under root.
    """
    return list(root.rglob("metrics_summary.json"))


def collect_method_results(summary_files: List[Path]) -> Dict:
    """
    Load and group metrics summaries by method.
    """
    results: Dict[str, List[dict]] = {}

    for path in summary_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception as e:
            print(f"[CollectResults] WARNING: Failed to load {path}: {e}")
            continue

        method = summary.get("method") or "Unknown"
        results.setdefault(method, []).append(summary)

    return results


def save_all_methods_results(results: Dict, output_path: Path) -> Path:
    """
    Save aggregated results to JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[CollectResults] Saved all methods results to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Collect metrics_summary.json files into a unified all_methods_results.json"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./experiment_results",
        help="Root directory containing per-experiment results (default: ./experiment_results)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./experiment_results/all_methods_results.json",
        help="Output JSON file for aggregated results",
    )

    args = parser.parse_args()

    root = Path(args.root)
    output_path = Path(args.output)

    if not root.exists():
        print(f"[CollectResults] Root directory {root} does not exist, nothing to collect.")
        return

    summary_files = find_summary_files(root)
    if not summary_files:
        print(f"[CollectResults] No metrics_summary.json files found under {root}")
        return

    print(f"[CollectResults] Found {len(summary_files)} metrics_summary.json files.")
    results = collect_method_results(summary_files)
    save_all_methods_results(results, output_path)


if __name__ == "__main__":
    main()



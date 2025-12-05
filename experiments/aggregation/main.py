#!/usr/bin/env python3
"""
Main entrypoint for the aggregation part of the experiments.

Running this script will scan the experiment results root for
`metrics_summary.json` files and produce a unified
`all_methods_results.json` file.
"""

from pathlib import Path

from .collect_results import main as collect_main


if __name__ == "__main__":
    collect_main()



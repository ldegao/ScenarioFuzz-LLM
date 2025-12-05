#!/usr/bin/env python3
"""
Main entrypoint for the analysis part of the experiments.

Running this script will:
  1. Read aggregated metrics from all_methods_results.json
  2. Generate figures into ./reports/figs
  3. Generate Markdown/JSON reports into ./reports
"""

from .generate_figures import main as figures_main
from .generate_reports import main as reports_main


def main():
    # First generate figures, then reports (both read the same results file)
    figures_main()
    reports_main()


if __name__ == "__main__":
    main()



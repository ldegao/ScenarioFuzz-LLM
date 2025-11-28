#!/usr/bin/env python3
"""
Paper entrypoint for TM-Fuzzer baseline experiments.

Usage (module form, recommended):
  python -m experiments.runners.run_tmfuzzer ...
"""

from experiments.runners.tmfuzzer.runner import main


if __name__ == "__main__":
    main()



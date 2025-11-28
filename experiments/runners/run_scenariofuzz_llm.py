#!/usr/bin/env python3
"""
Paper entrypoint for ScenarioFuzz-LLM experiments.

Usage (module form, recommended):
  python -m experiments.runners.run_scenariofuzz_llm ...
"""

from experiments.runners.scenariofuzz_llm.runner import main


if __name__ == "__main__":
    main()



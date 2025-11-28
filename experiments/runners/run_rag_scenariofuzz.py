#!/usr/bin/env python3
"""
Paper entrypoint for RAG-ScenarioFuzz experiments.

Usage (module form, recommended):
  python -m experiments.runners.run_rag_scenariofuzz ...
"""

from experiments.runners.rag_scenariofuzz.runner import main


if __name__ == "__main__":
    main()



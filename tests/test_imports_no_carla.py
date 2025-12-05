"""
Basic import tests for modules that do not require the CARLA Python API.

These tests are primarily smoke tests to ensure that refactors
did not break module imports or top-level initialization logic.
"""

import importlib
import types


MODULES_NO_CARLA = [
    "config",
    "constants",
    "gpt",
    "states",
    "clip",
    "rag_module",
    "rag_module.knowledge_base",
    "rag_module.scenario_encoder",
    "rag_module.vector_store",
    "rag_module.rag_engine",
    "rag_module.enhanced_rag_engine",
    "rag_module.hybrid_retriever",
    "rag_module.reranker",
    # metrics module depends on scenario/npc which require CARLA;
    # we deliberately exclude them from the "no CARLA" import list.
    "visualization",
    "visualization.plot_metrics",
    "visualization.report_generator",
]


def test_import_modules_no_carla():
    """
    Ensure that all listed modules can be imported without raising exceptions.

    This helps catch issues such as missing dependencies or invalid top-level code.
    """
    for name in MODULES_NO_CARLA:
        module = importlib.import_module(name)
        assert isinstance(module, types.ModuleType)



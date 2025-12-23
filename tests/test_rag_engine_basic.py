"""
Basic functional tests for RAGEngine and EnhancedRAGEngine.

These tests use the built-in mock data and mock encoders/vector stores
when heavy dependencies (sentence-transformers, faiss, dtaidistance) are
not available, so they should be lightweight and self-contained.
"""

import pytest
import importlib.util

HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None
HAS_FAISS = importlib.util.find_spec("faiss") is not None

# Skip the whole module if optional heavy dependencies are missing
pytestmark = pytest.mark.skipif(
    not (HAS_SENTENCE_TRANSFORMERS and HAS_FAISS),
    reason="sentence-transformers or faiss not installed; skipping RAG tests."
)

from rag_module import (
    KnowledgeBase,
    ScenarioEncoder,
    VectorStore,
    RAGEngine,
    EnhancedRAGEngine,
    PRESET_MODELS,
)


def test_knowledge_base_mock_data():
    kb = KnowledgeBase()
    assert kb.size() == 0

    ok = kb.load_mock_data()
    assert ok
    assert kb.size() > 0

    descriptions = kb.get_scenario_descriptions()
    assert len(descriptions) == kb.size()
    assert all(isinstance(d, str) for d in descriptions)


def test_scenario_encoder_mock_or_real_model():
    encoder = ScenarioEncoder()
    vec = encoder.encode("a simple test scenario")
    assert vec is not None
    # Ensure we have a 1D vector with the expected dimension
    assert vec.ndim == 1
    assert encoder.get_vector_dim() == vec.shape[0]


def test_scenario_encoder_presets_load():
    """
    Ensure that the three preset encoders can be instantiated.

    If sentence-transformers is not installed in the environment,
    skip this test gracefully.
    """
    preset_keys = [
        "paraphrase-multilingual",
        "all-minilm",
        "multi-qa-mpnet",
    ]

    try:
        # Quick import check so we can skip if dependency is missing.
        import sentence_transformers  # noqa: F401
    except Exception:
        pytest.skip("sentence-transformers not available; skipping preset tests.")

    for key in preset_keys:
        assert key in PRESET_MODELS
        encoder = ScenarioEncoder(model_name=key)
        vec = encoder.encode("preset encoder test")
        assert vec is not None
        assert vec.ndim == 1
        assert encoder.get_vector_dim() == vec.shape[0]


def test_vector_store_build_and_search():
    encoder = ScenarioEncoder()
    kb = KnowledgeBase()
    kb.load_mock_data()

    texts = kb.get_scenario_descriptions()
    vectors = encoder.encode_batch(texts)

    vs = VectorStore(vector_dim=vectors.shape[1])
    vs.build_index(vectors, texts)

    # Search for the first text using its own vector
    query_vec = vectors[0]
    results = vs.search(query_vec, k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "text" in results[0]


def test_rag_engine_retrieval_and_format():
    rag = RAGEngine(top_k=3)
    rag.initialize(load_mock_data=True)

    seed = "A vehicle suddenly brakes in front of the ego vehicle"
    retrieved = rag.retrieve_relevant_scenarios(seed_scenario=seed, k=3)
    assert isinstance(retrieved, list)
    assert 0 < len(retrieved) <= 3

    scenario_dict_str = rag.retrieve_and_format_for_prompt(seed, k=3)
    assert isinstance(scenario_dict_str, str)
    # Should look like a dict literal
    assert scenario_dict_str.startswith("{")
    assert "0" in scenario_dict_str or "1" in scenario_dict_str


def test_enhanced_rag_engine_hybrid_search():
    # Use hybrid search but disable reranking to keep the test lightweight
    enhanced = EnhancedRAGEngine(
        top_k=3,
        use_hybrid_search=True,
        hybrid_alpha=0.7,
        use_reranking=False,
    )
    enhanced.initialize(load_mock_data=True)

    seed = "Dense fog reduces visibility on the highway"
    retrieved = enhanced.retrieve_relevant_scenarios(seed_scenario=seed, k=3)
    assert isinstance(retrieved, list)
    assert 0 < len(retrieved) <= 3

    formatted = enhanced.retrieve_and_format_for_prompt(seed, k=3)
    assert isinstance(formatted, str)
    assert formatted.startswith("{")


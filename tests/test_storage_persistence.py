"""
Tests for data migration and storage components:

- KnowledgeBase JSON save/load
- VectorStore index save/load
"""

import json

import numpy as np

from rag_module import KnowledgeBase, VectorStore


def test_knowledge_base_save_and_load(tmp_path):
    kb = KnowledgeBase()
    # Load built-in mock data and add an extra scenario
    kb.load_mock_data()
    extra = {
        "id": "custom_001",
        "description": "Custom scenario for persistence test",
        "type": "test",
        "risk_level": "low",
    }
    kb.add_scenario(extra)
    original_size = kb.size()

    out_file = tmp_path / "kb.json"
    ok = kb.save_to_json(str(out_file))
    assert ok
    assert out_file.exists()

    # Load into a fresh knowledge base
    kb2 = KnowledgeBase()
    ok2 = kb2.load_from_json(str(out_file))
    assert ok2
    assert kb2.size() == original_size

    descriptions = kb2.get_scenario_descriptions()
    assert any("Custom scenario for persistence test" in d for d in descriptions)


def test_vector_store_save_and_load(tmp_path):
    # Simple synthetic data
    texts = ["scenario A", "scenario B", "scenario C"]
    metadata = [{"id": 1}, {"id": 2}, {"id": 3}]
    vectors = np.eye(len(texts), dtype=np.float32)

    vs = VectorStore(vector_dim=vectors.shape[1])
    vs.build_index(vectors, texts, metadata)

    save_dir = tmp_path / "index"
    vs.save_index(str(save_dir))

    # Load into a new VectorStore
    vs2 = VectorStore(vector_dim=vectors.shape[1])
    vs2.load_index(str(save_dir))

    assert vs2.texts == texts
    assert vs2.metadata == metadata
    # Size should match number of texts, regardless of faiss availability
    assert vs2.size() == len(texts)



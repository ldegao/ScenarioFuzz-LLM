"""
Unit tests for GPT helper utilities and configuration objects.

These tests avoid actual network calls and only exercise pure-Python logic.
"""

import json
import os
from collections import OrderedDict
from types import SimpleNamespace

import gpt
from config import Config


def test_extract_json_success_and_failure():
    payload = {"a": 1, "b": 2}
    text = f"some prefix\n{json.dumps(payload)}\n some suffix"
    extracted = gpt.extract_json(text)
    assert extracted == payload

    # Invalid JSON should return None
    bad = gpt.extract_json("no json here")
    assert gpt.extract_json(bad or "") is None


def test_add_answer1_to_database_rotation():
    database = OrderedDict()
    max_size = 3

    for i in range(5):
        response_json = {"answer1": {"Description": f"scenario {i}"}}
        database = gpt.add_answer1_to_database(response_json, database, max_size=max_size)
        assert isinstance(database, OrderedDict)
        # Size should never exceed max_size
        assert len(database) <= max_size

    # Oldest entries should have been rotated out and newest kept
    descriptions = list(database.values())
    # Values in the database are dicts like {"Description": "..."}
    desc_texts = [d.get("Description", "") for d in descriptions]
    # Ensure we have exactly max_size entries
    assert len(desc_texts) == max_size
    # Newest scenario (scenario 4) should be present
    assert "scenario 4" in desc_texts


def test_get_frame_data_random_and_specific(tmp_path):
    data = {
        "min_dist_frame": 2,
        "0": {"v": 0},
        "1": {"v": 1},
        "2": {"v": 2},
    }
    p = tmp_path / "frames.json"
    p.write_text(json.dumps(data))

    frame = gpt.get_frame_data(str(p))
    assert isinstance(frame, dict)
    assert frame.get("v") == 2

    # Implementation always prefers `min_dist_frame` when present,
    # even if a default_frame_number is provided.
    frame2 = gpt.get_frame_data(str(p), default_frame_number=1)
    assert frame2.get("v") == 2


def test_config_defaults_and_rag_flags():
    conf = Config()
    # Basic defaults
    assert conf.enable_rag is False
    assert conf.rag_k == 5

    # Enhanced RAG flags added recently
    assert hasattr(conf, "use_enhanced_rag")
    assert hasattr(conf, "use_hybrid_search")
    assert hasattr(conf, "hybrid_alpha")
    assert hasattr(conf, "use_reranking")
    assert hasattr(conf, "reranker_model")



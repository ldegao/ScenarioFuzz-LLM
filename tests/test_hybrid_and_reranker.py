"""
Unit tests for HybridRetriever, BM25Retriever, and Reranker.

These tests verify that hybrid retrieval and reranking pipelines can run
end-to-end on small in-memory examples.
"""

import numpy as np
import importlib.util
import pytest

HAS_FAISS = importlib.util.find_spec("faiss") is not None

pytestmark = pytest.mark.skipif(
    not HAS_FAISS,
    reason="faiss not installed; skipping hybrid retriever tests."
)

from rag_module import VectorStore
from rag_module.hybrid_retriever import HybridRetriever, BM25Retriever
from rag_module.reranker import Reranker


def test_bm25_retriever_basic():
    docs = [
        "ego vehicle follows a leading car on a straight road",
        "heavy rain causes low visibility in the city",
        "a pedestrian crosses at an intersection",
    ]
    bm25 = BM25Retriever()
    bm25.fit(docs)

    results = bm25.retrieve("rain and low visibility", k=2)
    assert isinstance(results, list)
    # At least one relevant document should be returned
    assert len(results) >= 1
    idx, score = results[0]
    assert 0 <= idx < len(docs)
    assert score > 0


def test_hybrid_retriever_with_vector_store():
    texts = [
        "ego vehicle follows a leading car on a straight road",
        "heavy rain causes low visibility in the city",
        "a pedestrian crosses at an intersection",
    ]
    # Use simple numeric vectors for testing
    vectors = np.eye(len(texts), dtype=np.float32)

    vs = VectorStore(vector_dim=vectors.shape[1])
    vs.build_index(vectors, texts)

    hybrid = HybridRetriever(vector_store=vs, alpha=0.5)
    hybrid.fit_bm25(texts)

    query = "rain and low visibility"
    # Query vector close to doc index 1
    query_vec = vectors[1]
    results = hybrid.retrieve(query=query, query_vector=query_vec, k=2, use_hybrid=True)

    assert isinstance(results, list)
    assert len(results) > 0
    first = results[0]
    assert "text" in first and "score" in first


def test_reranker_simple_overlap():
    docs = [
        "ego vehicle follows a car",
        "heavy rain and fog",
        "pedestrian crossing",
    ]
    # Use SimpleReranker by disabling cross encoder
    reranker = Reranker(use_cross_encoder=False)
    results = reranker.rerank("heavy rain", docs, top_k=2)

    assert isinstance(results, list)
    assert len(results) == 2
    # Expect the most relevant doc to mention "rain"
    assert "rain" in results[0]["text"]



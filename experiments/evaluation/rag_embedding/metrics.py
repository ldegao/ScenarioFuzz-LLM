"""
Evaluation metrics for RAG embedding experiments.

We focus on:
  - Hit@k for high-similarity scenarios
  - nDCG@k with graded relevance (high/medium/low)
  - Same-class ratio@k by a chosen label key (e.g., accident_type)
"""

from __future__ import annotations

import math
from typing import Dict, List


def relevance_grade(label_i: Dict[str, str], label_j: Dict[str, str]) -> int:
    """
    Compute graded relevance between two scenario labels.

    Rules:
      - High (2): same accident_type and same environment
      - Medium (1): same accident_type with different environment, OR
                    same environment and same risk_level with different accident_type
      - Low (0): all other combinations
    """
    if not label_i or not label_j:
        return 0

    ai = label_i.get("accident_type")
    aj = label_j.get("accident_type")
    ei = label_i.get("environment")
    ej = label_j.get("environment")
    ri = label_i.get("risk_level")
    rj = label_j.get("risk_level")

    if ai == aj and ei == ej:
        return 2

    if ai == aj and ei != ej:
        return 1

    if ei == ej and ri == rj and ai != aj:
        return 1

    return 0


def compute_hit_at_k(
    labels: List[Dict[str, str]],
    retrieved_indices: List[List[int]],
    k: int,
    relevance_level: str = "high",
) -> float:
    """
    Compute Hit@k: fraction of queries whose top-k results contain
    at least one item with the desired relevance level.

    relevance_level:
      - "high": grade >= 2
      - "medium_or_high": grade >= 1
    """
    if not labels or not retrieved_indices:
        return 0.0

    if relevance_level == "high":
        threshold = 2
    elif relevance_level == "medium_or_high":
        threshold = 1
    else:
        threshold = 1

    num_queries = len(labels)
    hits = 0

    for i, indices in enumerate(retrieved_indices):
        if not indices:
            continue
        query_label = labels[i]
        top_k = indices[:k]
        found = False
        for j in top_k:
            if j < 0 or j >= len(labels):
                continue
            grade = relevance_grade(query_label, labels[j])
            if grade >= threshold:
                found = True
                break
        if found:
            hits += 1

    return hits / float(num_queries) if num_queries > 0 else 0.0


def _dcg(scores: List[float]) -> float:
    dcg = 0.0
    for idx, s in enumerate(scores):
        # rank positions are 1-based in the log denominator
        dcg += (2.0 ** s - 1.0) / math.log2(idx + 2.0)
    return dcg


def compute_ndcg_at_k(
    labels: List[Dict[str, str]],
    retrieved_indices: List[List[int]],
    k: int,
) -> float:
    """
    Compute mean nDCG@k across all queries.
    """
    if not labels or not retrieved_indices:
        return 0.0

    num_queries = len(labels)
    total_ndcg = 0.0

    for i, indices in enumerate(retrieved_indices):
        if not indices:
            continue

        q_label = labels[i]
        top_k = indices[:k]
        gains = []
        for j in top_k:
            if j < 0 or j >= len(labels):
                gains.append(0.0)
            else:
                gains.append(float(relevance_grade(q_label, labels[j])))

        dcg = _dcg(gains)
        ideal_gains = sorted(gains, reverse=True)
        idcg = _dcg(ideal_gains)

        if idcg <= 0.0:
            ndcg = 0.0
        else:
            ndcg = dcg / idcg

        total_ndcg += ndcg

    return total_ndcg / float(num_queries) if num_queries > 0 else 0.0


def compute_same_class_ratio(
    labels: List[Dict[str, str]],
    retrieved_indices: List[List[int]],
    k: int,
    key: str = "accident_type",
) -> float:
    """
    Compute the average proportion of top-k results that share the same
    class label (specified by `key`) with the query.
    """
    if not labels or not retrieved_indices:
        return 0.0

    num_queries = len(labels)
    total_ratio = 0.0

    for i, indices in enumerate(retrieved_indices):
        query_label = labels[i].get(key)
        if not indices:
            continue
        top_k = indices[:k]
        same = 0
        count = 0
        for j in top_k:
            if j < 0 or j >= len(labels):
                continue
            count += 1
            if labels[j].get(key) == query_label:
                same += 1
        if count > 0:
            total_ratio += same / float(count)

    return total_ratio / float(num_queries) if num_queries > 0 else 0.0



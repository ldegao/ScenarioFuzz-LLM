"""
CLI runner for RAG embedding evaluation experiments.

This script:
  - Builds or loads a synthetic scenario dataset (no CARLA required)
  - Encodes scenario descriptions using different ScenarioEncoder backbones
  - Builds vector and hybrid (vector + BM25) retrieval backends
  - Evaluates retrieval quality using Hit@5, nDCG@10, and same-class ratio@10
  - Saves a JSON summary into experiment_results/rag_embedding_eval
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from rag_module import (
    ScenarioEncoder,
    VectorStore,
    HybridRetriever,
    PRESET_MODELS,
)
from experiments.evaluation.rag_embedding.dataset import (
    build_or_load_dataset,
    extract_descriptions_and_labels,
)
from experiments.evaluation.rag_embedding.metrics import (
    compute_hit_at_k,
    compute_ndcg_at_k,
    compute_same_class_ratio,
)


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def default_output_root() -> str:
    root = _project_root()
    out_dir = os.path.join(root, "experiment_results", "rag_embedding_eval")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline RAG embedding benchmark (no CARLA required).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=1000,
        help="Number of synthetic scenarios to generate / load.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k to retrieve for evaluation.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=default_output_root(),
        help="Directory to store evaluation results.",
    )
    parser.add_argument(
        "--encoders",
        type=str,
        nargs="*",
        default=["paraphrase-multilingual", "all-minilm", "multi-qa-mpnet"],
        help=(
            "Encoder presets to evaluate. Each must be a key in PRESET_MODELS "
            "(e.g., paraphrase-multilingual, all-minilm, multi-qa-mpnet)."
        ),
    )
    parser.add_argument(
        "--include-hybrid",
        action="store_true",
        help="Include hybrid (vector + BM25) retrieval in addition to vector-only.",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable hybrid retrieval; only evaluate vector-only.",
    )
    return parser


def _prepare_backends(
    descriptions: List[str],
    encoder_preset: str,
) -> Tuple[ScenarioEncoder, np.ndarray, VectorStore, HybridRetriever]:
    """
    Instantiate encoder, encode all descriptions, and create retrieval backends.
    """
    if encoder_preset not in PRESET_MODELS:
        raise ValueError(f"Unknown encoder preset: {encoder_preset}")

    print(f"[EmbeddingEval] Initializing encoder preset '{encoder_preset}' -> {PRESET_MODELS[encoder_preset]}")
    encoder = ScenarioEncoder.from_preset(encoder_preset)
    vectors = encoder.encode_batch(descriptions)

    if vectors.ndim != 2:
        raise RuntimeError(
            f"Expected 2D array of vectors, got shape {vectors.shape}"
        )

    vector_dim = int(vectors.shape[1])
    print(f"[EmbeddingEval] Vectors shape: {vectors.shape}, dim={vector_dim}")

    # Build vector store
    vs = VectorStore(vector_dim=vector_dim)
    vs.build_index(vectors, descriptions)

    # Build hybrid retriever (BM25 over descriptions)
    hybrid = HybridRetriever(vector_store=vs, alpha=0.7)
    hybrid.fit_bm25(descriptions)

    return encoder, vectors, vs, hybrid


def _run_vector_only(
    vectors: np.ndarray,
    vector_store: VectorStore,
    k: int,
) -> List[List[int]]:
    """
    Run vector-only retrieval for each scenario and return retrieved indices.
    Self-matches are removed from the top-k list.
    """
    n = vectors.shape[0]
    retrieved_indices: List[List[int]] = []

    for i in range(n):
        query_vec = vectors[i]
        # Retrieve one extra to increase chance of including non-self neighbors
        results = vector_store.search(query_vec, k=min(k + 1, n))
        indices = [r["index"] for r in results if r.get("index", -1) != i][:k]
        retrieved_indices.append(indices)

    return retrieved_indices


def _run_hybrid(
    descriptions: List[str],
    vectors: np.ndarray,
    hybrid: HybridRetriever,
    k: int,
) -> List[List[int]]:
    """
    Run hybrid (vector + BM25) retrieval for each scenario and return indices.
    Self-matches are removed from the top-k list.
    """
    n = len(descriptions)
    retrieved_indices: List[List[int]] = []

    for i, desc in enumerate(descriptions):
        query_vec = vectors[i]
        results = hybrid.retrieve(
            query=desc,
            query_vector=query_vec,
            k=min(k + 1, hybrid.vector_store.size()),
            use_hybrid=True,
        )
        indices = [r.get("index", -1) for r in results if r.get("index", -1) != i][:k]
        retrieved_indices.append(indices)

    return retrieved_indices


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Determine which retrieval modes to run
    include_hybrid = args.include_hybrid and not args.no_hybrid
    if args.no_hybrid:
        include_hybrid = False

    print(
        f"[EmbeddingEval] Starting evaluation with num_scenarios={args.num_scenarios}, "
        f"k={args.k}, encoders={args.encoders}, hybrid={include_hybrid}"
    )

    # Build/load dataset
    dataset = build_or_load_dataset(num_scenarios=args.num_scenarios)
    descriptions, labels = extract_descriptions_and_labels(dataset)
    n = len(descriptions)
    print(f"[EmbeddingEval] Dataset loaded with {n} scenarios.")

    methods_results: Dict[str, Dict[str, Any]] = {}

    for encoder_preset in args.encoders:
        encoder_preset = encoder_preset.strip()
        if not encoder_preset:
            continue

        if encoder_preset not in PRESET_MODELS:
            print(f"[EmbeddingEval] Warning: unknown encoder preset '{encoder_preset}', skipping.")
            continue

        (
            encoder,
            vectors,
            vector_store,
            hybrid,
        ) = _prepare_backends(descriptions, encoder_preset)

        # Vector-only retrieval
        print(f"[EmbeddingEval] Running vector-only retrieval for preset '{encoder_preset}'...")
        vec_indices = _run_vector_only(vectors, vector_store, k=args.k)

        hit5 = compute_hit_at_k(labels, vec_indices, k=5, relevance_level="high")
        ndcg10 = compute_ndcg_at_k(labels, vec_indices, k=min(10, args.k))
        same_ratio10 = compute_same_class_ratio(
            labels,
            vec_indices,
            k=min(10, args.k),
            key="accident_type",
        )

        method_name_vec = f"{encoder_preset}|vector"
        methods_results[method_name_vec] = {
            "encoder_preset": encoder_preset,
            "encoder_model": encoder.model_name,
            "mode": "vector",
            "hit@5_high": hit5,
            "ndcg@10": ndcg10,
            "same_class_ratio@10_accident_type": same_ratio10,
        }

        # Hybrid retrieval (optional)
        if include_hybrid:
            print(f"[EmbeddingEval] Running hybrid retrieval for preset '{encoder_preset}'...")
            hyb_indices = _run_hybrid(descriptions, vectors, hybrid, k=args.k)

            hit5_h = compute_hit_at_k(labels, hyb_indices, k=5, relevance_level="high")
            ndcg10_h = compute_ndcg_at_k(labels, hyb_indices, k=min(10, args.k))
            same_ratio10_h = compute_same_class_ratio(
                labels,
                hyb_indices,
                k=min(10, args.k),
                key="accident_type",
            )

            method_name_hyb = f"{encoder_preset}|hybrid"
            methods_results[method_name_hyb] = {
                "encoder_preset": encoder_preset,
                "encoder_model": encoder.model_name,
                "mode": "hybrid",
                "hit@5_high": hit5_h,
                "ndcg@10": ndcg10_h,
                "same_class_ratio@10_accident_type": same_ratio10_h,
            }

    summary: Dict[str, Any] = {
        "num_scenarios": n,
        "k": args.k,
        "encoders": args.encoders,
        "include_hybrid": include_hybrid,
        "timestamp": timestamp,
        "methods": methods_results,
    }

    out_path = os.path.join(args.output_root, f"embedding_eval_results_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[EmbeddingEval] Finished. Results written to: {out_path}")


if __name__ == "__main__":
    main()



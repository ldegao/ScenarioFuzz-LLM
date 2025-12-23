"""
Reranker Module
Re-ranks retrieved documents using cross-encoder models for better relevance
"""

import numpy as np
from typing import List, Dict, Any, Optional
import re
import importlib


class CrossEncoderReranker:
    """
    Cross-encoder based reranker for better relevance scoring
    Uses a cross-encoder model to score query-document pairs
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker
        
        Args:
            model_name: Name of the cross-encoder model
        """
        self.model_name = model_name
        self.model = None
    
    def _load_model(self):
        """Lazy load the cross-encoder model (defer heavy import)."""
        try:
            ce = _get_cross_encoder()
            self.model = ce(self.model_name)
            print(f"[Reranker] Loaded cross-encoder model: {self.model_name}")
        except ImportError as e:
            raise ImportError(
                "[Reranker] sentence-transformers is required for CrossEncoderReranker"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"[Reranker] Failed to load cross-encoder model {self.model_name}: {e}"
            ) from e
    
    def rerank(self, 
               query: str,
               documents: List[str],
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank documents for a query
        
        Args:
            query: Query text
            documents: List of document texts to rerank
            top_k: Number of top documents to return (None = return all)
            
        Returns:
            List of dictionaries with 'text', 'score', 'index' sorted by score descending
        """
        if not documents:
            return []
        
        if self.model is None:
            self._load_model()
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get scores from cross-encoder
        scores = self.model.predict(pairs)
        
        # Sort by score descending
        indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for i, (orig_idx, score) in enumerate(indexed_scores):
            if top_k is not None and i >= top_k:
                break
            results.append({
                'text': documents[orig_idx],
                'score': score,
                'index': orig_idx,
                'rerank_position': i
            })
        
        return results


def _get_cross_encoder():
    """
    Lazily import CrossEncoder to avoid heavy imports during test collection.
    """
    spec = importlib.util.find_spec("sentence_transformers")
    if spec is None:
        raise ImportError("[Reranker] sentence-transformers is required for CrossEncoderReranker")
    module = importlib.import_module("sentence_transformers")
    return module.CrossEncoder


class SimpleReranker:
    """
    Simple reranker based on keyword overlap and length normalization
    Useful when cross-encoder is not available
    """
    
    def __init__(self):
        """Initialize simple reranker"""
        pass
    
    def _calculate_overlap_score(self, query: str, document: str) -> float:
        """
        Calculate keyword overlap score between query and document
        
        Args:
            query: Query text
            document: Document text
            
        Returns:
            Overlap score between 0 and 1
        """
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        doc_words = set(re.findall(r'\b\w+\b', document.lower()))
        
        if not query_words:
            return 0.0
        
        # Jaccard similarity
        intersection = query_words & doc_words
        union = query_words | doc_words
        
        if not union:
            return 0.0
        
        jaccard = len(intersection) / len(union)
        
        # Also consider term frequency
        doc_lower = document.lower()
        term_freq_score = sum(1 for word in query_words if word in doc_lower) / len(query_words)
        
        # Combined score
        score = 0.6 * jaccard + 0.4 * term_freq_score
        
        return score
    
    def rerank(self,
               query: str,
               documents: List[str],
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank documents using keyword overlap
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of top documents to return
            
        Returns:
            List of dictionaries with 'text', 'score', 'index'
        """
        if not documents:
            return []
        
        # Calculate scores
        scores = []
        for i, doc in enumerate(documents):
            score = self._calculate_overlap_score(query, doc)
            scores.append((i, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for i, (orig_idx, score) in enumerate(scores):
            if top_k is not None and i >= top_k:
                break
            results.append({
                'text': documents[orig_idx],
                'score': score,
                'index': orig_idx,
                'rerank_position': i
            })
        
        return results


class Reranker:
    """
    Main reranker class that can use either cross-encoder or simple reranker
    """
    
    def __init__(self, 
                 use_cross_encoder: bool = True,
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker
        
        Args:
            use_cross_encoder: Whether to use cross-encoder (True) or simple reranker (False)
            cross_encoder_model: Model name for cross-encoder
        """
        self.use_cross_encoder = use_cross_encoder
        if use_cross_encoder:
            self.cross_encoder = CrossEncoderReranker(model_name=cross_encoder_model)
            self.simple_reranker = None
        else:
            self.cross_encoder = None
            self.simple_reranker = SimpleReranker()
    
    def rerank(self,
               query: str,
               documents: List[str],
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank documents for a query
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents with scores
        """
        if self.use_cross_encoder and self.cross_encoder:
            return self.cross_encoder.rerank(query, documents, top_k)
        else:
            return self.simple_reranker.rerank(query, documents, top_k)
    
    def rerank_results(self,
                      query: str,
                      retrieval_results: List[Dict[str, Any]],
                      top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank retrieval results (preserving metadata)
        
        Args:
            query: Query text
            retrieval_results: List of retrieval result dictionaries
            top_k: Number of top results to return
            
        Returns:
            List of reranked results with updated scores
        """
        if not retrieval_results:
            return []
        
        # Extract documents
        documents = [r['text'] for r in retrieval_results]
        
        # Rerank
        reranked = self.rerank(query, documents, top_k)
        
        # Merge with original metadata
        results = []
        for rerank_item in reranked:
            orig_idx = rerank_item['index']
            if orig_idx < len(retrieval_results):
                orig_result = retrieval_results[orig_idx].copy()
                orig_result['score'] = rerank_item['score']
                orig_result['rerank_position'] = rerank_item.get('rerank_position', 0)
                results.append(orig_result)
        
        return results


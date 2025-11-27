"""
Hybrid Retriever Module
Combines vector retrieval (semantic similarity) with keyword retrieval (BM25)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import math
import re


class BM25Retriever:
    """
    BM25 (Best Matching 25) keyword-based retriever
    Implements the BM25 ranking function for keyword search
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.avg_doc_len: float = 0.0
        self.vocab: set = set()
        self._initialized = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words (simple whitespace-based)"""
        # Convert to lowercase and split by whitespace/punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def fit(self, documents: List[str]):
        """
        Fit BM25 on a collection of documents
        
        Args:
            documents: List of document texts
        """
        self.documents = documents
        self.doc_freqs = []
        doc_lens = []
        
        # Tokenize all documents and build vocabulary
        for doc in documents:
            tokens = self._tokenize(doc)
            doc_lens.append(len(tokens))
            self.vocab.update(tokens)
            
            # Count term frequencies in this document
            term_freq = Counter(tokens)
            self.doc_freqs.append(dict(term_freq))
        
        # Calculate average document length
        self.avg_doc_len = sum(doc_lens) / len(doc_lens) if doc_lens else 0.0
        
        # Calculate IDF (Inverse Document Frequency)
        doc_count = len(documents)
        for term in self.vocab:
            # Count how many documents contain this term
            df = sum(1 for doc_freq in self.doc_freqs if term in doc_freq)
            # IDF formula: log((N - df + 0.5) / (df + 0.5))
            self.idf[term] = math.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)
        
        self._initialized = True
        print(f"[BM25] Fitted on {len(documents)} documents, vocabulary size: {len(self.vocab)}")
    
    def get_scores(self, query: str) -> List[float]:
        """
        Get BM25 scores for all documents given a query
        
        Args:
            query: Query text
            
        Returns:
            List of BM25 scores for each document
        """
        if not self._initialized:
            raise ValueError("BM25 retriever not fitted. Call fit() first.")
        
        query_tokens = self._tokenize(query)
        scores = []
        
        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = len(self.documents[i].split())
            
            for term in query_tokens:
                if term not in self.idf:
                    continue
                
                # Term frequency in document
                tf = doc_freq.get(term, 0)
                
                if tf == 0:
                    continue
                
                # BM25 formula
                numerator = self.idf[term] * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += numerator / denominator
            
            scores.append(score)
        
        return scores
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents for a query
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            List of tuples (document_index, score) sorted by score descending
        """
        scores = self.get_scores(query)
        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        results = [(int(idx), float(scores[idx])) for idx in top_k_indices if scores[idx] > 0]
        return results


class HybridRetriever:
    """
    Hybrid retriever that combines vector retrieval and BM25 keyword retrieval
    """
    
    def __init__(self, 
                 vector_store,  # VectorStore instance
                 alpha: float = 0.7,
                 bm25_k1: float = 1.5,
                 bm25_b: float = 0.75):
        """
        Initialize hybrid retriever
        
        Args:
            vector_store: VectorStore instance for semantic retrieval
            alpha: Weight for vector retrieval (1-alpha for BM25), default 0.7
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
        """
        self.vector_store = vector_store
        self.alpha = alpha
        self.bm25 = BM25Retriever(k1=bm25_k1, b=bm25_b)
        self._bm25_fitted = False
    
    def fit_bm25(self, documents: List[str]):
        """
        Fit BM25 on documents
        
        Args:
            documents: List of document texts
        """
        self.bm25.fit(documents)
        self._bm25_fitted = True
    
    def retrieve(self, 
                 query: str,
                 query_vector: np.ndarray,
                 k: int = 5,
                 use_hybrid: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid search
        
        Args:
            query: Query text for BM25
            query_vector: Query vector for semantic search
            k: Number of documents to retrieve
            use_hybrid: Whether to use hybrid search (True) or only vector search (False)
            
        Returns:
            List of result dictionaries with 'text', 'score', 'index', 'metadata'
        """
        if not use_hybrid or not self._bm25_fitted:
            # Fallback to vector-only retrieval
            return self.vector_store.search(query_vector, k=k)
        
        # Retrieve more candidates from each method
        candidate_k = min(k * 3, self.vector_store.size())
        
        # Vector retrieval
        vector_results = self.vector_store.search(query_vector, k=candidate_k)
        
        # BM25 retrieval
        bm25_results = self.bm25.retrieve(query, k=candidate_k)
        
        # Normalize scores
        vector_scores = {}
        for result in vector_results:
            idx = result['index']
            # Convert distance to similarity (assuming L2 distance)
            # Smaller distance = higher similarity
            distance = result.get('distance', float('inf'))
            similarity = 1.0 / (1.0 + distance)  # Simple conversion
            vector_scores[idx] = similarity
        
        bm25_scores = {}
        max_bm25 = max([score for _, score in bm25_results], default=1.0)
        for idx, score in bm25_results:
            # Normalize BM25 scores to [0, 1]
            normalized_score = score / max_bm25 if max_bm25 > 0 else 0.0
            bm25_scores[idx] = normalized_score
        
        # Combine scores
        combined_scores = {}
        all_indices = set(vector_scores.keys()) | set(bm25_scores.keys())
        
        for idx in all_indices:
            vector_score = vector_scores.get(idx, 0.0)
            bm25_score = bm25_scores.get(idx, 0.0)
            # Weighted combination
            combined_score = self.alpha * vector_score + (1 - self.alpha) * bm25_score
            combined_scores[idx] = combined_score
        
        # Get top-k by combined score
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Build results
        results = []
        for idx, score in sorted_indices:
            # Get metadata from vector results if available
            metadata = {}
            text = ""
            for vr in vector_results:
                if vr['index'] == idx:
                    text = vr['text']
                    metadata = vr.get('metadata', {})
                    break
            
            if not text:
                # Fallback: get from BM25 documents
                if idx < len(self.bm25.documents):
                    text = self.bm25.documents[idx]
            
            results.append({
                'text': text,
                'score': score,
                'index': idx,
                'metadata': metadata,
                'vector_score': vector_scores.get(idx, 0.0),
                'bm25_score': bm25_scores.get(idx, 0.0)
            })
        
        return results


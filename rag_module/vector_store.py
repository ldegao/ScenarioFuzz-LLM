"""
Vector Store Module
Uses Faiss to build vector index and support KNN retrieval
"""

import numpy as np
from typing import List, Dict, Any, Optional
import os
import pickle


class VectorStore:
    """
    Manages vector storage and retrieval using Faiss
    """
    
    def __init__(self, index_type: str = "IndexFlatL2", vector_dim: int = 384):
        """
        Initialize the vector store
        
        Args:
            index_type: Type of Faiss index to use
            vector_dim: Dimension of vectors
        """
        self.index_type = index_type
        self.vector_dim = vector_dim
        self.index = None
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the Faiss index"""
        try:
            import faiss
            if self.index_type == "IndexFlatL2":
                self.index = faiss.IndexFlatL2(self.vector_dim)
            elif self.index_type == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(self.vector_dim)
            else:
                print(f"[VectorStore] Warning: Unknown index type {self.index_type}, using IndexFlatL2")
                self.index = faiss.IndexFlatL2(self.vector_dim)
            print(f"[VectorStore] Initialized {self.index_type} index with dimension {self.vector_dim}")
        except ImportError:
            print("[VectorStore] Warning: faiss not installed. Using mock index.")
            self.index = None
    
    def build_index(self, vectors: np.ndarray, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Build the index from vectors and associated texts
        
        Args:
            vectors: numpy array of shape (n, vector_dim)
            texts: List of text descriptions corresponding to vectors
            metadata: Optional list of metadata dictionaries
        """
        if vectors.shape[1] != self.vector_dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.vector_dim}, got {vectors.shape[1]}")
        
        if len(texts) != vectors.shape[0]:
            raise ValueError(f"Text count mismatch: {len(texts)} texts for {vectors.shape[0]} vectors")
        
        if self.index is None:
            # Mock index for testing without faiss
            self._mock_build_index(vectors, texts, metadata)
            return
        
        # Normalize vectors for cosine similarity (if using IP index)
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(vectors)
        
        # Clear existing index if rebuilding
        if self.index.ntotal > 0:
            print("[VectorStore] Clearing existing index")
            self._initialize_index()
        
        # Add vectors to index
        self.index.add(vectors.astype(np.float32))
        self.texts = texts.copy()
        self.metadata = metadata.copy() if metadata else [{}] * len(texts)
        
        print(f"[VectorStore] Built index with {self.index.ntotal} vectors")
    
    def _mock_build_index(self, vectors: np.ndarray, texts: List[str], metadata: Optional[List[Dict[str, Any]]]):
        """Mock index building for testing without faiss"""
        self.texts = texts.copy()
        self.metadata = metadata.copy() if metadata else [{}] * len(texts)
        # Store vectors in memory for mock search
        self._mock_vectors = vectors.copy()
        print(f"[VectorStore] Built mock index with {len(texts)} vectors")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for k nearest neighbors
        
        Args:
            query_vector: Query vector of shape (vector_dim,) or (1, vector_dim)
            k: Number of nearest neighbors to retrieve
            
        Returns:
            List of dictionaries containing 'text', 'distance', 'index', and 'metadata'
        """
        if self.index is None:
            return self._mock_search(query_vector, k)
        
        # Reshape query vector if needed
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        if query_vector.shape[1] != self.vector_dim:
            raise ValueError(f"Query vector dimension mismatch: expected {self.vector_dim}, got {query_vector.shape[1]}")
        
        # Normalize for cosine similarity if using IP index
        if self.index_type == "IndexFlatIP":
            query_vector = query_vector.copy().astype(np.float32)
            faiss.normalize_L2(query_vector)
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vector.astype(np.float32), k)
        
        # Build results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                result = {
                    'text': self.texts[idx],
                    'distance': float(distances[0][i]),
                    'index': int(idx),
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
                }
                results.append(result)
        
        return results
    
    def _mock_search(self, query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Mock search for testing without faiss"""
        if not hasattr(self, '_mock_vectors'):
            return []
        
        # Reshape query vector if needed
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Calculate L2 distances
        distances = np.linalg.norm(self._mock_vectors - query_vector, axis=1)
        
        # Get top k
        k = min(k, len(distances))
        top_k_indices = np.argsort(distances)[:k]
        
        results = []
        for idx in top_k_indices:
            result = {
                'text': self.texts[idx],
                'distance': float(distances[idx]),
                'index': int(idx),
                'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
            }
            results.append(result)
        
        return results
    
    def save_index(self, path: str):
        """
        Save the index and associated data to disk
        
        Args:
            path: Directory path to save index files
        """
        os.makedirs(path, exist_ok=True)
        
        if self.index is not None:
            try:
                import faiss
                index_path = os.path.join(path, "index.faiss")
                faiss.write_index(self.index, index_path)
                print(f"[VectorStore] Saved index to {index_path}")
            except ImportError:
                pass
        
        # Save texts and metadata
        data_path = os.path.join(path, "data.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump({
                'texts': self.texts,
                'metadata': self.metadata,
                'vector_dim': self.vector_dim,
                'index_type': self.index_type
            }, f)
        print(f"[VectorStore] Saved data to {data_path}")
    
    def load_index(self, path: str):
        """
        Load the index and associated data from disk
        
        Args:
            path: Directory path containing index files
        """
        # Load data
        data_path = os.path.join(path, "data.pkl")
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.texts = data['texts']
                self.metadata = data['metadata']
                self.vector_dim = data.get('vector_dim', self.vector_dim)
                self.index_type = data.get('index_type', self.index_type)
            print(f"[VectorStore] Loaded data from {data_path}")
        
        # Load index
        index_path = os.path.join(path, "index.faiss")
        if os.path.exists(index_path):
            try:
                import faiss
                self.index = faiss.read_index(index_path)
                print(f"[VectorStore] Loaded index from {index_path}")
            except ImportError:
                print("[VectorStore] Warning: faiss not installed, cannot load index")
            except Exception as e:
                print(f"[VectorStore] Error loading index: {e}")
    
    def size(self) -> int:
        """Get the number of vectors in the index"""
        if self.index is not None:
            return self.index.ntotal
        return len(self.texts)


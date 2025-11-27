"""
Scenario Encoder Module
Uses Sentence-BERT to encode scenario descriptions into vectors
"""

import numpy as np
from typing import List, Union
import os


class ScenarioEncoder:
    """
    Encodes scenario text descriptions into dense vectors using Sentence-BERT
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize the encoder with a Sentence-BERT model
        
        Args:
            model_name: Name of the Sentence-BERT model to use
        """
        self.model_name = model_name
        self.model = None
        self.vector_dim = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load the Sentence-BERT model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            # Get vector dimension by encoding a dummy text
            dummy_vector = self.model.encode("test", convert_to_numpy=True)
            self.vector_dim = len(dummy_vector)
            print(f"[RAG] Loaded encoder model: {self.model_name}, vector_dim={self.vector_dim}")
        except ImportError:
            print("[RAG] Warning: sentence-transformers not installed. Using mock encoder.")
            self.model = None
            self.vector_dim = 384  # Default dimension for MiniLM
    
    def encode(self, scenario_text: str) -> np.ndarray:
        """
        Encode a single scenario description into a vector
        
        Args:
            scenario_text: Text description of the scenario
            
        Returns:
            numpy array of shape (vector_dim,)
        """
        if self.model is None:
            # Mock encoding for testing without sentence-transformers
            return self._mock_encode(scenario_text)
        
        vector = self.model.encode(scenario_text, convert_to_numpy=True)
        return vector
    
    def encode_batch(self, scenarios: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple scenario descriptions into vectors
        
        Args:
            scenarios: List of scenario text descriptions
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of shape (num_scenarios, vector_dim)
        """
        if self.model is None:
            # Mock encoding for testing
            return np.array([self._mock_encode(s) for s in scenarios])
        
        vectors = self.model.encode(
            scenarios,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(scenarios) > 100
        )
        return vectors
    
    def _mock_encode(self, text: str) -> np.ndarray:
        """
        Mock encoding function for testing without sentence-transformers
        Uses simple hash-based encoding
        """
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        # Create a vector of fixed dimension
        vector = np.frombuffer(hash_bytes * (self.vector_dim // 16 + 1), dtype=np.uint8)[:self.vector_dim]
        vector = vector.astype(np.float32) / 255.0  # Normalize to [0, 1]
        return vector
    
    def get_vector_dim(self) -> int:
        """Get the dimension of encoded vectors"""
        return self.vector_dim


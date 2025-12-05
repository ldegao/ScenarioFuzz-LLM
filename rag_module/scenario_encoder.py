"""
Scenario Encoder Module
Uses Sentence-BERT to encode scenario descriptions into vectors.

This module now supports a small set of preset model names so that
experiments can easily switch between different backbones.
"""

from typing import List
import hashlib

import numpy as np

# Preset model aliases for convenient switching in experiments.
# Users can still pass any full HuggingFace / sentence-transformers
# model name directly; unknown keys fall back to the raw string.
PRESET_MODELS = {
    # Current default multilingual baseline
    "paraphrase-multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # English, small, fast
    "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
    # English, retrieval-optimized
    "multi-qa-mpnet": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
}


class ScenarioEncoder:
    """
    Encodes scenario text descriptions into dense vectors using Sentence-BERT.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Initialize the encoder with a Sentence-BERT model.

        Args:
            model_name: Name of the Sentence-BERT model to use. Can be either:
                - A full HuggingFace / sentence-transformers model name, or
                - One of the preset keys in PRESET_MODELS, e.g.:
                  "paraphrase-multilingual", "all-minilm", "multi-qa-mpnet".
        """
        # Resolve preset alias if provided
        resolved_name = PRESET_MODELS.get(model_name, model_name)
        self.preset_key = model_name if model_name in PRESET_MODELS else None
        self.model_name = resolved_name
        self.model = None
        self.vector_dim = None
        self._load_model()

    @classmethod
    def from_preset(cls, preset_name: str) -> "ScenarioEncoder":
        """
        Convenience constructor using a preset key from PRESET_MODELS.

        Example:
            ScenarioEncoder.from_preset("all-minilm")
        """
        return cls(model_name=preset_name)

    def _load_model(self):
        """Lazy load the Sentence-BERT model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            # Get vector dimension by encoding a dummy text
            dummy_vector = self.model.encode("test", convert_to_numpy=True)
            self.vector_dim = int(len(dummy_vector))
            print(
                f"[RAG] Loaded encoder model: {self.model_name}, "
                f"vector_dim={self.vector_dim}"
            )
        except ImportError as e:
            # Fail-fast: encoder requires sentence-transformers in normal usage.
            raise ImportError(
                "[RAG] sentence-transformers is required for ScenarioEncoder"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"[RAG] Failed to load encoder model {self.model_name}: {e}"
            ) from e

    def encode(self, scenario_text: str) -> np.ndarray:
        """
        Encode a single scenario description into a vector.

        Args:
            scenario_text: Text description of the scenario

        Returns:
            numpy array of shape (vector_dim,)
        """
        if self.model is None:
            raise RuntimeError(
                "[RAG] Encoder model is not loaded. "
                "Ensure sentence-transformers is installed and _load_model() succeeded."
            )
        vector = self.model.encode(scenario_text, convert_to_numpy=True)
        return vector

    def encode_batch(self, scenarios: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple scenario descriptions into vectors.

        Args:
            scenarios: List of scenario text descriptions
            batch_size: Batch size for encoding

        Returns:
            numpy array of shape (num_scenarios, vector_dim)
        """
        if self.model is None:
            # Mock encoding for testing (should rarely be used in practice).
            if self.vector_dim is None:
                # Default to a small dimension if we truly have no model info.
                self.vector_dim = 32
            return np.array([self._mock_encode(s) for s in scenarios])

        vectors = self.model.encode(
            scenarios,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(scenarios) > 100,
        )
        return vectors

    def _mock_encode(self, text: str) -> np.ndarray:
        """
        Mock encoding function for testing without sentence-transformers.
        Uses simple hash-based encoding.
        """
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        dim = int(self.vector_dim) if self.vector_dim is not None else 32
        # Create a vector of fixed dimension
        repeats = dim // len(hash_bytes) + 1
        vector = np.frombuffer(hash_bytes * repeats, dtype=np.uint8)[:dim]
        vector = vector.astype(np.float32) / 255.0  # Normalize to [0, 1]
        return vector

    def get_vector_dim(self) -> int:
        """Get the dimension of encoded vectors."""
        if self.vector_dim is None:
            raise RuntimeError("[RAG] Encoder vector dimension is not initialized.")
        return int(self.vector_dim)


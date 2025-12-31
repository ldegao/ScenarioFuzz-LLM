"""
Enhanced RAG Engine Module
Improved RAG engine with hybrid retrieval (vector + BM25) and reranking
"""

from typing import List, Dict, Any, Optional
import time
import json
import os
from .scenario_encoder import ScenarioEncoder
from .knowledge_base import KnowledgeBase
from .vector_store import VectorStore
from .hybrid_retriever import HybridRetriever
from .reranker import Reranker


class EnhancedRAGEngine:
    """
    Enhanced RAG engine with hybrid retrieval and reranking capabilities
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 knowledge_base_path: str = "./data/knowledge_base/",
                 index_type: str = "IndexFlatL2",
                 top_k: int = 5,
                 use_hybrid_search: bool = True,
                 hybrid_alpha: float = 0.7,
                 use_reranking: bool = True,
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 similarity_threshold: Optional[float] = None):
        """
        Initialize the enhanced RAG engine
        
        Args:
            model_name: Sentence-BERT model name for encoding
            knowledge_base_path: Path to knowledge base directory
            index_type: Type of Faiss index
            top_k: Number of top scenarios to retrieve
            use_hybrid_search: Whether to use hybrid search (vector + BM25)
            hybrid_alpha: Weight for vector retrieval in hybrid search (0-1)
            use_reranking: Whether to use reranking after retrieval
            reranker_model: Cross-encoder model name for reranking
            similarity_threshold: Optional distance/score threshold to filter neighbors
        """
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking
        
        # Initialize components
        self.encoder = ScenarioEncoder(model_name=model_name)
        self.knowledge_base = KnowledgeBase(knowledge_base_path=knowledge_base_path)
        self.vector_store = VectorStore(
            index_type=index_type,
            vector_dim=self.encoder.get_vector_dim()
        )
        
        # Initialize hybrid retriever if enabled
        if use_hybrid_search:
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                alpha=hybrid_alpha
            )
        else:
            self.hybrid_retriever = None
        
        # Initialize reranker if enabled
        if use_reranking:
            self.reranker = Reranker(
                use_cross_encoder=True,
                cross_encoder_model=reranker_model
            )
        else:
            self.reranker = None
        
        self.initialized = False
    
    def initialize(self, load_mock_data: bool = True, load_from_json: Optional[str] = None):
        """
        Initialize the RAG engine by building the knowledge base and vector index
        
        Args:
            load_mock_data: Whether to load mock data
            load_from_json: Optional path to JSON file with scenarios
        """
        if self.initialized:
            print("[EnhancedRAGEngine] Already initialized")
            return
        
        # Load knowledge base
        if load_mock_data:
            self.knowledge_base.load_mock_data()
        
        if load_from_json and os.path.exists(load_from_json):
            self.knowledge_base.load_from_json(load_from_json)
        
        if self.knowledge_base.size() == 0:
            print("[EnhancedRAGEngine] Warning: Knowledge base is empty")
            self.initialized = True
            return
        
        # Encode all scenarios
        scenario_descriptions = self.knowledge_base.get_scenario_descriptions()
        print(f"[EnhancedRAGEngine] Encoding {len(scenario_descriptions)} scenarios...")
        vectors = self.encoder.encode_batch(scenario_descriptions)
        
        # Build vector index
        scenarios = self.knowledge_base.get_all_scenarios()
        self.vector_store.build_index(vectors, scenario_descriptions, scenarios)
        
        # Fit BM25 if hybrid search is enabled
        if self.use_hybrid_search and self.hybrid_retriever:
            print("[EnhancedRAGEngine] Fitting BM25 retriever...")
            self.hybrid_retriever.fit_bm25(scenario_descriptions)
        
        self.initialized = True
        print(f"[EnhancedRAGEngine] Initialized with {self.knowledge_base.size()} scenarios")
        print(f"[EnhancedRAGEngine] Hybrid search: {self.use_hybrid_search}, Reranking: {self.use_reranking}")
    
    def _filter_by_threshold(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optionally filter retrieval results by a distance/score threshold (τ gate).
        """
        if self.similarity_threshold is None:
            return results
        filtered = []
        for r in results:
            if 'distance' in r and r['distance'] <= self.similarity_threshold:
                filtered.append(r)
        return filtered

    def aggregate_neighbor_analysis(self, neighbors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Placeholder aggregation with priority awareness; extend with risk/diversity metrics.
        """
        if not neighbors:
            return {}
        summary = {
            "neighbor_count": len(neighbors),
            "top_priority": max((n.get("metadata", {}).get("priority", "low") for n in neighbors), default="low"),
        }
        return summary

    def update_access_stats(self, neighbors: List[Dict[str, Any]]):
        """Update access counters in the knowledge base for retrieved neighbors."""
        indices = [n.get("index", -1) for n in neighbors if "index" in n]
        self.knowledge_base.update_access_stats(indices)

    def insert_memory(self, vec: Any, snapshot: Any, analysis: Any, priority: str = "low", rebuild_index: bool = False):
        """
        Insert a new memory with priority metadata and optionally rebuild the index/BM25 model.
        """
        scenario = {
            "description": str(snapshot),
            "analysis": analysis,
            "priority": priority,
            "access_count": 0,
            "last_access_ts": time.time(),
        }
        self.knowledge_base.add_scenario(scenario)

        if rebuild_index:
            scenario_descriptions = self.knowledge_base.get_scenario_descriptions()
            vectors = self.encoder.encode_batch(scenario_descriptions)
            scenarios = self.knowledge_base.get_all_scenarios()
            self.vector_store.build_index(vectors, scenario_descriptions, scenarios)

            if self.use_hybrid_search and self.hybrid_retriever:
                self.hybrid_retriever.fit_bm25(scenario_descriptions)

    def retrieve_relevant_scenarios(self, 
                                    seed_scenario: str, 
                                    k: Optional[int] = None,
                                    return_metadata: bool = False) -> List[str]:
        """
        Retrieve k most relevant scenarios for a seed scenario
        
        Args:
            seed_scenario: Text description of the seed scenario
            k: Number of scenarios to retrieve (defaults to self.top_k)
            return_metadata: Whether to return full metadata (if True, returns List[Dict])
            
        Returns:
            List of retrieved scenario description texts (or List[Dict] if return_metadata=True)
        """
        if not self.initialized:
            print("[EnhancedRAGEngine] Warning: Engine not initialized, calling initialize()")
            self.initialize()
        
        if k is None:
            k = self.top_k
        
        # Encode seed scenario
        query_vector = self.encoder.encode(seed_scenario)
        
        # Retrieve using hybrid search or vector-only search
        if self.use_hybrid_search and self.hybrid_retriever:
            # Use hybrid retrieval
            results = self.hybrid_retriever.retrieve(
                query=seed_scenario,
                query_vector=query_vector,
                k=k * 2 if self.use_reranking else k,  # Retrieve more if reranking
                use_hybrid=True
            )
        else:
            # Use vector-only retrieval
            results = self.vector_store.search(query_vector, k=k * 2 if self.use_reranking else k)
        
        # Apply reranking if enabled
        if self.use_reranking and self.reranker:
            documents = [r['text'] for r in results]
            reranked = self.reranker.rerank_results(seed_scenario, results, top_k=k)
            results = reranked
        
        # Limit to top-k and apply optional threshold
        results = results[:k]
        results = self._filter_by_threshold(results)
        
        if return_metadata:
            return results
        else:
            # Extract text descriptions
            retrieved_texts = [result['text'] for result in results]
            return retrieved_texts
    
    def generate_enhanced_prompt(self, seed: str, retrieved: List[str]) -> str:
        """
        Build an enhanced prompt with retrieved context
        
        Args:
            seed: Seed scenario description
            retrieved: List of retrieved scenario descriptions
            
        Returns:
            Enhanced prompt string
        """
        prompt_parts = [
            "You are analyzing autonomous driving scenarios. Below are some related scenarios from the knowledge base:",
            ""
        ]
        
        for i, scenario in enumerate(retrieved, 1):
            prompt_parts.append(f"{i}. {scenario}")
        
        prompt_parts.extend([
            "",
            "Based on the above related scenarios and the following seed scenario:",
            f"Seed: {seed}",
            "",
            "Please design a novel and high-risk scenario that:",
            "1. Is semantically related to the seed scenario",
            "2. Incorporates elements from the retrieved scenarios",
            "3. Introduces new variations to enhance diversity",
            "4. Represents a realistic but challenging driving situation",
            "",
            "Provide your response in JSON format with the following structure:",
            "{",
            '  "scenario_description": "Detailed description of the scenario",',
            '  "key_elements": ["element1", "element2", ...],',
            '  "risk_factors": ["factor1", "factor2", ...],',
            '  "novel_aspects": "What makes this scenario novel"',
            "}"
        ])
        
        return "\n".join(prompt_parts)
    
    def format_scenario_dict_for_prompt(self, retrieved_scenarios: List[Dict[str, Any]]) -> str:
        """
        Format retrieved scenarios as Scenario-dict format for prompt
        
        Args:
            retrieved_scenarios: List of scenario dictionaries from retrieval
            
        Returns:
            Formatted string similar to Scenario-database format
        """
        if not retrieved_scenarios:
            return "{}"
        
        formatted_dict = {}
        for i, scenario in enumerate(retrieved_scenarios):
            # Extract description from scenario dict or metadata
            if isinstance(scenario, dict):
                description = scenario.get('description', scenario.get('text', str(scenario)))
            else:
                description = str(scenario)
            
            # Use index as key (similar to Scenario_database)
            formatted_dict[str(i)] = description
        
        return str(formatted_dict)
    
    def retrieve_and_format_for_prompt(self, query_scenario: str, k: Optional[int] = None) -> str:
        """
        Retrieve relevant scenarios and format as Scenario-dict for prompt
        
        Args:
            query_scenario: Current scenario description to query
            k: Number of scenarios to retrieve (defaults to self.top_k)
            
        Returns:
            Formatted Scenario-dict string ready for prompt insertion
        """
        if k is None:
            k = self.top_k
        
        # Retrieve scenarios with metadata
        retrieved_results = self.retrieve_relevant_scenarios(
            query_scenario, 
            k=k, 
            return_metadata=True
        )
        
        # Get full scenario metadata if available
        retrieved_scenarios = []
        for result in retrieved_results:
            # Try to find matching scenario in knowledge base
            idx = result.get('index', -1)
            if idx >= 0 and idx < len(self.knowledge_base.get_all_scenarios()):
                scenarios = self.knowledge_base.get_all_scenarios()
                retrieved_scenarios.append(scenarios[idx])
            else:
                # If not found, create a simple dict from result
                retrieved_scenarios.append({
                    'description': result.get('text', ''),
                    'metadata': result.get('metadata', {})
                })
        
        # Format as Scenario-dict
        return self.format_scenario_dict_for_prompt(retrieved_scenarios)
    
    def generate_scenario(self, seed_scenario: str, use_gpt: bool = True) -> Dict[str, Any]:
        """
        Complete RAG generation pipeline
        
        Args:
            seed_scenario: Seed scenario description
            use_gpt: Whether to use GPT for final generation (if False, returns retrieved scenarios)
            
        Returns:
            Dictionary containing generated scenario information
        """
        if not self.initialized:
            self.initialize()
        
        # Retrieve relevant scenarios
        retrieved = self.retrieve_relevant_scenarios(seed_scenario, k=self.top_k)
        
        if not use_gpt:
            # Return retrieved scenarios without GPT generation
            return {
                'seed': seed_scenario,
                'retrieved_scenarios': retrieved,
                'generated': None
            }
        
        # Generate enhanced prompt
        enhanced_prompt = self.generate_enhanced_prompt(seed_scenario, retrieved)
        
        # Call GPT (this would be integrated with gpt.py)
        # For now, return the enhanced prompt structure
        return {
            'seed': seed_scenario,
            'retrieved_scenarios': retrieved,
            'enhanced_prompt': enhanced_prompt,
            'generated': None  # Will be filled by GPT call
        }
    
    def add_scenario_to_knowledge_base(self, scenario: Dict[str, Any], rebuild_index: bool = False):
        """
        Add a new scenario to the knowledge base
        
        Args:
            scenario: Scenario dictionary
            rebuild_index: Whether to rebuild the index immediately
        """
        self.knowledge_base.add_scenario(scenario)
        
        if rebuild_index:
            # Rebuild index with new scenario
            scenario_descriptions = self.knowledge_base.get_scenario_descriptions()
            vectors = self.encoder.encode_batch(scenario_descriptions)
            scenarios = self.knowledge_base.get_all_scenarios()
            self.vector_store.build_index(vectors, scenario_descriptions, scenarios)
            
            # Refit BM25 if hybrid search is enabled
            if self.use_hybrid_search and self.hybrid_retriever:
                self.hybrid_retriever.fit_bm25(scenario_descriptions)
            
            print(f"[EnhancedRAGEngine] Rebuilt index with {self.knowledge_base.size()} scenarios")
    
    def save_knowledge_base(self, file_path: str):
        """Save the knowledge base to a JSON file"""
        self.knowledge_base.save_to_json(file_path)
    
    def save_vector_index(self, path: str):
        """Save the vector index to disk"""
        self.vector_store.save_index(path)


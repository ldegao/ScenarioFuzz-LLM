"""
RAG Engine Module
Main RAG engine that integrates encoding, retrieval, and generation
"""

from typing import List, Dict, Any, Optional
import json
import os
from .scenario_encoder import ScenarioEncoder
from .knowledge_base import KnowledgeBase
from .vector_store import VectorStore


class RAGEngine:
    """
    Main RAG engine that combines encoding, retrieval, and generation
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 knowledge_base_path: str = "./data/knowledge_base/",
                 index_type: str = "IndexFlatL2",
                 top_k: int = 5):
        """
        Initialize the RAG engine
        
        Args:
            model_name: Sentence-BERT model name for encoding
            knowledge_base_path: Path to knowledge base directory
            index_type: Type of Faiss index
            top_k: Number of top scenarios to retrieve
        """
        self.top_k = top_k
        self.encoder = ScenarioEncoder(model_name=model_name)
        self.knowledge_base = KnowledgeBase(knowledge_base_path=knowledge_base_path)
        self.vector_store = VectorStore(
            index_type=index_type,
            vector_dim=self.encoder.get_vector_dim()
        )
        self.initialized = False
    
    def initialize(self, load_mock_data: bool = True, load_from_json: Optional[str] = None):
        """
        Initialize the RAG engine by building the knowledge base and vector index
        
        Args:
            load_mock_data: Whether to load mock data
            load_from_json: Optional path to JSON file with scenarios
        """
        if self.initialized:
            print("[RAGEngine] Already initialized")
            return
        
        # Load knowledge base
        if load_mock_data:
            self.knowledge_base.load_mock_data()
        
        if load_from_json and os.path.exists(load_from_json):
            self.knowledge_base.load_from_json(load_from_json)
        
        if self.knowledge_base.size() == 0:
            print("[RAGEngine] Warning: Knowledge base is empty")
            self.initialized = True
            return
        
        # Encode all scenarios
        scenario_descriptions = self.knowledge_base.get_scenario_descriptions()
        print(f"[RAGEngine] Encoding {len(scenario_descriptions)} scenarios...")
        vectors = self.encoder.encode_batch(scenario_descriptions)
        
        # Build vector index
        scenarios = self.knowledge_base.get_all_scenarios()
        self.vector_store.build_index(vectors, scenario_descriptions, scenarios)
        
        self.initialized = True
        print(f"[RAGEngine] Initialized with {self.knowledge_base.size()} scenarios")
    
    def retrieve_relevant_scenarios(self, seed_scenario: str, k: Optional[int] = None) -> List[str]:
        """
        Retrieve k most relevant scenarios for a seed scenario
        
        Args:
            seed_scenario: Text description of the seed scenario
            k: Number of scenarios to retrieve (defaults to self.top_k)
            
        Returns:
            List of retrieved scenario description texts
        """
        if not self.initialized:
            print("[RAGEngine] Warning: Engine not initialized, calling initialize()")
            self.initialize()
        
        if k is None:
            k = self.top_k
        
        # Encode seed scenario
        query_vector = self.encoder.encode(seed_scenario)
        
        # Search for similar scenarios
        results = self.vector_store.search(query_vector, k=k)
        
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
        
        # Retrieve scenarios
        retrieved_texts = self.retrieve_relevant_scenarios(query_scenario, k=k)
        
        # Get full scenario metadata if available
        retrieved_scenarios = []
        for text in retrieved_texts:
            # Try to find matching scenario in knowledge base
            for scenario in self.knowledge_base.get_all_scenarios():
                if scenario.get('description', '') == text or scenario.get('text', '') == text:
                    retrieved_scenarios.append(scenario)
                    break
            else:
                # If not found, create a simple dict
                retrieved_scenarios.append({'description': text})
        
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
            print(f"[RAGEngine] Rebuilt index with {self.knowledge_base.size()} scenarios")
    
    def save_knowledge_base(self, file_path: str):
        """Save the knowledge base to a JSON file"""
        self.knowledge_base.save_to_json(file_path)
    
    def save_vector_index(self, path: str):
        """Save the vector index to disk"""
        self.vector_store.save_index(path)


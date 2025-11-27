"""
Knowledge Base Module
Manages scenario knowledge base, supports loading from multiple data sources
"""

import json
import os
from typing import List, Dict, Any
from pathlib import Path


class KnowledgeBase:
    """
    Manages a knowledge base of driving scenarios
    Supports loading from JSON files and mock data
    """
    
    def __init__(self, knowledge_base_path: str = "./data/knowledge_base/"):
        """
        Initialize the knowledge base
        
        Args:
            knowledge_base_path: Path to the knowledge base directory
        """
        self.knowledge_base_path = knowledge_base_path
        self.scenarios: List[Dict[str, Any]] = []
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure the knowledge base directory exists"""
        os.makedirs(self.knowledge_base_path, exist_ok=True)
    
    def load_from_json(self, file_path: str) -> bool:
        """
        Load scenarios from a JSON file
        
        Args:
            file_path: Path to JSON file containing scenarios
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                self.scenarios.extend(data)
            elif isinstance(data, dict) and 'scenarios' in data:
                self.scenarios.extend(data['scenarios'])
            else:
                print(f"[KnowledgeBase] Warning: Unexpected JSON format in {file_path}")
                return False
            
            print(f"[KnowledgeBase] Loaded {len(data) if isinstance(data, list) else len(data.get('scenarios', []))} scenarios from {file_path}")
            return True
        except Exception as e:
            print(f"[KnowledgeBase] Error loading {file_path}: {e}")
            return False
    
    def load_mock_data(self) -> bool:
        """
        Load mock scenario data for initial implementation
        
        Returns:
            True if successful
        """
        mock_scenarios = [
            {
                "id": "mock_001",
                "description": "A vehicle is changing lanes in front of the ego vehicle during heavy rain",
                "type": "lane_change",
                "weather": "rain",
                "risk_level": "high"
            },
            {
                "id": "mock_002",
                "description": "A pedestrian suddenly crosses the road at an intersection",
                "type": "pedestrian",
                "weather": "clear",
                "risk_level": "high"
            },
            {
                "id": "mock_003",
                "description": "Multiple vehicles are merging from an on-ramp simultaneously",
                "type": "merging",
                "weather": "clear",
                "risk_level": "medium"
            },
            {
                "id": "mock_004",
                "description": "A vehicle runs a red light at a busy intersection",
                "type": "traffic_violation",
                "weather": "clear",
                "risk_level": "high"
            },
            {
                "id": "mock_005",
                "description": "The ego vehicle encounters dense fog reducing visibility to less than 50 meters",
                "type": "weather",
                "weather": "fog",
                "risk_level": "high"
            },
            {
                "id": "mock_006",
                "description": "A vehicle suddenly brakes hard in front causing a potential rear-end collision",
                "type": "emergency_brake",
                "weather": "clear",
                "risk_level": "high"
            },
            {
                "id": "mock_007",
                "description": "Construction zone with narrowed lanes and reduced speed limit",
                "type": "construction",
                "weather": "clear",
                "risk_level": "medium"
            },
            {
                "id": "mock_008",
                "description": "A vehicle cuts in front of the ego vehicle with minimal safe distance",
                "type": "cut_in",
                "weather": "clear",
                "risk_level": "high"
            },
            {
                "id": "mock_009",
                "description": "Roundabout with multiple vehicles entering and exiting simultaneously",
                "type": "roundabout",
                "weather": "clear",
                "risk_level": "medium"
            },
            {
                "id": "mock_010",
                "description": "A vehicle is reversing out of a parking space into the ego vehicle's path",
                "type": "reversing",
                "weather": "clear",
                "risk_level": "high"
            },
            {
                "id": "mock_011",
                "description": "High-speed highway scenario with vehicles changing lanes frequently",
                "type": "highway",
                "weather": "clear",
                "risk_level": "medium"
            },
            {
                "id": "mock_012",
                "description": "Slippery road conditions after rain causing reduced traction",
                "type": "weather",
                "weather": "rain",
                "risk_level": "high"
            },
            {
                "id": "mock_013",
                "description": "A vehicle makes an illegal U-turn in front of the ego vehicle",
                "type": "traffic_violation",
                "weather": "clear",
                "risk_level": "high"
            },
            {
                "id": "mock_014",
                "description": "School zone with children crossing and reduced speed limit",
                "type": "pedestrian",
                "weather": "clear",
                "risk_level": "high"
            },
            {
                "id": "mock_015",
                "description": "Multiple vehicles racing and weaving through traffic",
                "type": "aggressive_driving",
                "weather": "clear",
                "risk_level": "high"
            },
            {
                "id": "mock_016",
                "description": "A large truck blocks visibility while making a wide turn",
                "type": "visibility",
                "weather": "clear",
                "risk_level": "medium"
            },
            {
                "id": "mock_017",
                "description": "Night driving scenario with limited street lighting",
                "type": "lighting",
                "weather": "clear",
                "risk_level": "medium"
            },
            {
                "id": "mock_018",
                "description": "A vehicle stops suddenly in the middle of the road without warning",
                "type": "emergency_stop",
                "weather": "clear",
                "risk_level": "high"
            },
            {
                "id": "mock_019",
                "description": "Intersection with malfunctioning traffic lights",
                "type": "infrastructure",
                "weather": "clear",
                "risk_level": "high"
            },
            {
                "id": "mock_020",
                "description": "A vehicle swerves to avoid an obstacle and enters the ego vehicle's lane",
                "type": "swerving",
                "weather": "clear",
                "risk_level": "high"
            }
        ]
        
        self.scenarios.extend(mock_scenarios)
        print(f"[KnowledgeBase] Loaded {len(mock_scenarios)} mock scenarios")
        return True
    
    def add_scenario(self, scenario: Dict[str, Any]) -> bool:
        """
        Add a single scenario to the knowledge base
        
        Args:
            scenario: Dictionary containing scenario information
            
        Returns:
            True if successful
        """
        if not isinstance(scenario, dict):
            print("[KnowledgeBase] Error: scenario must be a dictionary")
            return False
        
        # Ensure scenario has required fields
        if 'description' not in scenario:
            print("[KnowledgeBase] Warning: scenario missing 'description' field")
        
        self.scenarios.append(scenario)
        return True
    
    def get_all_scenarios(self) -> List[Dict[str, Any]]:
        """
        Get all scenarios in the knowledge base
        
        Returns:
            List of scenario dictionaries
        """
        return self.scenarios.copy()
    
    def get_scenario_descriptions(self) -> List[str]:
        """
        Extract description texts from all scenarios
        
        Returns:
            List of scenario description strings
        """
        descriptions = []
        for scenario in self.scenarios:
            if 'description' in scenario:
                descriptions.append(scenario['description'])
            elif 'text' in scenario:
                descriptions.append(scenario['text'])
            else:
                # Fallback: use string representation
                descriptions.append(str(scenario))
        return descriptions
    
    def save_to_json(self, file_path: str) -> bool:
        """
        Save all scenarios to a JSON file
        
        Args:
            file_path: Path to save the JSON file
            
        Returns:
            True if successful
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.scenarios, f, indent=2, ensure_ascii=False)
            print(f"[KnowledgeBase] Saved {len(self.scenarios)} scenarios to {file_path}")
            return True
        except Exception as e:
            print(f"[KnowledgeBase] Error saving to {file_path}: {e}")
            return False
    
    def clear(self):
        """Clear all scenarios from the knowledge base"""
        self.scenarios = []
        print("[KnowledgeBase] Cleared all scenarios")
    
    def size(self) -> int:
        """Get the number of scenarios in the knowledge base"""
        return len(self.scenarios)


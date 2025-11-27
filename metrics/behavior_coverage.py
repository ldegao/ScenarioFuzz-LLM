"""
Physical Behavior Equivalence Class Coverage (PEC) Metric
Evaluates coverage of physical behavior equivalence classes
"""

import numpy as np
from typing import List, Dict, Set
from sklearn.cluster import KMeans, DBSCAN
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario import Scenario
from states import ScenarioState


class BehaviorCoverage:
    """
    Calculates physical behavior equivalence class coverage
    PEC = |independent behavior classes| / |total behavior classes|
    """
    
    def __init__(self, n_clusters: int = 10, use_dbscan: bool = False):
        """
        Initialize behavior coverage calculator
        
        Args:
            n_clusters: Number of clusters for K-means (if not using DBSCAN)
            use_dbscan: Whether to use DBSCAN instead of K-means
        """
        self.n_clusters = n_clusters
        self.use_dbscan = use_dbscan
        self.clusterer = None
        self.behavior_classes: Set[int] = set()
    
    def extract_features(self, scenario_state: ScenarioState) -> np.ndarray:
        """
        Extract behavioral features from scenario state
        
        Args:
            scenario_state: ScenarioState object
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Speed features
        if scenario_state.speed and len(scenario_state.speed) > 0:
            features.append(np.mean(scenario_state.speed))
            features.append(np.std(scenario_state.speed))
            features.append(np.max(scenario_state.speed))
            features.append(np.min(scenario_state.speed))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Acceleration features (derived from speed)
        if scenario_state.speed and len(scenario_state.speed) > 1:
            accelerations = np.diff(scenario_state.speed)
            features.append(np.mean(accelerations))
            features.append(np.std(accelerations))
            features.append(np.max(accelerations))
            features.append(np.min(accelerations))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Steering/yaw features
        if scenario_state.yaw_list and len(scenario_state.yaw_list) > 0:
            features.append(np.mean(scenario_state.yaw_list))
            features.append(np.std(scenario_state.yaw_list))
            if len(scenario_state.yaw_list) > 1:
                yaw_rates = np.diff(scenario_state.yaw_list)
                features.append(np.mean(yaw_rates))
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Minimum distance feature
        if hasattr(scenario_state, 'min_dist'):
            features.append(scenario_state.min_dist)
        else:
            features.append(99999.0)
        
        # Error state features
        features.append(1.0 if scenario_state.crashed else 0.0)
        features.append(1.0 if scenario_state.laneinvaded else 0.0)
        features.append(1.0 if scenario_state.speeding else 0.0)
        features.append(1.0 if scenario_state.stuck else 0.0)
        
        return np.array(features)
    
    def cluster_behaviors(self, features: np.ndarray) -> List[int]:
        """
        Cluster behaviors based on feature vectors
        
        Args:
            features: Feature matrix of shape (n_scenarios, n_features)
            
        Returns:
            List of cluster labels
        """
        if features.shape[0] < 2:
            return [0] * features.shape[0]
        
        if self.use_dbscan:
            self.clusterer = DBSCAN(eps=0.5, min_samples=2)
        else:
            n_clusters = min(self.n_clusters, features.shape[0])
            self.clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        labels = self.clusterer.fit_predict(features)
        
        # Handle DBSCAN noise points (-1 labels)
        if self.use_dbscan:
            max_label = np.max(labels)
            labels[labels == -1] = max_label + 1
        
        return labels.tolist()
    
    def calculate_coverage(self, scenarios: List[Scenario]) -> float:
        """
        Calculate behavior coverage for a list of scenarios
        
        Args:
            scenarios: List of Scenario objects
            
        Returns:
            Coverage ratio between 0 and 1
        """
        if len(scenarios) == 0:
            return 0.0
        
        # Extract features from all scenarios
        feature_matrix = []
        for scenario in scenarios:
            if hasattr(scenario, 'state') and scenario.state:
                features = self.extract_features(scenario.state)
                feature_matrix.append(features)
        
        if len(feature_matrix) == 0:
            return 0.0
        
        feature_matrix = np.array(feature_matrix)
        
        # Cluster behaviors
        labels = self.cluster_behaviors(feature_matrix)
        
        # Track unique behavior classes
        self.behavior_classes.update(labels)
        
        # Calculate coverage
        # Coverage is the ratio of unique classes to total possible classes
        # For simplicity, we use the number of unique classes found
        unique_classes = len(set(labels))
        total_possible = self.n_clusters if not self.use_dbscan else unique_classes
        
        # Normalize coverage
        coverage = unique_classes / max(total_possible, 1)
        
        return float(min(coverage, 1.0))
    
    def get_behavior_classes(self) -> Set[int]:
        """
        Get the set of behavior class labels found
        
        Returns:
            Set of behavior class labels
        """
        return self.behavior_classes.copy()
    
    def reset(self):
        """Reset the coverage tracking"""
        self.behavior_classes.clear()
        self.clusterer = None


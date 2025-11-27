"""
Trajectory Pattern Diversity (TCD) Metric
Evaluates trajectory diversity using DTW distance, clustering, and entropy
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario import Scenario
from states import ScenarioState


class TrajectoryDiversity:
    """
    Calculates trajectory pattern diversity using DTW + clustering + entropy
    TCD measures the diversity of trajectory patterns through entropy
    """
    
    def __init__(self, n_clusters: int = 10, use_dtw: bool = True):
        """
        Initialize trajectory diversity calculator
        
        Args:
            n_clusters: Number of clusters for trajectory grouping
            use_dtw: Whether to use DTW distance (if False, uses Euclidean)
        """
        self.n_clusters = n_clusters
        self.use_dtw = use_dtw
        self.trajectories: List[np.ndarray] = []
        self.cluster_labels: List[int] = []
    
    def extract_trajectories(self, scenarios: List[Scenario]) -> List[np.ndarray]:
        """
        Extract trajectory data from scenarios
        
        Args:
            scenarios: List of Scenario objects
            
        Returns:
            List of trajectory arrays, each of shape (n_points, 2) for (x, y) coordinates
        """
        trajectories = []
        
        for scenario in scenarios:
            if not hasattr(scenario, 'state') or not scenario.state:
                continue
            
            state = scenario.state
            
            # Extract trajectory from position data
            # Use yaw_list and speed to reconstruct trajectory if position not available
            trajectory_points = []
            
            if hasattr(state, 'lon_speed_list') and hasattr(state, 'lat_speed_list'):
                # Reconstruct from speed components
                if len(state.lon_speed_list) > 0 and len(state.lat_speed_list) > 0:
                    # Integrate speeds to get positions (simplified)
                    dt = 1.0 / 25.0  # Assuming 25 FPS
                    x, y = 0.0, 0.0
                    for lon_speed, lat_speed in zip(state.lon_speed_list, state.lat_speed_list):
                        x += lon_speed * dt
                        y += lat_speed * dt
                        trajectory_points.append([x, y])
            
            elif hasattr(state, 'yaw_list') and state.speed:
                # Reconstruct from yaw and speed
                dt = 1.0 / 25.0
                x, y = 0.0, 0.0
                for i, (yaw, speed) in enumerate(zip(state.yaw_list[:len(state.speed)], state.speed)):
                    dx = speed * np.cos(yaw) * dt
                    dy = speed * np.sin(yaw) * dt
                    x += dx
                    y += dy
                    trajectory_points.append([x, y])
            
            if len(trajectory_points) > 0:
                trajectories.append(np.array(trajectory_points))
        
        self.trajectories = trajectories
        return trajectories
    
    def _dtw_distance(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """
        Calculate DTW distance between two trajectories
        
        Args:
            traj1: First trajectory array
            traj2: Second trajectory array
            
        Returns:
            DTW distance
        """
        try:
            from dtaidistance import dtw
            # Flatten trajectories for DTW (use Euclidean distance in 2D space)
            distance = dtw.distance(traj1, traj2)
            return distance
        except ImportError:
            # Fallback to Euclidean distance if dtaidistance not available
            # Interpolate to same length
            n = max(len(traj1), len(traj2))
            if len(traj1) < n:
                indices = np.linspace(0, len(traj1) - 1, n).astype(int)
                traj1 = traj1[indices]
            if len(traj2) < n:
                indices = np.linspace(0, len(traj2) - 1, n).astype(int)
                traj2 = traj2[indices]
            
            # Calculate Euclidean distance
            return np.linalg.norm(traj1 - traj2)
    
    def calculate_dtw_matrix(self, trajectories: List[np.ndarray]) -> np.ndarray:
        """
        Calculate pairwise DTW distance matrix
        
        Args:
            trajectories: List of trajectory arrays
            
        Returns:
            Distance matrix of shape (n, n)
        """
        n = len(trajectories)
        if n == 0:
            return np.array([])
        
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.use_dtw:
                    distance = self._dtw_distance(trajectories[i], trajectories[j])
                else:
                    # Use Euclidean distance on interpolated trajectories
                    traj1, traj2 = trajectories[i], trajectories[j]
                    max_len = max(len(traj1), len(traj2))
                    if len(traj1) < max_len:
                        indices = np.linspace(0, len(traj1) - 1, max_len).astype(int)
                        traj1 = traj1[indices]
                    if len(traj2) < max_len:
                        indices = np.linspace(0, len(traj2) - 1, max_len).astype(int)
                        traj2 = traj2[indices]
                    distance = np.linalg.norm(traj1 - traj2)
                
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        return distance_matrix
    
    def cluster_trajectories(self, dtw_matrix: np.ndarray) -> List[int]:
        """
        Cluster trajectories based on DTW distance matrix
        
        Args:
            dtw_matrix: Distance matrix
            
        Returns:
            List of cluster labels
        """
        if dtw_matrix.shape[0] < 2:
            return [0] * dtw_matrix.shape[0]
        
        # Convert to condensed distance matrix for linkage
        condensed_distances = squareform(dtw_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distances, method='ward')
        
        # Cut tree to get clusters
        n_clusters = min(self.n_clusters, dtw_matrix.shape[0])
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        self.cluster_labels = labels.tolist()
        return labels.tolist()
    
    def calculate_entropy(self, clusters: List[int]) -> float:
        """
        Calculate entropy of cluster distribution
        
        Args:
            clusters: List of cluster labels
            
        Returns:
            Entropy value H = -Σ p_i * log(p_i)
        """
        if len(clusters) == 0:
            return 0.0
        
        # Count occurrences of each cluster
        unique_labels, counts = np.unique(clusters, return_counts=True)
        
        # Calculate probabilities
        probabilities = counts / len(clusters)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        return float(entropy)
    
    def calculate_coverage(self, scenarios: List[Scenario]) -> Dict[str, float]:
        """
        Calculate trajectory diversity metrics for scenarios
        
        Args:
            scenarios: List of Scenario objects
            
        Returns:
            Dictionary containing 'entropy' and 'num_clusters'
        """
        # Extract trajectories
        trajectories = self.extract_trajectories(scenarios)
        
        if len(trajectories) < 2:
            return {'entropy': 0.0, 'num_clusters': 0, 'diversity_score': 0.0}
        
        # Calculate distance matrix
        dtw_matrix = self.calculate_dtw_matrix(trajectories)
        
        # Cluster trajectories
        labels = self.cluster_trajectories(dtw_matrix)
        
        # Calculate entropy
        entropy = self.calculate_entropy(labels)
        
        # Calculate diversity score (normalized entropy)
        max_entropy = np.log(len(set(labels))) if len(set(labels)) > 1 else 1.0
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return {
            'entropy': entropy,
            'num_clusters': len(set(labels)),
            'diversity_score': float(diversity_score)
        }
    
    def reset(self):
        """Reset the trajectory tracking"""
        self.trajectories.clear()
        self.cluster_labels.clear()


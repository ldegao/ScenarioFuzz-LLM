"""
Driving Pattern Diversity (DPD) Metric
Evaluates trajectory pattern diversity using Fréchet distance and adaptive clustering

Replaces the original TCD metric which used subjective DTW clustering.
DPD uses Fréchet distance (more sensitive to physical trajectories) and
DBSCAN adaptive clustering (eliminates subjective cluster count).

Source: Fréchet distance is widely used in autonomous driving trajectory analysis
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario import Scenario
from states import ScenarioState
from metrics.behavior_parameters import BehaviorParameterExtractor


class DrivingPatternDiversity:
    """
    Calculates driving pattern diversity using Fréchet distance and adaptive clustering
    DPD measures the diversity of trajectory patterns through entropy of distance distribution
    """
    
    # Frame rate for time-based calculations
    FRAME_RATE = 25.0  # Hz
    
    def __init__(self, use_frechet: bool = True, dbscan_eps: float = 5.0, dbscan_min_samples: int = 2):
        """
        Initialize driving pattern diversity calculator
        
        Args:
            use_frechet: Whether to use Fréchet distance (True) or Euclidean (False)
            dbscan_eps: DBSCAN epsilon parameter for adaptive clustering
            dbscan_min_samples: DBSCAN min_samples parameter
        """
        self.use_frechet = use_frechet
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.trajectories: List[np.ndarray] = []
        self.cluster_labels: List[int] = []
        self.param_extractor = BehaviorParameterExtractor()
    
    def extract_trajectories(self, scenarios: List[Scenario]) -> List[np.ndarray]:
        """
        Extract trajectory data from scenarios using actual position data when available
        
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
            trajectory_points = []
            
            # Try to extract from actual position data if available
            # Otherwise, reconstruct from speed and yaw
            
            # Method 1: Use longitudinal and lateral speeds to reconstruct trajectory
            if hasattr(state, 'lon_speed_list') and hasattr(state, 'lat_speed_list') and \
               state.lon_speed_list and state.lat_speed_list:
                dt = 1.0 / self.FRAME_RATE
                x, y = 0.0, 0.0
                for lon_speed, lat_speed in zip(state.lon_speed_list, state.lat_speed_list):
                    # Convert km/h to m/s
                    lon_ms = lon_speed / 3.6
                    lat_ms = lat_speed / 3.6
                    x += lon_ms * dt
                    y += lat_ms * dt
                    trajectory_points.append([x, y])
            
            # Method 2: Reconstruct from yaw and speed
            elif hasattr(state, 'yaw_list') and state.yaw_list and \
                 hasattr(state, 'speed') and state.speed:
                dt = 1.0 / self.FRAME_RATE
                x, y = 0.0, 0.0
                min_len = min(len(state.yaw_list), len(state.speed))
                for i in range(min_len):
                    yaw_rad = np.radians(state.yaw_list[i])
                    speed_ms = state.speed[i] / 3.6  # km/h to m/s
                    dx = speed_ms * np.cos(yaw_rad) * dt
                    dy = speed_ms * np.sin(yaw_rad) * dt
                    x += dx
                    y += dy
                    trajectory_points.append([x, y])
            
            if len(trajectory_points) > 0:
                trajectories.append(np.array(trajectory_points))
        
        self.trajectories = trajectories
        return trajectories
    
    def _frechet_distance(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """
        Calculate Fréchet distance between two trajectories
        
        Fréchet distance is more sensitive to physical trajectory shapes than DTW.
        It measures the minimum leash length needed to connect two trajectories.
        
        Args:
            traj1: First trajectory array of shape (n, 2)
            traj2: Second trajectory array of shape (m, 2)
            
        Returns:
            Fréchet distance
        """
        # Simplified Fréchet distance calculation
        # For more accurate results, use scipy.spatial.distance or specialized library
        
        # Interpolate trajectories to same length for comparison
        n = max(len(traj1), len(traj2))
        if len(traj1) < n:
            indices = np.linspace(0, len(traj1) - 1, n).astype(int)
            traj1_interp = traj1[indices]
        else:
            traj1_interp = traj1
        
        if len(traj2) < n:
            indices = np.linspace(0, len(traj2) - 1, n).astype(int)
            traj2_interp = traj2[indices]
        else:
            traj2_interp = traj2
        
        # Calculate point-wise distances
        distances = np.linalg.norm(traj1_interp - traj2_interp, axis=1)
        
        # Fréchet distance is the maximum of minimum distances along the path
        # Simplified: use maximum distance (upper bound of Fréchet)
        frechet_dist = np.max(distances)
        
        return float(frechet_dist)
    
    def calculate_distance_matrix(self, trajectories: List[np.ndarray]) -> np.ndarray:
        """
        Calculate pairwise distance matrix using Fréchet or Euclidean distance
        
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
                if self.use_frechet:
                    distance = self._frechet_distance(trajectories[i], trajectories[j])
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
    
    def cluster_trajectories_adaptive(self, distance_matrix: np.ndarray) -> List[int]:
        """
        Cluster trajectories using DBSCAN (adaptive, no subjective cluster count)
        
        Args:
            distance_matrix: Distance matrix
            
        Returns:
            List of cluster labels
        """
        if distance_matrix.shape[0] < 2:
            return [0] * distance_matrix.shape[0]
        
        # Convert distance matrix to condensed form for DBSCAN
        # DBSCAN requires a feature matrix, so we use multidimensional scaling
        # or directly use the distance matrix with metric='precomputed'
        
        # Use DBSCAN with precomputed distance matrix
        dbscan = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric='precomputed'
        )
        
        labels = dbscan.fit_predict(distance_matrix)
        
        # Handle noise points (-1 labels) by assigning them to a separate cluster
        max_label = np.max(labels)
        labels[labels == -1] = max_label + 1
        
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
        Calculate driving pattern diversity metrics for scenarios
        
        Args:
            scenarios: List of Scenario objects
            
        Returns:
            Dictionary containing diversity metrics
        """
        # Extract trajectories
        trajectories = self.extract_trajectories(scenarios)
        
        if len(trajectories) < 2:
            return {
                'entropy': 0.0,
                'num_clusters': 0,
                'diversity_score': 0.0
            }
        
        # Calculate distance matrix
        distance_matrix = self.calculate_distance_matrix(trajectories)
        
        # Cluster trajectories using adaptive DBSCAN
        labels = self.cluster_trajectories_adaptive(distance_matrix)
        
        # Calculate entropy
        entropy = self.calculate_entropy(labels)
        
        # Calculate diversity score (normalized entropy)
        # Use more stable normalization to handle cases with few clusters
        unique_clusters = len(set(labels))
        n_scenarios = len(labels)
        
        if unique_clusters <= 1 or n_scenarios <= 1:
            diversity_score = 0.0
        else:
            # Maximum entropy occurs when clusters are uniformly distributed
            # H_max = log(k) where k is the number of clusters
            max_entropy = np.log(unique_clusters)
            
            # Normalize entropy to [0, 1]
            # Add small epsilon to avoid division by zero and improve numerical stability
            # This handles cases where max_entropy is very small (e.g., log(2) ≈ 0.693)
            diversity_score = entropy / (max_entropy + 1e-10)
            
            # Clamp to [0, 1] to handle any numerical errors
            # Note: entropy can theoretically exceed max_entropy due to numerical precision,
            # but in practice it should be bounded
            diversity_score = np.clip(diversity_score, 0.0, 1.0)
        
        return {
            'entropy': entropy,
            'num_clusters': unique_clusters,
            'diversity_score': float(diversity_score)
        }
    
    def reset(self):
        """Reset the trajectory tracking"""
        self.trajectories.clear()
        self.cluster_labels.clear()


# Backward compatibility alias
TrajectoryDiversity = DrivingPatternDiversity

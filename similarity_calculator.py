#!/usr/bin/env python3
"""
Similarity Calculator Module
Implements three non-GPT similarity scoring methods: embedding, feature, and hybrid.
"""

import json
import os
import re
import math
import numpy as np
from typing import Dict, List, Optional, Tuple


def normalize_similarity_score(score: float, min_val: float = -1.0, max_val: float = 1.0) -> int:
    """
    Normalize similarity score to 0-100 range.
    
    Args:
        score: Raw similarity score
        min_val: Minimum possible value for the score
        max_val: Maximum possible value for the score
        
    Returns:
        Normalized score in 0-100 range
    """
    if max_val == min_val:
        return 50  # Default to middle if range is zero
    
    # Normalize to [0, 1] then scale to [0, 100]
    normalized = (score - min_val) / (max_val - min_val)
    normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
    return int(normalized * 100)


def extract_features_from_json(time_record_path: str, frame_number: int = -1) -> Dict:
    """
    Extract features from time_record JSON file.
    
    Args:
        time_record_path: Path to time_record JSON file
        frame_number: Frame number to extract (default: -1 for min_dist_frame)
        
    Returns:
        Dictionary containing extracted features
    """
    try:
        if not os.path.exists(time_record_path):
            return {}
        
        with open(time_record_path, 'r') as f:
            data = json.load(f)
        
        # Determine frame number
        if frame_number == -1:
            frame_number = data.get("min_dist_frame", -1)
            if frame_number == -1:
                # Use last frame if min_dist_frame not available
                frame_keys = [int(k) for k in data.keys() if k.isdigit()]
                if frame_keys:
                    frame_number = max(frame_keys)
                else:
                    return {}
        
        frame_key = str(frame_number)
        if frame_key not in data:
            return {}
        
        frame_data = data[frame_key]
        
        # Extract player (ADS vehicle) features
        player_features = {}
        if "player" in frame_data:
            player = frame_data["player"]
            if "transform" in player and "location" in player["transform"]:
                loc = player["transform"]["location"]
                player_features["position"] = (loc.get("x", 0.0), loc.get("y", 0.0))
            
            if "velocity" in player:
                vel = player["velocity"]
                speed = math.sqrt(vel.get("x", 0)**2 + vel.get("y", 0)**2 + vel.get("z", 0)**2)
                player_features["speed"] = speed
            
            # Calculate angular acceleration if available
            if "angular_velocity" in player:
                ang_vel = player["angular_velocity"]
                player_features["angular_velocity"] = ang_vel.get("z", 0.0)
        
        # Extract NPC features (highest risk vehicle)
        npc_features = []
        if "NPC" in frame_data:
            for npc in frame_data["NPC"]:
                npc_feat = {}
                if "transform" in npc and "location" in npc["transform"]:
                    loc = npc["transform"]["location"]
                    npc_feat["position"] = (loc.get("x", 0.0), loc.get("y", 0.0))
                    
                    # Calculate relative position to player
                    if "position" in player_features:
                        player_pos = player_features["position"]
                        npc_feat["relative_position"] = (
                            loc.get("x", 0.0) - player_pos[0],
                            loc.get("y", 0.0) - player_pos[1]
                        )
                
                if "velocity" in npc:
                    vel = npc["velocity"]
                    speed = math.sqrt(vel.get("x", 0)**2 + vel.get("y", 0)**2 + vel.get("z", 0)**2)
                    npc_feat["speed"] = speed
                
                if "angular_velocity" in npc:
                    ang_vel = npc["angular_velocity"]
                    npc_feat["angular_velocity"] = ang_vel.get("z", 0.0)
                
                npc_features.append(npc_feat)
        
        # Find highest risk NPC (closest to player)
        highest_risk_npc = None
        min_distance = float('inf')
        if "position" in player_features and npc_features:
            player_pos = player_features["position"]
            for npc in npc_features:
                if "position" in npc:
                    npc_pos = npc["position"]
                    distance = math.sqrt(
                        (npc_pos[0] - player_pos[0])**2 + 
                        (npc_pos[1] - player_pos[1])**2
                    )
                    if distance < min_distance:
                        min_distance = distance
                        highest_risk_npc = npc
        
        return {
            "player": player_features,
            "highest_risk_npc": highest_risk_npc,
            "all_npcs": npc_features
        }
    
    except Exception as e:
        print(f"[Similarity] Warning: Failed to extract features from JSON: {e}")
        return {}


def extract_features_from_text(scenario_text: str) -> Dict:
    """
    Extract features from scenario description text.
    Tries to parse JSON format if available, otherwise returns empty dict.
    
    Args:
        scenario_text: Scenario description text
        
    Returns:
        Dictionary containing extracted features
    """
    try:
        # Try to parse as JSON first
        if scenario_text.strip().startswith('{'):
            data = json.loads(scenario_text)
            
            features = {}
            if "ADS Vehicle" in data:
                ads = data["ADS Vehicle"]
                if "Location" in ads:
                    loc_str = ads["Location"]
                    # Parse "(x, y)" format
                    match = re.match(r'\(([-\d.]+),\s*([-\d.]+)\)', loc_str)
                    if match:
                        features["ads_position"] = (float(match.group(1)), float(match.group(2)))
                
                if "Speed" in ads:
                    speed_str = ads["Speed"]
                    # Extract numeric value
                    speed_match = re.search(r'([\d.]+)', speed_str)
                    if speed_match:
                        features["ads_speed"] = float(speed_match.group(1))
                
                if "Angular Acceleration" in ads:
                    ang_str = ads["Angular Acceleration"]
                    ang_match = re.search(r'([-\d.]+)', ang_str)
                    if ang_match:
                        features["ads_angular_accel"] = float(ang_match.group(1))
            
            if "Highest Risk Background Vehicle" in data:
                npc = data["Highest Risk Background Vehicle"]
                if "Location" in npc:
                    loc_str = npc["Location"]
                    match = re.match(r'\(([-\d.]+),\s*([-\d.]+)\)', loc_str)
                    if match:
                        features["npc_position"] = (float(match.group(1)), float(match.group(2)))
                        
                        # Calculate relative position
                        if "ads_position" in features:
                            ads_pos = features["ads_position"]
                            npc_pos = features["npc_position"]
                            features["relative_position"] = (
                                npc_pos[0] - ads_pos[0],
                                npc_pos[1] - ads_pos[1]
                            )
                
                if "Speed" in npc:
                    speed_str = npc["Speed"]
                    speed_match = re.search(r'([\d.]+)', speed_str)
                    if speed_match:
                        features["npc_speed"] = float(speed_match.group(1))
                
                if "Relative Distance" in npc:
                    dist_str = npc["Relative Distance"]
                    dist_match = re.search(r'([\d.]+)', dist_str)
                    if dist_match:
                        features["relative_distance"] = float(dist_match.group(1))
            
            return features
    
    except Exception as e:
        # If JSON parsing fails, return empty dict
        pass
    
    return {}


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score in [-1, 1]
    """
    if vec1.shape != vec2.shape:
        return 0.0
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def calculate_embedding_similarity(
    current_scenario_text: str,
    scenario_database: Dict,
    rag_engine
) -> int:
    """
    Calculate embedding-based similarity score.
    
    Args:
        current_scenario_text: Current scenario description text
        scenario_database: Dictionary of existing scenarios
        rag_engine: Initialized RAG engine with encoder
        
    Returns:
        Similarity score in 0-100 range
    """
    try:
        if not scenario_database or len(scenario_database) == 0:
            return 0
        
        # Encode current scenario
        if not hasattr(rag_engine, 'encoder'):
            print("[Similarity] Warning: RAG engine has no encoder, returning 0")
            return 0
        
        current_vector = rag_engine.encoder.encode(current_scenario_text)
        
        # Calculate similarity with all scenarios in database
        max_similarity = -1.0
        for scenario_id, scenario_text in scenario_database.items():
            try:
                scenario_vector = rag_engine.encoder.encode(str(scenario_text))
                similarity = calculate_cosine_similarity(current_vector, scenario_vector)
                max_similarity = max(max_similarity, similarity)
            except Exception as e:
                print(f"[Similarity] Warning: Failed to encode scenario {scenario_id}: {e}")
                continue
        
        # Normalize to 0-100 (cosine similarity is in [-1, 1])
        return normalize_similarity_score(max_similarity, min_val=-1.0, max_val=1.0)
    
    except Exception as e:
        print(f"[Similarity] Error calculating embedding similarity: {e}")
        return 0


def calculate_feature_similarity(
    current_scenario_features: Dict,
    scenario_database: Dict,
    weights: Dict
) -> int:
    """
    Calculate feature-based similarity score.
    
    Args:
        current_scenario_features: Extracted features from current scenario
        scenario_database: Dictionary of existing scenarios (text descriptions)
        weights: Dictionary with weights for each feature dimension
            - position_weight (default: 0.3)
            - speed_weight (default: 0.3)
            - angular_accel_weight (default: 0.2)
            - relative_position_weight (default: 0.2)
        
    Returns:
        Similarity score in 0-100 range
    """
    try:
        if not scenario_database or len(scenario_database) == 0:
            return 0
        
        if not current_scenario_features:
            return 0
        
        # Extract weights with defaults
        pos_weight = weights.get("position_weight", 0.3)
        speed_weight = weights.get("speed_weight", 0.3)
        ang_accel_weight = weights.get("angular_accel_weight", 0.2)
        rel_pos_weight = weights.get("relative_position_weight", 0.2)
        
        # Normalize weights to sum to 1.0
        total_weight = pos_weight + speed_weight + ang_accel_weight + rel_pos_weight
        if total_weight > 0:
            pos_weight /= total_weight
            speed_weight /= total_weight
            ang_accel_weight /= total_weight
            rel_pos_weight /= total_weight
        
        max_overall_similarity = 0.0
        
        # Compare with each scenario in database
        for scenario_id, scenario_text in scenario_database.items():
            try:
                # Extract features from scenario text
                scenario_features = extract_features_from_text(str(scenario_text))
                
                if not scenario_features:
                    continue
                
                # Calculate similarity for each dimension
                position_sim = 0.0
                speed_sim = 0.0
                angular_accel_sim = 0.0
                relative_position_sim = 0.0
                
                # Position similarity (Euclidean distance normalized)
                if "ads_position" in current_scenario_features and "ads_position" in scenario_features:
                    curr_pos = current_scenario_features["ads_position"]
                    scen_pos = scenario_features["ads_position"]
                    distance = math.sqrt(
                        (curr_pos[0] - scen_pos[0])**2 + 
                        (curr_pos[1] - scen_pos[1])**2
                    )
                    # Normalize distance to similarity (assuming max distance of 1000m)
                    position_sim = max(0.0, 1.0 - distance / 1000.0)
                
                # Speed similarity
                if "ads_speed" in current_scenario_features and "ads_speed" in scenario_features:
                    curr_speed = current_scenario_features["ads_speed"]
                    scen_speed = scenario_features["ads_speed"]
                    speed_diff = abs(curr_speed - scen_speed)
                    # Normalize speed difference to similarity (assuming max speed of 50 m/s)
                    speed_sim = max(0.0, 1.0 - speed_diff / 50.0)
                
                # Angular acceleration similarity
                if "ads_angular_accel" in current_scenario_features and "ads_angular_accel" in scenario_features:
                    curr_ang = current_scenario_features["ads_angular_accel"]
                    scen_ang = scenario_features["ads_angular_accel"]
                    ang_diff = abs(curr_ang - scen_ang)
                    # Normalize angular difference to similarity (assuming max angular accel of 10 rad/s^2)
                    angular_accel_sim = max(0.0, 1.0 - ang_diff / 10.0)
                
                # Relative position similarity
                if "relative_position" in current_scenario_features and "relative_position" in scenario_features:
                    curr_rel = current_scenario_features["relative_position"]
                    scen_rel = scenario_features["relative_position"]
                    rel_distance = math.sqrt(
                        (curr_rel[0] - scen_rel[0])**2 + 
                        (curr_rel[1] - scen_rel[1])**2
                    )
                    # Normalize relative distance to similarity (assuming max relative distance of 100m)
                    relative_position_sim = max(0.0, 1.0 - rel_distance / 100.0)
                
                # Weighted combination
                overall_similarity = (
                    pos_weight * position_sim +
                    speed_weight * speed_sim +
                    ang_accel_weight * angular_accel_sim +
                    rel_pos_weight * relative_position_sim
                )
                
                max_overall_similarity = max(max_overall_similarity, overall_similarity)
            
            except Exception as e:
                print(f"[Similarity] Warning: Failed to calculate feature similarity for scenario {scenario_id}: {e}")
                continue
        
        # Convert to 0-100 range
        return normalize_similarity_score(max_overall_similarity, min_val=0.0, max_val=1.0)
    
    except Exception as e:
        print(f"[Similarity] Error calculating feature similarity: {e}")
        return 0


def calculate_hybrid_similarity(
    current_scenario_text: str,
    current_scenario_features: Dict,
    scenario_database: Dict,
    rag_engine,
    weights: Dict
) -> int:
    """
    Calculate hybrid similarity score combining embedding and feature similarities.
    
    Args:
        current_scenario_text: Current scenario description text
        current_scenario_features: Extracted features from current scenario
        scenario_database: Dictionary of existing scenarios
        rag_engine: Initialized RAG engine with encoder
        weights: Dictionary with weights
            - embedding_weight (default: 0.6)
            - feature_weights: nested dict with feature weights
        
    Returns:
        Similarity score in 0-100 range
    """
    try:
        embedding_weight = weights.get("embedding_weight", 0.6)
        feature_weight = 1.0 - embedding_weight
        
        # Calculate embedding similarity
        embedding_sim = calculate_embedding_similarity(
            current_scenario_text,
            scenario_database,
            rag_engine
        )
        
        # Calculate feature similarity
        feature_weights = weights.get("feature_weights", {})
        feature_sim = calculate_feature_similarity(
            current_scenario_features,
            scenario_database,
            feature_weights
        )
        
        # Weighted combination
        hybrid_score = embedding_weight * (embedding_sim / 100.0) + feature_weight * (feature_sim / 100.0)
        
        # Convert back to 0-100 range
        return normalize_similarity_score(hybrid_score, min_val=0.0, max_val=1.0)
    
    except Exception as e:
        print(f"[Similarity] Error calculating hybrid similarity: {e}")
        return 0


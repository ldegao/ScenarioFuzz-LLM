"""
Synthetic dataset utilities for RAG embedding evaluation.

We generate a small corpus of textual autonomous driving scenarios that
roughly follow the style of data/scenario_db.json, but are fully
synthetic and do not require CARLA.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Tuple


ACCIDENT_TYPES = [
    "rear_end",
    "side_collision",
    "intersection",
    "reversing",
    "rollover",
    "scraping",
    "pedestrian",
    "blind_spot",
]

ENVIRONMENTS = [
    "highway",
    "intersection",
    "urban",
    "parking_lot",
    "roundabout",
]

RISK_LEVELS = ["low", "medium", "high"]


def _project_root() -> str:
    """Return the repository root directory (one level above experiments)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def default_dataset_path(num_scenarios: int) -> str:
    """
    Default path for the synthetic dataset JSON file.

    Example:
        data/rag_embedding_eval/synthetic_scenarios_1000.json
    """
    root = _project_root()
    out_dir = os.path.join(root, "data", "rag_embedding_eval")
    os.makedirs(out_dir, exist_ok=True)
    filename = f"synthetic_scenarios_{num_scenarios}.json"
    return os.path.join(out_dir, filename)


def _generate_single_scenario(idx: int, rng: random.Random) -> Dict[str, Any]:
    """
    Generate a single synthetic scenario dict, mimicking scenario_db.json style.
    """
    accident_type = rng.choice(ACCIDENT_TYPES)
    environment = rng.choice(ENVIRONMENTS)
    risk_level = rng.choice(RISK_LEVELS)

    # Basic numeric attributes
    speed = round(rng.uniform(0.0, 15.0), 2)
    bg_speed = round(rng.uniform(0.0, 20.0), 2)
    ang_acc = round(rng.uniform(-2.0, 2.0), 3)
    bg_ang_acc = round(rng.uniform(-3.0, 3.0), 3)
    x = round(rng.uniform(-200.0, 200.0), 2)
    y = round(rng.uniform(-200.0, 200.0), 2)
    bg_x = round(x + rng.uniform(-50.0, 50.0), 2)
    bg_y = round(y + rng.uniform(-50.0, 50.0), 2)
    distance = round(((bg_x - x) ** 2 + (bg_y - y) ** 2) ** 0.5, 2)

    if risk_level == "high":
        severity = rng.randint(70, 100)
    elif risk_level == "medium":
        severity = rng.randint(40, 69)
    else:
        severity = rng.randint(0, 39)

    # Simple mapping to impact positions
    impact_position = {
        "rear_end": "Rear Bumper",
        "side_collision": "Left Door",
        "intersection": "Front Bumper",
        "reversing": "Rear Bumper",
        "rollover": "Left Front Fender",
        "scraping": "Right Door",
        "pedestrian": "Front Bumper",
        "blind_spot": "Right Rear Fender",
    }.get(accident_type, "Front Bumper")

    # Natural-language description template
    env_phrase = {
        "highway": "on a multi-lane highway",
        "intersection": "at a signalized intersection",
        "urban": "on an urban arterial road",
        "parking_lot": "inside a crowded parking lot",
        "roundabout": "in a multi-exit roundabout",
    }[environment]

    desc_risk = {
        "low": "posing limited immediate risk",
        "medium": "creating a moderate risk of collision",
        "high": "creating a high-risk situation with strong potential for collision",
    }[risk_level]

    description = (
        f"The ADS vehicle is driving {env_phrase} at approximately {speed} km/h with "
        f"angular acceleration around {ang_acc} rad/s^2. A background vehicle is located "
        f"nearby at a relative distance of about {distance} m, traveling at {bg_speed} km/h "
        f"with angular acceleration {bg_ang_acc} rad/s^2. The configuration resembles a "
        f"{accident_type.replace('_', ' ')} scenario, {desc_risk} in this environment."
    )

    scenario = {
        "id": f"synthetic_{idx:04d}",
        "Description": description,
        "ADS Vehicle": {
            "Location": f"({x}, {y})",
            "Speed": f"{speed} km/h",
            "Angular Acceleration": f"{ang_acc} rad/s^2",
        },
        "Highest Risk Background Vehicle": {
            "Vehicle ID": "synthetic_bg",
            "Location": f"({bg_x}, {bg_y})",
            "Speed": f"{bg_speed} km/h",
            "Angular Acceleration": f"{bg_ang_acc} rad/s^2",
            "Relative Distance": f"{distance} m",
        },
        "Potential Accident Data": {
            "Collision Severity": severity,
            "ADS Vehicle Impact Position": impact_position,
        },
        # Explicit labels for evaluation
        "accident_type": accident_type,
        "environment": environment,
        "risk_level": risk_level,
    }
    return scenario


def build_or_load_dataset(
    num_scenarios: int = 1000,
    output_path: str | None = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Build (or load) a synthetic scenario dataset for embedding evaluation.

    If output_path exists, the dataset is loaded from it; otherwise a new
    dataset is generated and saved.
    """
    if output_path is None:
        output_path = default_dataset_path(num_scenarios)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # We accept either a list or a dict keyed by IDs; normalize to list.
        if isinstance(data, dict):
            scenarios = [v for _, v in sorted(data.items(), key=lambda kv: kv[0])]
        else:
            scenarios = list(data)
        return scenarios

    rng = random.Random(seed)
    scenarios = [
        _generate_single_scenario(i, rng) for i in range(num_scenarios)
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)

    return scenarios


def extract_descriptions_and_labels(
    dataset: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Extract plain-text descriptions and label dicts from the dataset.

    Returns:
        descriptions: list of scenario description strings
        labels: list of dicts with keys: accident_type, environment, risk_level
    """
    descriptions: List[str] = []
    labels: List[Dict[str, str]] = []

    for item in dataset:
        descriptions.append(item.get("Description", ""))
        labels.append(
            {
                "accident_type": item.get("accident_type", "unknown"),
                "environment": item.get("environment", "unknown"),
                "risk_level": item.get("risk_level", "unknown"),
            }
        )

    return descriptions, labels



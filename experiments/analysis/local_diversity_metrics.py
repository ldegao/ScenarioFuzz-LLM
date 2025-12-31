#!/usr/bin/env python3
"""
Local Diversity Metrics (LRD / SCD / TER; legacy keys lms/sed/oscr)
-------------------------------------------------------------------

用于“同初始场景下，GPT 指导变异 vs 随机变异”的局部多样性度量。
以结构化物理特征向量 φ(x) 计算三类指标（计算逻辑保持原样）：
  - LRD（Local Rao Diversity，原 LMS）：同一 seed 的成对距离均值，对应 Rao quadratic entropy 族的均匀权重情形
  - SCD（Seed-Conditional Distortion，原 SED）：变异样本到 seed 代表点的平均距离，对应失真/率失真中的 expected distortion
  - TER（Typical-set Escape Rate，原 OSCR）：样本落在 seed 典型半径 τ 之外的比例，等价于典型集外的概率质量

计算不变，输出字段仍使用兼容键 lms/sed/oscr。

特征构建基于 Scenario.state 中的基础运动学信号（速度、横/纵向速度、偏航率等），
并使用 BehaviorParameterExtractor 提取 TTC/THW 等补充信息。
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# 复用已有标准化的行为特征提取器
from metrics.behavior_parameters import BehaviorParameterExtractor
from states import ScenarioState


@dataclass
class LocalDiversityResult:
    lms: float = 0.0
    sed: float = 0.0
    oscr: float = 0.0
    num_valid: int = 0
    num_seeds: int = 0
    num_seeds_with_pairs: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "lms": float(self.lms),
            "sed": float(self.sed),
            "oscr": float(self.oscr),
            "num_valid": int(self.num_valid),
            "num_seeds": int(self.num_seeds),
            "num_seeds_with_pairs": int(self.num_seeds_with_pairs),
        }


def _safe_stats(arr: Iterable[float]) -> Tuple[float, float, float]:
    """返回数组的均值/标准差/最大值，空数组时为 0。"""
    vals = np.asarray(list(arr), dtype=float)
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals)), float(np.max(vals))


def _extract_feature_vector(state: ScenarioState, extractor: BehaviorParameterExtractor) -> Optional[np.ndarray]:
    """
    从 ScenarioState 提取结构化物理特征向量。
    采用稳健的一阶/二阶统计量，缺失值填 0。
    """
    if state is None:
        return None

    speed_mean, speed_std, speed_max = _safe_stats(state.speed if hasattr(state, "speed") else [])
    lat_mean, lat_std, lat_max = _safe_stats(state.lat_speed_list if hasattr(state, "lat_speed_list") else [])
    lon_mean, lon_std, lon_max = _safe_stats(state.lon_speed_list if hasattr(state, "lon_speed_list") else [])
    yaw_rate_mean, yaw_rate_std, yaw_rate_max = _safe_stats(state.yaw_rate_list if hasattr(state, "yaw_rate_list") else [])

    # 提取 TTC / THW，缺失置 0
    ttc = extractor.extract_ttc(state) or 0.0
    thw = extractor.extract_thw(state) or 0.0

    # 最小距离，若无值则置 0
    min_dist = getattr(state, "min_dist", 0.0) or 0.0
    if min_dist >= 99998:
        min_dist = 0.0

    # 将速度从 km/h 转为 m/s，以统一尺度
    kmh_to_ms = 1.0 / 3.6
    features = np.array(
        [
            speed_mean * kmh_to_ms,
            speed_std * kmh_to_ms,
            speed_max * kmh_to_ms,
            lat_mean * kmh_to_ms,
            lat_std * kmh_to_ms,
            lat_max * kmh_to_ms,
            lon_mean * kmh_to_ms,
            lon_std * kmh_to_ms,
            lon_max * kmh_to_ms,
            yaw_rate_mean,
            yaw_rate_std,
            yaw_rate_max,
            min_dist,
            ttc,
            thw,
        ],
        dtype=float,
    )
    return features


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """对特征矩阵按列做 z-score，忽略 NaN，防止量纲差异带来的主导效应。"""
    if matrix.size == 0:
        return matrix
    mat = matrix.copy()
    # 先填充 NaN 为列均值
    col_mean = np.nanmean(mat, axis=0)
    inds = np.where(np.isnan(mat))
    mat[inds] = np.take(col_mean, inds[1])
    col_std = np.nanstd(mat, axis=0)
    col_std[col_std == 0] = 1.0
    return (mat - col_mean) / col_std


def _derive_seed_key(scenario: Any) -> str:
    """
    根据 seed_data 生成稳定的 seed key。
    若缺失 seed_data，则回退到 generation_id。
    """
    if hasattr(scenario, "seed_data") and isinstance(scenario.seed_data, dict):
        seed = scenario.seed_data
        key_tuple = (
            seed.get("sp_x"),
            seed.get("sp_y"),
            seed.get("sp_z"),
            seed.get("wp_x"),
            seed.get("wp_y"),
            seed.get("map"),
            seed.get("yaw"),
            seed.get("roll"),
            seed.get("pitch"),
        )
        return str(key_tuple)
    return f"gen:{getattr(scenario, 'generation_id', -1)}"


def _select_seed_representative(indices: List[int], scenario_ids: List[int], vectors: np.ndarray) -> int:
    """
    选择该 seed 的代表样本（默认 scenario_id 最小者）。
    """
    if not indices:
        return -1
    best_idx = indices[0]
    best_sid = scenario_ids[best_idx]
    for idx in indices[1:]:
        sid = scenario_ids[idx]
        if sid < best_sid:
            best_sid = sid
            best_idx = idx
    return best_idx


def compute_local_diversity_metrics(scenarios: List[Any]) -> LocalDiversityResult:
    """
    核心计算函数：给定一批场景（同一实验），返回 LMS/SED/OSCR（对应 LRD/SCD/TER）。
    """
    if not scenarios:
        return LocalDiversityResult()

    extractor = BehaviorParameterExtractor()
    feature_vectors: List[np.ndarray] = []
    seed_keys: List[str] = []
    scenario_ids: List[int] = []

    for sc in scenarios:
        state = getattr(sc, "state", None)
        vec = _extract_feature_vector(state, extractor)
        if vec is None:
            continue
        feature_vectors.append(vec)
        seed_keys.append(_derive_seed_key(sc))
        scenario_ids.append(getattr(sc, "scenario_id", -1))

    if not feature_vectors:
        return LocalDiversityResult()

    matrix = np.vstack(feature_vectors)
    norm_matrix = _normalize_matrix(matrix)

    # 按 seed 分组
    seed_to_indices: Dict[str, List[int]] = {}
    for idx, key in enumerate(seed_keys):
        seed_to_indices.setdefault(key, []).append(idx)

    num_seeds = len(seed_to_indices)
    num_valid = len(feature_vectors)

    # 计算 LRD（兼容键 lms）
    lms_values: List[float] = []
    for indices in seed_to_indices.values():
        if len(indices) < 2:
            continue
        sub = norm_matrix[indices]
        # 成对欧氏距离均值
        dists = []
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                dists.append(np.linalg.norm(sub[i] - sub[j]))
        if dists:
            lms_values.append(float(np.mean(dists)))

    lms = float(np.mean(lms_values)) if lms_values else 0.0

    # 选择每个 seed 的代表向量
    seed_reps: List[np.ndarray] = []
    for key, indices in seed_to_indices.items():
        rep_idx = _select_seed_representative(indices, scenario_ids, norm_matrix)
        if rep_idx >= 0:
            seed_reps.append(norm_matrix[rep_idx])

    # 计算 SCD（兼容键 sed）：场景到其 seed 代表的平均距离
    sed_values: List[float] = []
    for key, indices in seed_to_indices.items():
        rep_idx = _select_seed_representative(indices, scenario_ids, norm_matrix)
        if rep_idx < 0:
            continue
        rep_vec = norm_matrix[rep_idx]
        for idx in indices:
            sed_values.append(float(np.linalg.norm(norm_matrix[idx] - rep_vec)))
    sed = float(np.mean(sed_values)) if sed_values else 0.0

    # 计算 TER（兼容键 oscr）：基于 seed 代表之间的典型间距 τ
    oscr = 0.0
    if len(seed_reps) >= 2:
        seed_reps_arr = np.vstack(seed_reps)
        nn_dists: List[float] = []
        for i in range(len(seed_reps_arr)):
            others = np.delete(seed_reps_arr, i, axis=0)
            if others.size == 0:
                continue
            dist = np.linalg.norm(others - seed_reps_arr[i], axis=1)
            nn_dists.append(float(np.min(dist)))
        if nn_dists:
            tau = float(np.median(nn_dists))
            # 场景到最近 seed 代表的距离
            out_count = 0
            for vec in norm_matrix:
                nearest = np.linalg.norm(seed_reps_arr - vec, axis=1)
                if np.min(nearest) > tau:
                    out_count += 1
            oscr = float(out_count) / float(len(norm_matrix))

    return LocalDiversityResult(
        lms=lms,
        sed=sed,
        oscr=oscr,
        num_valid=num_valid,
        num_seeds=num_seeds,
        num_seeds_with_pairs=len(lms_values),
    )


def summarize_local_diversity(scenarios: List[Any]) -> Dict[str, float]:
    """
    便捷包装，直接返回 dict。
    """
    return compute_local_diversity_metrics(scenarios).to_dict()


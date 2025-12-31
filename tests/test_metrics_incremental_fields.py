import json
from pathlib import Path

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2] / "experiment_results" / "SimilarityComparison"
RECENT_EXPS = [
    "SimilarityComparison_answer2_20251215_110047",
    "SimilarityComparison_embedding_20251215_115037",
    "SimilarityComparison_feature_20251215_124633",
    "SimilarityComparison_hybrid_20251215_132204",
]


def load_records(exp_dir: Path):
    records_path = exp_dir / "metrics" / "metrics_records.jsonl"
    if not records_path.exists():
        pytest.skip(f"{records_path} 不存在，跳过")
    with open(records_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("exp_name", RECENT_EXPS)
def test_incremental_fields_exist_and_monotonic(exp_name):
    exp_dir = EXPERIMENT_ROOT / exp_name
    records = load_records(exp_dir)
    assert records, f"{exp_name} 记录为空"

    for rec in records:
        for key in ["pc", "pec", "tcd", "bcm", "pc_incremental", "pec_incremental", "tcd_incremental", "bcm_incremental"]:
            assert key in rec, f"{exp_name} 缺少字段 {key}"

    # 累积值应非下降（覆盖/多样性不应减少）
    for metric in ["pc", "pec", "tcd", "bcm"]:
        vals = [r[metric] for r in records]
        assert all(vals[i] >= vals[i - 1] - 1e-9 for i in range(1, len(vals))), f"{exp_name} {metric} 非单调"

    # 增量之和≈最终值
    for metric in ["pc", "pec", "tcd", "bcm"]:
        inc_vals = [r[f"{metric}_incremental"] for r in records]
        final_val = records[-1][metric]
        assert abs(sum(inc_vals) - final_val) < 1e-6, f"{exp_name} {metric} 增量求和不等于最终值"


def test_parameter_coverage_bins_enhanced():
    from metrics import ParameterCoverage

    pc = ParameterCoverage()
    grid = pc.parameter_grid
    assert len(grid["a_long"]) == 21  # 0.5 m/s² bins -6..4
    assert len(grid["a_lat"]) == 17
    assert len(grid["jerk_long"]) == 13
    assert len(grid["jerk_lat"]) == 9
    assert len(grid["yaw_rate"]) == 21  # 10 deg/s
    assert len(grid["ttc"]) == 13  # 0.25s


import pytest

import experiments.core.experiment_manager as experiment_manager
import fuzzer


def test_fuzzer_parser_ablation_flags_defaults():
    parser = fuzzer.set_args()
    args = parser.parse_args([])
    assert hasattr(args, "disable_similarity")
    assert hasattr(args, "disable_guided_mutation")
    assert args.disable_similarity is False
    assert args.disable_guided_mutation is False


def test_fuzzer_parser_ablation_flags_set():
    parser = fuzzer.set_args()
    args = parser.parse_args(["--disable-similarity", "--disable-guided-mutation"])
    assert args.disable_similarity is True
    assert args.disable_guided_mutation is True


def test_experiment_manager_passes_ablation_flags(tmp_path):
    manager = experiment_manager.ExperimentManager(output_base_dir=tmp_path)
    args = manager._create_args(
        method_name="ScenarioFuzz-LLM",
        output_dir=tmp_path / "out",
        target="behavior",
        town=3,
        timeout=60,
        sim_port=2000,
        max_scenarios=0,
        rag_k=5,
        similarity_scoring_method="answer2",
        hybrid_embedding_weight=0.6,
        feature_position_weight=0.3,
        feature_speed_weight=0.3,
        feature_angular_accel_weight=0.2,
        feature_relative_position_weight=0.2,
        disable_similarity=True,
        disable_guided_mutation=True,
    )
    assert args.disable_similarity is True
    assert args.disable_guided_mutation is True


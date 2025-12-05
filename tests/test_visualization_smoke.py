"""
Smoke tests for visualization utilities.

These tests ensure that plotting functions and report generation
run without raising exceptions on small synthetic data.

If optional visualization dependencies (seaborn, plotly, matplotlib)
are not installed, these tests will be skipped gracefully.
"""

from pathlib import Path

import pytest

# Skip this module entirely if visualization dependencies are missing
seaborn = pytest.importorskip("seaborn")
plotly = pytest.importorskip("plotly")
matplotlib = pytest.importorskip("matplotlib")

from visualization import (
    plot_parameter_coverage,
    plot_behavior_coverage,
    plot_trajectory_diversity,
    plot_behavior_matrix,
    ReportGenerator,
)


def test_plot_functions(tmp_path):
    # Simple data for plotting
    pc_scores = {"MethodA": 0.8, "MethodB": 0.6}
    pec_scores = {
        "MethodA": {"num_classes": 10, "coverage": 0.7},
        "MethodB": {"num_classes": 8, "coverage": 0.5},
    }
    tcd_scores = {
        "MethodA": {"entropy": 0.9, "diversity_score": 0.8},
        "MethodB": {"entropy": 0.4, "diversity_score": 0.3},
    }
    import numpy as np

    bcm_matrix = np.array([[1, 0, 1], [0, 1, 0]], dtype=float)
    behavior_labels = ["collision", "lane_invasion"]

    out_pc = tmp_path / "pc.png"
    out_pec = tmp_path / "pec.png"
    out_tcd = tmp_path / "tcd.png"
    out_bcm = tmp_path / "bcm.png"

    plot_parameter_coverage(pc_scores, output_path=str(out_pc))
    plot_behavior_coverage(pec_scores, output_path=str(out_pec))
    plot_trajectory_diversity(tcd_scores, output_path=str(out_tcd))
    plot_behavior_matrix(bcm_matrix, behavior_labels, output_path=str(out_bcm))

    for p in [out_pc, out_pec, out_tcd, out_bcm]:
        assert p.exists()
        assert p.stat().st_size > 0


def test_report_generator_markdown(tmp_path):
    output_dir = tmp_path / "reports"
    rg = ReportGenerator(output_dir=str(output_dir))

    summary = {
        "pc": {"MethodA": 0.8},
        "pec": {"MethodA": 0.7},
        "tcd": {"MethodA": 0.9},
        "bcm": {"MethodA": 0.75},
    }

    md_path = rg.generate_markdown_report("test_report", summary)
    assert Path(md_path).exists()
    assert Path(md_path).read_text().strip() != ""



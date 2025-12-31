#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics Calculator
------------------

Calculates metrics (PCE, BCE, DPE, CCE) from scenario data.
This script is decoupled from the fuzzing loop and can be run offline.
Legacy output keys remain `pc/pec/tcd/bcm` for compatibility (mapping to
PCE/BCE/DPE/CCE respectively).

Usage:
    python -m experiments.analysis.calculate_metrics \\
        --experiment-dir ./experiment_results/SimilarityComparison/... \\
        [--output-dir ./experiment_results/.../metrics] \\
        [--incremental] \\
        [--recalculate]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.scenario_loader import ScenarioDataLoader, load_all_scenarios
from experiments.aggregation.metrics_aggregator import (
    load_records_from_jsonl,
    aggregate_run_metrics,
    save_run_summary
)
from experiments.analysis.local_diversity_metrics import summarize_local_diversity


def calculate_metrics_from_scenarios(
    experiment_dir: Path,
    output_dir: Optional[Path] = None,
    incremental: bool = False,
    recalculate: bool = False
) -> Dict:
    """
    Calculate metrics from scenario data in an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory (should contain queue/ subdirectory)
        output_dir: Optional output directory for metrics (default: experiment_dir/metrics)
        incremental: If True, only calculate metrics for new scenarios
        recalculate: If True, recalculate all metrics even if they exist
        
    Returns:
        Dictionary with calculation results and statistics
    """
    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory {experiment_dir} does not exist")
    
    # Determine output directory
    if output_dir is None:
        output_dir = experiment_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    records_path = output_dir / "metrics_records.jsonl"
    
    # Load existing records if incremental mode
    existing_records = {}
    if incremental and records_path.exists() and not recalculate:
        existing_records_list = load_records_from_jsonl(str(records_path))
        for record in existing_records_list:
            key = (record.get('generation_id', -1), record.get('scenario_id', -1))
            existing_records[key] = record
    
    # Load all scenarios
    print(f"[CalculateMetrics] Loading scenarios from {experiment_dir}")
    loader = ScenarioDataLoader()
    prefer_pickle_env = os.getenv("SCENARIO_LOADER_PREFER_PICKLE", "1")
    prefer_pickle = prefer_pickle_env not in ("0", "false", "False")
    scenarios = loader.load_all_scenarios(experiment_dir, prefer_pickle=prefer_pickle)
    
    if not scenarios:
        print(f"[CalculateMetrics] WARNING: No scenarios found in {experiment_dir}")
        return {
            'num_scenarios': 0,
            'num_calculated': 0,
            'num_skipped': 0,
            'errors': []
        }
    
    # Filter scenarios for incremental mode
    scenarios_to_calculate = []
    if incremental and not recalculate:
        for scenario in scenarios:
            key = (getattr(scenario, 'generation_id', -1), getattr(scenario, 'scenario_id', -1))
            if key not in existing_records:
                scenarios_to_calculate.append(scenario)
    else:
        scenarios_to_calculate = scenarios
    
    if not scenarios_to_calculate:
        print(f"[CalculateMetrics] No new scenarios to calculate (incremental mode)")
        return {
            'num_scenarios': len(scenarios),
            'num_calculated': 0,
            'num_skipped': len(scenarios),
            'errors': []
        }
    
    print(f"[CalculateMetrics] Calculating metrics for {len(scenarios_to_calculate)} scenario(s)")
    
    # Import metrics calculators
    try:
        from metrics import (
            ParameterConfigurationEntropy,
            BehaviorCategoryEntropy,
            DrivingPatternEntropy,
            CombinationCoverageEntropy,
        )
    except ImportError as e:
        raise ImportError(f"Failed to import metrics modules: {e}. Make sure metrics package is available.")
    
    # Initialize calculators
    pce_calculator = ParameterConfigurationEntropy()
    bce_calculator = BehaviorCategoryEntropy()
    dpe_calculator = DrivingPatternEntropy()
    cce_calculator = CombinationCoverageEntropy()
    
    # Calculate cumulative metrics
    # We need to use ALL scenarios (not just new ones) for cumulative calculation
    # But only save records for new scenarios
    all_scenarios_for_calculation = scenarios  # Use all scenarios for cumulative metrics
    
    # Helper to compute metrics for a list of scenarios (cumulative up to that point)
    def _compute_metrics_for_subset(subset):
        pce_val = ParameterConfigurationEntropy().calculate_coverage(subset)
        bce_val = BehaviorCategoryEntropy().calculate_coverage(subset)
        dpe_res = DrivingPatternEntropy().calculate_coverage(subset)
        dpe_val = dpe_res.get('diversity_score', 0.0) if isinstance(dpe_res, dict) else float(dpe_res)
        cce_res = CombinationCoverageEntropy().calculate_coverage(subset)
        cce_val = cce_res.get('coverage_ratio', 0.0) if isinstance(cce_res, dict) else float(cce_res)
        return pce_val, bce_val, dpe_val, cce_val
    
    print(f"[CalculateMetrics] Calculating cumulative and incremental metrics for {len(all_scenarios_for_calculation)} scenario(s)")
    
    records_to_save = []
    errors = []
    
    last_pc = last_pec = last_tcd = last_bcm = 0.0
    
    for scenario in scenarios_to_calculate:
        try:
            # Validate scenario has required data
            if not loader.validate_scenario_for_metrics(scenario):
                errors.append(f"Scenario gid:{getattr(scenario, 'generation_id', -1)} sid:{getattr(scenario, 'scenario_id', -1)} missing required state data")
                continue
            
            # 找到该场景在全量列表中的位置，确保增量计算正确
            try:
                scenario_pos = scenarios.index(scenario)
            except ValueError:
                scenario_pos = len(scenarios) - 1
            
            # Scenarios up to current (cumulative)
            scenarios_so_far = scenarios[:scenario_pos + 1]
            pc_cum, pec_cum, tcd_cum, bcm_cum = _compute_metrics_for_subset(scenarios_so_far)
            
            # Scenarios before current (for incremental contribution)
            if scenario_pos > 0:
                scenarios_before = scenarios[:scenario_pos]
                pc_prev, pec_prev, tcd_prev, bcm_prev = _compute_metrics_for_subset(scenarios_before)
            else:
                pc_prev = pec_prev = tcd_prev = bcm_prev = 0.0
            
            record = {
                "generation_id": getattr(scenario, "generation_id", -1),
                "scenario_id": getattr(scenario, "scenario_id", -1),
                # 累积值（保持向后兼容）
                "pc": float(pc_cum),
                "pec": float(pec_cum),
                "tcd": float(tcd_cum),
                "bcm": float(bcm_cum),
                # 增量贡献（新增）
                "pc_incremental": float(pc_cum - pc_prev),
                "pec_incremental": float(pec_cum - pec_prev),
                "tcd_incremental": float(tcd_cum - tcd_prev),
                "bcm_incremental": float(bcm_cum - bcm_prev),
                "num_accumulated_scenarios": len(scenarios_so_far),
            }
            last_pc, last_pec, last_tcd, last_bcm = pc_cum, pec_cum, tcd_cum, bcm_cum
            records_to_save.append(record)
        except Exception as e:
            errors.append(f"Failed to process scenario gid:{getattr(scenario, 'generation_id', -1)} sid:{getattr(scenario, 'scenario_id', -1)}: {e}")
    
    # Save records
    if recalculate:
        # Overwrite existing file
        mode = 'w'
    else:
        # Append to existing file
        mode = 'a'
    
    with open(records_path, mode, encoding='utf-8') as f:
        for record in records_to_save:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"[CalculateMetrics] Saved {len(records_to_save)} metric record(s) to {records_path}")
    
    # Aggregate and save summary
    all_records = load_records_from_jsonl(str(records_path))
    summary = aggregate_run_metrics(all_records)

    # 附加局部多样性指标（LRD/SCD/TER；字段名兼容 lms/sed/oscr）
    local_diversity = summarize_local_diversity(scenarios)
    summary.update(local_diversity)
    
    # Extract experiment ID from directory name
    experiment_id = experiment_dir.name
    
    summary_path = output_dir / "metrics_summary.json"
    save_run_summary(
        summary,
        str(summary_path),
        method_name=None,  # Will be set by caller if needed
        experiment_id=experiment_id,
        num_scenarios=len(all_records)
    )
    
    print(f"[CalculateMetrics] Saved summary to {summary_path}")
    
    return {
        'num_scenarios': len(scenarios),
        'num_calculated': len(records_to_save),
        'num_skipped': len(scenarios) - len(records_to_save),
        'metrics': {
            'pc': float(last_pc),
            'pec': float(last_pec),
            'tcd': float(last_tcd),
            'bcm': float(last_bcm),
            'lms': float(local_diversity.get('lms', 0.0)),
            'sed': float(local_diversity.get('sed', 0.0)),
            'oscr': float(local_diversity.get('oscr', 0.0)),
        },
        'errors': errors
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate metrics (PCE, BCE, DPE, CCE) from scenario data"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Path to experiment directory (should contain queue/ subdirectory)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for metrics (default: experiment_dir/metrics)"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only calculate metrics for new scenarios (skip existing ones)"
    )
    parser.add_argument(
        "--recalculate",
        action="store_true",
        help="Recalculate all metrics even if they already exist"
    )
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    try:
        result = calculate_metrics_from_scenarios(
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            incremental=args.incremental,
            recalculate=args.recalculate
        )
        
        print(f"\n[CalculateMetrics] Calculation complete!")
        print(f"  Total scenarios: {result['num_scenarios']}")
        print(f"  Calculated: {result['num_calculated']}")
        print(f"  Skipped: {result['num_skipped']}")
        if result['errors']:
            print(f"  Errors: {len(result['errors'])}")
            for error in result['errors'][:5]:  # Show first 5 errors
                print(f"    - {error}")
        
        if result['metrics']:
            print(f"\n  Final metrics (legacy keys pc/pec/tcd/bcm map to PCE/BCE/DPE/CCE):")
            print(f"    PCE:  {result['metrics']['pc']:.6f}")
            print(f"    BCE:  {result['metrics']['pec']:.6f}")
            print(f"    DPE:  {result['metrics']['tcd']:.6f}")
            print(f"    CCE:  {result['metrics']['bcm']:.6f}")
        
    except Exception as e:
        print(f"[CalculateMetrics] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


"""复核 Phase 1.5B 的科学判定，无需重跑优化器。"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vetm.transfer_matrix import classify_effect


def analyze(results_dir: Path = ROOT / "results") -> dict:
    rows = list(csv.DictReader((results_dir / "phase1_5B_transfer_matrix.csv").open(encoding="utf-8")))
    summary = json.loads((results_dir / "phase1_5B_summary.json").read_text(encoding="utf-8"))
    health = json.loads((results_dir / "metric_health_report.json").read_text(encoding="utf-8"))
    hv = json.loads((results_dir / "phase1_5B_hv_report.json").read_text(encoding="utf-8"))
    bad_tasks = {row["task_id"] for row in health["tasks"] if row["invalid"]}
    assert len(rows) == summary["raw_row_count"]
    assert all(row["function_evaluations_equal"] == "True" for row in rows)
    assert all(row["baseline_status"] == "ok" and row["intervention_status"] == "ok" for row in rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task_id"], row["intervention_id"])].append(float(row["delta_normalized_HV"]))
    states = defaultdict(list)
    for (task_id, intervention_id), values in grouped.items():
        result = classify_effect(np.asarray(values), tolerance=0.01)
        if task_id not in bad_tasks:
            states[intervention_id].append((task_id, result["state"]))

    stable_reversals = {
        intervention_id: entries
        for intervention_id, entries in states.items()
        if {state for _, state in entries} >= {"positive", "negative"}
    }
    heldout = summary["heldout_transfer_signal"]
    actual = np.asarray([row["actual_delta"] for row in heldout])
    predicted = np.asarray([row["predicted_delta"] for row in heldout])
    heldout_mae = float(np.mean(np.abs(actual - predicted)))
    zero_mae = float(np.mean(np.abs(actual)))
    analysis = {
        "before_zero_HV_ratio": hv["before"].get("raw_zero_HV_ratio"),
        "after_zero_HV_ratio": health["zero_HV_ratio"],
        "zero_HV_ratio_decreased": health["zero_HV_ratio"] < hv["before"].get("raw_zero_HV_ratio", 1.0),
        "invalid_tasks": sorted(bad_tasks),
        "healthy_task_count": len(health["tasks"]) - len(bad_tasks),
        "stable_sign_reversals_on_healthy_tasks": stable_reversals,
        "heldout_condition_count": len(heldout),
        "heldout_mae_global_intervention_mean": heldout_mae,
        "heldout_mae_zero_gain_baseline": zero_mae,
        "heldout_signal_above_zero_baseline": heldout_mae < zero_mae,
        "all_runs_budget_equal": True,
        "go_no_go": "NO-GO",
        "reason": "有两个健康任务上的符号翻转，但 3 个任务 metric invalid，且 family-held-out 预测劣于零增益基线。",
        "comparison_caveat": "before 与 after 来自不同种子数、预算和干预集合，零 HV 比率变化不能归因为单一因素。",
    }
    (results_dir / "phase1_5B_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return analysis


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))

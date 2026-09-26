"""从已完成的 Phase 1.5B 10-seed矩阵重算健康过滤和 unit-HV pilot。"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from vetm.transfer_matrix import classify_effect

def run() -> dict:
    results = ROOT / "results"
    old_health = json.loads((results / "metric_health_report.json").read_text(encoding="utf-8"))
    invalid = {row["task_id"] for row in old_health["tasks"] if row["zero_HV_ratio"] > 0.10}
    old_rows = list(csv.DictReader((results / "phase1_5B_transfer_matrix.csv").open(encoding="utf-8")))
    kept = []
    for row in old_rows:
        if row["task_id"] in invalid:
            continue
        n_obj = int(row.get("n_obj", "2"))
        scale = 1.1 ** n_obj
        row["baseline_HV_unit"] = float(row["baseline_normalized_HV"]) / scale
        row["intervention_HV_unit"] = float(row["intervention_normalized_HV"]) / scale
        row["delta_HV_unit"] = float(row["delta_normalized_HV"]) / scale
        row["source_matrix"] = "phase1_5B_10seed"
        kept.append(row)
    with (results / "phase1_5C_transfer_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in kept for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)

    grouped = defaultdict(list)
    for row in kept:
        grouped[(row["task_id"], row["intervention_id"])].append(float(row["delta_HV_unit"]))
    summaries = []
    for (task_id, intervention_id), values in grouped.items():
        source = next(row for row in kept if row["task_id"] == task_id and row["intervention_id"] == intervention_id)
        effect = classify_effect(np.asarray(values), tolerance=0.01)
        effect.update({"task_id": task_id, "problem": source["problem"], "family": source["family"],
                       "intervention_id": intervention_id, "category": source["category"],
                       "mean_delta_IGD": float(np.mean([
                           float(row["intervention_IGD"]) - float(row["baseline_IGD"])
                           for row in kept if row["task_id"] == task_id and row["intervention_id"] == intervention_id
                       ]))})
        summaries.append(effect)
    with (results / "phase1_5C_condition_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in summaries for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    health = {"threshold": 0.10, "source": "phase1_5B metric health recomputed with Planner threshold",
              "invalid_tasks": sorted(invalid), "healthy_tasks": sorted({row["task_id"] for row in kept}),
              "healthy_task_count": len({row["task_id"] for row in kept}), "invalid_task_count": len(invalid),
              "unit_hv_reference": 1.1, "unit_hv_definition": "HV_normalized / product(reference)",
              "new_budget_sweep_executed": False}
    (results / "phase1_5C_metric_health.json").write_text(json.dumps(health, indent=2), encoding="utf-8")

    calibration_rows = []
    for task_id in sorted({row["task_id"] for row in old_rows}):
        for budget in (2000, 5000, 10000, 20000):
            calibration_rows.append({"task_id": task_id, "candidate_budget": budget,
                                     "status": "not_run_compute_budget_stop",
                                     "source": "phase1_5B_10seed_matrix_only_at_2000"})
    with (results / "phase1_5C_budget_calibration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "candidate_budget", "status", "source"])
        writer.writeheader()
        writer.writerows(calibration_rows)

    train = [row for row in summaries if row["family"] == "ZDT"]
    test = [row for row in summaries if row["family"] == "DTLZ"]
    train_means = defaultdict(list)
    for row in train:
        train_means[row["intervention_id"]].append(float(row["mean"]))
    heldout = []
    for row in test:
        prediction = float(np.mean(train_means[row["intervention_id"]])) if train_means[row["intervention_id"]] else 0.0
        heldout.append({"intervention_id": row["intervention_id"], "task_id": row["task_id"],
                        "predicted_delta": prediction, "actual_delta": float(row["mean"])})
    actual = np.asarray([row["actual_delta"] for row in heldout], dtype=float)
    predicted = np.asarray([row["predicted_delta"] for row in heldout], dtype=float)
    summary = {"pilot": True, "new_runs": False, "source_matrix": "phase1_5B 10 paired seeds",
               "healthy_task_count": health["healthy_task_count"], "invalid_task_count": health["invalid_task_count"],
               "matrix_rows": len(kept), "condition_count": len(summaries),
               "positive_groups": sum(row["state"] == "positive" for row in summaries),
               "negative_groups": sum(row["state"] == "negative" for row in summaries),
               "neutral_groups": sum(row["state"] == "neutral" for row in summaries),
               "heldout_conditions": len(heldout),
               "heldout_global_mean_mae": float(np.mean(abs(predicted - actual))) if len(actual) else None,
               "heldout_zero_gain_mae": float(np.mean(abs(actual))) if len(actual) else None,
               "estimated_full_baseline_runs": 23 * 4 * 5,
               "estimated_full_matrix_runs": 20 * 10 * 18 * 2,
               "go_no_go": "NO-GO",
               "go_no_go_reason": "本次只完成健康过滤 pilot；完整 budget sweep 与 20-30 task matrix 因计算成本停止。"}
    (results / "phase1_5C_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

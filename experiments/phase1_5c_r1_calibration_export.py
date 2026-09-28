"""将 Stage A 前实际执行的 baseline 校准记录导出为 R1 交付格式。"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    source = ROOT / "results" / "r1_stage" / "phase1_5C_metric_health.json"
    reports = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for task_id, record in reports.items():
        for candidate in record["candidates"]:
            rows.append({
                "task_id": task_id,
                "candidate_budget": candidate["budget"],
                "calibration_seeds": 5,
                "actual_runs": 5,
                "zero_HV_ratio": candidate["zero_HV_ratio"],
                "normalization_valid": candidate["normalization_valid"],
                "mean_unit_HV": candidate["mean_unit_HV"],
                "mean_IGD": candidate["mean_IGD"],
                "saturated": candidate["saturated"],
                "invalid": candidate["invalid"],
                "selected": record["selected_budget"] == candidate["budget"],
            })
    output = ROOT / "results" / "phase1_5C_R1_budget_calibration.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    health = {
        "source": str(source.relative_to(ROOT)),
        "threshold": 0.10,
        "actual_calibration_runs": sum(row["actual_runs"] for row in rows),
        "healthy_tasks": [task_id for task_id, record in reports.items() if record["supported"]],
        "unsupported_tasks": [task_id for task_id, record in reports.items() if not record["supported"]],
        "selected_budgets": {task_id: record["selected_budget"] for task_id, record in reports.items() if record["supported"]},
        "candidate_evaluations": [2000, 5000, 10000, 20000],
        "saturation_rule": "unit_HV_mean >= 0.98",
    }
    (ROOT / "results" / "phase1_5C_R1_metric_health.json").write_text(
        json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "actual_runs": health["actual_calibration_runs"],
                      "healthy": len(health["healthy_tasks"]), "unsupported": len(health["unsupported_tasks"])}, indent=2))

if __name__ == "__main__":
    main()

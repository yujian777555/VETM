"""运行 Phase 1.5B：校准 HV、健康检查和配对 transfer matrix。"""
from __future__ import annotations
import csv, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vetm.interventions import active_interventions
from vetm.metric_calibration import calibrate_task, metric_health, normalize_objectives
from vetm.nsga2 import NSGA2Config, run_nsga2
from vetm.problems import get_problem, reference_front
from vetm.transfer_matrix import classify_effect, hypervolume, igd

def _family(problem_name: str) -> str:
    return "DTLZ" if problem_name.startswith("DTLZ") else "ZDT"

def _hash_config(task: dict, intervention: dict, base: NSGA2Config) -> str:
    payload = {"task": task, "intervention": intervention, "base": base.__dict__}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

def _load_before_metrics() -> dict:
    path = ROOT / "results" / "phase1_5_transfer_matrix.csv"
    if not path.exists():
        return {"available": False}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return {"available": False}
    hv = np.asarray([float(row["baseline_hv"]) for row in rows], dtype=float)
    return {"available": True, "source": str(path), "raw_zero_HV_ratio": float(np.mean(np.isclose(hv, 0.0)))}

def run(config_path: Path = ROOT / "configs" / "phase1_5b_tasks.json", output_dir: Path = ROOT / "results") -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = list(range(int(config["seed_count"])))
    base_config = NSGA2Config(
        population_size=int(config["population_size"]),
        generations=int(config["generations"]),
    )
    tolerance = float(config["practical_tolerance"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    grouped = {}
    calibrations = {}
    health_rows = []

    for task in config["tasks"]:
        problem = get_problem(task["problem"], n_var=task["n_var"], n_obj=task["n_obj"])
        calibration = calibrate_task(problem)
        reference = reference_front(problem)
        calibrations[task["task_id"]] = {
            "problem": task["problem"],
            "n_var": task["n_var"],
            "n_obj": task["n_obj"],
            "ideal": calibration.ideal.tolist(),
            "nadir": calibration.nadir.tolist(),
            "raw_reference": calibration.raw_reference.tolist(),
            "normalized_reference": calibration.normalized_reference.tolist(),
        }
        baseline_runs = {}
        baseline_raw_hv = []
        baseline_norm_hv = []
        baseline_objectives = []
        for seed in seeds:
            run_result = run_nsga2(problem, base_config, seed=seed)
            normalized = normalize_objectives(run_result["objectives"], calibration)
            baseline_runs[seed] = run_result
            baseline_raw_hv.append(hypervolume(run_result["objectives"], calibration.raw_reference))
            baseline_norm_hv.append(hypervolume(normalized, calibration.normalized_reference))
            baseline_objectives.append(run_result["objectives"])
        interventions = active_interventions(base_config, task["n_var"])
        for intervention in interventions:
            deltas = []
            for seed in seeds:
                baseline = baseline_runs[seed]
                try:
                    intervention_run = run_nsga2(problem, intervention.apply(base_config), seed=seed)
                    normalized_intervention = normalize_objectives(intervention_run["objectives"], calibration)
                    raw_base = hypervolume(baseline["objectives"], calibration.raw_reference)
                    raw_intervention = hypervolume(intervention_run["objectives"], calibration.raw_reference)
                    norm_base = hypervolume(normalize_objectives(baseline["objectives"], calibration), calibration.normalized_reference)
                    norm_intervention = hypervolume(normalized_intervention, calibration.normalized_reference)
                    delta = norm_intervention - norm_base
                    deltas.append(delta)
                    rows.append({
                        "task_id": task["task_id"], "problem": task["problem"],
                        "family": _family(task["problem"]),
                        "intervention_id": intervention.intervention_id,
                        "category": intervention.category, "seed": seed,
                        "baseline_raw_HV": raw_base, "intervention_raw_HV": raw_intervention,
                        "baseline_normalized_HV": norm_base,
                        "intervention_normalized_HV": norm_intervention,
                        "delta_normalized_HV": delta,
                        "baseline_IGD": igd(baseline["front"], reference),
                        "intervention_IGD": igd(intervention_run["front"], reference),
                        "function_evaluations_equal": baseline["function_evaluations"] == intervention_run["function_evaluations"],
                        "baseline_status": "ok", "intervention_status": "ok",
                        "config_hash": _hash_config(task, intervention.parameters, base_config),
                    })
                except Exception as error:
                    rows.append({
                        "task_id": task["task_id"], "problem": task["problem"],
                        "family": _family(task["problem"]),
                        "intervention_id": intervention.intervention_id,
                        "category": intervention.category, "seed": seed,
                        "baseline_status": "ok", "intervention_status": f"failed:{type(error).__name__}",
                        "function_evaluations_equal": False,
                        "config_hash": _hash_config(task, intervention.parameters, base_config),
                    })
            if deltas:
                stats = classify_effect(np.asarray(deltas), tolerance=tolerance)
                stats.update({"task_id": task["task_id"], "problem": task["problem"],
                              "family": _family(task["problem"]),
                              "intervention_id": intervention.intervention_id,
                              "category": intervention.category})
                grouped[(task["task_id"], intervention.intervention_id)] = stats
        task_health = metric_health(
            np.vstack(baseline_objectives),
            baseline_raw_hv,
            baseline_norm_hv,
            calibration.valid,
        )
        task_health.update({"task_id": task["task_id"], "problem": task["problem"]})
        health_rows.append(task_health)

    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "phase1_5B_transfer_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    patterns = defaultdict(list)
    for stats in grouped.values():
        patterns[stats["intervention_id"]].append(stats["state"])
    sign_reversal = {
        intervention: {"states": states, "has_positive_and_negative": ("positive" in states and "negative" in states)}
        for intervention, states in patterns.items()
    }
    matrix_values = list(grouped.values())
    zdt_groups = [row for row in matrix_values if row["family"] == "ZDT"]
    dtlz_groups = [row for row in matrix_values if row["family"] == "DTLZ"]
    train_by_intervention = defaultdict(list)
    for row in zdt_groups:
        train_by_intervention[row["intervention_id"]].append(row["mean"])
    heldout_predictions = []
    for row in dtlz_groups:
        prediction = float(np.mean(train_by_intervention[row["intervention_id"]]))
        heldout_predictions.append({"intervention_id": row["intervention_id"],
                                    "task_id": row["task_id"],
                                    "predicted_delta": prediction,
                                    "actual_delta": row["mean"],
                                    "actual_state": row["state"]})
    health_report = {"tasks": health_rows,
                     "aggregate_invalid_task_count": int(sum(item["invalid"] for item in health_rows)),
                     "zero_HV_ratio": float(np.mean([item["zero_HV_ratio"] for item in health_rows])),
                     "normalization_valid": bool(all(item["normalization_valid"] for item in health_rows))}
    (output_dir / "metric_health_report.json").write_text(json.dumps(health_report, indent=2), encoding="utf-8")
    hv_report = {"tasks": calibrations, "before": _load_before_metrics(),
                 "after": {"normalized_HV_reference": 1.1, "task_count": len(calibrations)}}
    (output_dir / "phase1_5B_hv_report.json").write_text(json.dumps(hv_report, indent=2), encoding="utf-8")
    summary = {
        "pilot": False, "task_count": len(config["tasks"]), "seed_count": len(seeds),
        "population_size": base_config.population_size,
        "evaluation_budget": base_config.population_size * (base_config.generations + 1),
        "condition_count": len(grouped),
        "raw_row_count": len(rows),
        "positive_groups": int(sum(row["state"] == "positive" for row in matrix_values)),
        "negative_groups": int(sum(row["state"] == "negative" for row in matrix_values)),
        "neutral_groups": int(sum(row["state"] == "neutral" for row in matrix_values)),
        "interventions_with_sign_reversal": int(sum(item["has_positive_and_negative"] for item in sign_reversal.values())),
        "sign_reversal_fraction": float(np.mean([item["has_positive_and_negative"] for item in sign_reversal.values()])) if sign_reversal else 0.0,
        "heldout_transfer_signal": heldout_predictions,
        "metric_health": health_report,
        "go_no_go": "NO-GO",
        "go_no_go_reason": "Phase 1.5B establishes calibrated paired data; Phase 2 requires held-out predictor validation and stable sign reversals.",
    }
    (output_dir / "phase1_5B_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

if __name__ == "__main__":
    run()

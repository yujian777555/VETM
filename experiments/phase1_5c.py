"""Phase 1.5C：健康任务校准、迁移矩阵和简单可预测性基线。"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vetm.interventions import active_interventions
from vetm.metric_calibration import calibrate_task, metric_health, normalize_objectives
from vetm.metrics import auroc, evaluate_predictions
from vetm.nsga2 import NSGA2Config, run_nsga2
from vetm.problems import get_problem, reference_front
from vetm.transfer_matrix import classify_effect, hypervolume, igd, unit_hypervolume
from vetm.validators import MLPValidityPredictor, NearestNeighborValidator

def family(name: str) -> str:
    return "DTLZ" if name.startswith("DTLZ") else "ZDT"

def task_features(task: dict, budget: int, intervention, population_size: int = 50) -> list[float]:
    p = intervention.parameters
    return [float(task["n_var"]), float(task["n_obj"]), float(population_size), float(budget),
            float(p.get("mutation_probability", 1.0 / task["n_var"])), float(p.get("eta_m", 20.0)),
            float(p.get("crossover_probability", .9)), float(p.get("eta_c", 15.0)),
            float(p.get("tournament_size", 2))]

def calibrate(tasks, config):
    reports, selected = {}, {}
    for task in tasks:
        problem = get_problem(task["problem"], n_var=task["n_var"], n_obj=task["n_obj"])
        calibration = calibrate_task(problem)
        reference = reference_front(problem)
        candidates = []
        for budget in config["candidate_budgets"]:
            generations = budget // config["population_size"] - 1
            if generations < 1 or budget % config["population_size"]:
                continue
            raw, normalized, objectives, igds = [], [], [], []
            for seed in range(config["calibration_seeds"]):
                result = run_nsga2(problem, NSGA2Config(population_size=config["population_size"], generations=generations), seed=seed)
                norm = normalize_objectives(result["objectives"], calibration)
                raw.append(hypervolume(result["objectives"], calibration.raw_reference))
                normalized.append(unit_hypervolume(norm, calibration.normalized_reference))
                objectives.append(result["objectives"])
                igds.append(igd(result["front"], reference))
            health = metric_health(np.vstack(objectives), raw, normalized, calibration.valid)
            health.update({"budget": budget, "mean_unit_HV": float(np.mean(normalized)), "std_unit_HV": float(np.std(normalized)), "mean_IGD": float(np.mean(igds)), "saturated": bool(np.ptp(normalized) < 1e-12)})
            candidates.append(health)
            if not health["invalid"] and not health["saturated"]:
                selected[task["task_id"]] = budget
                break
        reports[task["task_id"]] = {"task": task, "candidates": candidates, "selected_budget": selected.get(task["task_id"]), "supported": task["task_id"] in selected}
    return reports, selected

def ridge_fit_predict(train_x, train_y, test_x, alpha=1e-2):
    mean, scale = train_x.mean(0), train_x.std(0); scale[scale < 1e-12] = 1
    x = (train_x - mean) / scale; z = (test_x - mean) / scale
    w = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ train_y)
    return z @ w

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--max-healthy-tasks", type=int, default=20); parser.add_argument("--config", default="configs/phase1_5c_tasks.json"); parser.add_argument("--results", default="results")
    args = parser.parse_args(); results_dir = ROOT / args.results; results_dir.mkdir(exist_ok=True)
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8")); tasks = config["tasks"]
    health_reports, selected = calibrate(tasks, config)
    (results_dir / "phase1_5C_metric_health.json").write_text(json.dumps(health_reports, indent=2), encoding="utf-8")
    healthy_tasks = [t for t in tasks if t["task_id"] in selected][:args.max_healthy_tasks]
    rows, summaries = [], []
    for task in healthy_tasks:
        budget = selected[task["task_id"]]; generations = budget // config["population_size"] - 1
        problem = get_problem(task["problem"], n_var=task["n_var"], n_obj=task["n_obj"]); cal = calibrate_task(problem); ref = reference_front(problem)
        base_runs = {seed: run_nsga2(problem, NSGA2Config(population_size=config["population_size"], generations=generations), seed=seed) for seed in range(config["matrix_seeds"])}
        for intervention in active_interventions(NSGA2Config(population_size=config["population_size"], generations=generations), task["n_var"]):
            deltas, delta_igd = [], []
            for seed, base in base_runs.items():
                try:
                    result = run_nsga2(problem, intervention.apply(NSGA2Config(population_size=config["population_size"], generations=generations)), seed=seed)
                    bnorm = normalize_objectives(base["objectives"], cal); tnorm = normalize_objectives(result["objectives"], cal)
                    base_hv = unit_hypervolume(bnorm, cal.normalized_reference); transfer_hv = unit_hypervolume(tnorm, cal.normalized_reference); delta = transfer_hv - base_hv
                    di = igd(result["front"], ref) - igd(base["front"], ref); deltas.append(delta); delta_igd.append(di)
                    rows.append({"task_id":task["task_id"], "problem":task["problem"], "family":family(task["problem"]), "n_var":task["n_var"], "n_obj":task["n_obj"], "budget":budget, "intervention_id":intervention.intervention_id, "category":intervention.category, "seed":seed, "baseline_raw_HV":hypervolume(base["objectives"], cal.raw_reference), "intervention_raw_HV":hypervolume(result["objectives"], cal.raw_reference), "baseline_HV_unit":base_hv, "intervention_HV_unit":transfer_hv, "delta_HV_unit":delta, "delta_IGD":di, "function_evaluations_equal":base["function_evaluations"] == result["function_evaluations"], "status":"ok", "config_hash":hashlib.sha256(json.dumps({"task":task,"intervention":intervention.parameters},sort_keys=True).encode()).hexdigest()[:16]})
                except Exception as error:
                    rows.append({"task_id":task["task_id"], "intervention_id":intervention.intervention_id, "seed":seed, "status":f"failed:{type(error).__name__}"})
            if deltas:
                effect = classify_effect(np.asarray(deltas), config["practical_tolerance"]); effect.update({"task_id":task["task_id"], "problem":task["problem"], "family":family(task["problem"]), "n_var":task["n_var"], "n_obj":task["n_obj"], "budget":budget, "intervention_id":intervention.intervention_id, "category":intervention.category, "mean_delta_IGD":float(np.mean(delta_igd)), "igd_agreement":bool((effect["state"] == "positive" and np.mean(delta_igd) < 0) or (effect["state"] == "negative" and np.mean(delta_igd) > 0) or effect["state"] == "neutral" )}); summaries.append(effect)
    fieldnames = sorted({k for r in rows for k in r})
    with (results_dir / "phase1_5C_transfer_matrix.csv").open("w", newline="", encoding="utf-8") as f: w=csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    with (results_dir / "phase1_5C_condition_summary.csv").open("w", newline="", encoding="utf-8") as f: w=csv.DictWriter(f, fieldnames=sorted(summaries[0])); w.writeheader(); w.writerows(summaries)
    # Family-held-out simple baselines on aggregated conditions, excluding invalid tasks by construction.
    train=[s for s in summaries if s["family"] == "ZDT"]; test=[s for s in summaries if s["family"] == "DTLZ"]
    if train and test:
        lookup=defaultdict(list)
        for item in train: lookup[item["intervention_id"]].append(item["mean"])
        train_x=np.asarray([task_features(item, item["budget"], next(i for i in active_interventions(NSGA2Config(population_size=50,generations=item["budget"]//50-1),item["n_var"]) if i.intervention_id==item["intervention_id"])) for item in train],float); train_y=np.asarray([item["mean"] for item in train])
        test_x=np.asarray([task_features(item,item["budget"],next(i for i in active_interventions(NSGA2Config(population_size=50,generations=item["budget"]//50-1),item["n_var"]) if i.intervention_id==item["intervention_id"])) for item in test],float); test_y=np.asarray([item["mean"] for item in test])
        global_pred=np.asarray([np.mean(lookup[item["intervention_id"]]) for item in test]); ridge_pred=ridge_fit_predict(train_x,train_y,test_x)
        prediction = {
            "train_conditions": len(train),
            "test_conditions": len(test),
            "global_intervention_mean_mae": float(np.mean(abs(global_pred - test_y))),
            "ridge_mae": float(np.mean(abs(ridge_pred - test_y))),
            "zero_gain_mae": float(np.mean(abs(test_y))),
        }
    else: prediction={"available":False}
    summary={"healthy_task_count":len(healthy_tasks),"unsupported_task_count":len(tasks)-len(selected),"selected_budgets":selected,"matrix_rows":len(rows),"condition_count":len(summaries),"positive_groups":sum(s["state"]=="positive" for s in summaries),"negative_groups":sum(s["state"]=="negative" for s in summaries),"confirmed_igd_agreement":sum(s["igd_agreement"] for s in summaries),"predictability":prediction,"go_no_go":"NO-GO"}
    (results_dir / "phase1_5C_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
if __name__ == "__main__": main()

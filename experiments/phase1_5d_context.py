"""生成 dedicated probe seeds 100--104 的 early dynamic task context。"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vetm.metric_calibration import calibrate_task, normalize_objectives
from vetm.nsga2 import NSGA2Config, run_nsga2
from vetm.problems import get_problem, reference_front
from vetm.transfer_matrix import igd, unit_hypervolume

def main():
    rows = list(csv.DictReader((ROOT / "results" / "phase1_5C_R1_transfer_matrix.csv").open(encoding="utf-8")))
    tasks = {}
    for row in rows:
        tasks[row["task_id"]] = {
            "task_id": row["task_id"], "problem": row["problem"],
            "n_var": int(row["n_var"]), "n_obj": int(row["n_obj"]),
            "budget": int(row["budget"]),
        }
    context_rows, static_rows = [], []
    for task in tasks.values():
        problem = get_problem(task["problem"], n_var=task["n_var"], n_obj=task["n_obj"])
        calibration = calibrate_task(problem); reference = reference_front(problem)
        probe_budget = min(1000, max(400, int(0.2 * task["budget"]) // 50 * 50))
        probe_cfg = NSGA2Config(population_size=50, generations=probe_budget // 50 - 1)
        probe_features = []
        for seed in range(100, 105):
            result = run_nsga2(problem, probe_cfg, seed=seed)
            hvs, igds, nds, spreads = [], [], [], []
            for front, history in zip(result["trajectory_fronts"], result["history"]):
                normalized = normalize_objectives(front, calibration)
                hvs.append(unit_hypervolume(normalized, calibration.normalized_reference))
                igds.append(igd(front, reference))
                nds.append(float(history["nondominated_count"]) / history["population_size"])
                spreads.append(float(np.mean(np.std(front, axis=0))) if len(front) else 0.0)
            improvement = np.diff(hvs)
            probe_features.append({
                "hv_start": hvs[0], "hv_end": hvs[-1], "hv_slope": (hvs[-1] - hvs[0]) / max(1, len(hvs)-1),
                "hv_auc": float(np.mean(hvs)), "igd_start": igds[0], "igd_end": igds[-1],
                "igd_slope": (igds[-1] - igds[0]) / max(1, len(igds)-1),
                "nd_ratio_start": nds[0], "nd_ratio_end": nds[-1],
                "nd_ratio_slope": (nds[-1] - nds[0]) / max(1, len(nds)-1),
                "spread_start": spreads[0], "spread_end": spreads[-1],
                "stagnation_rate": float(np.mean(improvement <= 1e-12)) if len(improvement) else 1.0,
            })
        aggregate = {"task_id": task["task_id"], "problem": task["problem"], "n_var": task["n_var"],
                     "n_obj": task["n_obj"], "budget": task["budget"], "probe_budget": probe_budget}
        for key in probe_features[0]:
            values = np.asarray([item[key] for item in probe_features])
            aggregate[key] = float(values.mean()); aggregate[key + "_std"] = float(values.std(ddof=1))
        context_rows.append(aggregate)
        lower, upper = np.asarray(problem.lower), np.asarray(problem.upper)
        static_rows.append({"task_id": task["task_id"], "n_var": task["n_var"], "n_obj": task["n_obj"],
                            "budget": task["budget"], "population_size": 50,
                            "lower_min": float(lower.min()), "lower_max": float(lower.max()),
                            "lower_mean": float(lower.mean()), "lower_std": float(lower.std()),
                            "upper_min": float(upper.min()), "upper_max": float(upper.max()),
                            "upper_mean": float(upper.mean()), "upper_std": float(upper.std()),
                            "bound_width_mean": float((upper-lower).mean()),
                            "bound_width_std": float((upper-lower).std())})
    for name, data in [("phase1_5D_task_context.csv", context_rows), ("phase1_5D_static_features.csv", static_rows)]:
        with (ROOT / "results" / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(data[0])); writer.writeheader(); writer.writerows(data)
    print(json.dumps({"tasks": len(tasks), "probe_seeds": 5}, indent=2))

if __name__ == "__main__":
    main()

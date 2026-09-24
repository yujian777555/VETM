"""运行 Phase 1.5 的小规模 paired transfer matrix pilot。"""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vetm.interventions import intervention_registry
from vetm.nsga2 import NSGA2Config, run_nsga2
from vetm.problems import get_problem, reference_front
from vetm.transfer_matrix import classify_effect, hypervolume, igd

def task_config(row, generations):
    return NSGA2Config(population_size=row["population_size"], generations=generations)

def run(pilot_tasks=8, pilot_interventions=8, seeds=(0,1,2), out=Path("results")):
    cfg=json.loads(Path("configs/phase1_5_tasks.json").read_text(encoding="utf-8"))
    tasks=cfg["tasks"][:pilot_tasks]; interventions=intervention_registry()[:pilot_interventions]
    out.mkdir(parents=True,exist_ok=True); rows=[]; grouped={}
    for task in tasks:
        problem=get_problem(task["problem"],n_var=task["n_var"],n_obj=task["n_obj"])
        ref=np.asarray(task.get("reference_point", [1.2]*task["n_obj"]),dtype=float)
        front=reference_front(problem)
        for intervention in interventions:
            deltas=[]; raw=[]
            for seed in seeds:
                base_cfg=task_config(task,cfg["generations"])
                base=run_nsga2(problem,base_cfg,seed=seed)
                intervention_run=run_nsga2(problem,intervention.apply(base_cfg),seed=seed)
                base_hv=hypervolume(base["objectives"],ref); int_hv=hypervolume(intervention_run["objectives"],ref)
                base_igd=igd(base["front"],front); int_igd=igd(intervention_run["front"],front)
                delta= int_hv-base_hv; deltas.append(delta)
                raw.append({"task_id":task["task_id"],"intervention_id":intervention.intervention_id,"seed":seed,"baseline_hv":base_hv,"intervention_hv":int_hv,"delta_hv":delta,"baseline_igd":base_igd,"intervention_igd":int_igd,"function_evaluations_equal":base["function_evaluations"]==intervention_run["function_evaluations"],"config_hash":hashlib.sha256(json.dumps({"task":task,"intervention":intervention.parameters},sort_keys=True).encode()).hexdigest()[:12]})
            stats=classify_effect(np.asarray(deltas)); stats.update({"task_id":task["task_id"],"intervention_id":intervention.intervention_id,"category":intervention.category})
            grouped[(task["task_id"],intervention.intervention_id)]=stats; rows.extend(raw)
    with (out/"phase1_5_transfer_matrix.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=sorted(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/"phase1_5_summary.json").write_text(json.dumps({"pilot":True,"task_count":len(tasks),"intervention_count":len(interventions),"seed_count":len(seeds),"conditions":len(grouped),"groups":list(grouped.values())},indent=2),encoding="utf-8")
    return grouped
if __name__=="__main__":
    run()

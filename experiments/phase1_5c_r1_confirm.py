"""对 R1 候选 reversal 条件补跑 seed 10--19。"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vetm.interventions import active_interventions
from vetm.metric_calibration import calibrate_task, normalize_objectives
from vetm.nsga2 import NSGA2Config, run_nsga2
from vetm.problems import get_problem, reference_front
from vetm.transfer_matrix import classify_effect, hypervolume, unit_hypervolume, igd

def main():
    results=ROOT/"results"
    summary=json.loads((results/"phase1_5C_R1_summary.json").read_text())
    candidates=set(summary["reversals"]["within_family"]+summary["reversals"]["cross_family"])
    conditions=list(csv.DictReader((results/"phase1_5C_R1_condition_summary.csv").open(encoding="utf-8")))
    conditions=[row for row in conditions if row["intervention_id"] in candidates and row["state"]!="neutral"]
    tasks={t["task_id"]:t for t in json.loads((ROOT/"configs/phase1_5c_tasks.json").read_text())["tasks"]}
    tasks.update({t["task_id"]:t for t in json.loads((ROOT/"configs/phase1_5c_stageb_tasks.json").read_text())["tasks"]})
    rows=[]; groups={}
    for row in conditions:
        task=tasks[row["task_id"]]; problem=get_problem(task["problem"],n_var=task["n_var"],n_obj=task["n_obj"])
        budget=int(row["budget"]); cfg=NSGA2Config(population_size=50,generations=budget//50-1)
        cal=calibrate_task(problem); ref=reference_front(problem)
        intervention=next(i for i in active_interventions(cfg,task["n_var"]) if i.intervention_id==row["intervention_id"])
        deltas=[]; delta_igd=[]
        for seed in range(10,20):
            base=run_nsga2(problem,cfg,seed=seed)
            inter=run_nsga2(problem,intervention.apply(cfg),seed=seed)
            bnorm=normalize_objectives(base["objectives"],cal); tnorm=normalize_objectives(inter["objectives"],cal)
            b=unit_hypervolume(bnorm,cal.normalized_reference); t=unit_hypervolume(tnorm,cal.normalized_reference)
            d=t-b; di=igd(inter["front"],ref)-igd(base["front"],ref); deltas.append(d); delta_igd.append(di)
            rows.append({"task_id":task["task_id"],"problem":task["problem"],"family":"DTLZ" if task["problem"].startswith("DTLZ") else "ZDT","intervention_id":intervention.intervention_id,"seed":seed,"delta_HV_unit":d,"delta_IGD":di,"function_evaluations_equal":base["function_evaluations"]==inter["function_evaluations"],"status":"ok"})
        effect=classify_effect(np.asarray(deltas),tolerance=.01)
        groups[f"{task['task_id']}::{intervention.intervention_id}"]={**effect,"task_id":task["task_id"],"intervention_id":intervention.intervention_id,"mean_delta_IGD":float(np.mean(delta_igd))}
    out=results/"phase1_5C_R1_reversal_confirmation.csv"
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=sorted(rows[0])); w.writeheader(); w.writerows(rows)
    (results/"phase1_5C_R1_reversal_confirmation.json").write_text(json.dumps({"candidate_condition_count":len(conditions),"rows":len(rows),"groups":groups},indent=2),encoding="utf-8")
    print(json.dumps({"candidate_condition_count":len(conditions),"rows":len(rows),"confirmed_groups":sum(v["state"]!="neutral" for v in groups.values())},indent=2))
if __name__=="__main__": main()

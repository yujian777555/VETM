"""对 Phase 1.5D 原模型重建 20 个任务配置级误差块。"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vetm.interventions import intervention_registry
from vetm.nsga2 import NSGA2Config

def ridge(train, target, test):
    mean=train.mean(0); scale=train.std(0); scale[scale<1e-12]=1
    x=(train-mean)/scale; z=(test-mean)/scale; center=target.mean()
    return center+z@np.linalg.solve(x.T@x+np.eye(x.shape[1]),x.T@(target-center))

def boot(x, seed=42, n=5000):
    x=np.asarray(x,float); rng=np.random.default_rng(seed)
    samples=x[rng.integers(0,len(x),size=(n,len(x)))].mean(1)
    return {"mean":float(x.mean()),"ci95_low":float(np.quantile(samples,.025)),
            "ci95_high":float(np.quantile(samples,.975)),"n_task_blocks":len(x),"resamples":n}

def run():
    results=ROOT/"results"
    with (results/"phase1_5C_R1_condition_summary.csv").open(encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    with (results/"phase1_5D_task_context.csv").open(encoding="utf-8") as handle:
        dynamic={r["task_id"]:r for r in csv.DictReader(handle)}
    with (results/"phase1_5D_static_features.csv").open(encoding="utf-8") as handle:
        static={r["task_id"]:r for r in csv.DictReader(handle)}
    registry={item.intervention_id:item for item in intervention_registry()}
    static_keys=[k for k in next(iter(static.values())) if k!="task_id"]
    dynamic_keys=[k for k in next(iter(dynamic.values())) if k not in ("task_id","problem")]
    intervention=[]
    for row in rows:
        cfg=NSGA2Config(population_size=50,generations=int(row["budget"])//50-1)
        desc=registry[row["intervention_id"]].descriptor(cfg,int(row["n_var"]))
        intervention.append(
            [float(desc["category"]==c) for c in ("mutation","crossover","selection")]
            +[float(desc["parameter"]==p) for p in ("mutation_probability","eta_m","crossover_probability","eta_c","tournament_size")]
            +[desc["absolute_value"],desc["signed_change"],desc["relative_magnitude"]]
        )
    inter=np.asarray(intervention,float)
    stat=np.asarray([[float(static[r["task_id"]][k]) for k in static_keys] for r in rows],float)
    dyn=np.asarray([[float(dynamic[r["task_id"]][k]) for k in dynamic_keys] for r in rows],float)
    y=np.asarray([float(r["mean"]) for r in rows],float)
    indices_by_problem=defaultdict(list)
    for i,row in enumerate(rows): indices_by_problem[row["problem"]].append(i)
    predictions={key:np.full(len(rows),np.nan) for key in ("static_dynamic","intervention_only","global")}
    for problem,test_list in indices_by_problem.items():
        test=np.asarray(test_list); train=np.asarray([i for i in range(len(rows)) if i not in set(test_list)])
        predictions["static_dynamic"][test]=ridge(np.c_[stat[train],dyn[train],inter[train]],y[train],
                                                  np.c_[stat[test],dyn[test],inter[test]])
        predictions["intervention_only"][test]=ridge(inter[train],y[train],inter[test])
        historical=defaultdict(list)
        for i in train: historical[rows[i]["intervention_id"]].append(y[i])
        predictions["global"][test]=[
            np.mean(historical[rows[i]["intervention_id"]]) if historical[rows[i]["intervention_id"]] else y[train].mean()
            for i in test
        ]
    per_task={}
    for task_id in sorted({row["task_id"] for row in rows}):
        idx=np.asarray([i for i,row in enumerate(rows) if row["task_id"]==task_id])
        per_task[task_id]={
            "static_dynamic_mae":float(np.mean(abs(predictions["static_dynamic"][idx]-y[idx]))),
            "zero_mae":float(np.mean(abs(y[idx]))),
            "intervention_only_mae":float(np.mean(abs(predictions["intervention_only"][idx]-y[idx]))),
            "global_mae":float(np.mean(abs(predictions["global"][idx]-y[idx]))),
        }
    comparisons={
        name:boot([row["static_dynamic_mae"]-row[name+"_mae"] for row in per_task.values()])
        for name in ("zero","intervention_only","global")
    }
    out={"status":"diagnostic_only","source_phase":"Phase 1.5D","task_configuration_blocks":len(per_task),
         "per_task":per_task,"comparisons":comparisons,
         "warning":"只修正统计单位；原 Phase 1.5D 特征与模型选择保持不变，包含当时的 oracle 特征，不可用于 Phase 1.5E 输入。"}
    (results/"phase1_5D_R1_taskblock_audit.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"task_blocks":len(per_task),"comparisons":comparisons},indent=2))

if __name__=="__main__":run()
